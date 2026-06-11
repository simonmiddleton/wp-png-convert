#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "Pillow>=10.0.0",
#     "pymysql>=1.1.0",
#     "rich>=13.0.0",
# ]
# ///

"""Convert PNG, BMP, and TIFF WordPress uploads to JPEG and update the database.

Dry-run by default — reports what would change without touching any files or the DB.
Pass --execute to apply changes.

Usage:
    uv run convert_images_to_jpg.py              # dry run
    uv run convert_images_to_jpg.py --execute    # apply changes
"""

import argparse
import re
import select
import shutil
import sys
import termios
import tty
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pymysql
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()

CONVERTIBLE_EXTS = {'.png', '.bmp', '.tif', '.tiff'}

# Matches WordPress-generated size variants: name-WxH.ext or name-scaled.ext
VARIANT_RE = re.compile(r'(-\d+x\d+|-scaled)\.[^.]+$', re.IGNORECASE)

# All MIME types that map to JPEG
SOURCE_MIMES = {
    'image/png', 'image/bmp', 'image/x-bmp', 'image/x-ms-bmp',
    'image/tiff', 'image/x-tiff',
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageGroup:
    """One master image and all its WordPress-generated size variants."""
    master: Path
    variants: list[Path] = field(default_factory=list)

    @property
    def all_files(self) -> list[Path]:
        return [self.master] + self.variants

    @property
    def original_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.all_files if f.exists())


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def parse_wp_config(wp_root: Path) -> dict:
    """Extract DB credentials and table prefix from wp-config.php files."""
    config: dict = {}
    define_re = re.compile(r"define\s*\(\s*['\"](\w+)['\"]\s*,\s*['\"]([^'\"]*)['\"]")
    prefix_re = re.compile(r"\$table_prefix\s*=\s*['\"]([^'\"]+)['\"]")

    # Local config takes priority (it overrides the main config)
    for cfg_path in [wp_root / 'local' / 'wp-config.php', wp_root / 'wp-config.php']:
        if not cfg_path.exists():
            continue
        text = cfg_path.read_text()
        for m in define_re.finditer(text):
            key, val = m.group(1), m.group(2)
            if key not in config:
                config[key] = val
        if 'table_prefix' not in config:
            if pm := prefix_re.search(text):
                config['table_prefix'] = pm.group(1)

    config.setdefault('table_prefix', 'wp_')
    return config


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def is_variant(path: Path) -> bool:
    return bool(VARIANT_RE.search(path.name))


def find_image_groups(uploads_dir: Path) -> list[ImageGroup]:
    """Find all convertible images and group variants with their master."""
    all_images: list[Path] = []
    for ext in CONVERTIBLE_EXTS:
        all_images.extend(uploads_dir.rglob(f'*{ext}'))
        all_images.extend(uploads_dir.rglob(f'*{ext.upper()}'))
    all_images = sorted(set(all_images))

    masters = {p: ImageGroup(master=p) for p in all_images if not is_variant(p)}
    orphans: list[Path] = []

    for variant in (p for p in all_images if is_variant(p)):
        # Strip variant suffix (e.g. -150x150.png) to find the master basename
        base_name = VARIANT_RE.sub('', variant.name)
        parent = variant.parent
        matched = False
        for ext in CONVERTIBLE_EXTS:
            for candidate in [parent / (base_name + ext), parent / (base_name + ext.upper())]:
                if candidate in masters:
                    masters[candidate].variants.append(variant)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            orphans.append(variant)

    if orphans:
        console.print(f'[yellow]Warning: {len(orphans)} variant(s) have no master — will convert but not back up:[/yellow]')
        for o in orphans:
            console.print(f'  [dim]{o}[/dim]')

    # Orphaned variants become singleton groups (no backup needed)
    result = list(masters.values())
    result += [ImageGroup(master=o) for o in orphans]
    return result


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

def backup_master(master: Path, backup_dir: Path, uploads_dir: Path, dry_run: bool) -> Path:
    rel = master.relative_to(uploads_dir)
    dest = backup_dir / rel
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(master, dest)
    return dest


def convert_to_jpg(src: Path, quality: int, dry_run: bool) -> tuple[int, int]:
    """Convert src to JPEG alongside the original. Returns (orig_bytes, new_bytes)."""
    orig_bytes = src.stat().st_size
    dst = src.with_suffix('.jpg')

    if dry_run:
        return orig_bytes, 0

    if dst.exists():
        console.print(f'    [yellow]Skipped (JPG already exists): {dst.name}[/yellow]')
        return orig_bytes, dst.stat().st_size

    img = Image.open(src)
    # Flatten transparency onto white before converting to JPEG
    if img.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        converted = img.convert('RGBA') if img.mode == 'P' else img
        bg.paste(converted, mask=converted.split()[-1])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    img.save(dst, 'JPEG', quality=quality, optimize=True)
    return orig_bytes, dst.stat().st_size


def regenerate_variant_from_master(variant: Path, master: Path, quality: int, dry_run: bool) -> tuple[int, int]:
    """Re-create a truncated size variant by resizing the master image.
    Deletes the truncated source on success."""
    orig_bytes = variant.stat().st_size
    dst = variant.with_suffix('.jpg')

    if dry_run:
        return orig_bytes, 0

    # Get target dimensions from filename (e.g. photo-150x150.png → 150×150).
    # For -scaled variants the dimensions aren't in the name, so try reading
    # the truncated header; fall back to WordPress's default threshold.
    m = re.search(r'-(\d+)x(\d+)\.[^.]+$', variant.name)
    if m:
        target_w, target_h = int(m.group(1)), int(m.group(2))
    else:
        from PIL import ImageFile as _PIF
        _PIF.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(variant) as tmp:
                target_w, target_h = tmp.size
        except Exception:
            target_w, target_h = 2560, 2560  # WordPress big_image_size_threshold default
        finally:
            _PIF.LOAD_TRUNCATED_IMAGES = False

    img = Image.open(master)
    if img.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        converted = img.convert('RGBA') if img.mode == 'P' else img
        bg.paste(converted, mask=converted.split()[-1])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    img.save(dst, 'JPEG', quality=quality, optimize=True)
    variant.unlink()
    return orig_bytes, dst.stat().st_size


# ---------------------------------------------------------------------------
# PHP serialised data helpers
# ---------------------------------------------------------------------------

def serialized_replace(data: str, old: str, new: str) -> str:
    """Replace a string value inside PHP serialised data, fixing the s:N: length prefix."""
    old_token = f's:{len(old.encode())}:"{old}"'
    new_token = f's:{len(new.encode())}:"{new}"'
    return data.replace(old_token, new_token)


def build_serialized_replacements(group: ImageGroup, uploads_dir: Path) -> list[tuple[str, str]]:
    """
    Return (old, new) pairs for every string that needs updating in
    _wp_attachment_metadata for this group.
    """
    pairs: list[tuple[str, str]] = []

    master = group.master
    ext = master.suffix.lower()

    # Full relative path stored in the top-level "file" key
    rel_path = str(master.relative_to(uploads_dir))
    pairs.append((rel_path, str(Path(rel_path).with_suffix('.jpg'))))

    # Each filename (basename only) stored inside the "sizes" sub-array
    for f in group.all_files:
        if f.suffix.lower() != '.jpg':
            pairs.append((f.name, f.with_suffix('.jpg').name))

    # MIME type strings inside each size entry
    if ext in ('.png',):
        pairs.append(('image/png', 'image/jpeg'))
    elif ext in ('.bmp',):
        pairs += [('image/bmp', 'image/jpeg'), ('image/x-bmp', 'image/jpeg'),
                  ('image/x-ms-bmp', 'image/jpeg')]
    elif ext in ('.tif', '.tiff'):
        pairs += [('image/tiff', 'image/jpeg'), ('image/x-tiff', 'image/jpeg')]

    return pairs


# ---------------------------------------------------------------------------
# Database updater
# ---------------------------------------------------------------------------

class DatabaseUpdater:
    def __init__(self, config: dict, dry_run: bool):
        self.dry_run = dry_run
        self.prefix = config.get('table_prefix', 'wp_')
        self._conn = None
        self._cur = None
        self._post_content_replacements: list[tuple[str, str]] = []

        if not dry_run:
            self._conn = pymysql.connect(
                host=config.get('DB_HOST', '127.0.0.1'),
                user=config['DB_USER'],
                password=config['DB_PASSWORD'],
                database=config['DB_NAME'],
                charset='utf8mb4',
                autocommit=False,
            )
            self._cur = self._conn.cursor()

    # ------------------------------------------------------------------
    def _run(self, sql: str, args: tuple = ()) -> int:
        if self.dry_run or self._cur is None:
            return 0
        self._cur.execute(sql, args) if args else self._cur.execute(sql)
        return self._cur.rowcount

    def _fetchall(self, sql: str, args: tuple = ()) -> list:
        if self.dry_run or self._cur is None:
            return []
        self._cur.execute(sql, args) if args else self._cur.execute(sql)
        return self._cur.fetchall()

    # ------------------------------------------------------------------
    def update_for_group(self, group: ImageGroup, uploads_dir: Path) -> int:
        """Update guid, mime type, and attachment metadata for one converted group.
        Post_content updates are queued here and applied in a single pass by flush_post_content()."""
        master = group.master
        rel_path = str(master.relative_to(uploads_dir))
        new_rel_path = str(Path(rel_path).with_suffix('.jpg'))
        old_name = master.name
        new_name = master.with_suffix('.jpg').name

        # Queue every filename in this group for the post_content batch pass
        for f in group.all_files:
            self._post_content_replacements.append((f.name, f.with_suffix('.jpg').name))

        if self.dry_run:
            return 1  # placeholder so callers can count groups

        # 1. Resolve post_id via exact match on _wp_attached_file — uses meta_key index,
        #    no LIKE scan needed since we know the exact relative path stored by WordPress.
        rows = self._fetchall(
            f"SELECT post_id FROM {self.prefix}postmeta"
            f" WHERE meta_key = '_wp_attached_file' AND meta_value = %s LIMIT 1",
            (rel_path,),
        )
        if not rows:
            # Older WP versions used _wp_attachment_file; try that too
            rows = self._fetchall(
                f"SELECT post_id FROM {self.prefix}postmeta"
                f" WHERE meta_key = '_wp_attachment_file' AND meta_value = %s LIMIT 1",
                (rel_path,),
            )
        if not rows:
            console.print(f'  [yellow]No attachment post found for {rel_path} — skipping guid/metadata[/yellow]')
            return 0

        post_id = rows[0][0]
        affected = 0

        # 2. Update guid and mime_type using primary key — no table scan
        affected += self._run(
            f"UPDATE {self.prefix}posts"
            f" SET guid = REPLACE(guid, %s, %s), post_mime_type = 'image/jpeg'"
            f" WHERE ID = %s",
            (old_name, new_name, post_id),
        )

        # 3. Update _wp_attached_file by post_id — indexed lookup
        affected += self._run(
            f"UPDATE {self.prefix}postmeta SET meta_value = %s"
            f" WHERE post_id = %s AND meta_key = '_wp_attached_file'",
            (new_rel_path, post_id),
        )

        # 4. Update _wp_attachment_metadata by post_id — indexed lookup, serialised PHP
        meta_rows = self._fetchall(
            f"SELECT meta_value FROM {self.prefix}postmeta"
            f" WHERE post_id = %s AND meta_key = '_wp_attachment_metadata'",
            (post_id,),
        )
        replacements = build_serialized_replacements(group, uploads_dir)
        for (meta_value,) in meta_rows:
            updated = meta_value
            for old, new in replacements:
                updated = serialized_replace(updated, old, new)
            if updated != meta_value:
                affected += self._run(
                    f"UPDATE {self.prefix}postmeta SET meta_value = %s"
                    f" WHERE post_id = %s AND meta_key = '_wp_attachment_metadata'",
                    (updated, post_id),
                )

        return affected

    def flush_post_content(self) -> int:
        """Single-pass post_content update across all live posts (revisions excluded).
        Fetches every candidate post once, applies all queued replacements in Python,
        then writes back only the rows that actually changed."""
        if self.dry_run or not self._post_content_replacements:
            return 0

        rows = self._fetchall(
            f"""SELECT ID, post_content
                  FROM {self.prefix}posts
                 WHERE post_type   NOT IN ('revision', 'attachment')
                   AND post_status NOT IN ('auto-draft', 'trash')
                   AND (   post_content LIKE '%.png%'
                        OR post_content LIKE '%.bmp%'
                        OR post_content LIKE '%.tif%')"""
        )

        affected = 0
        for post_id, content in rows:
            updated = content
            for old, new in self._post_content_replacements:
                if old in updated:
                    updated = updated.replace(old, new)
            if updated != content:
                self._run(
                    f"UPDATE {self.prefix}posts SET post_content = %s WHERE ID = %s",
                    (updated, post_id),
                )
                affected += 1

        return affected

    def commit(self):
        if self._conn:
            self._conn.commit()

    def close(self):
        if self._cur:
            self._cur.close()
        if self._conn:
            self._conn.close()


# ---------------------------------------------------------------------------
# Keypress detection
# ---------------------------------------------------------------------------

@contextmanager
def raw_stdin():
    """Put stdin in cbreak mode so individual keypresses are available immediately.
    No-op when stdin is not a TTY (e.g. piped output)."""
    if not sys.stdin.isatty():
        yield
        return
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def keypress_pending() -> bool:
    """Return True if a key has been pressed (non-blocking)."""
    if not sys.stdin.isatty():
        return False
    return bool(select.select([sys.stdin], [], [], 0)[0])


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_bytes(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if abs(n) < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024  # type: ignore[assignment]
    return f'{n:.1f} TB'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert PNG/BMP/TIFF WordPress uploads to JPEG and update the database.',
        epilog='Dry-run by default. Pass --execute to apply changes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--execute', action='store_true',
        help='Apply conversions and database updates (default: dry run)',
    )
    parser.add_argument(
        '--wp-root', type=Path, default=Path('.'),
        help='WordPress root directory (default: current directory)',
    )
    parser.add_argument(
        '--uploads-dir', type=Path, default=None,
        help='Uploads directory (default: <wp-root>/wp-content/uploads)',
    )
    parser.add_argument(
        '--backup-dir', type=Path, default=None,
        help='Directory for original master backups (default: <wp-root>/image-backups)',
    )
    parser.add_argument(
        '--quality', type=int, default=90, metavar='1-95',
        help='JPEG quality (default: 90)',
    )
    args = parser.parse_args()

    wp_root    = args.wp_root.resolve()
    uploads    = (args.uploads_dir  or wp_root / 'wp-content' / 'uploads').resolve()
    backup_dir = (args.backup_dir   or wp_root / 'image-backups').resolve()
    dry_run    = not args.execute

    if dry_run:
        console.rule('[bold yellow]DRY RUN — no files or database rows will be changed[/bold yellow]')
        console.print('Pass [bold]--execute[/bold] to apply.\n')
    else:
        console.rule('[bold red]EXECUTE MODE — changes will be written[/bold red]')
        console.print(f'Backups → [bold]{backup_dir}[/bold]\n')

    console.print(f'Scanning [cyan]{uploads}[/cyan] …')
    groups = find_image_groups(uploads)

    if not groups:
        console.print('[green]No convertible images found.[/green]')
        return

    n_masters  = sum(1 for g in groups if not is_variant(g.master))
    n_variants = sum(len(g.variants) for g in groups)
    n_total    = len(groups) + n_variants   # groups includes orphan singletons

    console.print(
        f'Found [bold]{n_total}[/bold] file(s) to convert '
        f'([bold]{n_masters}[/bold] masters + [bold]{n_variants}[/bold] variants)\n'
    )

    wp_config = parse_wp_config(wp_root)
    if not dry_run and 'DB_USER' not in wp_config:
        console.print('[red]Could not read DB credentials from wp-config.php — aborting.[/red]')
        sys.exit(1)

    db = DatabaseUpdater(wp_config, dry_run)

    total_orig    = 0
    total_new     = 0
    total_db      = 0
    n_processed   = 0
    stopped_early = False
    errors: list[tuple[Path, str]] = []

    console.print('[dim](press any key to stop cleanly after the current image)[/dim]\n')

    with raw_stdin():
        for group in groups:
            if keypress_pending():
                sys.stdin.read(1)   # consume the keypress
                console.print('\n[yellow]Keypress detected — stopping after current image.[/yellow]')
                stopped_early = True
                break

            n_processed += 1
            master   = group.master
            is_orph  = is_variant(master)   # orphaned variant treated as master
            rel_path = master.relative_to(wp_root)

            console.print(f'[cyan]{rel_path}[/cyan]')

            # --- Backup master ---
            if not is_orph:
                backup_path = backup_master(master, backup_dir, uploads, dry_run)
                if dry_run:
                    console.print(f'  [dim]backup → {backup_path.relative_to(wp_root)}[/dim]')
                else:
                    console.print(f'  backed up → {backup_path.relative_to(wp_root)}')

            # --- Convert all files in group ---
            group_orig = 0
            group_new  = 0
            for f in group.all_files:
                label = 'master' if f == master else 'variant'
                try:
                    orig, new = convert_to_jpg(f, args.quality, dry_run)
                    group_orig += orig
                    group_new  += new
                    if dry_run:
                        console.print(
                            f'  [dim][{label}] {f.name} → {f.with_suffix(".jpg").name}'
                            f'  ({fmt_bytes(orig)})[/dim]'
                        )
                    else:
                        saving = orig - new
                        pct    = (saving / orig * 100) if orig else 0
                        console.print(
                            f'  [{label}] {f.name} → {f.with_suffix(".jpg").name}'
                            f'  {fmt_bytes(orig)} → {fmt_bytes(new)}'
                            f'  ([green]-{pct:.0f}%[/green])'
                        )
                        f.unlink()
                except OSError as exc:
                    if 'truncated' in str(exc).lower() and f != master:
                        console.print(f'  [yellow]Truncated {f.name} — regenerating from master …[/yellow]')
                        try:
                            orig, new = regenerate_variant_from_master(f, master, args.quality, dry_run)
                            group_orig += orig
                            group_new  += new
                            if dry_run:
                                console.print(
                                    f'  [dim][{label}] {f.name} → {f.with_suffix(".jpg").name}'
                                    f'  (regenerated, {fmt_bytes(orig)})[/dim]'
                                )
                            else:
                                saving = orig - new
                                pct    = (saving / orig * 100) if orig else 0
                                console.print(
                                    f'  [{label}] {f.name} → {f.with_suffix(".jpg").name}'
                                    f'  {fmt_bytes(orig)} → {fmt_bytes(new)}'
                                    f'  ([green]-{pct:.0f}%[/green]) [yellow](regenerated)[/yellow]'
                                )
                        except Exception as regen_exc:
                            errors.append((f, str(regen_exc)))
                            console.print(f'  [red]ERROR regenerating {f.name}: {regen_exc}[/red]')
                    else:
                        errors.append((f, str(exc)))
                        console.print(f'  [red]ERROR {f.name}: {exc}[/red]')
                except Exception as exc:
                    errors.append((f, str(exc)))
                    console.print(f'  [red]ERROR {f.name}: {exc}[/red]')

            total_orig += group_orig
            total_new  += group_new

            # --- Database (guid, mime type, metadata) ---
            if dry_run:
                console.print(f'  [dim]would update DB refs for {master.name}[/dim]')
                total_db += 1   # placeholder count
            else:
                n = db.update_for_group(group, uploads)
                db.commit()
                total_db += n
                console.print(f'  DB: {n} row(s) updated')

            # --- Periodic post_content flush every 1000 groups ---
            if not dry_run and n_processed % 1000 == 0:
                n_rep = len(db._post_content_replacements)
                console.print(f'\n[dim]post_content flush ({n_rep} replacements, checkpoint at {n_processed} groups) …[/dim]')
                n = db.flush_post_content()
                db._post_content_replacements.clear()
                db.commit()
                total_db += n
                console.print(f'  [dim]{n} post(s) updated[/dim]\n')

    # --- Final post_content batch pass for remaining groups ---
    n_replacements = len(db._post_content_replacements)
    if dry_run:
        console.print(
            f'\n[dim]Would scan post_content of live posts for '
            f'{n_replacements} filename replacement(s) (revisions excluded).[/dim]'
        )
    else:
        console.print(f'\nUpdating post_content ({n_replacements} replacements, revisions excluded) …')
        n = db.flush_post_content()
        db.commit()
        total_db += n
        console.print(f'  {n} post(s) updated')

    db.close()

    # --- Summary ---
    console.print()
    console.rule('[bold]Summary[/bold]')

    if stopped_early:
        console.print(f'[yellow]Stopped after {n_processed} of {len(groups)} master(s).[/yellow]')

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    files_label = f'[bold]{n_processed}[/bold] of {len(groups)} master(s) processed'
    if not stopped_early:
        files_label = f'[bold]{n_total}[/bold]  ({n_masters} masters + {n_variants} variants)'
    tbl.add_row('Files converted', files_label)
    tbl.add_row('DB rows updated', f'[bold]{total_db}[/bold]' if not dry_run else '[dim](dry run)[/dim]')

    if dry_run:
        tbl.add_row('Total original size', f'[bold]{fmt_bytes(total_orig)}[/bold]')
        tbl.add_row('Estimated savings', '[dim]run with --execute to measure[/dim]')
    else:
        saved = total_orig - total_new
        pct   = (saved / total_orig * 100) if total_orig else 0
        tbl.add_row('Size before', f'[bold]{fmt_bytes(total_orig)}[/bold]')
        tbl.add_row('Size after',  f'[bold]{fmt_bytes(total_new)}[/bold]')
        tbl.add_row('Space saved', f'[bold green]{fmt_bytes(saved)} ({pct:.1f}%)[/bold green]')
        tbl.add_row('', '[dim](backups not counted)[/dim]')

    console.print(tbl)

    if errors:
        console.print(f'\n[red bold]{len(errors)} error(s):[/red bold]')
        for f, msg in errors:
            console.print(f'  {f.relative_to(wp_root)}: {msg}')
        sys.exit(1)


if __name__ == '__main__':
    main()
