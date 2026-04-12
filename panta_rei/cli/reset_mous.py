"""CLI entry point for resetting MOUSes for re-download.

Mirrors the interface of the legacy ``reset_mous.py``.

This script:
1. Resets specified UIDs to 'pending' status in the database
2. Optionally deletes their extracted directories
3. Cleans up stale duplicate entries with non-canonical UID formats

After running this, run your normal retrieval service/script to re-download.

Usage:
    # List MOUSes that would be affected (dry-run)
    panta-rei-reset --base-dir /path/to/2025.1.00383.L --dry-run

    # Reset specific MOUSes
    panta-rei-reset --base-dir /path/to/2025.1.00383.L \\
        --uids uid___a001_x3833_x64d2 uid___a001_x3833_x64de

    # Reset and delete extracted files
    panta-rei-reset --base-dir /path/to/2025.1.00383.L \\
        --uids uid___a001_x3833_x64d2 --delete-extracted
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

from panta_rei.core.logging import setup_logging
from panta_rei.core.text import now_iso
from panta_rei.core.uid import canonical_uid

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_extracted_dir(data_dir: Path, uid: str) -> Path | None:
    """Find the extracted member directory for a MOUS UID."""
    parts = uid.split("_")
    if len(parts) < 3:
        return None
    suffix = parts[-1]  # e.g., x64d2

    matches: list[Path] = []
    for suffix_variant in [suffix.upper(), suffix.capitalize(), suffix]:
        pattern = f"science_goal.*/group.*/member.uid___A001_X3833_{suffix_variant}"
        matches = list(data_dir.glob(pattern))
        if matches:
            break

    if matches:
        return matches[0]
    return None


def _clean_stale_duplicates(db_path: Path, dry_run: bool = True) -> int:
    """Remove entries with non-canonical UID format (uid://...)."""
    with sqlite3.connect(db_path) as con:
        count = con.execute(
            "SELECT COUNT(*) FROM obs WHERE uid LIKE 'uid://%'"
        ).fetchone()[0]

        if count == 0:
            log.info("No stale duplicate entries found")
            return 0

        if dry_run:
            log.info("[DRY-RUN] Would delete %d stale entries with uid:// format", count)
            rows = con.execute(
                "SELECT uid, status FROM obs WHERE uid LIKE 'uid://%'"
            ).fetchall()
            for uid, status in rows:
                log.info("  - %s (%s)", uid, status)
        else:
            con.execute("DELETE FROM obs WHERE uid LIKE 'uid://%'")
            log.info("Deleted %d stale duplicate entries", count)

        return count


def _reset_mous(
    db_path: Path,
    data_dir: Path,
    uids: list[str],
    delete_extracted: bool = False,
    dry_run: bool = True,
) -> int:
    """Reset specified MOUSes to pending status."""
    reset_count = 0

    with sqlite3.connect(db_path) as con:
        for uid in uids:
            uidc = canonical_uid(uid) or ""
            if not uidc:
                log.warning("Could not normalise UID: %s", uid)
                continue

            row = con.execute(
                "SELECT status, extracted_root, n_extracted FROM obs WHERE uid = ?",
                (uidc,)
            ).fetchone()

            if not row:
                log.warning("UID not found in database: %s", uidc)
                continue

            status, extracted_root, n_extracted = row
            log.info("Processing %s (current: %s, %s files)", uidc, status, n_extracted)

            if delete_extracted:
                member_dir = _find_extracted_dir(data_dir, uidc)
                if member_dir and member_dir.exists():
                    if dry_run:
                        log.info("  [DRY-RUN] Would delete: %s", member_dir)
                    else:
                        log.info("  Deleting: %s", member_dir)
                        shutil.rmtree(member_dir)
                else:
                    log.info("  No extracted directory found")

            if dry_run:
                log.info("  [DRY-RUN] Would reset to pending")
            else:
                con.execute("""
                    UPDATE obs SET
                        status = 'pending',
                        tar_path = NULL,
                        tar_deleted = 0,
                        extracted_root = NULL,
                        n_extracted = NULL,
                        n_skipped = NULL,
                        weblog_staged = 0,
                        weblog_path = NULL,
                        weblog_url = NULL,
                        weblog_staged_at = NULL,
                        updated_at = ?
                    WHERE uid = ?
                """, (now_iso(), uidc))
                log.info("  Reset to pending")

            reset_count += 1

    return reset_count


def _show_status(db_path: Path) -> None:
    """Show current database status summary."""
    with sqlite3.connect(db_path) as con:
        log.info("=== Database Status ===")

        rows = con.execute(
            "SELECT status, COUNT(*) FROM obs GROUP BY status ORDER BY COUNT(*) DESC"
        ).fetchall()
        for status, count in rows:
            log.info("  %s: %d", status, count)

        dup_count = con.execute(
            "SELECT COUNT(*) FROM obs WHERE uid LIKE 'uid://%'"
        ).fetchone()[0]
        if dup_count:
            log.warning("  Stale duplicates (uid://...): %d", dup_count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reset MOUSes for re-download",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--base-dir", required=True,
        help="Base directory for the project (contains alma_retrieval_state.sqlite3)",
    )
    ap.add_argument(
        "--db", default=None,
        help="Path to SQLite DB (default: <base-dir>/alma_retrieval_state.sqlite3)",
    )
    ap.add_argument(
        "--data-dir", default=None,
        help="Path to extracted data (default: <base-dir>/<project-code>)",
    )
    ap.add_argument(
        "--project-code", default="2025.1.00383.L",
        help="Project code (default: 2025.1.00383.L)",
    )
    ap.add_argument(
        "--uids", nargs="+",
        help="UIDs to reset (e.g., uid___a001_x3833_x64d2)",
    )
    ap.add_argument(
        "--delete-extracted", action="store_true",
        help="Delete extracted directories (required for re-extraction to work)",
    )
    ap.add_argument(
        "--clean-duplicates", action="store_true",
        help="Clean up stale duplicate entries with uid:// format",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making changes",
    )
    ap.add_argument(
        "--status", action="store_true",
        help="Show database status and exit",
    )

    return ap


def main() -> int:
    """Entry point for ``panta-rei-reset``."""
    setup_logging()
    args = _build_parser().parse_args()

    base_dir = Path(args.base_dir).resolve()
    db_path = Path(args.db) if args.db else (base_dir / "alma_retrieval_state.sqlite3")
    data_dir = Path(args.data_dir) if args.data_dir else (base_dir / args.project_code)

    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        return 1

    if args.status:
        _show_status(db_path)
        return 0

    if args.dry_run:
        log.info("=== DRY-RUN MODE ===")

    # Clean duplicates if requested
    if args.clean_duplicates:
        _clean_stale_duplicates(db_path, dry_run=args.dry_run)

    # Reset specified UIDs
    if args.uids:
        _reset_mous(
            db_path=db_path,
            data_dir=data_dir,
            uids=args.uids,
            delete_extracted=args.delete_extracted,
            dry_run=args.dry_run,
        )
    elif not args.clean_duplicates and not args.status:
        log.warning("No UIDs specified. Use --uids to specify MOUSes to reset.")
        log.info("Example: --uids uid___a001_x3833_x64d2 uid___a001_x3833_x64de")

    if not args.dry_run:
        _show_status(db_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
