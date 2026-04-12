"""CLI entry point for migrating database paths after a data move.

Mirrors the interface of the legacy ``migrate_db_paths.py``.

Updates absolute paths stored in the SQLite database after migrating
the project data to a new location.

Tables and columns updated:
    - obs: tar_path, extracted_root
    - pi_runs: script_path, cwd, log_path

Columns NOT updated (weblog paths on separate NAS):
    - obs: weblog_path, weblog_url

Usage:
    # Dry run (default) - shows what would change
    panta-rei-migrate-db /path/to/alma_retrieval_state.sqlite3

    # Apply changes
    panta-rei-migrate-db /path/to/alma_retrieval_state.sqlite3 --apply

    # Custom path prefixes
    panta-rei-migrate-db /path/to/db.sqlite3 \\
        --old-prefix /old/path/prefix \\
        --new-prefix /new/path/prefix
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from panta_rei.core.logging import setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_backup(db_path: Path) -> Path:
    """Create a timestamped backup using SQLite's built-in .backup API.

    This is safe even if another process has the DB open (unlike plain cp/copy).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_suffix(f".backup_{timestamp}.sqlite3")
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(backup_path)
    src.backup(dst)
    dst.close()
    src.close()
    return backup_path


def _get_table_columns(con: sqlite3.Connection, table: str) -> set:
    """Get the set of column names for a table."""
    cursor = con.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def _count_paths_to_migrate(
    con: sqlite3.Connection,
    table: str,
    column: str,
    old_prefix: str,
) -> int:
    """Count rows where the column starts with *old_prefix*."""
    cursor = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} LIKE ?",
        (old_prefix + "%",),
    )
    return cursor.fetchone()[0]


def _get_paths_to_migrate(
    con: sqlite3.Connection,
    table: str,
    column: str,
    pk_column: str,
    old_prefix: str,
) -> List[Tuple[str, str]]:
    """Return ``(pk, current_path)`` tuples for rows matching *old_prefix*."""
    cursor = con.execute(
        f"SELECT {pk_column}, {column} FROM {table} WHERE {column} LIKE ?",
        (old_prefix + "%",),
    )
    return cursor.fetchall()


def _verify_path_exists(path: str) -> bool:
    return Path(path).exists()


def _migrate_column(
    con: sqlite3.Connection,
    table: str,
    column: str,
    pk_column: str,
    old_prefix: str,
    new_prefix: str,
    apply: bool,
    verify: bool,
) -> Tuple[int, int, List[str]]:
    """Migrate paths in a single column.

    Returns ``(migrated_count, missing_count, missing_paths)``.
    """
    rows = _get_paths_to_migrate(con, table, column, pk_column, old_prefix)

    migrated = 0
    missing_paths: List[str] = []

    for pk, old_path in rows:
        new_path = old_path.replace(old_prefix, new_prefix, 1)

        if verify and not _verify_path_exists(new_path):
            missing_paths.append(new_path)

        if apply:
            con.execute(
                f"UPDATE {table} SET {column} = ? WHERE {pk_column} = ?",
                (new_path, pk),
            )
        migrated += 1

    return migrated, len(missing_paths), missing_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate database paths from old prefix to new prefix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "database",
        type=Path,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--old-prefix",
        default="/scratch/nas_aces1/dwalker/panta_rei",
        help="Old path prefix to replace (default: /scratch/nas_aces1/dwalker/panta_rei)",
    )
    parser.add_argument(
        "--new-prefix",
        default=None,
        help="New path prefix (default: PANTA_REI_BASE from .env / environment)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run mode)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification that new paths exist on disk",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup (not recommended)",
    )

    return parser


def main() -> None:
    """Entry point for ``panta-rei-migrate-db``."""
    setup_logging()
    args = _build_parser().parse_args()

    # Resolve --new-prefix default from PipelineConfig
    if args.new_prefix is None:
        try:
            from panta_rei.config import PipelineConfig

            cfg = PipelineConfig.from_env()
            args.new_prefix = str(cfg.panta_rei_base)
        except Exception:
            print(
                "Error: Could not determine PANTA_REI_BASE from .env. "
                "Please provide --new-prefix explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Validate database exists
    if not args.database.exists():
        print(f"Error: Database not found: {args.database}", file=sys.stderr)
        sys.exit(1)

    # Validate prefixes are different
    if args.old_prefix == args.new_prefix:
        print("Error: Old and new prefixes are identical", file=sys.stderr)
        sys.exit(1)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"=== Database Path Migration ({mode}) ===\n")
    print(f"Database: {args.database}")
    print(f"Old prefix: {args.old_prefix}")
    print(f"New prefix: {args.new_prefix}")
    print(f"Verify paths: {not args.no_verify}")
    print()

    # Create backup if applying changes
    if args.apply and not args.no_backup:
        backup_path = _create_backup(args.database)
        print(f"Backup created: {backup_path}\n")

    # Define columns to migrate
    # Note: weblog_path and weblog_url are intentionally excluded
    migrations = [
        ("obs", "tar_path", "uid"),
        ("obs", "extracted_root", "uid"),
        ("pi_runs", "script_path", "id"),
        ("pi_runs", "cwd", "id"),
        ("pi_runs", "log_path", "id"),
    ]

    con = sqlite3.connect(args.database)

    try:
        total_to_migrate = 0
        total_migrated = 0
        total_missing = 0
        all_missing_paths: List[str] = []

        print("Scanning for paths to migrate...\n")

        for table, column, pk_col in migrations:
            # Check if table exists
            cursor = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if not cursor.fetchone():
                print(f"  {table}.{column}: table does not exist, skipping")
                continue

            # Check if column exists
            columns = _get_table_columns(con, table)
            if column not in columns:
                print(f"  {table}.{column}: column does not exist, skipping")
                continue

            # Count paths to migrate
            count = _count_paths_to_migrate(con, table, column, args.old_prefix)
            total_to_migrate += count

            if count == 0:
                print(f"  {table}.{column}: no paths to migrate")
                continue

            print(f"  {table}.{column}: {count} paths to migrate")

            # Perform migration
            migrated, missing, missing_list = _migrate_column(
                con, table, column, pk_col,
                args.old_prefix, args.new_prefix,
                apply=args.apply,
                verify=not args.no_verify,
            )

            total_migrated += migrated
            total_missing += missing
            all_missing_paths.extend(missing_list)

        if args.apply:
            con.commit()

        print()
        print("=" * 50)
        print("Summary:")
        print(f"  Total paths found: {total_to_migrate}")
        print(f"  Paths {'migrated' if args.apply else 'to migrate'}: {total_migrated}")

        if not args.no_verify:
            print(f"  Missing paths: {total_missing}")
            if all_missing_paths:
                print("\nWarning: The following new paths do not exist on disk:")
                for p in all_missing_paths[:10]:
                    print(f"    {p}")
                if len(all_missing_paths) > 10:
                    print(f"    ... and {len(all_missing_paths) - 10} more")

        if not args.apply and total_to_migrate > 0:
            print(f"\nThis was a dry run. To apply changes, run with --apply")

        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()
