"""CLI entry point for collecting standalone total-power (TP) cubes.

Builds ``<base-dir>/total_power/member.uid___A001_<mous>/`` containing hard
links to every delivered ``*.cube.I.sd.fits`` single-dish cube, plus a
``tp_manifest.csv`` recording where each one came from.

The tree exists so the standalone TP cubes can be QA'd and synced to
Globus in one place — they are otherwise scattered one level deep inside
each MOUS delivery. Hard links mean the tree costs no extra disk and can
never drift from the delivered originals.

Filenames are left exactly as ALMA delivered them. These cubes are for
research/QA, not for re-delivery to the archive, so the strict naming
convention that governs the feathered 12m+7m+TP products does not apply.

Typical use::

    panta-rei-collect-tp --base-dir /path/to/2025.1.00383.L --dry-run
    panta-rei-collect-tp --base-dir /path/to/2025.1.00383.L
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from panta_rei.core.logging import setup_logging
from panta_rei.imaging.collect_tp import (
    DEFAULT_COLLECT_DIRNAME,
    collect,
    discover_tp_cubes,
    load_freq_ranges,
    verify,
    write_manifest,
)

logger = logging.getLogger("panta_rei.cli.run_collect_tp")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Collect delivered standalone total-power cubes into a flat "
            "per-member tree of hard links for QA and Globus sync."
        ),
    )
    ap.add_argument(
        "--base-dir", required=True,
        help="Project base directory (e.g. ./2025.1.00383.L)",
    )
    ap.add_argument(
        "--data-dir", default=None,
        help="Delivery tree root (default: <base-dir>/<base-dir name>)",
    )
    ap.add_argument(
        "--dest-dir", default=None,
        help=f"Collected tree (default: <base-dir>/{DEFAULT_COLLECT_DIRNAME})",
    )
    ap.add_argument(
        "--csv", default=None,
        help="targets_by_array.csv (default: <base-dir>/targets_by_array.csv)",
    )
    ap.add_argument(
        "--imaging-db", default=None,
        help=(
            "imaging.sqlite3, read-only, used only to add frequency ranges to "
            "the manifest (default: <base-dir>/imaging.sqlite3)"
        ),
    )
    ap.add_argument(
        "--member", action="append", default=None, metavar="SUBSTR",
        help="Substring filter on MOUS id (repeatable).",
    )
    ap.add_argument(
        "--source", action="append", default=None, metavar="SUBSTR",
        help="Substring filter on source name (repeatable).",
    )
    ap.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Collect at most N cubes (debugging).",
    )
    ap.add_argument(
        "--expect", type=int, default=None, metavar="N",
        help=(
            "Assert exactly N cubes are collected; exit non-zero otherwise. "
            "Use to pin the known-good total (2590 for 2025.1.00383.L)."
        ),
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Replace existing links that point at a different inode.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report planned actions without creating links or the manifest.",
    )
    ap.add_argument(
        "--no-verify", action="store_true",
        help="Skip the post-run link verification pass.",
    )
    ap.add_argument(
        "--log-file", default=None,
        help=(
            "Optional log file (default: <dest-dir>/.collect_tp.log). "
            "Pass an empty string to disable file logging."
        ),
    )
    return ap


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    base_dir = Path(args.base_dir).resolve()
    data_dir = (
        Path(args.data_dir).resolve() if args.data_dir
        else base_dir / base_dir.name
    )
    dest_dir = (
        Path(args.dest_dir).resolve() if args.dest_dir
        else base_dir / DEFAULT_COLLECT_DIRNAME
    )
    csv_path = (
        Path(args.csv).resolve() if args.csv
        else base_dir / "targets_by_array.csv"
    )
    imaging_db = (
        Path(args.imaging_db).resolve() if args.imaging_db
        else base_dir / "imaging.sqlite3"
    )
    return base_dir, data_dir, dest_dir, csv_path, imaging_db


def _resolve_log_file(args: argparse.Namespace, dest_dir: Path) -> Path | None:
    if args.log_file is None:
        return dest_dir / ".collect_tp.log"
    if args.log_file == "":
        return None
    return Path(args.log_file)


def main() -> int:
    args = _build_parser().parse_args()
    _, data_dir, dest_dir, csv_path, imaging_db = _resolve_paths(args)

    log_file = None if args.dry_run else _resolve_log_file(args, dest_dir)
    if log_file is not None:
        dest_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=log_file)

    if not data_dir.is_dir():
        logger.error("delivery tree does not exist: %s", data_dir)
        return 2
    if not csv_path.is_file():
        logger.error("targets CSV does not exist: %s", csv_path)
        return 2

    logger.info("Delivery tree: %s", data_dir)
    logger.info("Destination:   %s%s", dest_dir, " [dry-run]" if args.dry_run else "")

    cubes = discover_tp_cubes(
        data_dir, csv_path, member=args.member, source=args.source,
    )
    if args.limit is not None:
        cubes = cubes[:args.limit]
    if not cubes:
        logger.warning(
            "No TP cubes matched (member=%s, source=%s)", args.member, args.source,
        )
        return 1

    freq_ranges = load_freq_ranges(imaging_db)

    stats, rows = collect(
        cubes, dest_dir,
        freq_ranges=freq_ranges, force=args.force, dry_run=args.dry_run,
    )
    write_manifest(rows, dest_dir, dry_run=args.dry_run)

    logger.info(
        "%s: %d linked, %d already present, %d replaced, %d conflicts, %d failed "
        "(%d cubes across %d members)",
        "Would collect" if args.dry_run else "Collected",
        stats.linked, stats.already, stats.replaced, stats.conflict, stats.failed,
        len(cubes), len({c.mous_id for c in cubes}),
    )

    exit_code = 0
    if stats.failed or stats.conflict:
        exit_code = 1

    if args.expect is not None and len(cubes) != args.expect:
        logger.error(
            "expected %d cubes, discovered %d", args.expect, len(cubes),
        )
        exit_code = 1

    if args.dry_run or args.no_verify:
        return exit_code

    n_links, problems = verify(dest_dir, expected=stats.total_present)
    if problems:
        for p in problems[:20]:
            logger.error("verify: %s", p)
        if len(problems) > 20:
            logger.error("verify: ... and %d more", len(problems) - 20)
        return 1
    logger.info("Verified %d hard links in %s", n_links, dest_dir)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
