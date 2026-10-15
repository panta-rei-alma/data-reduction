"""CLI entry point for continuum quicklook-PNG generation.

Discovers ``*.cont.tt0.pbcor.fits`` science images under
``<base-dir>/imaging/output/group.*.lp_nperetto/`` and renders, beside each
FITS, an asinh flux quicklook plus a S/N-masked spectral-index map (when the
unit has significant continuum). PNGs are named to match the FITS stem so the
portal indexer attaches them as previews.

Tracking is filesystem-only — a PNG is skipped when newer than its FITS
input(s). Use ``--force`` to regenerate.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from panta_rei.analysis.continuum_quicklook import (
    discover_units,
    process_unit,
)
from panta_rei.core.logging import setup_logging

logger = logging.getLogger("panta_rei.cli.run_continuum_quicklooks")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate continuum quicklook PNGs (asinh flux + S/N-masked "
            "spectral index) beside the 12m+7m MFS FITS products. Idempotent: "
            "re-runs only regenerate PNGs older than their FITS input(s)."
        ),
    )
    ap.add_argument(
        "--base-dir", required=True,
        help="Project base directory (e.g. ./2025.1.00383.L)",
    )
    ap.add_argument(
        "--imaging-dir", default=None,
        help="Imaging output dir (default: <base-dir>/imaging/output)",
    )
    ap.add_argument(
        "--group", action="append", default=None, metavar="SUBSTR",
        help=(
            "Substring filter on group dirname (repeatable; e.g. 'X64cf'). "
            "Default: all group.*.lp_nperetto dirs."
        ),
    )
    ap.add_argument(
        "--match", default=None, metavar="REGEX",
        help="Regex filter on the tt0.pbcor filename (e.g. 'AG301').",
    )
    ap.add_argument(
        "--snr", type=float, default=5.0, metavar="SIGMA",
        help="S/N threshold for the alpha mask, on the flat image (default: 5).",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Regenerate PNGs even when newer than the input FITS.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="List planned actions without writing files.",
    )
    ap.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help="Parallel units to process (default: 1). 4 is a safe ceiling on iris1.",
    )
    ap.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process at most N units (debugging).",
    )
    ap.add_argument(
        "--log-file", default=None,
        help=(
            "Optional log file (default: <imaging-dir>/.continuum_quicklooks.log). "
            "Pass an empty string to disable file logging."
        ),
    )
    return ap


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    base_dir = Path(args.base_dir).resolve()
    imaging_dir = (
        Path(args.imaging_dir).resolve() if args.imaging_dir
        else base_dir / "imaging" / "output"
    )
    return base_dir, imaging_dir


def _resolve_log_file(args: argparse.Namespace, imaging_dir: Path) -> Path | None:
    if args.log_file is None:
        return imaging_dir / ".continuum_quicklooks.log"
    if args.log_file == "":
        return None
    return Path(args.log_file)


def _filter_units(units: list[Path], match_regex: str | None, limit: int | None) -> list[Path]:
    if match_regex:
        pattern = re.compile(match_regex)
        units = [u for u in units if pattern.search(u.name)]
    if limit is not None:
        units = units[:limit]
    return units


def _summarize(results: list) -> tuple[int, int, int]:
    """Return (n_written, n_skipped, n_failed) over all product statuses."""
    n_written = n_skipped = n_failed = 0
    for r in results:
        for status in r.products.values():
            if status == "written":
                n_written += 1
            elif status.startswith("skipped"):
                n_skipped += 1
            elif status.startswith("failed:"):
                n_failed += 1
    logger.info(
        "Done: %d unit(s) processed; PNGs written=%d skipped=%d failed=%d",
        len(results), n_written, n_skipped, n_failed,
    )
    return n_written, n_skipped, n_failed


def _run_serial(units, snr, force, dry_run):
    return [
        process_unit(u, snr=snr, force=force, dry_run=dry_run)
        for u in units
    ]


def _run_parallel(units, snr, force, dry_run, jobs):
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(process_unit, u, snr=snr, force=force, dry_run=dry_run): u
            for u in units
        }
        for fut in as_completed(futures):
            unit = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                logger.error("%s: worker crashed: %s", unit.name, exc)
    return results


def main() -> int:
    """Entry point for ``panta-rei-continuum-quicklooks``."""
    args = _build_parser().parse_args()

    base_dir, imaging_dir = _resolve_paths(args)
    log_file = _resolve_log_file(args, imaging_dir)
    setup_logging(log_file=log_file)

    if not imaging_dir.is_dir():
        logger.error("imaging dir does not exist: %s", imaging_dir)
        return 2

    units = discover_units(imaging_dir, args.group)
    units = _filter_units(units, args.match, args.limit)
    if not units:
        logger.warning(
            "No continuum units matched (imaging_dir=%s, group=%s, match=%r)",
            imaging_dir, args.group, args.match,
        )
        return 0

    logger.info(
        "Processing %d continuum unit(s) [snr=%g, force=%s, dry_run=%s, jobs=%d]",
        len(units), args.snr, args.force, args.dry_run, args.jobs,
    )

    if args.jobs <= 1:
        results = _run_serial(units, args.snr, args.force, args.dry_run)
    else:
        results = _run_parallel(units, args.snr, args.force, args.dry_run, args.jobs)

    _, _, n_failed = _summarize(results)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
