"""CLI entry point for QA moment-map / mean-spectrum generation.

Discovers ``*.cube.pbcor.fits`` files under
``<base-dir>/imaging/output/group.*.lp_nperetto/`` and produces the
configured QA products under ``<base-dir>/analysis/<group_dir>/``.

Tracking is filesystem-only — outputs are skipped when newer than their
input cube. Use ``--force`` to regenerate.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from panta_rei.analysis.moments import (
    CUBE_GLOB,
    DEFAULT_ARRAY_COMBOS,
    PRODUCT_KINDS,
    discover_cubes,
    process_cube,
)
from panta_rei.core.logging import setup_logging

logger = logging.getLogger("panta_rei.cli.run_moments")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate QA moment-0, peak-intensity, and mean-spectrum FITS "
            "products from imaging cubes. Idempotent: re-runs only "
            "regenerate outputs older than their input cube."
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
        "--analysis-dir", default=None,
        help="Analysis output dir (default: <base-dir>/analysis)",
    )
    ap.add_argument(
        "--group", action="append", default=None, metavar="SUBSTR",
        help=(
            "Substring filter on group dirname (repeatable; e.g. 'X64b9'). "
            "Default: all group.*.lp_nperetto dirs."
        ),
    )
    ap.add_argument(
        "--array-combo", action="append", default=None, metavar="COMBO",
        help=(
            "Array-combination token to include, matched as '.<combo>.' "
            "in the cube filename (repeatable). Default: "
            f"{','.join(DEFAULT_ARRAY_COMBOS)}. "
            "Pass 'all' to disable the array-combo filter."
        ),
    )
    ap.add_argument(
        "--match", default=None, metavar="REGEX",
        help="Regex filter on cube filename (e.g. '102\\.5' or 'AG221')",
    )
    ap.add_argument(
        "--products", nargs="+", default=list(PRODUCT_KINDS),
        choices=PRODUCT_KINDS,
        help="Products to generate (default: all three).",
    )
    ap.add_argument(
        "--spectral-unit", choices=["auto", "freq", "velocity"], default="auto",
        help=(
            "Spectral-axis convention. 'auto' (default) uses km/s when "
            "RESTFRQ is in the header, else falls back to Hz with a warning."
        ),
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Regenerate outputs even when newer than the input cube.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="List planned actions without writing files.",
    )
    ap.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help=(
            "Parallel cubes to process (default: 1). Each worker memmaps "
            "its cube; tune to host RAM (4 is a safe ceiling on iris1)."
        ),
    )
    ap.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process at most N cubes (debugging).",
    )
    ap.add_argument(
        "--log-file", default=None,
        help=(
            "Optional log file (default: <analysis-dir>/.moments.log). "
            "Pass an empty string to disable file logging."
        ),
    )
    return ap


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    base_dir = Path(args.base_dir).resolve()
    imaging_dir = (
        Path(args.imaging_dir).resolve() if args.imaging_dir
        else base_dir / "imaging" / "output"
    )
    analysis_dir = (
        Path(args.analysis_dir).resolve() if args.analysis_dir
        else base_dir / "analysis"
    )
    return base_dir, imaging_dir, analysis_dir


def _resolve_log_file(args: argparse.Namespace, analysis_dir: Path) -> Path | None:
    if args.log_file is None:
        return analysis_dir / ".moments.log"
    if args.log_file == "":
        return None
    return Path(args.log_file)


def _filter_cubes(
    cubes: list[Path],
    match_regex: str | None,
    limit: int | None,
) -> list[Path]:
    if match_regex:
        pattern = re.compile(match_regex)
        cubes = [c for c in cubes if pattern.search(c.name)]
    if limit is not None:
        cubes = cubes[:limit]
    return cubes


def _summarize(results: list, total: int) -> tuple[int, int, int]:
    """Return (n_written, n_skipped, n_failed) tallies."""
    n_written = n_skipped = n_failed = 0
    for r in results:
        for status in r.products.values():
            if status == "written":
                n_written += 1
            elif status == "skipped":
                n_skipped += 1
            elif status.startswith("failed:"):
                n_failed += 1
            # dry-run is not tallied as written/skipped/failed
    logger.info(
        "Done: %d cubes processed; products written=%d skipped=%d failed=%d",
        len(results), n_written, n_skipped, n_failed,
    )
    return n_written, n_skipped, n_failed


def _run_serial(cubes, analysis_dir, products, spectral_unit, force, dry_run):
    results = []
    for cube_fits in cubes:
        result = process_cube(
            cube_fits,
            analysis_dir,
            products=products,
            spectral_unit=spectral_unit,
            force=force,
            dry_run=dry_run,
        )
        results.append(result)
    return results


def _run_parallel(cubes, analysis_dir, products, spectral_unit, force, dry_run, jobs):
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                process_cube,
                cube_fits,
                analysis_dir,
                products=products,
                spectral_unit=spectral_unit,
                force=force,
                dry_run=dry_run,
            ): cube_fits
            for cube_fits in cubes
        }
        for fut in as_completed(futures):
            cube_fits = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                logger.error("%s: worker crashed: %s", cube_fits.name, exc)
    return results


def main() -> int:
    """Entry point for ``panta-rei-moments``."""
    args = _build_parser().parse_args()

    base_dir, imaging_dir, analysis_dir = _resolve_paths(args)
    log_file = _resolve_log_file(args, analysis_dir)
    if log_file is not None:
        analysis_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=log_file)

    if not imaging_dir.is_dir():
        logger.error("imaging dir does not exist: %s", imaging_dir)
        return 2

    combos_arg = args.array_combo if args.array_combo is not None else list(DEFAULT_ARRAY_COMBOS)
    array_combos: tuple[str, ...] | None
    if any(c.lower() == "all" for c in combos_arg):
        array_combos = None
    else:
        array_combos = tuple(combos_arg)

    cubes = discover_cubes(imaging_dir, args.group, array_combos=array_combos)
    cubes = _filter_cubes(cubes, args.match, args.limit)
    if not cubes:
        logger.warning(
            "No cubes matched (imaging_dir=%s, group=%s, array_combos=%s, match=%r)",
            imaging_dir, args.group, array_combos, args.match,
        )
        return 0

    logger.info(
        "Processing %d cube(s) -> %s [array_combos=%s, products=%s, "
        "spectral_unit=%s, force=%s, dry_run=%s, jobs=%d]",
        len(cubes), analysis_dir,
        ",".join(array_combos) if array_combos else "all",
        ",".join(args.products),
        args.spectral_unit, args.force, args.dry_run, args.jobs,
    )

    products = tuple(args.products)
    if args.jobs <= 1:
        results = _run_serial(
            cubes, analysis_dir, products, args.spectral_unit,
            args.force, args.dry_run,
        )
    else:
        results = _run_parallel(
            cubes, analysis_dir, products, args.spectral_unit,
            args.force, args.dry_run, args.jobs,
        )

    _, _, n_failed = _summarize(results, len(cubes))
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
