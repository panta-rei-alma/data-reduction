"""CLI entry point for the 12m+7m clean-diagnostics QA pipeline.

Discovers ``*.cube.pbcor.fits`` files under
``<base-dir>/imaging/output/aux/group.*.lp_nperetto/`` (the aux subtree
where the deferred-aux-products patch publishes mask/residual/pb
alongside each 12m7m pbcor) and produces:

- 10 FITS products per cube (4 intensity maps in K / K·km/s, 2 uint8
  mask projections, 4 mean-spectrum BinTables)
- 8 single-panel PNGs per cube (4 maps, 2 masks, 2 paired-spectrum)
- 2 three-panel summary PNGs per cube (integrated, peak)

Output goes under ``<base-dir>/analysis/<group_dir>/{fits,png}/`` —
same shard layout as ``panta-rei-moments``. Idempotent: re-runs only
regenerate outputs older than the **newest** of their 4 input siblings.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from panta_rei.analysis.clean_diagnostics import (
    DEFAULT_ARRAY_COMBOS,
    PAIRED_SPECTRUM_PLOT_KINDS,
    PRODUCT_KINDS,
    SUMMARY_PLOT_KINDS,
    discover_cubes,
    process_cube,
)
from panta_rei.core.logging import setup_logging

logger = logging.getLogger("panta_rei.cli.run_clean_diagnostics")


ALL_PRODUCT_KINDS = PRODUCT_KINDS + SUMMARY_PLOT_KINDS + PAIRED_SPECTRUM_PLOT_KINDS


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate clean-imaging QA diagnostics (image / residual moment "
            "and peak maps, 2D mask projections, mask-averaged spectra, plus "
            "3-panel summary plots) for 12m+7m pbcor cubes."
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
        help="Substring filter on group dirname (repeatable).",
    )
    ap.add_argument(
        "--match", default=None, metavar="REGEX",
        help="Regex filter on pbcor filename (e.g. 'AG310.8797').",
    )
    ap.add_argument(
        "--products", nargs="+", default=list(ALL_PRODUCT_KINDS),
        choices=ALL_PRODUCT_KINDS,
        help="Products to generate (default: all).",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Regenerate outputs even when newer than the input siblings.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="List planned actions without writing files.",
    )
    ap.add_argument(
        "--plots", action=argparse.BooleanOptionalAction, default=True,
        help="Render PNGs alongside each FITS product (default on).",
    )
    ap.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help="Parallel cubes to process (default: 1; 4 is a safe ceiling).",
    )
    ap.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process at most N cubes (debugging).",
    )
    ap.add_argument(
        "--log-file", default=None,
        help=(
            "Optional log file (default: <analysis-dir>/.clean_diagnostics.log). "
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
        return analysis_dir / ".clean_diagnostics.log"
    if args.log_file == "":
        return None
    return Path(args.log_file)


def _filter_cubes(cubes, match_regex, limit):
    if match_regex:
        pattern = re.compile(match_regex)
        cubes = [c for c in cubes if pattern.search(c.name)]
    if limit is not None:
        cubes = cubes[:limit]
    return cubes


def _summarize(results, total):
    n_written = n_skipped = n_failed = 0
    for r in results:
        for status in r.products.values():
            if status == "written":
                n_written += 1
            elif status == "skipped":
                n_skipped += 1
            elif status.startswith("failed:"):
                n_failed += 1
    logger.info(
        "Done: %d cubes processed; products written=%d skipped=%d failed=%d",
        len(results), n_written, n_skipped, n_failed,
    )
    return n_written, n_skipped, n_failed


def _run_serial(cubes, analysis_dir, products, force, dry_run, plot):
    results = []
    for cube in cubes:
        results.append(process_cube(
            cube, analysis_dir, products=products,
            force=force, dry_run=dry_run, plot=plot,
        ))
    return results


def _run_parallel(cubes, analysis_dir, products, force, dry_run, jobs, plot):
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                process_cube, cube, analysis_dir,
                products=products, force=force, dry_run=dry_run, plot=plot,
            ): cube for cube in cubes
        }
        for fut in as_completed(futures):
            cube = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                logger.error("%s: worker crashed: %s", cube.name, exc)
    return results


def main() -> int:
    args = _build_parser().parse_args()

    _, imaging_dir, analysis_dir = _resolve_paths(args)
    log_file = _resolve_log_file(args, analysis_dir)
    if log_file is not None:
        analysis_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=log_file)

    if not imaging_dir.is_dir():
        logger.error("imaging dir does not exist: %s", imaging_dir)
        return 2

    cubes = discover_cubes(imaging_dir, args.group, array_combos=DEFAULT_ARRAY_COMBOS)
    cubes = _filter_cubes(cubes, args.match, args.limit)
    if not cubes:
        logger.warning(
            "No cubes matched (imaging_dir=%s, group=%s, match=%r)",
            imaging_dir, args.group, args.match,
        )
        return 0

    logger.info(
        "Processing %d cube(s) -> %s [products=%s, plots=%s, force=%s, "
        "dry_run=%s, jobs=%d]",
        len(cubes), analysis_dir, ",".join(args.products),
        args.plots, args.force, args.dry_run, args.jobs,
    )

    products = tuple(args.products)
    if args.jobs <= 1:
        results = _run_serial(
            cubes, analysis_dir, products, args.force, args.dry_run, args.plots,
        )
    else:
        results = _run_parallel(
            cubes, analysis_dir, products, args.force, args.dry_run, args.jobs,
            args.plots,
        )

    _, _, n_failed = _summarize(results, len(cubes))
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
