"""CLI entry point for joint imaging (tclean+feather or sdintimaging).

Usage::

    # Recover tclean parameters from TM weblogs
    panta-rei-imaging --base-dir ./2025.1.00383.L --step recover --dry-run

    # Advisory preflight (no casatasks required)
    panta-rei-imaging --base-dir ./2025.1.00383.L --step preflight

    # Full imaging with tclean+feather (default)
    panta-rei-imaging --base-dir ./2025.1.00383.L --step all \\
        --output-dir /path/to/output --limit 1

    # Parallel tclean via mpicasa (requires CASA_PATH)
    panta-rei-imaging --base-dir ./2025.1.00383.L --step image \\
        --output-dir /path/to/output --parallel --nproc 8

    # Legacy sdintimaging mode
    panta-rei-imaging --base-dir ./2025.1.00383.L --step all \\
        --output-dir /path/to/output --method sdintimaging --limit 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from panta_rei.config import PipelineConfig
from panta_rei.core.logging import setup_logging
from panta_rei.db.connection import DatabaseManager
from panta_rei.workflows.base import WorkflowContext
from panta_rei.workflows.imaging import ImagingOptions, run_imaging

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Joint 12m+7m+TP imaging (tclean+feather or sdintimaging).",
    )
    ap.add_argument(
        "--base-dir", required=True,
        help="Project base directory (e.g. ./2025.1.00383.L)",
    )
    ap.add_argument(
        "--db", default=None,
        help="Path to SQLite DB (default: <base-dir>/imaging.sqlite3, or IMAGING_DB from .env)",
    )
    ap.add_argument(
        "--weblog-dir", default=None,
        help="Weblog directory (default: from .env WEBLOG_DIR)",
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Output directory for imaging products (required for --step image/all)",
    )
    ap.add_argument(
        "--obs-csv", default=None,
        help="Path to targets_by_array.csv (default: <base-dir>/targets_by_array.csv)",
    )
    ap.add_argument(
        "--step",
        choices=["recover", "preflight", "image", "all"],
        default="all",
        help="Which step(s) to run (default: all)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without executing",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Max number of imaging units to process",
    )
    ap.add_argument(
        "--match", default=None,
        help="Regex filter on source name or UID",
    )
    ap.add_argument(
        "--re-run", action="store_true",
        help="Force re-run even if success exists",
    )
    ap.add_argument(
        "--sdgain", type=float, default=1.0,
        help="SD gain for sdintimaging (default: 1.0)",
    )
    ap.add_argument(
        "--deconvolver", default="multiscale",
        help="Deconvolver (default: multiscale)",
    )
    ap.add_argument(
        "--scales", default=None,
        help="Multiscale scales as JSON list (default: [0,5,10,15,20])",
    )
    ap.add_argument(
        "--include-sources", nargs="+", default=None,
        help="Only process these source names",
    )
    ap.add_argument(
        "--include-line-groups", nargs="+", default=None,
        help="Only process these line groups (e.g. N2H+ HCO+)",
    )
    ap.add_argument(
        "--keep-intermediates", action="store_true",
        help="Retain all CASA intermediate products",
    )
    ap.add_argument(
        "--method",
        choices=["tclean_feather", "sdintimaging"],
        default="tclean_feather",
        help="Imaging method (default: tclean_feather)",
    )
    ap.add_argument(
        "--parallel", action="store_true",
        help="Enable MPI parallelization for tclean via mpicasa (requires CASA_PATH)",
    )
    ap.add_argument(
        "--nproc", type=int, default=4,
        help="Number of MPI processes (default: 4, only with --parallel)",
    )
    return ap


def main() -> int:
    """Entry point for ``panta-rei-imaging``."""
    setup_logging()
    args = _build_parser().parse_args()

    base_dir = Path(args.base_dir).resolve()

    # DB resolution: --db flag > IMAGING_DB from .env > <base-dir>/imaging.sqlite3
    if args.db:
        db_path = Path(args.db)
    else:
        try:
            env_config = PipelineConfig.from_env()
            db_path = env_config.imaging_db_path
        except Exception:
            db_path = base_dir / "imaging.sqlite3"

    # Build config from --base-dir.  base_dir is the project dir
    # (e.g. ./2025.1.00383.L), so panta_rei_base is its parent.
    # Overlay weblog_dir and casa_path from .env if available.
    try:
        env_config = PipelineConfig.from_env()
        weblog_default = env_config.weblog_dir
        casa_path_default = env_config.casa_path
    except Exception:
        weblog_default = Path("/scratch/almanas/dwalker2/panta-rei/weblogs")
        casa_path_default = None

    config = PipelineConfig(
        panta_rei_base=base_dir.parent,
        weblog_dir=Path(args.weblog_dir) if args.weblog_dir else weblog_default,
        casa_path=casa_path_default,
    )

    db_manager = DatabaseManager(db_path)

    # Parse scales
    scales = [0, 5, 10, 15, 20]
    if args.scales:
        scales = json.loads(args.scales)

    # Validation
    if args.method == "tclean_feather" and args.sdgain != 1.0:
        log.warning("--sdgain is ignored for --method tclean_feather")
    if args.parallel and args.method == "sdintimaging":
        log.error("--parallel is not supported with --method sdintimaging")
        return 1
    if args.parallel and not config.casa_path:
        log.error(
            "--parallel requires CASA_PATH to be set in .env "
            "(pointing to monolithic CASA installation with mpicasa)"
        )
        return 1

    opts = ImagingOptions(
        weblog_dir=Path(args.weblog_dir) if args.weblog_dir else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        obs_csv=Path(args.obs_csv) if args.obs_csv else None,
        step=args.step,
        re_run=args.re_run,
        match=args.match,
        limit=args.limit,
        include_sources=args.include_sources,
        include_line_groups=args.include_line_groups,
        method=args.method,
        sdgain=args.sdgain,
        deconvolver=args.deconvolver,
        scales=scales,
        keep_intermediates=args.keep_intermediates,
        parallel=args.parallel,
        nproc=args.nproc,
    )

    ctx = WorkflowContext(
        config=config,
        db_manager=db_manager,
        dry_run=args.dry_run,
    )

    results = run_imaging(ctx, opts)

    any_failed = any(not r.success for r in results.values())
    for name, r in results.items():
        status = "OK" if r.success else "FAILED"
        log.info("Step %s [%s]: %s", name, status, r.summary)

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
