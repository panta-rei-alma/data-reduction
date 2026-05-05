"""CLI entry point for distributed tclean+feather imaging.

Usage::

    panta-rei-imaging-dispatch \\
        --base-dir <project-base> \\
        --output-dir <project-base>/imaging/output \\
        --machines-config <project-base>/machines.json

See ``--help`` for filters, transfer-method, overwrite, and reconcile flags.
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
from panta_rei.imaging.dispatch import (
    load_machines_config,
    run_dispatch,
    reconcile_prior,
)
from panta_rei.imaging.unit_selection import SelectionFilters

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Distributed tclean+feather imaging dispatcher.",
    )
    ap.add_argument("--base-dir", required=True,
                    help="Project base dir (e.g. ./2025.1.00383.L)")
    ap.add_argument("--output-dir", required=True,
                    help="Canonical NAS publish dir for output FITS")
    ap.add_argument("--machines-config", required=True,
                    help="Path to machines.json")
    ap.add_argument("--db", default=None,
                    help="Path to SQLite DB (default: <base-dir>/imaging.sqlite3)")
    ap.add_argument("--obs-csv", default=None,
                    help="Path to targets_by_array.csv (default: <base-dir>/targets_by_array.csv)")

    # Filters (forwarded to selection)
    ap.add_argument("--match", default=None, help="Regex on source_name/gous_uid")
    ap.add_argument("--include-sources", nargs="+", default=None)
    ap.add_argument("--include-line-groups", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--re-run", action="store_true",
                    help="Bypass success_exists() filter; requires --overwrite "
                         "to actually publish over existing FITS")
    ap.add_argument("--machines", nargs="+", default=None,
                    help="Subset of machines.json entries to use")

    # Imaging knobs
    ap.add_argument("--method", default="tclean_feather",
                    choices=["tclean_feather"],
                    help="MVP supports tclean_feather only")
    ap.add_argument("--deconvolver", default="multiscale")
    ap.add_argument("--scales", default=None,
                    help="JSON list, e.g. '[0,5,10,15,20]'")

    # Transport / publish
    ap.add_argument("--transfer-method", default="tar",
                    choices=["tar", "rsync", "cp"])
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow publish to overwrite existing FITS "
                         "(default: fail_if_exists)")

    # Lifecycle
    ap.add_argument("--reconcile-only", action="store_true",
                    help="Run reconciliation pass and exit")
    ap.add_argument("--abandon-prior", action="store_true",
                    help="Mark in-flight prior runs FAILED instead of adopting")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be dispatched, no SSH, no DB writes "
                         "for new rows")

    return ap


def _validate_re_run_overwrite(args) -> Optional[str]:
    """`--re-run` without `--overwrite` would be a publish-time surprise."""
    if args.re_run and not args.overwrite:
        return (
            "--re-run bypasses the success_exists() filter, but the "
            "dispatcher's default publish policy is fail_if_exists. "
            "Pass --overwrite to actually replace existing FITS, or drop "
            "--re-run."
        )
    return None


def main(argv=None) -> int:
    setup_logging()
    ap = _build_parser()
    args = ap.parse_args(argv)

    base_dir = Path(args.base_dir).resolve()
    publish_dir = Path(args.output_dir).resolve()

    err = _validate_re_run_overwrite(args)
    if err:
        log.error("%s", err)
        return 2

    # DB resolution
    if args.db:
        db_path = Path(args.db)
    else:
        try:
            env_config = PipelineConfig.from_env()
            db_path = env_config.imaging_db_path
        except Exception:
            db_path = base_dir / "imaging.sqlite3"
    db_manager = DatabaseManager(db_path)

    # machines.json
    cfg = load_machines_config(Path(args.machines_config))
    if args.machines:
        keep = {m: cfg.machines[m] for m in args.machines if m in cfg.machines}
        missing = [m for m in args.machines if m not in cfg.machines]
        if missing:
            log.error("--machines references entries not in config: %s", missing)
            return 2
        cfg.machines = keep

    if not cfg.machines:
        log.error("no machines selected")
        return 2

    # Reconcile-only mode
    if args.reconcile_only:
        adoptable = reconcile_prior(
            db_manager, base_dir, cfg.global_cfg,
            abandon=args.abandon_prior,
        )
        log.info("reconcile-only complete; adoptable=%d", len(adoptable))
        return 0

    # data_dir matches what JointImagingStep / PipelineConfig use:
    # <base_dir.parent>/<project_code>/<project_code>, where base_dir
    # is itself the project_dir.  See panta_rei/config.py:85.
    try:
        env_config = PipelineConfig.from_env()
        config = PipelineConfig(
            panta_rei_base=base_dir.parent,
            weblog_dir=env_config.weblog_dir,
            casa_path=env_config.casa_path,
            project_code=base_dir.name,
        )
    except Exception:
        config = PipelineConfig(
            panta_rei_base=base_dir.parent,
            project_code=base_dir.name,
        )
    data_dir = config.data_dir

    obs_csv = (Path(args.obs_csv).resolve() if args.obs_csv
               else base_dir / "targets_by_array.csv")
    scales = json.loads(args.scales) if args.scales else [0, 5, 10, 15, 20]
    filters = SelectionFilters(
        match=args.match,
        include_sources=args.include_sources,
        include_line_groups=args.include_line_groups,
        limit=args.limit,
        method=args.method,
        deconvolver=args.deconvolver,
        scales=scales,
        re_run=args.re_run,
        exclude_active=True,
    )

    publish_policy = "overwrite" if args.overwrite else "fail_if_exists"
    cli_args = " ".join(sys.argv[1:])

    summary = run_dispatch(
        base_dir=base_dir,
        publish_dir=publish_dir,
        cfg=cfg,
        db_manager=db_manager,
        selection_filters=filters,
        obs_csv=obs_csv,
        data_dir=data_dir,
        transfer_method=args.transfer_method,
        publish_policy=publish_policy,
        deconvolver=args.deconvolver,
        scales=scales,
        cli_args=cli_args,
        abandon_prior=args.abandon_prior,
        dry_run=args.dry_run,
    )
    log.info("dispatch summary: %s", json.dumps(summary, indent=2))
    return 0


# Imported lazily so this module can be imported without matplotlib etc.
from typing import Optional  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
