"""CLI entry point for continuum subtraction remediation.

Discovers MOUSs where ScriptForPI skipped continuum subtraction and
runs a CASA script to produce the missing ``*_targets_line.ms`` files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from panta_rei.config import PipelineConfig
from panta_rei.core.logging import setup_logging
from panta_rei.db.connection import DatabaseManager
from panta_rei.workflows.base import WorkflowContext
from panta_rei.workflows.contsub import ContsubOptions, run_contsub


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run continuum subtraction for calibrated MOUSs that are "
            "missing *_targets_line.ms (because ScriptForPI skipped the "
            "contsub branch)."
        ),
    )
    ap.add_argument(
        "--base-dir", required=True,
        help="Project base directory (e.g. ./2025.1.00383.L)",
    )
    ap.add_argument(
        "--db", default=None,
        help=(
            "Path to the SQLite DB "
            "(default: <base-dir>/alma_retrieval_state.sqlite3)"
        ),
    )
    ap.add_argument(
        "--casa-cmd", default=None,
        help=(
            "Base CASA command (default: from config or "
            "'casa --nologger --nogui --pipeline')"
        ),
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without executing CASA",
    )
    ap.add_argument(
        "--match", default=None,
        help="Regex filter on MOUS UID (e.g. 'X64bc')",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N MOUSs",
    )
    ap.add_argument(
        "--re-run", action="store_true",
        help="Force re-run even when _targets_line.ms already exists",
    )
    ap.add_argument(
        "--obs-csv", type=str, default=None,
        help="Path to targets_by_array.csv for metadata enrichment",
    )
    return ap


def main() -> int:
    """Entry point for ``panta-rei-contsub``."""
    setup_logging()
    args = _build_parser().parse_args()

    base_dir = Path(args.base_dir).resolve()
    db_path = (
        Path(args.db) if args.db
        else base_dir / "alma_retrieval_state.sqlite3"
    )
    db_manager = DatabaseManager(db_path)

    # Resolve CASA command
    casa_cmd = args.casa_cmd
    if casa_cmd is None:
        try:
            cfg = PipelineConfig.from_env()
            casa_cmd = cfg.casa_cmd
        except Exception:
            pass
    if casa_cmd is None:
        casa_cmd = "casa --nologger --nogui --pipeline"

    # Build data_dir: base_dir is project_dir, data_dir is project_dir/project_code
    # For contsub we need data_dir to point to where science_goal.* dirs live
    data_dir = base_dir / base_dir.name
    if not data_dir.is_dir():
        data_dir = base_dir

    ctx = WorkflowContext(
        config=PipelineConfig(panta_rei_base=base_dir.parent),
        db_manager=db_manager,
        dry_run=args.dry_run,
    )

    options = ContsubOptions(
        casa_cmd=casa_cmd,
        match=args.match,
        limit=args.limit,
        re_run=args.re_run,
        obs_csv=Path(args.obs_csv) if args.obs_csv else None,
    )

    result = run_contsub(ctx, options)

    if not result.success:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
