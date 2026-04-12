"""CLI entry point for the daily retrieval pipeline.

Mirrors the interface of the legacy ``run_retrieval_scripts.py``.

Steps:
    1. Download + extract new/updated UIDs (respects SQLite state)
    2. Rebuild the targets-by-array CSV
    3. Stage weblogs for public access
    4. Create/update GitHub issues for tracking
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from panta_rei.core.logging import setup_logging

log = logging.getLogger(__name__)

DEFAULT_PROJECT_CODE = "2025.1.00383.L"
DEFAULT_WEBLOG_DIR = "/scratch/almanas/dwalker2/panta-rei/weblogs"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Retrieve ALMA data and rebuild the observation table together (systemd/headless)."
    )
    ap.add_argument(
        "username", nargs="?",
        help="ALMA archive username (or set ALMA_USERNAME)",
    )
    ap.add_argument(
        "--project-code",
        default=DEFAULT_PROJECT_CODE,
        help=f"Project code (default: {DEFAULT_PROJECT_CODE})",
    )
    ap.add_argument(
        "--base-dir",
        default=None,
        help="Base directory (default: ./<project-code>)",
    )
    ap.add_argument(
        "--db",
        default=None,
        help="Path to SQLite state DB (default: <base-dir>/alma_retrieval_state.sqlite3)",
    )
    ap.add_argument(
        "--csv-out",
        default=None,
        help="Output CSV path for the table (default: <base-dir>/targets_by_array.csv)",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Path to extracted ALMA data directory containing science_goal.* subdirs "
            "(default: <base-dir>/<project-code>)"
        ),
    )

    # Headless credential controls
    ap.add_argument(
        "--non-interactive", action="store_true",
        help="Fail fast if creds are missing; never prompt",
    )

    # Weblog staging options
    ap.add_argument(
        "--weblog-dir",
        default=DEFAULT_WEBLOG_DIR,
        help=f"Directory for staged weblogs (default: {DEFAULT_WEBLOG_DIR})",
    )

    # Convenience toggles
    ap.add_argument(
        "--skip-download", action="store_true",
        help="Skip step 1 (download + extract)",
    )
    ap.add_argument(
        "--skip-table", action="store_true",
        help="Skip step 2 (rebuild table)",
    )
    ap.add_argument(
        "--skip-weblogs", action="store_true",
        help="Skip step 3 (stage weblogs)",
    )
    ap.add_argument(
        "--skip-issues", action="store_true",
        help="Skip step 4 (GitHub issues)",
    )

    return ap


def main() -> int:
    """Entry point for ``panta-rei-pipeline``."""
    # Disable desktop keyrings early, matching legacy behaviour
    os.environ.setdefault(
        "PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring"
    )

    setup_logging()
    args = _build_parser().parse_args()

    # -- Resolve credentials ------------------------------------------------
    from panta_rei.auth import (
        install_headless_password,
        resolve_alma_creds,
    )

    username, password = resolve_alma_creds(args.username)

    # Only require credentials when steps that need them are active
    needs_alma_creds = not (args.skip_download and args.skip_table)
    if args.non_interactive and needs_alma_creds and (not username or not password):
        log.error("No credentials available in non-interactive mode.")
        return 1

    if password:
        install_headless_password(password)

    # -- Resolve paths matching legacy run_retrieval_scripts.py semantics ----
    # IMPORTANT: --base-dir is the PROJECT directory (e.g. .../2025.1.00383.L),
    # not panta_rei_base.  Legacy: db, csv, and weblogs all live at this level.
    from panta_rei.config import PipelineConfig

    base_dir = Path(args.base_dir or f"./{args.project_code}").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db) if args.db else (base_dir / "alma_retrieval_state.sqlite3")
    csv_out = Path(args.csv_out) if args.csv_out else (base_dir / "targets_by_array.csv")
    data_dir = Path(args.data_dir) if args.data_dir else (base_dir / args.project_code)
    weblog_dir = Path(args.weblog_dir).resolve()

    # panta_rei_base is the PARENT of the project dir
    panta_rei_base = base_dir.parent
    cfg_kwargs: dict = {
        "panta_rei_base": panta_rei_base,
        "project_code": args.project_code,
        "weblog_dir": weblog_dir,
    }
    # Allow env-var overrides for GitHub target (needed for test-repo validation)
    gh_owner = os.environ.get("GH_OWNER")
    if gh_owner:
        cfg_kwargs["gh_owner"] = gh_owner
    gh_repo = os.environ.get("GH_REPO")
    if gh_repo:
        cfg_kwargs["gh_repo"] = gh_repo
    cfg = PipelineConfig(**cfg_kwargs)

    # Override derived paths so CLI args take effect end-to-end
    object.__setattr__(cfg, "project_dir", base_dir)
    object.__setattr__(cfg, "data_dir", data_dir)
    object.__setattr__(cfg, "state_db_path", db_path)
    object.__setattr__(cfg, "targets_csv_path", csv_out)

    # -- Create DatabaseManager ---------------------------------------------
    from panta_rei.db.connection import DatabaseManager

    db_manager = DatabaseManager(db_path)

    # -- Build WorkflowContext ----------------------------------------------
    from panta_rei.workflows.base import WorkflowContext

    skip_steps: set[str] = set()
    if args.skip_download:
        skip_steps.add("retrieve")
    if args.skip_table:
        skip_steps.add("build_table")
    if args.skip_weblogs:
        skip_steps.add("stage_weblogs")
    if args.skip_issues:
        skip_steps.add("update_issues")

    ctx = WorkflowContext(
        config=cfg,
        db_manager=db_manager,
        username=username,
        skip_steps=skip_steps,
        non_interactive=args.non_interactive,
    )

    # -- Run the retrieval pipeline -----------------------------------------
    from panta_rei.workflows.retrieval import run_retrieval

    results = run_retrieval(ctx)

    # -- Log results --------------------------------------------------------
    any_failed = False
    for step_name, result in results.items():
        status = "OK" if result.success else "FAILED"
        log.info(
            "Step %-15s [%s]: %s", step_name, status, result.summary,
        )
        if not result.success:
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
