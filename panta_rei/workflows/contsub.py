"""Continuum subtraction remediation workflow.

Detects MOUSs where ScriptForPI skipped the contsub branch and runs a
standalone CASA script to produce the missing ``*_targets_line.ms`` files.

Detection, subprocess invocation, and DB tracking follow the patterns
established in :mod:`panta_rei.workflows.calibration`.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import shlex
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from panta_rei.core.text import now_iso
from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import ContsubRunsQueries, PIRunStatus
from panta_rei.workflows.base import Step, StepResult, WorkflowContext
from panta_rei.workflows.calibration import (
    _load_obs_csv,
    discover_scriptforpi,
    parse_hierarchy_from_path,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _find_base_ms_files(working_dir: Path) -> List[Path]:
    """Return base calibrated MS dirs (not _targets*, .flagversions, .tbl)."""
    found: List[Path] = []
    for child in sorted(working_dir.glob("uid___A002_*.ms")):
        name = child.name
        if "_targets" in name:
            continue
        if not child.is_dir():
            continue
        found.append(child)
    return found


def needs_contsub(member_dir: Path) -> Tuple[bool, str]:
    """Check whether a MOUS needs continuum subtraction remediation.

    On-disk per-EB completeness is the authoritative check.

    Returns ``(True, reason)`` if remediation is needed, or
    ``(False, reason)`` with a diagnostic explanation otherwise.
    """
    working_dir = member_dir / "calibrated" / "working"
    if not working_dir.is_dir():
        return False, "no calibrated/working/ directory"

    base_ms = _find_base_ms_files(working_dir)
    if not base_ms:
        return False, "no calibrated MS files in working dir"

    # Check cont.dat existence (glob, matching ScriptForPI pattern)
    contfiles = list((member_dir / "calibration").glob("*cont.dat"))
    if not contfiles:
        return False, "no *cont.dat in calibration/ — pipeline did not identify continuum"

    # Per-EB completeness check
    missing: List[str] = []
    for ms_path in base_ms:
        expected_name = ms_path.name.replace(".ms", "_targets_line.ms")
        expected_path = working_dir / expected_name
        if not expected_path.is_dir():
            missing.append(expected_name)

    if not missing:
        return False, "all _targets_line.ms files present"

    return True, (
        f"{len(missing)} of {len(base_ms)} EBs missing _targets_line.ms: "
        + ", ".join(missing)
    )


# ---------------------------------------------------------------------------
# Single MOUS execution
# ---------------------------------------------------------------------------

def _run_one_contsub(
    db_manager: DatabaseManager,
    uid: str,
    member_dir: Path,
    casa_cmd: str,
    contsub_script: Path,
    extra_meta: Dict[str, Optional[str]],
    dry_run: bool = False,
) -> Tuple[str, int, Path]:
    """Run the contsub CASA script for one MOUS.

    Returns ``(status, retcode, log_path)``.
    """
    working_dir = member_dir / "calibrated" / "working"
    base_ms = _find_base_ms_files(working_dir)
    eb_count = len(base_ms)

    log_dir = member_dir / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"contsub_{ts}.log"

    full_cmd = f"{casa_cmd} -c {contsub_script} --member-dir {member_dir}"

    with db_manager.connect() as con:
        row_id = ContsubRunsQueries.insert_row(
            con,
            uid=uid,
            sg_uid=extra_meta.get("sg_uid"),
            gous_uid=extra_meta.get("gous_uid"),
            mous_uid=extra_meta.get("mous_uid") or uid,
            array=extra_meta.get("array"),
            sb_name=extra_meta.get("sb_name"),
            source_name=extra_meta.get("source_name"),
            line_group=extra_meta.get("line_group"),
            member_dir=str(member_dir),
            working_dir=str(working_dir),
            casa_cmd=full_cmd,
            log_path=str(log_path),
            started_at=now_iso(),
            status=PIRunStatus.QUEUED,
            hostname=socket.gethostname(),
            eb_count=eb_count,
        )
        con.commit()

    if dry_run:
        log.info(
            "[DRY] Would run contsub: UID=%s (%d EBs)\n"
            "      member_dir=%s\n      cmd=%s\n      log=%s",
            uid, eb_count, member_dir, full_cmd, log_path,
        )
        with db_manager.connect() as con:
            ContsubRunsQueries.mark_done(
                con, row_id,
                status=PIRunStatus.SKIPPED,
                retcode=0,
                finished_at=now_iso(),
                duration_sec=0.0,
            )
            con.commit()
        return (PIRunStatus.SKIPPED, 0, log_path)

    with db_manager.connect() as con:
        ContsubRunsQueries.mark_running(con, row_id)
        con.commit()

    t0 = dt.datetime.now()
    with open(log_path, "w") as lf:
        lf.write(
            f"# UID={uid}\n# started={t0.isoformat()}\n"
            f"# member_dir={member_dir}\n# cmd={full_cmd}\n\n"
        )
        try:
            proc = subprocess.run(
                shlex.split(full_cmd),
                cwd=str(member_dir),
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
            )
            ret = proc.returncode
        except FileNotFoundError as e:
            lf.write(f"\n[ERROR] {e}\n")
            ret = 127
        except Exception as e:
            lf.write(f"\n[EXCEPTION] {e}\n")
            ret = 1

    dt_sec = (dt.datetime.now() - t0).total_seconds()
    status = PIRunStatus.SUCCESS if ret == 0 else PIRunStatus.FAILED

    # Count created _targets_line.ms
    targets_line = list(working_dir.glob("*_targets_line.ms"))
    targets_line_count = len([p for p in targets_line if p.is_dir()])

    with db_manager.connect() as con:
        ContsubRunsQueries.mark_done(
            con, row_id,
            status=status,
            retcode=ret,
            finished_at=now_iso(),
            duration_sec=dt_sec,
            targets_line_count=targets_line_count,
        )
        con.commit()

    log.info(
        "Finished contsub %s: %s (ret=%s, %.1fs, %d targets_line.ms)  log=%s",
        uid, status, ret, dt_sec, targets_line_count, log_path,
    )

    return (status, ret, log_path)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class ContsubOptions:
    """Parameters for the contsub remediation step."""

    casa_cmd: str = "casa --nologger --nogui --pipeline"
    match: Optional[str] = None
    limit: Optional[int] = None
    re_run: bool = False
    obs_csv: Optional[Path] = None


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class ContsubStep(Step):
    """Run continuum subtraction for MOUSs missing ``_targets_line.ms``."""

    def __init__(self, options: ContsubOptions) -> None:
        self._options = options

    @property
    def name(self) -> str:
        return "contsub"

    @property
    def description(self) -> str:
        return (
            "Run continuum subtraction for calibrated MOUSs "
            "missing _targets_line.ms"
        )

    def run(self, ctx: WorkflowContext) -> StepResult:
        opts = self._options

        # Locate the CASA remediation script
        contsub_script = (
            Path(__file__).parent.parent / "casa" / "contsub_remediation.py"
        ).resolve()
        if not contsub_script.exists():
            return StepResult(
                success=False,
                summary=f"contsub script not found: {contsub_script}",
            )

        # Load obs CSV enrichment
        obs_data = _load_obs_csv(opts.obs_csv)

        # Discover all MOUSs via scriptForPI discovery
        match_re = re.compile(opts.match, re.IGNORECASE) if opts.match else None
        processed = 0
        skipped = 0
        errors: List[str] = []

        for uid, _script_path, mous_dir, hierarchy in discover_scriptforpi(
            ctx.data_dir
        ):
            # Apply match filter
            if match_re and not match_re.search(uid):
                continue

            # Check if contsub is needed (on-disk is authoritative)
            if not opts.re_run:
                needed, reason = needs_contsub(mous_dir)
                if not needed:
                    log.debug("Skipping %s: %s", uid, reason)
                    skipped += 1
                    continue

            # Apply limit
            if opts.limit is not None and processed >= opts.limit:
                log.info("Limit of %d reached, stopping.", opts.limit)
                break

            # Enrich metadata from CSV
            from panta_rei.workflows.calibration import _extract_xpair
            xpair = _extract_xpair(uid)
            csv_row = obs_data.get(xpair, {}) if xpair else {}
            extra_meta = {**hierarchy}
            if csv_row:
                extra_meta.setdefault("array", csv_row.get("array"))
                extra_meta.setdefault("sb_name", csv_row.get("sb_name"))
                extra_meta.setdefault("source_name", csv_row.get("source_name"))
                extra_meta.setdefault(
                    "line_group", csv_row.get("Line group")
                )

            needed, reason = needs_contsub(mous_dir)
            log.info(
                "Processing %s: %s (%s)",
                uid,
                "needs contsub" if needed else "complete",
                reason,
            )

            status, ret, log_path = _run_one_contsub(
                db_manager=ctx.db_manager,
                uid=uid,
                member_dir=mous_dir,
                casa_cmd=opts.casa_cmd,
                contsub_script=contsub_script,
                extra_meta=extra_meta,
                dry_run=ctx.dry_run,
            )

            processed += 1
            if status == PIRunStatus.FAILED:
                errors.append(f"{uid}: failed (ret={ret})")

        # Summary
        with ctx.db_manager.connect() as con:
            db_summary = ContsubRunsQueries.summary(con)

        summary = (
            f"contsub: processed={processed} skipped={skipped} "
            f"errors={len(errors)} | {db_summary}"
        )
        log.info(summary)

        return StepResult(
            success=len(errors) == 0,
            summary=summary,
            items_processed=processed,
            items_skipped=skipped,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Convenience orchestration
# ---------------------------------------------------------------------------

def run_contsub(ctx: WorkflowContext, options: ContsubOptions) -> StepResult:
    """Run the contsub remediation step."""
    step = ContsubStep(options)
    return step.run(ctx)
