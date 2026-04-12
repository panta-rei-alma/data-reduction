"""Manual calibration workflow.

Discovers ``ScriptForPI.py`` scripts under the project data directory and
runs them via CASA, tracking each run in the ``pi_runs`` database table.

The heavy lifting is delegated to the helpers already present in
``run_script_for_pi.py`` (discovery, execution, CSV enrichment).  This
module wraps them in a :class:`CalibrateStep` conforming to the
:class:`~panta_rei.workflows.base.Step` interface.
"""

from __future__ import annotations

import csv
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
from panta_rei.core.uid import UID_CORE_RE, canonical_uid
from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import PIRunsQueries, PIRunStatus
from panta_rei.workflows.base import Step, StepResult, WorkflowContext

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Discovery helpers (ported from run_script_for_pi.py)
# ---------------------------------------------------------------------------

_SCRIPT_NAME_RE = re.compile(r"scriptforpi.*\.py$", re.IGNORECASE)
_HIER_RE = re.compile(
    r"(science_goal|group|member)\.?(uid___a\d{3}_x[0-9a-f]+_x[0-9a-f]+)",
    re.IGNORECASE,
)


def _last_uid_in(s: str) -> Optional[str]:
    matches = list(UID_CORE_RE.finditer(s))
    return matches[-1].group(1).lower() if matches else None


def parse_hierarchy_from_path(p: Path) -> Dict[str, Optional[str]]:
    """Extract SG/GOUS/MOUS UIDs from a member directory path."""
    out: Dict[str, Optional[str]] = {
        "sg_uid": None,
        "gous_uid": None,
        "mous_uid": None,
    }
    for role, uid in _HIER_RE.findall(str(p)):
        role_lower = role.lower()
        uid_lower = uid.lower()
        if role_lower == "science_goal":
            out["sg_uid"] = uid_lower
        elif role_lower == "group":
            out["gous_uid"] = uid_lower
        elif role_lower == "member":
            out["mous_uid"] = uid_lower
    return out


def _extract_xpair(uid: str) -> Optional[str]:
    """From ``'uid___a001_x3833_x6571'`` return ``'x3833_x6571'``."""
    m = re.search(r"(x[0-9a-f]+_x[0-9a-f]+)$", uid.lower())
    return m.group(1) if m else None


def discover_scriptforpi(
    base_dir: Path,
) -> Iterator[Tuple[str, Path, Path, Dict[str, Optional[str]]]]:
    """Yield ``(uid, script_path, mous_dir, hierarchy)`` for each MOUS.

    UID priority:
      1. From script filename (member/MOUS UID).
      2. Last UID in the directory path.
      3. ``canonical_uid(str(mous_dir))``.
    """
    for script_path in base_dir.rglob("script/*.py"):
        if not _SCRIPT_NAME_RE.search(script_path.name):
            continue
        mous_dir = script_path.parent.parent

        uid = (
            canonical_uid(script_path.name)
            or _last_uid_in(str(mous_dir))
            or canonical_uid(str(mous_dir))
        )

        if not uid:
            log.debug("Skipping script with no parseable UID: %s", script_path)
            continue

        hierarchy = parse_hierarchy_from_path(mous_dir)
        yield (uid.lower(), script_path.resolve(), mous_dir.resolve(), hierarchy)


# ---------------------------------------------------------------------------
# On-disk completion check
# ---------------------------------------------------------------------------

def find_calibrated_directories(mous_dir: Path) -> List[Path]:
    """Return calibrated measurement-set directories produced by the pipeline."""
    caldir = mous_dir / "calibrated"
    if not caldir.is_dir():
        return []
    found: List[Path] = []
    for child in caldir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name.endswith(".ms") or name.endswith(".ms.split.cal"):
            found.append(child)
    return found


def already_completed(mous_dir: Path) -> bool:
    """Heuristic: calibration exists if ``calibrated/`` contains at least one MS."""
    return bool(find_calibrated_directories(mous_dir))


# ---------------------------------------------------------------------------
# CSV enrichment
# ---------------------------------------------------------------------------

def _load_obs_csv(csv_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """Load obs CSV keyed by ``x####_x####`` from the ``mous_ids`` column."""
    if not csv_path or not csv_path.exists():
        if csv_path:
            log.warning("obs CSV not found: %s (ignoring)", csv_path)
        return {}
    obs: Dict[str, Dict[str, str]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = str(row.get("mous_ids", "")).strip()
            if not token:
                continue
            key = token.lower()
            obs[key] = row
    return obs


# ---------------------------------------------------------------------------
# Back-fill synthetic success
# ---------------------------------------------------------------------------

def _ensure_db_success_for_existing_outputs(
    db_manager: DatabaseManager,
    uid: str,
    script_path: Path,
    mous_dir: Path,
    casa_cmd_tmpl: str,
    extra_meta: Dict[str, Optional[str]],
) -> None:
    """Record a synthetic success row when calibrated products already exist."""
    with db_manager.connect() as con:
        if PIRunsQueries.latest_success_exists(con, uid):
            return

    ms_dirs = find_calibrated_directories(mous_dir)
    if not ms_dirs:
        return

    log_dir = mous_dir / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"scriptforpi_auto_success_{ts}.log"
    with open(log_path, "w") as lf:
        lf.write(
            "# Auto-detected calibrated products; recording success without rerun.\n"
        )
        lf.write(f"# generated={now_iso()}\n")
        for child in ms_dirs:
            lf.write(f"found={child.name}\n")

    with db_manager.connect() as con:
        row_id = PIRunsQueries.insert_row(
            con,
            uid=uid,
            sg_uid=extra_meta.get("sg_uid"),
            gous_uid=extra_meta.get("gous_uid"),
            mous_uid=extra_meta.get("mous_uid") or uid,
            array=extra_meta.get("array"),
            sb_name=extra_meta.get("sb_name"),
            source_name=extra_meta.get("source_name"),
            line_group=extra_meta.get("line_group"),
            script_path=str(script_path),
            cwd=str(script_path.parent),
            casa_cmd=casa_cmd_tmpl.format(script=script_path.name),
            log_path=str(log_path),
            started_at=now_iso(),
            finished_at=None,
            retcode=None,
            status=PIRunStatus.QUEUED,
            hostname=socket.gethostname(),
            duration_sec=None,
        )
        PIRunsQueries.mark_done(
            con,
            row_id,
            status=PIRunStatus.SUCCESS,
            retcode=0,
            finished_at=now_iso(),
            duration_sec=0.0,
        )
        con.commit()

    log.info(
        "Recorded DB success for %s based on existing calibrated outputs (%d dirs).",
        uid,
        len(ms_dirs),
    )


# ---------------------------------------------------------------------------
# Single MOUS execution
# ---------------------------------------------------------------------------

def _run_one(
    db_manager: DatabaseManager,
    uid: str,
    script_path: Path,
    mous_dir: Path,
    casa_cmd_tmpl: str,
    extra_meta: Dict[str, Optional[str]],
    dry_run: bool = False,
) -> Tuple[str, int, Path]:
    """Execute one ScriptForPI run from within the ``script/`` directory.

    Returns ``(status, retcode, log_path)``.
    """
    script_dir = script_path.parent
    script_basename = script_path.name

    log_dir = mous_dir / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"scriptforpi_{ts}.log"

    casa_cmd_str = casa_cmd_tmpl.format(script=script_basename)

    with db_manager.connect() as con:
        row_id = PIRunsQueries.insert_row(
            con,
            uid=uid,
            sg_uid=extra_meta.get("sg_uid"),
            gous_uid=extra_meta.get("gous_uid"),
            mous_uid=extra_meta.get("mous_uid") or uid,
            array=extra_meta.get("array"),
            sb_name=extra_meta.get("sb_name"),
            source_name=extra_meta.get("source_name"),
            line_group=extra_meta.get("line_group"),
            script_path=str(script_path),
            cwd=str(script_dir),
            casa_cmd=casa_cmd_str,
            log_path=str(log_path),
            started_at=now_iso(),
            finished_at=None,
            retcode=None,
            status=PIRunStatus.QUEUED,
            hostname=socket.gethostname(),
            duration_sec=None,
        )
        con.commit()

    if dry_run:
        log.info(
            "[DRY] Would run: UID=%s\n      cwd=%s\n      cmd=%s\n      log=%s",
            uid,
            script_dir,
            casa_cmd_str,
            log_path,
        )
        with db_manager.connect() as con:
            PIRunsQueries.mark_done(
                con,
                row_id,
                status=PIRunStatus.SKIPPED,
                retcode=0,
                finished_at=now_iso(),
                duration_sec=0.0,
            )
            con.commit()
        return (PIRunStatus.SKIPPED, 0, log_path)

    with db_manager.connect() as con:
        PIRunsQueries.mark_running(con, row_id)
        con.commit()

    t0 = dt.datetime.now()
    with open(log_path, "w") as lf:
        lf.write(
            f"# UID={uid}\n# started={t0.isoformat()}\n"
            f"# cwd={script_dir}\n# cmd={casa_cmd_str}\n\n"
        )
        try:
            proc = subprocess.run(
                shlex.split(casa_cmd_str),
                cwd=str(script_dir),
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

    # Check for existing calibrated products on non-zero exit
    if status == PIRunStatus.FAILED:
        ms_dirs = find_calibrated_directories(mous_dir)
        if ms_dirs:
            status = PIRunStatus.SUCCESS
            original_ret = ret
            ret = 0
            try:
                with open(log_path, "a") as lf:
                    lf.write(
                        "\n[INFO] Detected existing calibrated products; "
                        "treating pipeline exit as success.\n"
                    )
                    lf.write(f"[INFO] original_retcode={original_ret}\n")
                    for child in ms_dirs:
                        lf.write(f"[INFO] existing={child.name}\n")
            except Exception:
                pass
            log.info(
                "Treating %s ret=%s as success because calibrated outputs "
                "already exist (%d dirs).",
                uid,
                original_ret,
                len(ms_dirs),
            )

    with db_manager.connect() as con:
        PIRunsQueries.mark_done(
            con,
            row_id,
            status=status,
            retcode=ret,
            finished_at=now_iso(),
            duration_sec=dt_sec,
        )
        con.commit()

    log.info(
        "Finished %s: %s (ret=%s, %.1fs)  log=%s",
        uid,
        status,
        ret,
        dt_sec,
        log_path,
    )

    # Symlink to latest log
    try:
        latest = log_dir / "last_scriptforpi.log"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log_path.name)
    except Exception:
        pass

    return (status, ret, log_path)


# ---------------------------------------------------------------------------
# Calibration step configuration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationOptions:
    """Extra parameters for the calibration step.

    These augment the general-purpose :class:`WorkflowContext` with
    calibration-specific knobs.
    """

    casa_cmd: str = 'casa --nologger --nogui --pipeline -c "{script}"'
    only_new: bool = False
    re_run: bool = False
    match: Optional[str] = None
    limit: Optional[int] = None
    obs_csv: Optional[Path] = None
    include_arrays: Optional[List[str]] = None
    skip_tp: bool = False


# ---------------------------------------------------------------------------
# CalibrateStep
# ---------------------------------------------------------------------------

class CalibrateStep(Step):
    """Discover ScriptForPI.py scripts and run them via CASA."""

    def __init__(self, options: Optional[CalibrationOptions] = None) -> None:
        self._opts = options or CalibrationOptions()

    @property
    def name(self) -> str:
        return "calibrate"

    @property
    def description(self) -> str:
        return "Run ScriptForPI calibration via CASA"

    def should_skip(self, ctx: WorkflowContext) -> str | None:
        base = super().should_skip(ctx)
        if base is not None:
            return base
        if not ctx.data_dir.is_dir():
            return f"data directory does not exist: {ctx.data_dir}"
        return None

    def run(self, ctx: WorkflowContext) -> StepResult:
        opts = self._opts

        # Use CASA command from config if available and user did not override
        casa_cmd_tmpl = opts.casa_cmd
        if ctx.config.casa_cmd is not None:
            # Build a template from the config's pre-formatted command
            casa_cmd_tmpl = ctx.config.casa_cmd + ' -c "{script}"'

        # Load optional obs CSV for enrichment/filtering
        obs = _load_obs_csv(opts.obs_csv)

        # Resolve array filter
        include_arrays = opts.include_arrays
        if include_arrays is None and opts.skip_tp:
            include_arrays = ["TM", "SM"]
        if include_arrays is not None and not obs:
            log.warning(
                "Array filter requested but no obs CSV available -- "
                "array filter will be ignored."
            )
            include_arrays = None

        # Compile optional regex filter
        regex: Optional[re.Pattern] = None
        if opts.match:
            try:
                regex = re.compile(opts.match, re.IGNORECASE)
            except re.error:
                regex = None

        def keep_uid(uid: str) -> bool:
            if not opts.match:
                return True
            if regex:
                return bool(regex.search(uid))
            return opts.match.lower() in uid.lower()

        # Discover scripts on disk
        discovered = list(discover_scriptforpi(ctx.data_dir))
        if not discovered:
            return StepResult(
                success=True,
                summary="No ScriptForPI files found",
                items_processed=0,
            )

        runs = 0
        skipped = 0
        errors: list[str] = []

        for uid, script_path, mous_dir, hierarchy in discovered:
            if not keep_uid(uid):
                skipped += 1
                continue

            # Build metadata dict
            meta: Dict[str, Optional[str]] = {
                "sg_uid": hierarchy.get("sg_uid"),
                "gous_uid": hierarchy.get("gous_uid"),
                "mous_uid": hierarchy.get("mous_uid") or uid,
                "array": None,
                "sb_name": None,
                "source_name": None,
                "line_group": None,
            }

            # CSV enrichment
            if obs:
                xpair = _extract_xpair(uid)
                row = obs.get(xpair) if xpair else None
                if row is None:
                    log.info("Skipping %s (not present in obs CSV).", uid)
                    skipped += 1
                    continue
                meta["array"] = str(row.get("array", "")).strip() or None
                meta["sb_name"] = str(row.get("sb_name", "")).strip() or None
                meta["source_name"] = (
                    str(row.get("source_name", "")).strip() or None
                )
                meta["line_group"] = (
                    str(row.get("Line group", "")).strip() or None
                )

                if (
                    include_arrays is not None
                    and meta["array"] not in include_arrays
                ):
                    log.info(
                        "Skipping %s (array %s not in %s).",
                        uid,
                        meta["array"],
                        include_arrays,
                    )
                    skipped += 1
                    continue

            # Idempotence: DB + on-disk heuristic
            with ctx.db_manager.connect() as con:
                db_success = PIRunsQueries.latest_success_exists(con, uid)

            if opts.only_new and not opts.re_run and db_success:
                log.info(
                    "Skipping %s (already successful in DB). Use re_run to force.",
                    uid,
                )
                skipped += 1
                continue

            if not opts.re_run and already_completed(mous_dir):
                if not ctx.dry_run:
                    _ensure_db_success_for_existing_outputs(
                        db_manager=ctx.db_manager,
                        uid=uid,
                        script_path=script_path,
                        mous_dir=mous_dir,
                        casa_cmd_tmpl=casa_cmd_tmpl,
                        extra_meta=meta,
                    )
                log.info(
                    "Skipping %s (appears already calibrated on disk). "
                    "Use re_run to force.",
                    uid,
                )
                skipped += 1
                continue

            status, _ret, _log_path = _run_one(
                db_manager=ctx.db_manager,
                uid=uid,
                script_path=script_path,
                mous_dir=mous_dir,
                casa_cmd_tmpl=casa_cmd_tmpl,
                extra_meta=meta,
                dry_run=ctx.dry_run,
            )
            runs += 1

            if status == PIRunStatus.FAILED:
                errors.append(f"{uid}: failed (ret={_ret})")

            if opts.limit and runs >= opts.limit:
                log.info("Hit limit=%d, stopping.", opts.limit)
                break

        summary_parts = [f"Processed {runs} MOUS(es)"]
        if skipped:
            summary_parts.append(f"{skipped} skipped")
        if errors:
            summary_parts.append(f"{len(errors)} failed")

        return StepResult(
            success=len(errors) == 0,
            summary="; ".join(summary_parts),
            items_processed=runs,
            items_skipped=skipped,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Convenience orchestration
# ---------------------------------------------------------------------------

def run_calibration(
    ctx: WorkflowContext,
    options: Optional[CalibrationOptions] = None,
) -> dict[str, StepResult]:
    """Run the calibration workflow (single step).

    Parameters
    ----------
    ctx:
        Shared workflow context.
    options:
        Calibration-specific parameters.  Defaults are used if *None*.

    Returns
    -------
    dict mapping ``"calibrate"`` to its :class:`StepResult`.
    """
    from panta_rei.workflows.base import run_workflow

    step = CalibrateStep(options=options)
    return run_workflow([step], ctx)
