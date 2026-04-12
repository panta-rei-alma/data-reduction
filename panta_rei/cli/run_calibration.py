"""CLI entry point for running CASA ScriptForPI per delivered MOUS.

Mirrors the interface of the legacy ``run_script_for_pi.py``.

Discovers ``script/*scriptforpi*.py`` files under ``--base-dir``, applies
optional filters (UID regex, array type, obs CSV), and runs CASA
calibration with idempotence tracking in the retrieval SQLite DB.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from panta_rei.core.logging import setup_logging
from panta_rei.core.text import now_iso
from panta_rei.core.uid import UID_CORE_RE, canonical_uid
from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import PIRunsQueries, PIRunStatus

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (ported from legacy run_script_for_pi.py)
# ---------------------------------------------------------------------------

_SCRIPT_NAME_RE = re.compile(r"scriptforpi.*\.py$", re.IGNORECASE)
_HIER_RE = re.compile(
    r"(science_goal|group|member)\.?(uid___a\d{3}_x[0-9a-f]+_x[0-9a-f]+)",
    re.IGNORECASE,
)


def _last_uid_in(s: str) -> Optional[str]:
    matches = list(UID_CORE_RE.finditer(s))
    return matches[-1].group(1).lower() if matches else None


def _parse_hierarchy_from_path(p: Path) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"sg_uid": None, "gous_uid": None, "mous_uid": None}
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
    m = re.search(r"(x[0-9a-f]+_x[0-9a-f]+)$", uid.lower())
    return m.group(1) if m else None


def _find_calibrated_directories(mous_dir: Path) -> List[Path]:
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


def _already_completed(mous_dir: Path) -> bool:
    return bool(_find_calibrated_directories(mous_dir))


def _ensure_db_success_for_existing_outputs(
    db_manager: DatabaseManager,
    uid: str,
    script_path: Path,
    mous_dir: Path,
    casa_cmd_tmpl: str,
    extra_meta: Dict[str, Optional[str]],
) -> None:
    with db_manager.connect() as con:
        if PIRunsQueries.latest_success_exists(con, uid):
            return

    ms_dirs = _find_calibrated_directories(mous_dir)
    if not ms_dirs:
        return

    log_dir = mous_dir / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"scriptforpi_auto_success_{ts}.log"
    with open(log_path, "w") as lf:
        lf.write("# Auto-detected calibrated products; recording success without rerun.\n")
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
            con, row_id,
            status=PIRunStatus.SUCCESS,
            retcode=0,
            finished_at=now_iso(),
            duration_sec=0.0,
        )
    log.info(
        "Recorded DB success for %s based on existing calibrated outputs (%d dirs).",
        uid, len(ms_dirs),
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_scriptforpi(
    base_dir: Path,
) -> Iterator[Tuple[str, Path, Path, Dict[str, Optional[str]]]]:
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

        hierarchy = _parse_hierarchy_from_path(mous_dir)
        yield (uid.lower(), script_path.resolve(), mous_dir.resolve(), hierarchy)


# ---------------------------------------------------------------------------
# CSV enrichment / filtering
# ---------------------------------------------------------------------------

def _load_obs_csv(csv_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not csv_path:
        return {}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        log.warning("obs CSV not found: %s (ignoring)", csv_path)
        return {}
    import csv as _csv

    obs: Dict[str, Dict[str, str]] = {}
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            token = str(row.get("mous_ids", "")).strip()
            if not token:
                continue
            key = token.lower()
            if not key.startswith("x"):
                key = key.replace("X", "x").replace("x", "x")
            obs[key] = row
    return obs


# ---------------------------------------------------------------------------
# Execution
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

    if dry_run:
        log.info(
            "[DRY] Would run: UID=%s\n      cwd=%s\n      cmd=%s\n      log=%s",
            uid, script_dir, casa_cmd_str, log_path,
        )
        with db_manager.connect() as con:
            PIRunsQueries.mark_done(
                con, row_id,
                status=PIRunStatus.SKIPPED,
                retcode=0,
                finished_at=now_iso(),
                duration_sec=0.0,
            )
        return ("skipped", 0, log_path)

    with db_manager.connect() as con:
        PIRunsQueries.mark_running(con, row_id)

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

    if status == PIRunStatus.FAILED:
        ms_dirs = _find_calibrated_directories(mous_dir)
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
                uid, original_ret, len(ms_dirs),
            )

    with db_manager.connect() as con:
        PIRunsQueries.mark_done(
            con, row_id,
            status=status,
            retcode=ret,
            finished_at=now_iso(),
            duration_sec=dt_sec,
        )
    log.info(
        "Finished %s: %s (ret=%s, %.1fs)  log=%s",
        uid, status, ret, dt_sec, log_path,
    )

    # Maintain a handy symlink to the latest log
    try:
        latest = log_dir / "last_scriptforpi.log"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log_path.name)
    except Exception:
        pass

    return (status, ret, log_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run ScriptForPI for delivered MOUSes and record results."
    )
    ap.add_argument(
        "--base-dir", required=True,
        help="Project base directory where tarballs were extracted (e.g. ./2025.1.00383.L)",
    )
    ap.add_argument(
        "--db", default=None,
        help="Path to the retrieval SQLite DB (default: <base-dir>/alma_retrieval_state.sqlite3)",
    )
    ap.add_argument(
        "--casa-cmd",
        default='casa --nologger --nogui --pipeline -c "{script}"',
        help='Command template to run CASA. Must include {script}.',
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Do not execute CASA; record as skipped",
    )
    ap.add_argument(
        "--only-new", action="store_true",
        help="Skip MOUSes with a previous successful run in DB",
    )
    ap.add_argument(
        "--re-run", action="store_true",
        help="Force re-run even if a success exists (overrides --only-new)",
    )
    ap.add_argument(
        "--match", default=None,
        help="Substring or regex to filter UIDs (e.g. 'X64bc' or 'uid___a001_x3833_x64bc')",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Run at most N MOUSes",
    )
    ap.add_argument(
        "--obs-csv", type=str, default=None,
        help="Path to obs_table.csv for filtering/enrichment",
    )
    ap.add_argument(
        "--include-arrays", nargs="+", choices=["TM", "SM", "TP"],
        help="Only run these arrays (requires --obs-csv). Default: all arrays.",
    )
    ap.add_argument(
        "--skip-tp", action="store_true",
        help="Convenience: if no --include-arrays given, treat as --include-arrays TM SM",
    )
    return ap


def main() -> int:
    """Entry point for ``panta-rei-calibrate``."""
    setup_logging()
    args = _build_parser().parse_args()

    base_dir = Path(args.base_dir).resolve()
    db_path = Path(args.db) if args.db else (base_dir / "alma_retrieval_state.sqlite3")
    db_manager = DatabaseManager(db_path)

    # Load CSV enrichment (optional)
    obs = _load_obs_csv(Path(args.obs_csv) if args.obs_csv else None)

    # Resolve array filter
    include_arrays: Optional[List[str]] = None
    if args.include_arrays:
        include_arrays = [s.upper() for s in args.include_arrays]
    elif args.skip_tp:
        include_arrays = ["TM", "SM"]

    if (include_arrays is not None) and not obs:
        log.warning(
            "--include-arrays/--skip-tp requested but no --obs-csv given; "
            "array filter will be ignored."
        )
        include_arrays = None

    # Compile optional regex filter
    regex = None
    if args.match:
        try:
            regex = re.compile(args.match, re.IGNORECASE)
        except re.error:
            regex = None

    def keep_uid(uid: str) -> bool:
        if not args.match:
            return True
        if regex:
            return bool(regex.search(uid))
        return args.match.lower() in uid.lower()

    # Discover everything on disk
    discovered = list(_discover_scriptforpi(base_dir))
    if not discovered:
        log.info("No ScriptForPI files found under: %s", base_dir)
        return 0

    runs = 0
    any_failed = False
    for uid, script_path, mous_dir, hierarchy in discovered:
        if not keep_uid(uid):
            continue

        meta: Dict[str, Optional[str]] = {
            "sg_uid": hierarchy.get("sg_uid"),
            "gous_uid": hierarchy.get("gous_uid"),
            "mous_uid": hierarchy.get("mous_uid") or uid,
            "array": None,
            "sb_name": None,
            "source_name": None,
            "line_group": None,
        }

        # CSV-based enrichment + filtering
        if obs:
            xpair = _extract_xpair(uid)
            row = obs.get(xpair) if xpair else None
            if row is None:
                log.info("Skipping %s (not present in obs CSV).", uid)
                continue
            meta["array"] = str(row.get("array", "")).strip() or None
            meta["sb_name"] = str(row.get("sb_name", "")).strip() or None
            meta["source_name"] = str(row.get("source_name", "")).strip() or None
            meta["line_group"] = str(row.get("Line group", "")).strip() or None

            if include_arrays is not None and meta["array"] not in include_arrays:
                log.info("Skipping %s (array %s not in %s).", uid, meta["array"], include_arrays)
                continue

        # Idempotence: DB + on-disk heuristic
        if args.only_new and not args.re_run:
            with db_manager.connect() as con:
                if PIRunsQueries.latest_success_exists(con, uid):
                    log.info("Skipping %s (already successful in DB). Use --re-run to force.", uid)
                    continue
        if not args.re_run and _already_completed(mous_dir):
            if not args.dry_run:
                _ensure_db_success_for_existing_outputs(
                    db_manager=db_manager,
                    uid=uid,
                    script_path=script_path,
                    mous_dir=mous_dir,
                    casa_cmd_tmpl=args.casa_cmd,
                    extra_meta=meta,
                )
            log.info("Skipping %s (appears already calibrated on disk). Use --re-run to force.", uid)
            continue

        status, ret, log_path = _run_one(
            db_manager=db_manager,
            uid=uid,
            script_path=script_path,
            mous_dir=mous_dir,
            casa_cmd_tmpl=args.casa_cmd,
            extra_meta=meta,
            dry_run=args.dry_run,
        )
        if status == "failed":
            any_failed = True
        runs += 1
        if args.limit and runs >= args.limit:
            log.info("Hit --limit=%d, stopping.", args.limit)
            break

    log.info("Done. MOUS processed this invocation: %d", runs)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
