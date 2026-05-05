"""Detached per-unit worker for distributed imaging.

Invoked via SSH by :mod:`panta_rei.imaging.dispatch`.  Each call processes
exactly one imaging unit on the remote machine, reading its job manifest
from NAS, staging inputs to the local /raid/, calling
:func:`panta_rei.imaging.runner.run_tclean_feather_parallel` with
``work_dir=/raid/...`` and ``publish_dir=<NAS canonical>``, then copying
provenance back to NAS before exit.

Two ownership rules avoid concurrent writers on shared state:

1. **Heartbeat thread** owns ``<state-dir>/heartbeat`` (an empty file
   that is touched).  It NEVER writes ``state.json``.
2. **Main thread** owns ``state.json`` — only writes it on phase
   transitions.  Atomic via temp + ``os.replace``.

Coordinator polls heartbeat mtime + state.json phase.  PID + PGID are
recorded once on first state.json write so the coordinator can verify
liveness via ``ssh kill -0 <pid>`` + ``/proc/<pid>/cmdline`` matching.

Two CLI modes:

- ``--preflight`` : check raid path, free space, conda env, NAS visibility.
  Returns JSON to stdout, exits.
- (default run mode) : full lifecycle described above.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("panta_rei.remote_worker")


# ---------------------------------------------------------------------------
# State machine constants
# ---------------------------------------------------------------------------

class Phase:
    STARTING = "starting"
    STAGING = "staging"
    RUNNING = "running"
    PUBLISHING = "publishing"
    DONE = "done"
    FAILED = "failed"


TERMINAL_PHASES = {Phase.DONE, Phase.FAILED}


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------

class _Heartbeat:
    """Touches a heartbeat file at fixed intervals.  Owns ONLY that file.

    Started before staging so 10+-minute MS copies don't trigger the
    coordinator's stale-heartbeat path.
    """

    def __init__(self, hb_path: Path, interval_sec: float) -> None:
        self.hb_path = Path(hb_path)
        self.interval = max(1.0, float(interval_sec))
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def start(self) -> None:
        self.hb_path.parent.mkdir(parents=True, exist_ok=True)
        self.hb_path.touch()
        self._t = threading.Thread(target=self._run, name="hb", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.hb_path.touch()
            except OSError as exc:
                log.warning("heartbeat touch failed: %s", exc)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# State.json owner (main thread only)
# ---------------------------------------------------------------------------

def write_state_atomic(state_path: Path, payload: dict) -> None:
    """Write state.json via temp + os.replace.  Main thread only."""
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(str(tmp), str(state_path))


def read_state(state_path: Path) -> dict:
    if not Path(state_path).exists():
        return {}
    try:
        return json.loads(Path(state_path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Preflight mode
# ---------------------------------------------------------------------------

def preflight(args: argparse.Namespace) -> int:
    """Validate raid path + free space + NAS visibility + conda env.

    Emits a single line of JSON on stdout, then exits.
    """
    raid = Path(args.raid_dir)
    nas_check = Path(args.nas_check_path) if args.nas_check_path else None
    required_gb = float(args.required_gb or 0)

    out: dict[str, Any] = {
        "ok": True,
        "host": socket.gethostname(),
        "raid_dir": str(raid),
    }

    try:
        raid.mkdir(parents=True, exist_ok=True)
        # Try a small write
        probe = raid / f".preflight.{os.getpid()}.tmp"
        probe.write_text("ok")
        probe.unlink()
        out["raid_writable"] = True
    except OSError as exc:
        out["ok"] = False
        out["raid_writable"] = False
        out["raid_error"] = str(exc)

    try:
        st = shutil.disk_usage(str(raid))
        free_gb = st.free / 1e9
        out["free_gb"] = round(free_gb, 1)
        if required_gb and free_gb < required_gb:
            out["ok"] = False
            out["disk_error"] = (
                f"required {required_gb:.1f} GB, free {free_gb:.1f} GB"
            )
    except OSError as exc:
        out["ok"] = False
        out["free_gb"] = None
        out["disk_error"] = str(exc)

    if nas_check is not None:
        try:
            out["nas_visible"] = nas_check.exists()
            if not out["nas_visible"]:
                out["ok"] = False
                out["nas_error"] = f"path not found: {nas_check}"
        except OSError as exc:
            out["ok"] = False
            out["nas_visible"] = False
            out["nas_error"] = str(exc)

    out["python"] = sys.executable
    out["pid"] = os.getpid()

    print(json.dumps(out))
    return 0 if out["ok"] else 1


# ---------------------------------------------------------------------------
# Main run mode
# ---------------------------------------------------------------------------

def _setup_pgid() -> int:
    """Detach into a new process group so coordinator can kill -PGID us."""
    try:
        os.setsid()
    except OSError:
        # Already a session leader — fine.
        pass
    try:
        return os.getpgrp()
    except OSError:
        return os.getpid()


def _patch_unit_paths(
    unit_dict: dict, gous_input_dir: Path,
) -> None:
    """Rewrite vis_tm/vis_sm/sdimage absolute paths to /raid/ counterparts.

    The dispatcher embeds the original NAS paths in the manifest; the
    worker stages each input under ``gous_input_dir / <bucket> / <name>``
    where bucket is "ms" or "tp".  We map basenames back to the staged
    location.
    """
    ms_dir = gous_input_dir / "ms"
    tp_dir = gous_input_dir / "tp"

    def remap(p: str, bucket_dir: Path) -> str:
        return str(bucket_dir / Path(p).name)

    unit_dict["vis_tm"] = [remap(p, ms_dir) for p in unit_dict.get("vis_tm", [])]
    unit_dict["vis_sm"] = [remap(p, ms_dir) for p in unit_dict.get("vis_sm", [])]
    sdimage = unit_dict.get("sdimage")
    if sdimage:
        unit_dict["sdimage"] = remap(sdimage, tp_dir)


def _build_imaging_unit(unit_dict: dict):
    """Reconstruct an ImagingUnit from the manifest dict (with raid paths)."""
    from panta_rei.imaging.matching import ImagingUnit
    return ImagingUnit(
        gous_uid=unit_dict["gous_uid"],
        source_name=unit_dict["source_name"],
        line_group=unit_dict.get("line_group"),
        spw_id=str(unit_dict["spw_id"]),
        params_id=int(unit_dict["params_id"]),
        recovered_params=unit_dict.get("recovered_params", {}),
        vis_tm=list(unit_dict.get("vis_tm", [])),
        vis_sm=list(unit_dict.get("vis_sm", [])),
        sdimage=unit_dict.get("sdimage"),
        spw_selection=list(unit_dict.get("spw_selection", [])),
        field_selection=list(unit_dict.get("field_selection", [])),
        datacolumn=unit_dict.get("datacolumn", "corrected"),
        mous_uids_tm=list(unit_dict.get("mous_uids_tm", [])),
        mous_uids_sm=list(unit_dict.get("mous_uids_sm", [])),
        mous_uids_tp=list(unit_dict.get("mous_uids_tp", [])),
        tp_freq_min=unit_dict.get("tp_freq_min"),
        tp_freq_max=unit_dict.get("tp_freq_max"),
        tp_nchan=unit_dict.get("tp_nchan"),
        ready=True,
        skip_reason=None,
    )


def _ensure_staged_inputs(
    gous_input_dir: Path,
    expected_inputs: list[dict],
    transfer_method: str,
    *,
    cache_root: Optional[Path] = None,
    cache_min_free_bytes: Optional[int] = None,
    token_lease: Optional["staging.TokenLease"] = None,
) -> dict:
    """Stage all inputs in ``expected_inputs`` (the GOUS union).

    Uses :func:`panta_rei.imaging.staging.stage_one` plus an mkdir-based
    per-(machine, GOUS) lock so concurrent slot workers cooperate.

    Returns a telemetry dict with per-source counts for state.json::

        {"existing": N, "cache_hit": N, "cache_hit_after_wait": N,
         "cache_miss": N, "nas_direct": N}
    """
    from panta_rei.imaging import staging

    gous_input_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = gous_input_dir / ".staging.manifest"
    counts = {
        "existing": 0, "cache_hit": 0, "cache_hit_after_wait": 0,
        "cache_miss": 0, "nas_direct": 0,
    }

    with staging.acquire_stage_lock(gous_input_dir, holder_meta={"pid": os.getpid()}):
        manifest = staging.read_manifest(manifest_path)
        expected_paths = sorted(set(
            list(manifest.get("expected", [])) + [e["src"] for e in expected_inputs]
        ))
        manifest["expected"] = expected_paths
        already = set(manifest.get("completed", []))

        for entry in expected_inputs:
            src = entry["src"]
            bucket = entry.get("bucket", "ms")
            if src in already:
                counts["existing"] += 1
                continue
            _staged_path, source = staging.stage_one(
                src, gous_input_dir,
                method=transfer_method, bucket=bucket,
                cache_root=cache_root,
                cache_min_free_bytes=cache_min_free_bytes,
                token_lease=token_lease,
            )
            counts[source] = counts.get(source, 0) + 1
            manifest.setdefault("completed", []).append(src)
            staging.atomic_write_json(manifest_path, manifest)

        manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        staging.atomic_write_json(manifest_path, manifest)
    return counts


def _copy_provenance_to_nas(
    work_run_dir: Path,
    log_path: Path,
    nas_unit_dir: Path,
) -> dict:
    """Copy job.json, result.json, log to NAS.  Return absolute NAS paths."""
    nas_unit_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Optional[str]] = {
        "job_json": None, "result_json": None, "log": None,
    }
    job_src = work_run_dir / "job.json"
    if job_src.exists():
        dst = nas_unit_dir / "job.json"
        shutil.copy2(str(job_src), str(dst))
        out["job_json"] = str(dst)
    result_src = work_run_dir / "result.json"
    if result_src.exists():
        dst = nas_unit_dir / "result.json"
        shutil.copy2(str(result_src), str(dst))
        out["result_json"] = str(dst)
    if log_path.exists():
        dst = nas_unit_dir / "unit.log"
        shutil.copy2(str(log_path), str(dst))
        out["log"] = str(dst)
    return out


def run_unit(args: argparse.Namespace) -> int:
    """Process a single imaging unit end-to-end."""
    pgid = _setup_pgid()

    manifest_path = Path(args.manifest)
    raid_dir = Path(args.raid_dir)
    run_id = int(args.run_id)
    dispatch_id = args.dispatch_id
    transfer_method = args.transfer_method or "tar"
    publish_policy = args.publish_policy or "fail_if_exists"
    heartbeat_interval = float(args.heartbeat_interval or 30)
    tokens_dir = Path(args.tokens_dir) if args.tokens_dir else None
    max_concurrent_staging = int(args.max_concurrent_staging or 4)
    cache_root = Path(args.cache_root) if args.cache_root else None
    cache_min_free_bytes = (
        int(args.cache_min_free_gb) * (1024 ** 3)
        if args.cache_min_free_gb else None
    )

    manifest = json.loads(manifest_path.read_text())
    unit_dict = manifest["unit"]
    expected_inputs = manifest["expected_inputs"]  # [{"src": "...", "bucket": "ms"}, ...]
    gous_uid = unit_dict["gous_uid"]
    publish_dir = Path(manifest["publish_dir"])
    nas_unit_dir = Path(manifest["nas_unit_dir"])
    nproc = int(manifest.get("nproc", 4))
    casa_path = manifest.get("casa_path")
    deconvolver = manifest.get("deconvolver", "multiscale")
    scales = manifest.get("scales", [0, 5, 10, 15, 20])

    # Per-unit work dir on /raid/
    work_dir = raid_dir / "work"
    gous_input_dir = raid_dir / "input" / gous_uid
    log_dir = raid_dir / "logs"
    log_path = log_dir / f"unit_{run_id}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # State + heartbeat live on NAS for coordinator visibility
    state_path = nas_unit_dir / "state.json"
    hb_path = nas_unit_dir / "heartbeat"

    # File-log handler in addition to stdout
    file_h = logging.FileHandler(log_path, mode="a")
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    logging.getLogger().addHandler(file_h)
    logging.getLogger().setLevel(logging.INFO)

    base_state = {
        "run_id": run_id,
        "dispatch_id": dispatch_id,
        "machine": socket.gethostname(),
        "hostname": socket.gethostname(),
        "worker_pid": os.getpid(),
        "worker_pgid": pgid,
        "raid_dir": str(raid_dir),
        "publish_dir": str(publish_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": Phase.STARTING,
    }
    write_state_atomic(state_path, base_state)

    hb = _Heartbeat(hb_path, heartbeat_interval)
    hb.start()

    # Lazy NAS staging token: acquired ONLY when stage_one needs to
    # actually read from NAS (cache miss / fallback).  Cache hits skip
    # the gate entirely.  Released in the outer ``finally``.
    from panta_rei.imaging import staging as _staging
    token_lease = _staging.TokenLease(
        tokens_dir, max_concurrent_staging,
        holder_id=f"{dispatch_id}/{run_id}",
    )

    try:
        # ---------- STAGING ----------
        base_state["phase"] = Phase.STAGING
        write_state_atomic(state_path, base_state)

        try:
            stage_counts = _ensure_staged_inputs(
                gous_input_dir, expected_inputs, transfer_method,
                cache_root=cache_root,
                cache_min_free_bytes=cache_min_free_bytes,
                token_lease=token_lease,
            )
        finally:
            token_lease.release()

        base_state["staging_stats"] = {**stage_counts, **token_lease.stats}
        if stage_counts.get("cache_hit", 0) or stage_counts.get("cache_hit_after_wait", 0):
            log.info(
                "staging summary: %s, token_acquires=%d, token_wait=%.1fs",
                stage_counts, token_lease.stats["token_acquires"],
                token_lease.stats["token_wait_sec"],
            )

        # ---------- PATCH PATHS + RUN ----------
        _patch_unit_paths(unit_dict, gous_input_dir)
        unit = _build_imaging_unit(unit_dict)

        base_state["phase"] = Phase.RUNNING
        write_state_atomic(state_path, base_state)

        from panta_rei.imaging.runner import run_tclean_feather_parallel

        success, msg, output_fits = run_tclean_feather_parallel(
            unit=unit,
            output_dir=publish_dir,
            row_id=run_id,
            nproc=nproc,
            casa_path=casa_path,
            deconvolver=deconvolver,
            scales=scales,
            keep_intermediates=manifest.get("keep_intermediates", False),
            dry_run=False,
            work_dir=work_dir,
            publish_policy=publish_policy,
            log_path=log_path,
            extra_env={
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )

        # ---------- PUBLISHING is done inside run_tclean_feather_parallel ----------
        base_state["phase"] = Phase.PUBLISHING
        write_state_atomic(state_path, base_state)

        # ---------- COPY PROVENANCE TO NAS ----------
        provenance = _copy_provenance_to_nas(
            work_run_dir=work_dir / "runs" / str(run_id),
            log_path=log_path,
            nas_unit_dir=nas_unit_dir,
        )

        # Resolved selections survive on the worker side because
        # run_trusted_preflight mutates `unit` in-place inside the runner.
        terminal: dict[str, Any] = {
            "phase": Phase.DONE if success else Phase.FAILED,
            "success": bool(success),
            "message": msg,
            "output_fits": output_fits,
            "spw_selection": unit.spw_selection,
            "field_selection": unit.field_selection,
            "datacolumn": unit.datacolumn,
            "provenance": provenance,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if not success:
            terminal["error_message"] = msg

        base_state.update(terminal)
        write_state_atomic(state_path, base_state)
        return 0 if success else 1

    except Exception as exc:
        # Any unexpected failure → mark FAILED + last-resort log
        log.exception("worker crashed")
        crash_tail = traceback.format_exc()[-2000:]
        base_state.update({
            "phase": Phase.FAILED,
            "success": False,
            "error_message": f"worker exception: {exc}: {crash_tail}",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        try:
            write_state_atomic(state_path, base_state)
        except OSError:
            pass
        # Best-effort provenance copy
        try:
            _copy_provenance_to_nas(
                work_run_dir=work_dir / "runs" / str(run_id),
                log_path=log_path,
                nas_unit_dir=nas_unit_dir,
            )
        except OSError:
            pass
        return 2

    finally:
        try:
            token_lease.release()
        except OSError:
            pass
        hb.stop()




# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="panta_rei.imaging.remote_worker",
        description="Detached worker for distributed tclean+feather imaging.",
    )
    sub = ap.add_subparsers(dest="mode")

    # Default (run) mode — flat args, no subcommand keyword required
    ap.add_argument("--manifest", help="Path to NAS unit manifest JSON")
    ap.add_argument("--raid-dir", help="Per-dispatch /raid/ work root")
    ap.add_argument("--run-id", type=int, help="imaging_runs row id")
    ap.add_argument("--dispatch-id", help="Dispatch identifier")
    ap.add_argument("--tokens-dir", default=None,
                    help="NAS staging-token directory (None disables global cap)")
    ap.add_argument("--max-concurrent-staging", type=int, default=4)
    ap.add_argument("--cache-root", default=None,
                    help="Per-host cross-dispatch staging cache root. "
                         "If unset, the cache is disabled.")
    ap.add_argument("--cache-min-free-gb", type=int, default=None,
                    help="Evict cache entries oldest-first to keep at "
                         "least this many GB free on /raid (default: "
                         "no eviction).")
    ap.add_argument("--transfer-method", default="tar", choices=["tar", "rsync", "cp"])
    ap.add_argument("--publish-policy", default="fail_if_exists",
                    choices=["fail_if_exists", "overwrite"])
    ap.add_argument("--heartbeat-interval", type=float, default=30)

    # Preflight subcommand
    pf = sub.add_parser("preflight", help="Check raid + NAS + free space.")
    pf.add_argument("--raid-dir", required=True)
    pf.add_argument("--required-gb", type=float, default=0)
    pf.add_argument("--nas-check-path", default=None)

    # Convenience: top-level --preflight switch (for callers that prefer flags)
    ap.add_argument("--preflight", action="store_true",
                    help="Run in preflight mode (alternative to subcommand).")
    ap.add_argument("--required-gb", type=float, default=0,
                    help="(preflight) required free GB on raid.")
    ap.add_argument("--nas-check-path", default=None,
                    help="(preflight) NAS path to check for visibility.")

    return ap


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = _build_parser()
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "preflight" or args.preflight:
        return preflight(args)

    if not (args.manifest and args.raid_dir and args.run_id and args.dispatch_id):
        ap.error("--manifest, --raid-dir, --run-id, --dispatch-id are required in run mode")

    return run_unit(args)


if __name__ == "__main__":
    raise SystemExit(main())
