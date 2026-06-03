"""Distributed-imaging coordinator.

Dispatches tclean+feather imaging jobs across a configurable set of
remote machines, staging MS inputs to fast local /raid/ storage and
publishing canonical FITS back to NAS.  See
``docs/`` and the project plan for the full design rationale.

Public entry point: :func:`run_dispatch` (called by the
``panta-rei-imaging-dispatch`` CLI).

The coordinator runs entirely in one Python process with the following
threads:

- 1 main thread — startup, reconciliation, scheduler kickoff, shutdown.
- 1 DB-writer thread — single SQLite owner; consumes a queue of events.
- N machine-worker threads (one per slot per machine) — pull units from
  a GOUS-affinity scheduler, SSH-launch the detached worker, poll the
  unit's NAS state file, enqueue DB events.
- 1 token-reaper thread — wipes staging tokens whose holder PID is gone.
- 1 cleanup thread — coordinator-driven per-(machine, GOUS) cleanup.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Optional

from panta_rei.core.text import now_iso
from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import (
    DispatchesQueries,
    DispatchState,
    ImagingRunsQueries,
    ImagingRunStatus,
)
from panta_rei.imaging.matching import ImagingUnit
from panta_rei.imaging.staging import (
    _atomic_reclaim,
    _read_token_fingerprint,
    token_fingerprint_from_fields,
    list_held_tokens,
)
from panta_rei.imaging.unit_selection import (
    SelectionFilters,
    SelectionResult,
    select_units_to_image,
)

log = logging.getLogger("panta_rei.dispatch")

WORKER_MODULE = "panta_rei.imaging.remote_worker"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GlobalCfg:
    max_concurrent_staging: int = 4
    # Worker staging-token acquire wait.  24h backstop (NOT a tight bound):
    # token contention with a live holder must be waited out — a single unit
    # can stage+image for hours, and historical cache-miss units waited 2-8h
    # for a token and succeeded.  Forwarded to the worker as
    # --token-wait-timeout-sec; see staging._DEFAULT_TOKEN_WAIT_TIMEOUT_SEC.
    token_wait_timeout_sec: float = 86400.0
    heartbeat_interval_sec: int = 30
    heartbeat_stale_threshold_sec: int = 300
    max_stale_alive_sec: int = 3600
    # Raised from 60s (v5 plan F2.e): cold-cache NFS-backed worker startup
    # can need several minutes under contention.  Bounded waste per failed
    # unit is now ~300s + verify-or-quarantine flow.
    state_appeared_timeout_sec: int = 300
    poll_interval_sec: int = 10
    cleanup_interval_sec: int = 30
    token_reaper_interval_sec: int = 60
    ssh_timeout_sec: int = 30
    # How long to wait for ``worker.pidfile`` to appear after a launch
    # whose wrapper_pid is unknown (ssh-timeout, ssh rc!=0, or successful
    # ssh with unparsable pidfile).  See F2.c + Delta 3.
    launch_pidfile_wait_sec: int = 30
    # Single grace knob applied to BOTH worker-side (graceful publish
    # finish) and orchestrator-side (SIGTERM-to-SIGKILL wait, plus 30s
    # buffer).  See v4 Delta 6.
    worker_shutdown_grace_sec: int = 300
    # If state.json reappears as ``publishing``/``running`` after the
    # initial state_appeared_timeout, resume polling for up to this many
    # seconds before quarantining the host.  See F2.h.
    late_state_grace_sec: int = 1800


@dataclass
class MachineCfg:
    name: str
    raid: str
    slots: int = 1
    nproc: int = 4
    # Cross-dispatch staging cache (see staging.py "Cross-dispatch
    # staging cache" section).  ``None`` disables the cache for this
    # machine; an int value is the per-host minimum free space (in GB)
    # the worker maintains by evicting oldest entries before each
    # populate.  Default 1 TB matches typical large /raid sizes; hosts
    # with smaller drives (e.g. 2 TB total) should override to ~512 GB
    # or disable.
    cache_min_free_gb: Optional[int] = 1024


@dataclass
class MachinesConfig:
    conda_env: str
    repo_path: str
    casa_path: Optional[str]
    global_cfg: GlobalCfg
    machines: dict[str, MachineCfg]


def load_machines_config(path: Path) -> MachinesConfig:
    """Parse + validate a machines.json file."""
    raw = json.loads(Path(path).read_text())
    if "conda_env" not in raw or "repo_path" not in raw:
        raise ValueError("machines.json missing 'conda_env' or 'repo_path'")
    if "machines" not in raw or not isinstance(raw["machines"], dict):
        raise ValueError("machines.json missing 'machines' dict")
    g_raw = raw.get("global", {})
    g = GlobalCfg(**{k: v for k, v in g_raw.items() if k in GlobalCfg().__dict__})
    machines: dict[str, MachineCfg] = {}
    for name, m in raw["machines"].items():
        if "raid" not in m:
            raise ValueError(f"machine {name!r}: missing 'raid' path")
        cache_free = m.get("cache_min_free_gb", 1024)
        # Allow ``"cache_min_free_gb": null`` to disable the cache per host.
        cache_free_val = (None if cache_free is None
                          else int(cache_free))
        machines[name] = MachineCfg(
            name=name,
            raid=m["raid"],
            slots=int(m.get("slots", 1)),
            nproc=int(m.get("nproc", 4)),
            cache_min_free_gb=cache_free_val,
        )
    return MachinesConfig(
        conda_env=raw["conda_env"],
        repo_path=raw["repo_path"],
        casa_path=raw.get("casa_path"),
        global_cfg=g,
        machines=machines,
    )


# ---------------------------------------------------------------------------
# Dispatch lock (per-base-dir flock)
# ---------------------------------------------------------------------------

class DispatcherLock:
    """Exclusive non-blocking flock on a base-dir lockfile."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = Path(lock_path)
        self._fp = None

    def acquire(self) -> None:
        import fcntl
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.lock_path, "w")
        try:
            fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = self.lock_path.read_text() if self.lock_path.exists() else "?"
            self._fp.close()
            self._fp = None
            raise RuntimeError(
                f"dispatcher lock held by {holder.strip()}; "
                f"another panta-rei-imaging-dispatch may be running"
            )
        self._fp.write(f"host={socket.gethostname()} pid={os.getpid()}\n")
        self._fp.flush()

    def release(self) -> None:
        import fcntl
        if self._fp is not None:
            try:
                fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
            finally:
                self._fp.close()
                self._fp = None


# ---------------------------------------------------------------------------
# DB-writer thread
# ---------------------------------------------------------------------------

DB_SHUTDOWN = object()


class DBWriter(threading.Thread):
    """Single-owner SQLite writer fed by a Queue.

    All other threads drop dicts with ``{"op": "...", **kwargs}`` onto
    the queue.  The writer applies them serially using one long-lived
    sqlite3 connection.  Workers never touch SQLite.
    """

    def __init__(self, db_manager: DatabaseManager, dispatch_id: str) -> None:
        super().__init__(name="db-writer", daemon=True)
        self.q: Queue = Queue()
        self.db_manager = db_manager
        self.dispatch_id = dispatch_id
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self.q.put(DB_SHUTDOWN)
        self._stop_event.set()

    def run(self) -> None:
        con = self.db_manager.connect()
        con.execute("PRAGMA busy_timeout=10000")
        try:
            while True:
                item = self.q.get()
                try:
                    if item is DB_SHUTDOWN:
                        return
                    try:
                        self._apply(con, item)
                        con.commit()
                    except Exception:
                        log.exception("DB writer failed on event: %r", item)
                finally:
                    # task_done() balances every q.put() so callers can
                    # use q.join() as a synchronous drain barrier.
                    self.q.task_done()
        finally:
            try:
                con.close()
            except sqlite3_close_silently:
                pass

    def _apply(self, con, ev: dict) -> None:
        op = ev["op"]
        if op == "INSERT_QUEUED":
            run_id = ImagingRunsQueries.insert_row(con, **ev["row"])
            ev["row_id_holder"]["id"] = run_id
        elif op == "MARK_RUNNING":
            ImagingRunsQueries.mark_running(con, ev["run_id"])
            ImagingRunsQueries.set_dispatch_meta(
                con, ev["run_id"],
                dispatch_id=self.dispatch_id,
                remote_workdir=ev.get("remote_workdir"),
                worker_pid=ev.get("worker_pid"),
                worker_pgid=ev.get("worker_pgid"),
                hostname=ev.get("hostname"),
            )
        elif op == "HEARTBEAT":
            ImagingRunsQueries.update_heartbeat(con, ev["run_id"], ev["ts"])
        elif op == "MARK_DONE":
            status = ev["status"]
            if ev.get("spw_selection") or ev.get("field_selection"):
                ImagingRunsQueries.update_resolved(
                    con, ev["run_id"],
                    spw_selection=ev.get("spw_selection"),
                    field_selection=ev.get("field_selection"),
                )
            if "job_json_path" in ev and ev["job_json_path"]:
                con.execute(
                    "UPDATE imaging_runs SET job_json_path=? WHERE id=?",
                    (ev["job_json_path"], ev["run_id"]),
                )
            ImagingRunsQueries.mark_done(
                con,
                ev["run_id"],
                status=status,
                retcode=int(ev.get("retcode", 1 if status == ImagingRunStatus.FAILED else 0)),
                finished_at=ev.get("finished_at") or now_iso(),
                duration_sec=float(ev.get("duration_sec", 0.0)),
                output_fits=ev.get("output_fits"),
                terminal_source=ev.get("terminal_source", TS_UNKNOWN),
            )
            ImagingRunsQueries.set_error_message(
                con, ev["run_id"], ev.get("error_message"),
            )
        elif op == "DISPATCH_TERMINAL":
            DispatchesQueries.mark_terminal(
                con, ev["dispatch_id"], state=ev.get("state", DispatchState.DONE),
            )
        elif op == "INSERT_DISPATCH":
            DispatchesQueries.insert(con, **ev["row"])
        else:
            log.warning("unknown DB op: %r", op)


class sqlite3_close_silently(BaseException):
    pass


# ---------------------------------------------------------------------------
# GOUS-affinity scheduler with staging gate
# ---------------------------------------------------------------------------

@dataclass
class SchedulerState:
    queue: list[ImagingUnit] = field(default_factory=list)
    gous_machine: dict[str, str] = field(default_factory=dict)
    in_flight: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    gous_staged_on: dict[tuple[str, str], bool] = field(default_factory=dict)
    # Every (machine, gous) pair the scheduler has ever assigned a unit to.
    # Used by the cleaner because in_flight entries are deleted once empty.
    seen_pairs: set[tuple[str, str]] = field(default_factory=set)
    # Terminal run_ids per (machine, gous), recorded synchronously by the
    # slot/adoption thread that calls mark_terminal.  The cleaner reads
    # from this rather than querying SQLite — that avoids a race where
    # the DB-writer queue hasn't yet committed MARK_DONE when the cleaner
    # sweeps and would otherwise miss SUCCESS work-dirs.
    success_run_ids: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    failed_run_ids: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    # The dispatch_id whose /raid/d_<id>/... paths the cleaner should
    # target for each (machine, gous) pair.  Adopted units inherit the
    # *prior* dispatch_id; new units use the current one.
    pair_dispatch_id: dict[tuple[str, str], str] = field(default_factory=dict)
    # Per-dispatch host quarantine (v3 F2.d): a host added here is
    # excluded from pick(), causes its MachineSlot.run() to exit, and
    # blocks GousCleaner / end-of-run sweep from touching anything on it.
    quarantined_machines: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def quarantine(self, machine: str, *, reason: str = "") -> None:
        with self.lock:
            self.quarantined_machines.add(machine)

    def quarantine_all(self, machines) -> None:
        with self.lock:
            for m in machines:
                if m:
                    self.quarantined_machines.add(m)

    def is_quarantined(self, machine: str) -> bool:
        with self.lock:
            return machine in self.quarantined_machines

    def queued_for_gous_on_machine(self, gous: str, machine: str) -> int:
        with self.lock:
            return sum(
                1 for u in self.queue
                if u.gous_uid == gous
                and self.gous_machine.get(u.gous_uid) == machine
            )

    def queued_for_gous_total(self, gous: str) -> int:
        with self.lock:
            return sum(1 for u in self.queue if u.gous_uid == gous)

    def pick(self, machine: str, run_id_assigner) -> Optional[ImagingUnit]:
        """Return a unit for *machine* per the affinity strategy."""
        with self.lock:
            # Quarantined host: refuse to assign new work (F2.d).
            if machine in self.quarantined_machines:
                return None
            # 1. mapped + staged on this machine
            for i, u in enumerate(self.queue):
                if (self.gous_machine.get(u.gous_uid) == machine
                        and self.gous_staged_on.get((machine, u.gous_uid))):
                    self.queue.pop(i)
                    return u
            # 2. mapped to this machine but not yet staged: only if no other
            #    unit of this GOUS is currently staging on this machine.
            for i, u in enumerate(self.queue):
                if (self.gous_machine.get(u.gous_uid) == machine
                        and not self.in_flight.get((machine, u.gous_uid))):
                    self.queue.pop(i)
                    return u
            # 3. unmapped GOUS — claim
            for i, u in enumerate(self.queue):
                if u.gous_uid not in self.gous_machine:
                    self.gous_machine[u.gous_uid] = machine
                    self.queue.pop(i)
                    return u
            # 4. fallback FIFO — but skip units whose GOUS is currently
            #    staging-blocked on THIS machine; otherwise we'd pick a
            #    sibling we already declined in step 2 and just block on
            #    the staging flock.
            for i, u in enumerate(self.queue):
                mapped = self.gous_machine.get(u.gous_uid)
                if (mapped == machine
                        and self.in_flight.get((machine, u.gous_uid))
                        and not self.gous_staged_on.get((machine, u.gous_uid))):
                    continue
                self.queue.pop(i)
                self.gous_machine.setdefault(u.gous_uid, machine)
                return u
            return None

    def mark_inflight(
        self, machine: str, gous: str, run_id: int,
        *, dispatch_id: Optional[str] = None,
    ) -> None:
        """Record an in-flight unit on (machine, gous).

        ``dispatch_id`` records which dispatch's /raid/d_<id>/ tree this
        pair lives under; the cleaner reads it back so adopted prior-
        dispatch units get cleaned up against their *original* /raid/
        layout, not the new dispatch's.
        """
        with self.lock:
            self.in_flight.setdefault((machine, gous), set()).add(run_id)
            self.seen_pairs.add((machine, gous))
            if dispatch_id is not None:
                self.pair_dispatch_id.setdefault((machine, gous), dispatch_id)

    def mark_staged(self, machine: str, gous: str) -> None:
        with self.lock:
            self.gous_staged_on[(machine, gous)] = True

    def mark_terminal(
        self, machine: str, gous: str, run_id: int,
        *, success: Optional[bool] = None,
    ) -> bool:
        """Drop run_id from in-flight; return True if (machine, gous) now empty.

        ``success`` records the run's outcome in-memory so the cleaner
        can act without a DB round-trip (and without racing the DB
        writer's queue).  ``None`` is acceptable for callers that don't
        yet know — adoptions populate this once polling completes.
        """
        with self.lock:
            s = self.in_flight.get((machine, gous))
            if s and run_id in s:
                s.discard(run_id)
            empty = not s
            if empty and (machine, gous) in self.in_flight:
                del self.in_flight[(machine, gous)]
            if success is True:
                self.success_run_ids.setdefault(
                    (machine, gous), set()
                ).add(run_id)
            elif success is False:
                self.failed_run_ids.setdefault(
                    (machine, gous), set()
                ).add(run_id)
            return empty


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def ssh_run(
    machine: str, remote_cmd: str, *,
    timeout: int = 30,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run ``ssh <machine> <cmd>``.  Caller composes the remote command.

    The command is base64-encoded and piped through ``bash`` on the
    remote so the caller can use bash syntax (redirections, ``$!``,
    ``if/then/fi``) regardless of the remote user's login shell — many
    of these hosts default to tcsh, which mis-parses ``>file 2>&1`` and
    history-expands ``!`` even inside single quotes.

    Returns the CompletedProcess; does NOT raise on non-zero exit.
    """
    encoded = base64.b64encode(remote_cmd.encode("utf-8")).decode("ascii")
    wrapped = f"echo {encoded} | base64 -d | bash"
    cmd = ["ssh", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=10",
           "-o", "StrictHostKeyChecking=accept-new",
           machine, wrapped]
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True, timeout=timeout,
    )


def ssh_pid_alive(
    machine: str,
    pid: int,
    expected_tokens: list[str] | str,
    timeout: int = 10,
) -> tuple[Optional[bool], str]:
    """Check whether *pid* on *machine* is alive AND owned by our worker.

    *expected_tokens* may be a single substring or a list of substrings;
    every substring must be present in ``/proc/<pid>/cmdline`` for the
    PID to be considered ours.  This is order-independent, so the
    launcher's actual flag order doesn't have to match a fixed template.

    NUL separators in cmdline are translated to spaces (``tr '\\0' ' '``)
    before substring matching.  Without that, distinct argv entries
    would concatenate and our space-bearing tokens (e.g. ``--run-id 42``)
    would never match.

    Returns ``(True, cmdline)`` if alive + all tokens present.
    Returns ``(False, cmdline)`` if dead OR token mismatch (PID reused).
    Returns ``(None, reason)`` if the check itself failed (network / ssh
    error) — coordinator must NOT mark FAILED in that case.
    """
    if not pid:
        return False, ""
    if isinstance(expected_tokens, str):
        tokens = [expected_tokens]
    else:
        tokens = list(expected_tokens)
    cmd = (
        f"if [ -e /proc/{int(pid)}/cmdline ]; then "
        f"  tr '\\0' ' ' </proc/{int(pid)}/cmdline; "
        f"else "
        f"  echo __DEAD__; "
        f"fi"
    )
    try:
        proc = ssh_run(machine, cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"ssh timeout to {machine}"
    if proc.returncode != 0:
        return None, f"ssh rc={proc.returncode}: {(proc.stderr or '')[-200:]}"
    text = (proc.stdout or "").strip()
    if text == "__DEAD__":
        return False, ""
    if all(tok in text for tok in tokens):
        return True, text
    # PID reused for unrelated process
    return False, text


def ssh_preflight_machine(
    machine: str,
    m: "MachineCfg",
    cfg: "MachinesConfig",
    *,
    required_gb: float = 0.0,
    nas_check_path: Optional[str] = None,
    timeout: int = 30,
) -> tuple[bool, dict]:
    """SSH ``machine`` and run ``remote_worker preflight``.

    Returns ``(ok, details)``.  ``details`` is the parsed JSON from the
    worker, or a synthetic dict on failure (no JSON / non-zero exit /
    SSH error).
    """
    cmd = (
        f"env PYTHONPATH={shlex.quote(cfg.repo_path)} "
        f"{shlex.quote(cfg.conda_env)}/bin/python -m {WORKER_MODULE} "
        f"preflight --raid-dir {shlex.quote(m.raid)} "
        f"--required-gb {float(required_gb)}"
    )
    if nas_check_path:
        cmd += f" --nas-check-path {shlex.quote(nas_check_path)}"
    try:
        proc = ssh_run(machine, cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, {"error": f"ssh timeout to {machine}"}
    if proc.returncode != 0 and not (proc.stdout or "").strip():
        return False, {
            "error": f"ssh rc={proc.returncode}: "
                     f"{(proc.stderr or '')[-300:]}"
        }
    text = (proc.stdout or "").strip()
    try:
        # Last line is the JSON output.
        last = text.splitlines()[-1] if text else ""
        details = json.loads(last)
    except (ValueError, IndexError):
        return False, {
            "error": f"could not parse preflight JSON: "
                     f"stdout={text[-300:]} stderr={(proc.stderr or '')[-300:]}"
        }
    return bool(details.get("ok")), details


def ssh_kill_pgid(machine: str, pgid: int, signal_name: str = "TERM",
                  timeout: int = 10) -> None:
    """Send SIG to the process group leader at *pgid* on *machine*.  Best-effort."""
    if not pgid:
        return
    try:
        ssh_run(machine, f"kill -{shlex.quote(signal_name)} -{int(pgid)}",
                timeout=timeout)
    except subprocess.SubprocessError:
        pass


# ---------------------------------------------------------------------------
# Verify-or-quarantine cleanup (shared by _dispatch_one + AdoptionPoller +
# reconcile_prior).  See plan v3 F2.b/v4 Delta 1.
# ---------------------------------------------------------------------------

# Outcomes from verify_and_kill_worker:
VERIFY_DEAD = "dead"                # confirmed not running (or PID reused)
VERIFY_KILLED = "killed"            # alive but TERM/KILL confirmed it dead
VERIFY_INCONCLUSIVE = "inconclusive"  # ssh-unreachable or kill didn't take


# ---------------------------------------------------------------------------
# terminal_source taxonomy (migration v16).  Set EXPLICITLY by each decision
# site and carried through to mark_done — never inferred from
# _is_observed_terminal/phase (the poll dead-pid branch sets phase="failed"
# with no synthetic reason, which would otherwise mislabel a coordinator-
# verified death as worker-observed).
# ---------------------------------------------------------------------------

# Worker wrote its own terminal state.json that we relay.
TS_WORKER_OBSERVED = "worker_observed"
# Coordinator proved the worker dead (dead pid / PID-reused / __DEAD__).
TS_VERIFIED_DEAD = "verified_dead"
# Coordinator killed an alive worker and confirmed death.
TS_VERIFIED_KILLED = "verified_killed"
# --abandon-prior force-fail; NOT death-proven.
TS_ABANDONED = "abandoned"
# reconcile failure with no host known; NOT death-proven.
TS_RECONCILE_NO_HOST = "reconcile_no_host"
# default / pre-launch crash / manual edits / legacy rows.
TS_UNKNOWN = "unknown"

# The reaper trusts ONLY these to reap a token without an ssh probe.
TRUSTED_TERMINAL_SOURCES = frozenset(
    {TS_WORKER_OBSERVED, TS_VERIFIED_DEAD, TS_VERIFIED_KILLED}
)


def _source_from_verify(outcome: str) -> str:
    """Map a verify_and_kill_worker outcome to a trusted terminal_source.

    VERIFY_INCONCLUSIVE never reaches here — its callers quarantine and
    write NO terminal row.  A KeyError on an unexpected outcome is a
    programming error we want to surface loudly.
    """
    return {
        VERIFY_DEAD: TS_VERIFIED_DEAD,
        VERIFY_KILLED: TS_VERIFIED_KILLED,
    }[outcome]


def verify_and_kill_worker(
    machine: str,
    wrapper_pid: Optional[int],
    expected_tokens: list[str],
    g: GlobalCfg,
    *,
    sleep_fn=time.sleep,
) -> tuple[str, str]:
    """Verify worker ownership, kill if needed, return outcome.

    Returns ``(outcome, message)`` where ``outcome`` is one of
    ``VERIFY_DEAD``, ``VERIFY_KILLED``, ``VERIFY_INCONCLUSIVE``.

    - ``VERIFY_DEAD``: worker confirmed not running (genuinely dead per
      ``__DEAD__`` or PID reused for unrelated cmdline).  Safe to
      MARK_DONE FAILED.
    - ``VERIFY_KILLED``: worker was alive and we successfully killed it
      (verified dead after TERM, or after escalation to KILL).  Safe to
      MARK_DONE FAILED.
    - ``VERIFY_INCONCLUSIVE``: ssh failed or kill didn't take.  Caller
      MUST quarantine the host — cleanup is unverified.
    """
    if not wrapper_pid:
        # No PID to verify or kill; caller should treat as inconclusive
        # (worker may exist with unknown pid).
        return VERIFY_INCONCLUSIVE, "no wrapper_pid to verify"

    alive, info = ssh_pid_alive(
        machine, int(wrapper_pid), expected_tokens, timeout=10,
    )
    if alive is None:
        return VERIFY_INCONCLUSIVE, f"ssh-unreachable: {info}"
    if alive is False:
        if info:
            # PID reused for an unrelated process; our worker is gone.
            return VERIFY_DEAD, f"pid_reused_or_token_mismatch: {info!r}"
        # __DEAD__ — worker is genuinely not running anymore.
        return VERIFY_DEAD, "worker_already_exited"

    # alive is True — send SIGTERM, wait the grace, verify.
    ssh_kill_pgid(machine, int(wrapper_pid), "TERM")
    sleep_fn(g.worker_shutdown_grace_sec + 30)
    alive2, info2 = ssh_pid_alive(
        machine, int(wrapper_pid), expected_tokens, timeout=10,
    )
    if alive2 is False:
        return VERIFY_KILLED, f"sigterm_killed: {info2!r}"
    if alive2 is None:
        return VERIFY_INCONCLUSIVE, f"ssh-unreachable post-TERM: {info2}"

    # Still alive — escalate to SIGKILL.
    ssh_kill_pgid(machine, int(wrapper_pid), "KILL")
    sleep_fn(5)
    alive3, info3 = ssh_pid_alive(
        machine, int(wrapper_pid), expected_tokens, timeout=10,
    )
    if alive3 is False:
        return VERIFY_KILLED, f"sigkill_killed: {info3!r}"
    if alive3 is None:
        return VERIFY_INCONCLUSIVE, f"ssh-unreachable post-KILL: {info3}"
    return VERIFY_INCONCLUSIVE, f"still alive after SIGKILL: {info3!r}"


def _read_pidfile(nas_unit_dir: Path) -> Optional[int]:
    """Return the wrapper_pid recorded in worker.pidfile, or None."""
    p = Path(nas_unit_dir) / "worker.pidfile"
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def _wait_for_pidfile(
    nas_unit_dir: Path,
    wait_sec: float,
    *,
    sleep_fn=time.sleep,
    poll_interval: float = 1.0,
) -> Optional[int]:
    """Poll for worker.pidfile up to wait_sec; return parsed PID or None."""
    deadline = time.monotonic() + max(0.0, float(wait_sec))
    while True:
        pid = _read_pidfile(nas_unit_dir)
        if pid is not None:
            return pid
        if time.monotonic() >= deadline:
            return None
        sleep_fn(poll_interval)


# ---------------------------------------------------------------------------
# Remote launch via a NAS-resident shell launcher script
# ---------------------------------------------------------------------------

def write_launcher_script(
    nas_unit_dir: Path,
    cfg: MachinesConfig,
    raid_dir: str,
    manifest_path: str,
    run_id: int,
    dispatch_id: str,
    transfer_method: str,
    publish_policy: str,
    tokens_dir: str,
    max_concurrent_staging: int,
    heartbeat_interval: int,
    token_wait_timeout_sec: float,
    cache_root: Optional[str] = None,
    cache_min_free_gb: Optional[int] = None,
) -> Path:
    """Drop a per-unit ``launch.sh`` on NAS that the SSH command runs.

    Putting all argument substitution into a shell script keeps the
    SSH command line trivial (``ssh host bash <quoted-launcher>``) and
    eliminates the temptation to interpolate user-controlled paths into
    the SSH command — which would be a shell-injection liability.
    """
    nas_unit_dir.mkdir(parents=True, exist_ok=True)
    launcher = nas_unit_dir / "launch.sh"
    launch_log = nas_unit_dir / "launch.log"

    cache_args = ""
    if cache_root:
        cache_args += f"--cache-root {shlex.quote(cache_root)} "
    if cache_min_free_gb is not None:
        cache_args += f"--cache-min-free-gb {int(cache_min_free_gb)} "

    content = (
        "#!/bin/bash\n"
        "set -u\n"
        "export OMP_NUM_THREADS=1\n"
        "export OPENBLAS_NUM_THREADS=1\n"
        "export MKL_NUM_THREADS=1\n"
        f"export PYTHONPATH={shlex.quote(cfg.repo_path)}\n"
        "exec setsid "
        f"{shlex.quote(cfg.conda_env)}/bin/python -m {WORKER_MODULE} "
        f"--manifest {shlex.quote(manifest_path)} "
        f"--raid-dir {shlex.quote(raid_dir)} "
        f"--run-id {int(run_id)} "
        f"--dispatch-id {shlex.quote(dispatch_id)} "
        f"--transfer-method {shlex.quote(transfer_method)} "
        f"--publish-policy {shlex.quote(publish_policy)} "
        f"--tokens-dir {shlex.quote(tokens_dir)} "
        f"--max-concurrent-staging {int(max_concurrent_staging)} "
        f"--token-wait-timeout-sec {float(token_wait_timeout_sec)} "
        f"--heartbeat-interval {int(heartbeat_interval)} "
        f"{cache_args}"
        f">{shlex.quote(str(launch_log))} 2>&1 "
        "</dev/null\n"
    )
    launcher.write_text(content)
    launcher.chmod(0o755)
    return launcher


def launch_detached(
    machine: str,
    launcher: Path,
    nas_unit_dir: Path,
    timeout: int = 30,
) -> tuple[bool, str, Optional[int]]:
    """SSH a machine and start the worker detached via ``nohup``.

    Returns ``(launched_ok, message, worker_pid)``.

    ``worker_pid`` is the worker's PID (best-effort: read from the
    launcher's ``pidfile`` after SSH returns).  None if the worker did
    not write its pidfile within the launch window.
    """
    nohup_log = nas_unit_dir / "nohup.log"
    pidfile = nas_unit_dir / "worker.pidfile"
    # We pipe `bash launcher` through nohup; the worker itself rewrites
    # state.json with its own pid+pgid as soon as it starts.  We capture
    # the wrapper's pid into the pidfile but the source of truth for
    # liveness checks is state.json's worker_pid (the actual python process).
    remote = (
        f"nohup bash {shlex.quote(str(launcher))} "
        f">{shlex.quote(str(nohup_log))} 2>&1 & "
        f"echo $! > {shlex.quote(str(pidfile))}"
    )
    try:
        proc = ssh_run(machine, remote, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"ssh timeout to {machine}", None
    if proc.returncode != 0:
        return False, f"ssh rc={proc.returncode}: {(proc.stderr or '')[-500:]}", None
    pid: Optional[int] = None
    try:
        if pidfile.exists():
            pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        pid = None
    return True, "launched", pid


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

# Synthetic reasons returned by poll_state_until_terminal when the caller
# must run the verify-or-quarantine flow (i.e. the worker state was NOT
# directly observed as terminal).
_SYNTHETIC_REASONS = frozenset({
    "state_missing_timeout",
    "heartbeat_stale_alive",
    "state_appeared_after_timeout",
})


def _is_observed_terminal(final: Optional[dict]) -> bool:
    """Return True iff *final* represents a worker-observed terminal state.

    Synthetic terminals produced by orchestrator timeouts (missing
    state.json, stale-alive heartbeat, grace expiry) are NOT observed:
    the remote worker may still be alive and a successor must NOT be
    spawned without first verifying or quarantining.
    """
    if not final:
        return False
    if final.get("phase") not in {"done", "failed"}:
        return False
    if final.get("reason") in _SYNTHETIC_REASONS:
        return False
    return True


def poll_state_until_terminal(
    machine: str,
    nas_unit_dir: Path,
    *,
    g: GlobalCfg,
    expected_tokens: list[str],
    on_phase_change=None,
    on_poll=None,
) -> dict:
    """Block until the unit reaches a terminal phase OR is declared dead.

    Returns the final state dict.

    Callbacks:
    - ``on_phase_change(phase, state)`` fires whenever ``state.json``
      ``phase`` transitions.
    - ``on_poll(state)`` fires every successful poll iteration (after
      ``state.json`` first appears) — used to push throttled HEARTBEAT
      events to the DB writer so ``imaging_runs.last_heartbeat`` stays
      fresh in DB even though NAS heartbeat is the real liveness source.
    """
    state_path = nas_unit_dir / "state.json"
    hb_path = nas_unit_dir / "heartbeat"

    last_phase: Optional[str] = None
    state_appeared = False
    started = time.monotonic()

    while True:
        time.sleep(g.poll_interval_sec)
        # 1. Wait for state.json to appear
        if not state_appeared:
            if state_path.exists():
                state_appeared = True
            else:
                if time.monotonic() - started > g.state_appeared_timeout_sec:
                    # Worker never wrote state — possibly cold-NFS slow
                    # startup (worker still alive) or crashed at launch.
                    # Caller (F2.b) decides between verify+kill and
                    # quarantine; we just signal the synthetic reason.
                    return {
                        "phase": "failed",
                        "success": False,
                        "reason": "state_missing_timeout",
                        "error_message": (
                            f"state.json never appeared within "
                            f"{g.state_appeared_timeout_sec}s"
                        ),
                    }
                continue

        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            # Mid-write race; retry.
            continue

        phase = state.get("phase")
        if phase != last_phase and on_phase_change is not None:
            try:
                on_phase_change(phase, state)
            except Exception:
                log.exception("on_phase_change callback failed")
            last_phase = phase

        if on_poll is not None:
            try:
                on_poll(state)
            except Exception:
                log.exception("on_poll callback failed")

        if phase in ("done", "failed"):
            # The worker wrote its own terminal state.json; relay it as
            # worker-observed (v5 Delta 1) unless the worker already stamped
            # a source itself.
            state.setdefault("terminal_source", TS_WORKER_OBSERVED)
            return state

        # Liveness check on stale heartbeat
        try:
            hb_age = time.time() - hb_path.stat().st_mtime
        except OSError:
            hb_age = float("inf")

        if hb_age > g.heartbeat_stale_threshold_sec:
            pid = state.get("worker_pid")
            alive, info = ssh_pid_alive(
                machine, pid, expected_tokens, timeout=10,
            )
            if alive is False:
                # Coordinator proved the worker dead/reused — this is a
                # verified death, NOT a worker-observed terminal.  Carry the
                # source explicitly (v5 Delta 1) so the enqueue site doesn't
                # have to infer it from phase/reason.
                return {
                    **state,
                    "phase": "failed",
                    "success": False,
                    "terminal_source": TS_VERIFIED_DEAD,
                    "error_message": (
                        f"worker dead pid {pid} (cmdline now: {info!r})"
                    ),
                }
            if alive is None:
                # Unreachable — keep waiting; coordinator will try again
                log.warning(
                    "ssh-unreachable while polling run_id=%s on %s: %s",
                    state.get("run_id"), machine, info,
                )
                continue
            # Alive but heartbeat stale — bound by max_stale_alive.
            # v4 Delta 1: do NOT kill here.  We return a synthetic reason
            # and let the caller run the verify-or-quarantine flow so an
            # unverified kill can never silently drop the slot.
            if hb_age > g.max_stale_alive_sec:
                return {
                    **state,
                    "phase": "failed",
                    "success": False,
                    "reason": "heartbeat_stale_alive",
                    "error_message": (
                        f"worker pid {pid} alive but heartbeat stale "
                        f"{hb_age:.0f}s > max_stale_alive "
                        f"{g.max_stale_alive_sec}s"
                    ),
                }


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def serialise_unit(u: ImagingUnit) -> dict:
    return {
        "gous_uid": u.gous_uid,
        "source_name": u.source_name,
        "line_group": u.line_group,
        "spw_id": u.spw_id,
        "params_id": u.params_id,
        "recovered_params": u.recovered_params,
        "vis_tm": list(u.vis_tm),
        "vis_sm": list(u.vis_sm),
        "sdimage": u.sdimage,
        "spw_selection": list(u.spw_selection),
        "field_selection": list(u.field_selection),
        "datacolumn": u.datacolumn,
        "mous_uids_tm": list(u.mous_uids_tm),
        "mous_uids_sm": list(u.mous_uids_sm),
        "mous_uids_tp": list(u.mous_uids_tp),
        "tp_freq_min": u.tp_freq_min,
        "tp_freq_max": u.tp_freq_max,
        "tp_nchan": u.tp_nchan,
    }


def union_inputs_for_gous(units: list[ImagingUnit]) -> list[dict]:
    """Compute the union of MS+TP inputs across all units of one GOUS."""
    ms = set()
    tp = set()
    for u in units:
        for p in u.vis_tm:
            ms.add(p)
        for p in u.vis_sm:
            ms.add(p)
        if u.sdimage:
            tp.add(u.sdimage)
    out = [{"src": p, "bucket": "ms"} for p in sorted(ms)]
    out += [{"src": p, "bucket": "tp"} for p in sorted(tp)]
    return out


def write_unit_manifest(
    nas_unit_dir: Path,
    *,
    unit: ImagingUnit,
    expected_inputs: list[dict],
    publish_dir: Path,
    nproc: int,
    casa_path: Optional[str],
    deconvolver: str,
    scales: list[int],
) -> Path:
    nas_unit_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = nas_unit_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "unit": serialise_unit(unit),
        "expected_inputs": expected_inputs,
        "publish_dir": str(publish_dir),
        "nas_unit_dir": str(nas_unit_dir),
        "nproc": int(nproc),
        "casa_path": casa_path,
        "deconvolver": deconvolver,
        "scales": list(scales),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest_path


# ---------------------------------------------------------------------------
# Reconciliation pass
# ---------------------------------------------------------------------------

@dataclass
class ReconcileResult:
    """Outcome of reconcile_prior().

    ``adoptable`` is the list of adopt-candidate dicts (back-compat with
    callers that ignore quarantine).  ``quarantined_hosts`` is a set of
    hosts whose prior-dispatch cleanup was inconclusive — the orchestrator
    must seed these into Scheduler.quarantined_machines before preflight
    and slot creation so they are excluded for the whole new dispatch.
    """
    adoptable: list[dict] = field(default_factory=list)
    quarantined_hosts: set[str] = field(default_factory=set)

    # Back-compat: iterate / len() should behave like the old `list[dict]`
    # return so callers that simply did ``adoptable = reconcile_prior(...)``
    # keep working until they migrate to the named field.
    def __iter__(self):
        return iter(self.adoptable)

    def __len__(self):
        return len(self.adoptable)

    def __bool__(self):
        return bool(self.adoptable) or bool(self.quarantined_hosts)

    def __getitem__(self, idx):
        return self.adoptable[idx]


def reconcile_prior(
    db_manager: DatabaseManager,
    base_dir: Path,
    g: GlobalCfg,
    *,
    abandon: bool = False,
) -> ReconcileResult:
    """Reconcile prior dispatches.  Returns a ReconcileResult.

    For each non-terminal dispatch in the DB:

    - For each ``state.json`` under that dispatch's NAS dir:
        - Terminal phase → apply MARK_DONE (idempotent via mark_done UPDATE).
        - Non-terminal + fresh heartbeat → adopt.
        - Non-terminal + stale heartbeat + ``ssh kill -0`` says alive
          (cmdline match) → adopt with warning.
        - Non-terminal + stale heartbeat + dead → MARK_FAILED.
        - Non-terminal + stale heartbeat + ssh-unreachable → keep,
          log warning (coordinator can't safely declare dead).
    - Additionally iterate DB rows in non-terminal state for which no
      state file exists.  v4 Delta 4: this is INCONCLUSIVE, not safe:
        - ``worker.pidfile`` present → verify+kill; quarantine on
          inconclusive cleanup.
        - ``worker.pidfile`` absent → quarantine the host AND leave
          the row ACTIVE for the next dispatch to recover.

    ``abandon=True`` skips all adoption and force-FAILs in-flight units
    of prior dispatches.
    """
    dispatch_root = base_dir / "imaging" / "dispatch"
    result = ReconcileResult()

    with db_manager.connect() as con:
        prior = DispatchesQueries.list_running(con)

    for d in prior:
        d_id = d["dispatch_id"]
        d_dir = dispatch_root / d_id
        units_dir = d_dir / "units"

        # Pass 1: state files
        seen_run_ids: set[int] = set()
        if units_dir.exists():
            for run_subdir in sorted(units_dir.iterdir()):
                if not run_subdir.is_dir():
                    continue
                state_path = run_subdir / "state.json"
                if not state_path.exists():
                    continue
                try:
                    state = json.loads(state_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                run_id = int(state.get("run_id") or 0)
                if not run_id:
                    continue
                seen_run_ids.add(run_id)
                _reconcile_state_entry(
                    db_manager, d_id, run_id, state, run_subdir, g,
                    result, abandon=abandon,
                )

        # Pass 2: DB rows for this dispatch with no state file
        with db_manager.connect() as con:
            db_rows = ImagingRunsQueries.list_running_for_dispatch(con, d_id)
        for row in db_rows:
            if row["id"] in seen_run_ids:
                continue
            if abandon:
                # --abandon-prior overrides Delta 4 — operator has
                # explicitly asked to drop the prior dispatch.  Not
                # death-proven → untrusted source.
                _mark_failed(
                    db_manager, row["id"],
                    error="abandoned by --abandon-prior (no state.json)",
                    terminal_source=TS_ABANDONED,
                )
                continue
            # F5 + Delta 4: no state.json may mean (a) launch never
            # wrote state and crashed, or (b) launch was in progress
            # when the prior orchestrator died, and the worker may
            # still be alive on a host we don't know.  Treat as
            # INCONCLUSIVE.
            unit_dir = units_dir / str(row["id"])
            pid = _read_pidfile(unit_dir) if unit_dir.exists() else None
            machine = (row.get("hostname") or "").split(".", 1)[0]
            if pid and machine:
                expected_tokens = [
                    f"--dispatch-id {d_id}",
                    f"--run-id {int(row['id'])}",
                ]
                outcome, detail = verify_and_kill_worker(
                    machine, pid, expected_tokens, g,
                )
                log.info(
                    "reconcile: pidfile verify run_id=%d machine=%s "
                    "outcome=%s detail=%s",
                    row["id"], machine, outcome, detail,
                )
                if outcome == VERIFY_INCONCLUSIVE:
                    result.quarantined_hosts.add(machine)
                    log.warning(
                        "reconcile: QUARANTINE machine=%s run_id=%d "
                        "(pidfile present, cleanup unverified) — leaving "
                        "row ACTIVE",
                        machine, row["id"],
                    )
                else:
                    # verify_and_kill_worker proved the worker dead/killed.
                    _mark_failed(
                        db_manager, row["id"],
                        error=f"reconcile: no state.json; "
                              f"worker {outcome} ({detail})",
                        terminal_source=_source_from_verify(outcome),
                    )
            else:
                # No pidfile — Delta 4: still inconclusive.  Quarantine
                # the host (if we know it) and leave the row active.
                if machine:
                    result.quarantined_hosts.add(machine)
                    log.warning(
                        "reconcile: QUARANTINE machine=%s run_id=%d "
                        "(no state.json, no pidfile) — leaving row ACTIVE "
                        "for next reconcile",
                        machine, row["id"],
                    )
                else:
                    # No host known — fall back to MARK_DONE FAILED so
                    # the row doesn't pile up forever (we have no host
                    # to quarantine and no evidence of liveness).  Not
                    # death-proven → untrusted source.
                    _mark_failed(
                        db_manager, row["id"],
                        error="no state.json found at reconciliation "
                              "(no pidfile, no recorded host)",
                        terminal_source=TS_RECONCILE_NO_HOST,
                    )

    # Mark prior dispatches done if they have no surviving non-terminal rows
    for d in prior:
        d_id = d["dispatch_id"]
        with db_manager.connect() as con:
            still_active = ImagingRunsQueries.list_running_for_dispatch(con, d_id)
        if not still_active:
            with db_manager.connect() as con:
                DispatchesQueries.mark_terminal(
                    con, d_id, state=DispatchState.ABORTED if abandon else DispatchState.DONE,
                )
                con.commit()
            if abandon:
                # Best-effort: reap orphaned staging tokens for this
                # dispatch and ssh-rm /raid/d_<id>/ on every machine in
                # its machines_json snapshot.  Without this, prior code
                # required manual rm -rf on each participating host.
                _cleanup_abandoned_dispatch(
                    d_id, dispatch_root / d_id,
                    d.get("machines_json") or "{}",
                )

    # When abandoning, also try to release a stale dispatcher lock left
    # by a coordinator that died (e.g. machine reboot).  Safe in normal
    # dispatch runs too: our own lock points at our live PID with our
    # own cmdline, which the helper detects and leaves alone.
    if abandon:
        _release_stale_dispatcher_lock(
            dispatch_root / ".dispatcher.lock",
            expected_tokens=["run_imaging_dispatch"],
        )

    return result


def _reconcile_state_entry(
    db_manager, dispatch_id, run_id, state, unit_dir,
    g: GlobalCfg, result: "ReconcileResult", *, abandon: bool,
) -> None:
    """Apply one prior unit's state to the DB; possibly mark for adoption."""
    phase = state.get("phase", "starting")
    # state.json stores ``socket.gethostname()``, which on some hosts is the
    # FQDN (e.g. ``host5.example.com``); normalise to the short name keyed
    # in machines.json so slot accounting and cleanup mappings match.
    machine = (state.get("machine") or state.get("hostname") or "").split(".", 1)[0]
    pid = state.get("worker_pid")
    expected_tokens = [
        f"--dispatch-id {dispatch_id}",
        f"--run-id {int(run_id)}",
    ]

    if phase in ("done", "failed"):
        _apply_terminal_to_db(db_manager, run_id, state, unit_dir)
        return

    if abandon:
        _mark_failed(db_manager, run_id,
                     error=f"abandoned by --abandon-prior at phase={phase}",
                     terminal_source=TS_ABANDONED)
        return

    # Heartbeat check
    hb_path = unit_dir / "heartbeat"
    try:
        hb_age = time.time() - hb_path.stat().st_mtime
    except OSError:
        hb_age = float("inf")

    if hb_age < g.heartbeat_stale_threshold_sec:
        result.adoptable.append({
            "machine": machine, "run_id": run_id,
            "unit_dir": unit_dir, "state": state,
            "expected_tokens": expected_tokens,
            "prior_dispatch_id": dispatch_id,
        })
        return

    alive, info = ssh_pid_alive(machine, pid, expected_tokens, timeout=10)
    if alive is True:
        log.warning(
            "adopting run_id=%d on %s: pid alive but heartbeat stale "
            "%.0fs (cmdline ok)",
            run_id, machine, hb_age,
        )
        result.adoptable.append({
            "machine": machine, "run_id": run_id,
            "unit_dir": unit_dir, "state": state,
            "expected_tokens": expected_tokens,
            "prior_dispatch_id": dispatch_id,
        })
        return
    if alive is False:
        # ssh proved the pid dead/reused — coordinator-verified death.
        _mark_failed(
            db_manager, run_id,
            error=f"abandoned (no heartbeat {hb_age:.0f}s, pid {pid} dead/reused: {info!r})",
            terminal_source=TS_VERIFIED_DEAD,
        )
        return
    # alive is None — ssh-unreachable.  Same fail-closed treatment as the
    # F5 no-pidfile path: we cannot prove the worker is dead, so quarantine
    # the host (if we know it) and leave the row ACTIVE for the next
    # reconcile_prior to recover.  Logging the warning is not enough on its
    # own; without quarantine the next dispatch would happily schedule new
    # work onto a host with an unverified live worker.
    if machine:
        result.quarantined_hosts.add(machine)
        log.warning(
            "reconcile: QUARANTINE machine=%s run_id=%d "
            "(stale heartbeat, ssh-unreachable: %s) — leaving row ACTIVE",
            machine, run_id, info,
        )
    else:
        log.warning(
            "ssh-unreachable during reconcile of run_id=%d on %s: %s; "
            "leaving row active (no host known to quarantine; rerun "
            "--reconcile or use --abandon-prior)",
            run_id, machine, info,
        )


def _apply_terminal_to_db(db_manager, run_id, state, unit_dir) -> None:
    """Idempotent terminal-state apply for a recovered worker."""
    success = bool(state.get("success"))
    status = ImagingRunStatus.SUCCESS if success else ImagingRunStatus.FAILED
    job_json = None
    prov = state.get("provenance") or {}
    if prov.get("job_json"):
        job_json = prov["job_json"]
    with db_manager.connect() as con:
        if state.get("spw_selection") or state.get("field_selection"):
            ImagingRunsQueries.update_resolved(
                con, run_id,
                spw_selection=json.dumps(state.get("spw_selection"))
                if state.get("spw_selection") else None,
                field_selection=json.dumps(state.get("field_selection"))
                if state.get("field_selection") else None,
            )
        if job_json:
            con.execute(
                "UPDATE imaging_runs SET job_json_path=? WHERE id=?",
                (job_json, run_id),
            )
        ImagingRunsQueries.mark_done(
            con, run_id, status=status,
            retcode=0 if success else 1,
            finished_at=state.get("finished_at") or now_iso(),
            duration_sec=0.0,
            output_fits=state.get("output_fits"),
            # Reconcile read the worker's own terminal state.json off NAS.
            terminal_source=TS_WORKER_OBSERVED,
        )
        ImagingRunsQueries.set_error_message(
            con, run_id, state.get("error_message"),
        )
        con.commit()


def _mark_failed(db_manager, run_id, *, error: str, terminal_source: str) -> None:
    # ``terminal_source`` is REQUIRED (v3 Delta 4 / v4): this helper is shared
    # across trusted (verified) and untrusted (abandon / no-host) callers, so
    # the source must be supplied explicitly by each caller — never defaulted.
    with db_manager.connect() as con:
        ImagingRunsQueries.mark_done(
            con, run_id,
            status=ImagingRunStatus.FAILED,
            retcode=1,
            finished_at=now_iso(),
            duration_sec=0.0,
            output_fits=None,
            terminal_source=terminal_source,
        )
        ImagingRunsQueries.set_error_message(con, run_id, error)
        con.commit()


# ---------------------------------------------------------------------------
# Cleanup helpers used by both ``reconcile_prior(abandon=True)`` and the
# running coordinator's TokenReaper.  Kept as module-level functions so
# the abandon path can invoke them without spawning a thread.
# ---------------------------------------------------------------------------

# A token whose ``host``/``pid`` files are missing is either fresh
# (worker mid-write between mkdir and metadata) or a worker that crashed
# in that window.  Reap only after this grace period so live workers
# aren't disrupted.
_TOKEN_MALFORMED_GRACE_SEC = 60.0


# Per-reaper throttled-WARN interval for inconclusive skips (the silent
# 20h skip in the incident is why this exists).
_TOKEN_WARN_THROTTLE_SEC = 300.0


@dataclass
class _SweepCounters:
    """Thread-safe sweep tallies, surfaced in the watchdog line."""
    tokens_reaped_db: int = 0
    tokens_reaped_ssh: int = 0
    tokens_skipped_inconclusive: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    # (slot, reason) -> last monotonic time we WARNed about it.
    _last_warn: dict = field(default_factory=dict)

    def add(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "tokens_reaped_db": self.tokens_reaped_db,
                "tokens_reaped_ssh": self.tokens_reaped_ssh,
                "tokens_skipped_inconclusive": self.tokens_skipped_inconclusive,
            }

    def should_warn(self, slot: str, reason: str) -> bool:
        now = time.monotonic()
        with self.lock:
            key = (slot, reason)
            last = self._last_warn.get(key)
            if last is None or (now - last) >= _TOKEN_WARN_THROTTLE_SEC:
                self._last_warn[key] = now
                return True
            return False


def _parse_holder(holder: str) -> tuple[Optional[str], Optional[int]]:
    """Parse a token ``holder`` (``<dispatch_id>/<run_id>``).

    Returns ``(dispatch_id, run_id)`` or ``(None, None)`` when the holder
    is empty/malformed.  ``dispatch_id`` itself may contain underscores
    (``d_<ts>_<hex>``) but never a slash, so we split on the LAST slash.
    """
    if not holder:
        return None, None
    sep = holder.rfind("/")
    if sep < 0:
        return None, None
    d_id = holder[:sep].strip()
    rid_str = holder[sep + 1:].strip()
    if not d_id or not rid_str:
        return None, None
    try:
        return d_id, int(rid_str)
    except ValueError:
        return None, None


def _resolve_token_host(
    recorded_host: Optional[str],
    machines: Optional[dict],
) -> Optional[str]:
    """Map a token's recorded ``host`` to a name we can ssh.

    The worker records ``socket.gethostname()``, which on some hosts is the
    FQDN — and the FQDN may carry a broken/changed known_hosts key while the
    short name used in machines.json does not (the incident's trigger).

    Resolution (never blind dot-split):
    - exact match against a machines key → use it verbatim;
    - else, if the short form (``host.split('.',1)[0]``) UNIQUELY matches one
      machines key, use that key;
    - else (ambiguous, or deployment intentionally uses an FQDN/alias not in
      machines) → keep the recorded host.
    """
    if not recorded_host:
        return recorded_host
    keys = set(machines or {})
    if recorded_host in keys:
        return recorded_host
    short = recorded_host.split(".", 1)[0]
    matches = [k for k in keys if k.split(".", 1)[0] == short]
    if len(matches) == 1:
        return matches[0]
    return recorded_host


def _holder_terminal_source(
    db_manager,
    holder_dispatch_id: str,
    run_id: int,
) -> Optional[str]:
    """Return the ``terminal_source`` of a terminal holder run, else None.

    Opens a SHORT-LIVED connection (the reaper thread must not hold one).
    Returns the source string only if the row exists, belongs to
    ``holder_dispatch_id``, AND is in a terminal status; otherwise None
    (non-terminal / missing / wrong dispatch / DB error → ssh fallback).
    """
    if db_manager is None:
        return None
    try:
        with db_manager.connect() as con:
            row = ImagingRunsQueries.get_by_id(con, run_id)
    except Exception:
        log.debug("reaper DB query failed for run_id=%s", run_id, exc_info=True)
        return None
    if not row:
        return None
    if row.get("dispatch_id") != holder_dispatch_id:
        return None
    if row.get("status") not in (
        ImagingRunStatus.SUCCESS, ImagingRunStatus.FAILED,
    ):
        return None
    return row.get("terminal_source") or TS_UNKNOWN


def _reap_token(slot_dir: Path, fp_snapshot: Optional[bytes]) -> bool:
    """Generation-checked reclaim of a token slot.  Returns True if reaped.

    Uses ``_atomic_reclaim`` (rename-then-verify) so a slot released +
    reacquired by a fresh holder between our decision and our delete is
    put back rather than wiped — NOT a blind ``shutil.rmtree``.
    """
    return _atomic_reclaim(
        Path(slot_dir),
        expected_fingerprint=fp_snapshot,
        read_fingerprint=_read_token_fingerprint,
        malformed_grace_sec=_TOKEN_MALFORMED_GRACE_SEC,
    )


def _sweep_tokens_once(
    tokens_dir: Path,
    expected_tokens: list[str],
    *,
    malformed_grace_sec: float = _TOKEN_MALFORMED_GRACE_SEC,
    ssh_timeout: int = 8,
    db_manager=None,
    machines: Optional[dict] = None,
    counters: Optional[_SweepCounters] = None,
) -> int:
    """Reap stale staging tokens under *tokens_dir*; return count reaped.

    Used by the running coordinator's :class:`TokenReaper` (every
    ``token_reaper_interval_sec``) and by ``reconcile_prior`` when
    cleaning up an abandoned dispatch.

    Per-token decision is DB-FIRST (v3 Delta A): if the holder run is
    terminal via a TRUSTED source (``worker_observed`` / ``verified_dead`` /
    ``verified_killed``) the token is reaped WITHOUT any ssh probe — the
    trusted terminal already proved the worker dead, so we never touch the
    fragile ssh path that leaked the token for 20h in the incident.

    Otherwise we fall back to ``ssh_pid_alive`` with expected tokens that
    include BOTH ``--dispatch-id`` AND ``--run-id`` (so a PID reused by a
    different worker of the same dispatch is not mistaken for the live
    holder): True→keep, False→reap, None→keep + throttled WARN.

    ``db_manager``/``machines`` default to None so the abandon-cleanup
    caller can keep ssh-only behaviour (``abandoned`` rows are UNTRUSTED).
    """
    reaped = 0
    if not Path(tokens_dir).exists():
        return 0

    def _bump(**kw):
        if counters is not None:
            counters.add(**kw)

    def _warn(slot, reason, msg, *args):
        if counters is None or counters.should_warn(slot, reason):
            log.warning(msg, *args)

    for t in list_held_tokens(tokens_dir):
        slot = t["slot"]
        host = t.get("host")
        pid = t.get("pid")
        slot_path = Path(t["path"])

        # Generation fingerprint derived from the SAME ``list_held_tokens``
        # read pass that produced host/pid/holder — NOT a second re-read of
        # the slot files.  A coherent snapshot is essential: if the slot is
        # released + reacquired by a fresh holder between reads, a re-read
        # fingerprint would describe the NEW holder while our DB/ssh decision
        # is about the OLD one, and ``_atomic_reclaim``'s post-rename verify
        # would then match — and delete — the live new holder.  Sharing one
        # snapshot makes identity and fingerprint describe the same
        # generation, so a race instead yields a mismatch → safe put-back.
        fp_snapshot = token_fingerprint_from_fields(
            t.get("pid"), t.get("starttime_ticks"),
        )

        # Missing host/pid: malformed-grace path (unchanged).  Distinct from
        # a bad/empty *holder* with valid host+pid (handled below).
        if not host or not pid:
            try:
                age = time.time() - slot_path.stat().st_mtime
            except OSError:
                continue
            if age >= malformed_grace_sec:
                if _reap_token(slot_path, fp_snapshot):
                    reaped += 1
                    _bump(tokens_reaped_ssh=1)
                    log.info(
                        "reaped malformed staging token %s "
                        "(no holder metadata after %.0fs)",
                        slot, age,
                    )
            continue

        ssh_host = _resolve_token_host(host, machines)
        holder = t.get("holder") or ""
        holder_did, run_id = _parse_holder(holder)

        if holder_did is None or run_id is None:
            # Bad/empty holder but valid host+pid: NO DB fallback (we can't
            # map to a run), NO malformed-rmtree.  ssh-coarse with just the
            # dispatch-id; keep on inconclusive.
            alive, info = ssh_pid_alive(
                ssh_host, pid, expected_tokens, timeout=ssh_timeout,
            )
            if alive is False:
                if _reap_token(slot_path, fp_snapshot):
                    reaped += 1
                    _bump(tokens_reaped_ssh=1)
                    log.info(
                        "reaped stale staging token %s "
                        "(bad holder %r, host=%s pid=%s ssh-dead)",
                        slot, holder, ssh_host, pid,
                    )
            elif alive is None:
                _bump(tokens_skipped_inconclusive=1)
                _warn(slot, "bad_holder_inconclusive",
                      "token reaper: KEEP slot %s (bad holder %r, "
                      "host=%s pid=%s) — ssh inconclusive: %s",
                      slot, holder, ssh_host, pid, info)
            continue

        # DB-FIRST: trusted terminal → reap without ssh.
        src = _holder_terminal_source(db_manager, holder_did, run_id)
        if src in TRUSTED_TERMINAL_SOURCES:
            if _reap_token(slot_path, fp_snapshot):
                reaped += 1
                _bump(tokens_reaped_db=1)
                log.info(
                    "reaped staging token %s via DB (holder=%s source=%s, "
                    "no ssh needed)",
                    slot, holder, src,
                )
            continue

        # Non-terminal / untrusted terminal / missing row → ssh fallback,
        # matching BOTH dispatch-id and run-id.
        per_token_tokens = [
            f"--dispatch-id {holder_did}",
            f"--run-id {run_id}",
        ]
        alive, info = ssh_pid_alive(
            ssh_host, pid, per_token_tokens, timeout=ssh_timeout,
        )
        if alive is True:
            continue
        if alive is False:
            if _reap_token(slot_path, fp_snapshot):
                reaped += 1
                _bump(tokens_reaped_ssh=1)
                log.info(
                    "reaped stale staging token %s (holder=%s host=%s "
                    "pid=%s ssh-dead)",
                    slot, holder, ssh_host, pid,
                )
            continue
        # alive is None — inconclusive.  Keep + throttled WARN.
        _bump(tokens_skipped_inconclusive=1)
        _warn(slot, "inconclusive",
              "token reaper: KEEP slot %s (holder=%s host=%s pid=%s "
              "db_source=%s) — ssh inconclusive: %s",
              slot, holder, ssh_host, pid, src, info)
    return reaped


def _release_stale_dispatcher_lock(
    lock_path: Path,
    expected_tokens: list[str],
    *,
    timeout: int = 8,
) -> bool:
    """Unlink the dispatcher lock file iff its recorded holder is dead.

    The lock file format written by :class:`DispatcherLock.acquire` is
    ``host=<fqdn> pid=<int>``.  We strip the FQDN to a short name and
    consult ``ssh_pid_alive`` with *expected_tokens* (a substring of our
    own dispatcher cmdline, e.g. ``run_imaging_dispatch``) to confirm
    the holder process is dead AND not a PID-reused unrelated process.
    Returns ``True`` iff we removed the file.

    Conservative on every error path: missing/malformed lock, ssh
    failure, or live holder all return ``False`` without touching the
    file.  Safe to call in a live-dispatcher run — we'll find our own
    PID alive with our own cmdline and leave the lock alone.
    """
    try:
        text = lock_path.read_text().strip()
    except OSError:
        return False
    m = re.search(r"host=(\S+)\s+pid=(\d+)", text)
    if not m:
        return False
    host = m.group(1).split(".", 1)[0]
    try:
        pid = int(m.group(2))
    except (TypeError, ValueError):
        return False
    alive, info = ssh_pid_alive(host, pid, expected_tokens, timeout=timeout)
    if alive is True:
        return False
    if alive is None:
        log.warning(
            "stale dispatcher lock not released: ssh-unreachable to %s "
            "(holder %r); leaving %s in place",
            host, text, lock_path,
        )
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return False
    log.info(
        "released stale dispatcher lock %s (holder %r was dead/reused)",
        lock_path, text,
    )
    return True


def _cleanup_abandoned_dispatch(
    dispatch_id: str,
    dispatch_dir: Path,
    machines_json_str: str,
    *,
    ssh_timeout: int = 30,
) -> dict:
    """Best-effort cleanup of a dispatch we just force-abandoned.

    1. Sweep orphaned staging tokens under
       ``<dispatch_dir>/staging_tokens/`` using ``_sweep_tokens_once``.
       The expected token is ``--dispatch-id <id>`` so a PID reused for
       an unrelated live process is correctly classified as dead.
    2. ``ssh <m> "rm -rf <raid>/d_<dispatch_id>/"`` on every machine in
       the ``machines_json`` snapshot stored on the ``dispatches`` row.

    Returns a summary dict.  Errors on any single host are logged and
    counted but never raised — this runs from ``reconcile_prior`` which
    must remain best-effort across a partial cluster.
    """
    summary = {"tokens_reaped": 0, "machines_swept": [], "machine_failures": {}}

    tokens_dir = dispatch_dir / "staging_tokens"
    expected = [f"--dispatch-id {dispatch_id}"]
    try:
        summary["tokens_reaped"] = _sweep_tokens_once(tokens_dir, expected)
    except Exception:
        log.exception(
            "token sweep failed for abandoned dispatch %s", dispatch_id,
        )

    try:
        machines = json.loads(machines_json_str or "{}")
    except (TypeError, ValueError):
        log.warning(
            "could not parse machines_json for dispatch %s", dispatch_id,
        )
        machines = {}

    for name, mcfg in machines.items():
        raid = (mcfg or {}).get("raid")
        if not raid:
            continue
        target = f"{raid}/d_{dispatch_id}"
        cmd = f"rm -rf -- {shlex.quote(target)}"
        try:
            proc = ssh_run(name, cmd, timeout=ssh_timeout)
        except subprocess.TimeoutExpired:
            summary["machine_failures"][name] = "ssh timeout"
            log.warning(
                "raid sweep timeout on %s for dispatch %s", name, dispatch_id,
            )
            continue
        except Exception as e:
            summary["machine_failures"][name] = str(e)
            log.warning(
                "raid sweep error on %s for dispatch %s: %s",
                name, dispatch_id, e,
            )
            continue
        if proc.returncode != 0:
            summary["machine_failures"][name] = (
                f"rc={proc.returncode}: {(proc.stderr or '')[-120:]}"
            )
            log.warning(
                "raid sweep rc=%d on %s for dispatch %s: %s",
                proc.returncode, name, dispatch_id,
                (proc.stderr or "")[-120:],
            )
            continue
        summary["machines_swept"].append(name)

    log.info(
        "abandoned dispatch %s cleaned: tokens_reaped=%d, swept=%s, failed=%s",
        dispatch_id, summary["tokens_reaped"],
        summary["machines_swept"], summary["machine_failures"],
    )
    return summary


# ---------------------------------------------------------------------------
# Token reaper (background thread)
# ---------------------------------------------------------------------------

class TokenReaper(threading.Thread):
    def __init__(
        self,
        tokens_dir: Path,
        g: GlobalCfg,
        expected_tokens: list[str],
        *,
        db_manager=None,
        machines: Optional[dict] = None,
    ):
        super().__init__(name="token-reaper", daemon=True)
        self.tokens_dir = Path(tokens_dir)
        self.g = g
        self.expected_tokens = list(expected_tokens)
        # DB-backed reaping: a SHORT-LIVED connection is opened per query
        # inside the sweep — the thread never holds a connection.
        self.db_manager = db_manager
        # machines.json snapshot for the hostname resolver (FQDN→short).
        self.machines = machines
        self.counters = _SweepCounters()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    # Kept as a class attribute for backwards-compat with code that may
    # have referenced ``TokenReaper.MALFORMED_GRACE_SEC``; the actual
    # sweep delegates to the module-level ``_sweep_tokens_once``.
    MALFORMED_GRACE_SEC = _TOKEN_MALFORMED_GRACE_SEC

    def run(self) -> None:
        while not self._stop_event.wait(self.g.token_reaper_interval_sec):
            try:
                self._sweep()
            except Exception:
                log.exception("token reaper sweep failed")

    def _sweep(self) -> int:
        """Thin wrapper around :func:`_sweep_tokens_once` — kept so
        existing tests that call ``reaper._sweep()`` still work."""
        return _sweep_tokens_once(
            self.tokens_dir,
            self.expected_tokens,
            malformed_grace_sec=self.MALFORMED_GRACE_SEC,
            ssh_timeout=self.g.ssh_timeout_sec,
            db_manager=self.db_manager,
            machines=self.machines,
            counters=self.counters,
        )


# ---------------------------------------------------------------------------
# Cleanup thread (coordinator-driven)
# ---------------------------------------------------------------------------

class GousCleaner(threading.Thread):
    def __init__(
        self,
        scheduler: SchedulerState,
        cfg: MachinesConfig,
        dispatch_id: str,
        g: GlobalCfg,
    ):
        super().__init__(name="gous-cleaner", daemon=True)
        self.scheduler = scheduler
        self.cfg = cfg
        self.dispatch_id = dispatch_id
        self.g = g
        self._stop_event = threading.Event()
        self._cleaned: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(self.g.cleanup_interval_sec):
            try:
                self._sweep()
            except Exception:
                log.exception("GOUS cleaner sweep failed")

    def force_run(self) -> None:
        try:
            self._sweep()
        except Exception:
            log.exception("GOUS cleaner force_run failed")

    def _sweep(self) -> None:
        # Iterate seen_pairs (not in_flight) so we still catch GOUSs whose
        # in_flight entry has been deleted by mark_terminal.
        with self.scheduler.lock:
            pairs = list(self.scheduler.seen_pairs)
        for machine, gous in pairs:
            with self._lock:
                if (machine, gous) in self._cleaned:
                    continue
            with self.scheduler.lock:
                in_flight = self.scheduler.in_flight.get((machine, gous), set())
            if in_flight:
                continue
            if self.scheduler.queued_for_gous_total(gous):
                continue
            with self._lock:
                if (machine, gous) in self._cleaned:
                    continue
                self._cleaned.add((machine, gous))
            self._clean_one(machine, gous)

    def _clean_one(self, machine: str, gous: str) -> None:
        m = self.cfg.machines.get(machine)
        if m is None:
            return
        # v3 F2.d: never rm -rf on a quarantined host — its worker may
        # still be alive and reading inputs.
        if self.scheduler.is_quarantined(machine):
            log.info(
                "cleaner: skipping cleanup for %s on quarantined %s",
                gous, machine,
            )
            return
        # Adopted prior-dispatch units live under /raid/d_<prior_id>/...
        # (not /raid/d_<new_id>/...), so we resolve the dispatch_id per
        # pair before building the rm path.
        with self.scheduler.lock:
            pair_did = self.scheduler.pair_dispatch_id.get(
                (machine, gous), self.dispatch_id,
            )
            ok_run_ids = sorted(self.scheduler.success_run_ids.get(
                (machine, gous), set(),
            ))
        gous_input = f"{m.raid}/d_{pair_did}/input/{gous}"
        cmds = [f"rm -rf -- {shlex.quote(gous_input)}"]
        # Also delete work/runs/<id> for SUCCESS units of this (machine,
        # gous).  We read from the scheduler's in-memory record rather
        # than the DB so a queued-but-not-yet-committed MARK_DONE event
        # can't cause us to miss a cleanup.  Failed-unit work dirs are
        # preserved for forensics; their logs are already on NAS via
        # remote_worker._copy_provenance_to_nas.
        if ok_run_ids:
            for rid in ok_run_ids:
                rd = f"{m.raid}/d_{pair_did}/work/runs/{int(rid)}"
                cmds.append(f"rm -rf -- {shlex.quote(rd)}")
        try:
            ssh_run(machine, " && ".join(cmds), timeout=60)
            log.info(
                "cleaned (dispatch=%s): input/%s + %d success work-dirs on %s",
                pair_did, gous, len(ok_run_ids), machine,
            )
        except subprocess.SubprocessError as exc:
            log.warning("cleanup ssh failed for %s:%s: %s", machine, gous, exc)


# ---------------------------------------------------------------------------
# Per-machine worker thread
# ---------------------------------------------------------------------------

class MachineSlot(threading.Thread):
    def __init__(
        self,
        slot_id: str,
        machine: MachineCfg,
        ctx: "DispatchContext",
    ):
        super().__init__(name=f"slot-{slot_id}", daemon=True)
        self.slot_id = slot_id
        self.machine = machine
        self.ctx = ctx
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            # Quarantine check BEFORE pick so a quarantine raced into the
            # scheduler between iterations also exits the slot.
            if self.ctx.scheduler.is_quarantined(self.machine.name):
                log.warning(
                    "slot %s exiting: %s quarantined",
                    self.slot_id, self.machine.name,
                )
                return
            unit = self.ctx.scheduler.pick(self.machine.name, self._next_run_id)
            if unit is None:
                # Drained — exit slot
                return
            # Non-blocking 5: another thread (e.g. an AdoptionPoller
            # finishing on this host, or _dispatch_one in a sibling slot)
            # may have quarantined this host between the pre-pick check
            # above and now.  Re-check immediately before dispatching so
            # we don't oversubscribe the host with one extra unit.  The
            # picked unit is dropped from this slot's view; the next
            # dispatch's reconcile_prior + selection will re-discover it
            # (we deliberately have not run mark_inflight / INSERT_QUEUED
            # for it yet, so there is no DB row to leave dangling).
            if self.ctx.scheduler.is_quarantined(self.machine.name):
                log.warning(
                    "slot %s exiting: %s quarantined between pick() and "
                    "dispatch; dropping unit %s/%s/spw%s (will be "
                    "re-selected by the next dispatch)",
                    self.slot_id, self.machine.name,
                    unit.gous_uid, unit.source_name, unit.spw_id,
                )
                return
            try:
                self._dispatch_one(unit)
            except Exception:
                log.exception("slot %s crashed on unit %s/%s/spw%s",
                              self.slot_id, unit.gous_uid, unit.source_name, unit.spw_id)
            # Post-dispatch quarantine check: _dispatch_one may have
            # quarantined the host on inconclusive cleanup (F2.d).
            if self.ctx.scheduler.is_quarantined(self.machine.name):
                log.warning(
                    "slot %s exiting: %s quarantined",
                    self.slot_id, self.machine.name,
                )
                return

    def _next_run_id(self) -> Optional[int]:  # not used by current scheduler
        return None

    def _dispatch_one(self, unit: ImagingUnit) -> None:
        c = self.ctx
        machine = self.machine.name

        # 1. Allocate run_id (insert QUEUED row)
        from panta_rei.imaging.runner import get_casa_version
        from panta_rei.imaging.matching import (
            _compute_tm_freq_range, build_output_path,
        )
        recovered = unit.recovered_params or {}
        tm_freq = _compute_tm_freq_range(
            recovered.get("start", ""),
            recovered.get("width", ""),
            recovered.get("nchan"),
        )
        freq_min = tm_freq[0] if tm_freq else (unit.tp_freq_min or 0)
        freq_max = tm_freq[1] if tm_freq else (unit.tp_freq_max or 0)
        canonical_fits = build_output_path(
            c.publish_dir, unit.gous_uid, unit.source_name, freq_min, freq_max,
        )
        imagename_stem = str(canonical_fits).replace(".pbcor.fits", "")
        scales_str = json.dumps(c.scales)

        row_id_holder: dict = {}
        c.db_writer.q.put({
            "op": "INSERT_QUEUED",
            "row": {
                "params_id": unit.params_id,
                "gous_uid": unit.gous_uid,
                "source_name": unit.source_name,
                "line_group": unit.line_group,
                "spw_id": unit.spw_id,
                "vis_tm": json.dumps(unit.vis_tm),
                "vis_sm": json.dumps(unit.vis_sm),
                "sdimage": unit.sdimage,
                "mous_uids_tm": json.dumps(unit.mous_uids_tm),
                "mous_uids_sm": json.dumps(unit.mous_uids_sm),
                "mous_uids_tp": json.dumps(unit.mous_uids_tp),
                "output_dir": str(c.publish_dir),
                "imagename": imagename_stem,
                "deconvolver": c.deconvolver,
                "scales": scales_str,
                "casa_version": get_casa_version(),
                "started_at": now_iso(),
                "status": ImagingRunStatus.QUEUED,
                "hostname": machine,
                "method": "tclean_feather",
                "parallel": 1,
                "dispatch_id": c.dispatch_id,
            },
            "row_id_holder": row_id_holder,
        })
        # Wait for the writer to populate the row id.
        for _ in range(200):
            if "id" in row_id_holder:
                break
            time.sleep(0.05)
        if "id" not in row_id_holder:
            log.error("DB writer did not return run_id within 10s; skipping unit")
            return
        run_id = int(row_id_holder["id"])

        # Sentinels covering everything that must be reconciled in the
        # ``finally`` below.  Initialised BEFORE the try so they're always
        # defined regardless of where in the pre-launch path we crash.
        #
        # Once mark_inflight is set we MUST either:
        #   - release it with mark_terminal (cleanup verified — FAILED
        #     or SUCCESS), or
        #   - quarantine the host (cleanup unverified — leave run row
        #     active for next dispatch's reconcile_prior to recover).
        #
        # Sentinels:
        #   ``terminal_recorded``     — mark_terminal was called.
        #   ``quarantine_recorded``   — _quarantine_and_log was called
        #                               (or the host is otherwise known
        #                               to have been deliberately
        #                               quarantined for this unit).  The
        #                               finally must not undo this by
        #                               re-running the post-launch
        #                               recovery path.
        #   ``worker_possibly_launched`` — launch_detached returned;
        #                                  remote worker may be alive
        #                                  even on SSH failure (Delta 2).
        #   ``mark_inflight_called``  — scheduler.mark_inflight succeeded;
        #                               only then is mark_terminal valid.
        terminal_recorded = False
        quarantine_recorded = False
        worker_possibly_launched = False
        mark_inflight_called = False
        wrapper_pid: Optional[int] = None
        nas_unit_dir: Optional[Path] = None

        expected_tokens = [
            f"--dispatch-id {c.dispatch_id}",
            f"--run-id {run_id}",
        ]

        def _emit_mark_done_failed(
            error_message: str, *, terminal_source: str, retcode: int = 1,
        ) -> None:
            # ``terminal_source`` is REQUIRED: the three call sites differ
            # (launch-failed verified → verified_dead; post-launch verified
            # → by outcome; pre-launch crash → unknown, NOT death-proven).
            log.info(
                "dispatch: MARK_DONE FAILED run_id=%d machine=%s reason=%s "
                "error=%s",
                run_id, machine, "synthetic", error_message,
            )
            log.warning(
                "slot %s unit run_id=%d failed: %s",
                self.slot_id, run_id, error_message,
            )
            c.db_writer.q.put({
                "op": "MARK_DONE",
                "run_id": run_id,
                "status": ImagingRunStatus.FAILED,
                "retcode": int(retcode),
                "finished_at": now_iso(),
                "duration_sec": 0.0,
                "error_message": error_message,
                "terminal_source": terminal_source,
            })

        def _quarantine_and_log(reason: str, detail: str) -> None:
            """Quarantine the host; do NOT MARK_DONE; do NOT mark_terminal.

            Sets ``quarantine_recorded`` so the outer ``finally`` does not
            undo this by re-running the post-launch recovery path (which
            would otherwise re-verify and possibly MARK_DONE the row that
            we have deliberately left ACTIVE).
            """
            nonlocal quarantine_recorded
            c.scheduler.quarantine(machine, reason=reason)
            quarantine_recorded = True
            log.warning(
                "dispatch: QUARANTINE machine=%s run_id=%d reason=%s detail=%s "
                "— run row left ACTIVE for next reconcile_prior",
                machine, run_id, reason, detail,
            )

        try:
            # 2. Build NAS unit dir + manifest.  Delta 2: this MUST live
            # inside the try/finally so a pre-launch crash (e.g. NAS write
            # failure during manifest/launcher generation) is recoverable.
            # A pre-launch failure provably did not launch a worker, so
            # the finally can safely MARK_DONE FAILED + mark_terminal
            # (gated on ``mark_inflight_called`` to avoid spurious
            # mark_terminal for an unregistered (machine, gous) pair).
            nas_unit_dir = c.dispatch_dir / "units" / str(run_id)
            nas_unit_dir.mkdir(parents=True, exist_ok=True)

            # Use the GOUS-union of inputs the coordinator pre-computed
            expected_inputs = c.gous_inputs[unit.gous_uid]

            manifest_path = write_unit_manifest(
                nas_unit_dir,
                unit=unit,
                expected_inputs=expected_inputs,
                publish_dir=c.publish_dir,
                nproc=self.machine.nproc,
                casa_path=c.cfg.casa_path,
                deconvolver=c.deconvolver,
                scales=c.scales,
            )

            # 3. Build per-unit launcher.sh on NAS
            raid_dir = f"{self.machine.raid}/d_{c.dispatch_id}"
            # Per-host cross-dispatch staging cache lives next to the
            # per-dispatch dir, scoped to this host's /raid only.
            cache_root = (
                f"{self.machine.raid}/cache"
                if self.machine.cache_min_free_gb is not None else None
            )
            launcher = write_launcher_script(
                nas_unit_dir, c.cfg,
                raid_dir=raid_dir,
                manifest_path=str(manifest_path),
                run_id=run_id,
                dispatch_id=c.dispatch_id,
                transfer_method=c.transfer_method,
                publish_policy=c.publish_policy,
                tokens_dir=str(c.tokens_dir),
                max_concurrent_staging=c.cfg.global_cfg.max_concurrent_staging,
                token_wait_timeout_sec=c.cfg.global_cfg.token_wait_timeout_sec,
                heartbeat_interval=c.cfg.global_cfg.heartbeat_interval_sec,
                cache_root=cache_root,
                cache_min_free_gb=self.machine.cache_min_free_gb,
            )

            # 4. Mark in-flight before SSH so the cleaner doesn't fire.
            # Tag with the *current* dispatch_id so cleanup hits the right
            # /raid/d_<id>/... tree.
            c.scheduler.mark_inflight(
                machine, unit.gous_uid, run_id, dispatch_id=c.dispatch_id,
            )
            mark_inflight_called = True

            # 5. Launch detached
            ok, msg, _wpid = launch_detached(
                machine, launcher, nas_unit_dir,
                timeout=c.cfg.global_cfg.ssh_timeout_sec,
            )
            wrapper_pid = _wpid
            # Delta 2: any return from launch_detached (ok or not) means
            # the SSH command may have backgrounded the worker before
            # returning.  Mark worker_possibly_launched so the crash/
            # finally path uses verify-or-quarantine.
            worker_possibly_launched = True

            # Delta 3: if wrapper_pid is unknown for any reason (ssh
            # timeout, ssh rc!=0, or ssh ok but unparsable pidfile),
            # wait briefly for worker.pidfile to appear and parse it.
            if wrapper_pid is None:
                wrapper_pid = _wait_for_pidfile(
                    nas_unit_dir,
                    c.cfg.global_cfg.launch_pidfile_wait_sec,
                )

            if not ok:
                # SSH launch did not return cleanly.  If we now have a
                # pidfile, the worker likely launched — run verify+kill.
                # If we don't, treat as inconclusive (Delta 3) and
                # quarantine.
                if wrapper_pid is not None:
                    outcome, detail = verify_and_kill_worker(
                        machine, wrapper_pid, expected_tokens,
                        c.cfg.global_cfg,
                    )
                    if outcome == VERIFY_INCONCLUSIVE:
                        _quarantine_and_log(
                            "ssh_launch_failed_unverified",
                            f"{msg}; {detail}",
                        )
                        return
                    # Verified dead/killed → safe to MARK_DONE FAILED.
                    _emit_mark_done_failed(
                        f"ssh launch failed: {msg}; cleanup={detail}",
                        terminal_source=_source_from_verify(outcome),
                        retcode=255,
                    )
                    c.scheduler.mark_terminal(
                        machine, unit.gous_uid, run_id, success=False,
                    )
                    terminal_recorded = True
                    return
                # No pidfile after wait — inconclusive.
                _quarantine_and_log(
                    "ssh_launch_failed_no_pidfile",
                    f"{msg}; no worker.pidfile after "
                    f"{c.cfg.global_cfg.launch_pidfile_wait_sec}s",
                )
                return

            # Delta 3 (non-blocking 4): ok=True but we never learned a
            # wrapper_pid, even after waiting for the pidfile.  Per v4
            # Delta 3 this is explicitly inconclusive — absence of the
            # pidfile is NOT proof the worker didn't launch.  Quarantine
            # rather than polling, to avoid silently scheduling more work
            # onto a host that may have a live worker we can't track.
            if wrapper_pid is None:
                _quarantine_and_log(
                    "launch_ok_no_pidfile",
                    f"launch_detached returned ok={ok} msg={msg!r} but no "
                    f"worker.pidfile appeared within "
                    f"{c.cfg.global_cfg.launch_pidfile_wait_sec}s",
                )
                return

            # 6. Poll until terminal
            t0 = time.monotonic()

            def _on_phase(phase, state):
                if phase == "running":
                    # Staging finished — let other slots pick this GOUS freely.
                    c.scheduler.mark_staged(machine, unit.gous_uid)
                # Push MARK_RUNNING the first time we see worker_pid in state
                if state.get("worker_pid"):
                    c.db_writer.q.put({
                        "op": "MARK_RUNNING",
                        "run_id": run_id,
                        "remote_workdir": raid_dir,
                        "worker_pid": int(state["worker_pid"]),
                        "worker_pgid": int(state.get("worker_pgid") or 0),
                        "hostname": machine,
                    })

            last_db_hb_state = {"t": 0.0}
            hb_throttle = max(
                5.0, c.cfg.global_cfg.heartbeat_interval_sec / 2,
            )

            def _on_poll(state):
                # Throttled DB heartbeat so imaging_runs.last_heartbeat is
                # observable from `panta-rei` queries.  NAS heartbeat is the
                # real liveness source; this is a visibility convenience.
                now = time.monotonic()
                if now - last_db_hb_state["t"] < hb_throttle:
                    return
                last_db_hb_state["t"] = now
                c.db_writer.q.put({
                    "op": "HEARTBEAT", "run_id": run_id,
                    "ts": now_iso(),
                })

            final = poll_state_until_terminal(
                machine, nas_unit_dir, g=c.cfg.global_cfg,
                expected_tokens=expected_tokens,
                on_phase_change=_on_phase,
                on_poll=_on_poll,
            )

            # 6b. Synthetic-terminal handling (F2.b + F2.h + Delta 1).
            # If final.reason is in _SYNTHETIC_REASONS, we did NOT observe
            # a real terminal — run verify-or-quarantine before MARK_DONE.
            if not _is_observed_terminal(final) and final.get("phase") == "failed":
                reason = final.get("reason")
                # F2.b.i: re-read state.json once.  If phase is now
                # publishing/done/running, resume polling for the late-state
                # grace window.  If grace expires without a terminal, quarantine.
                if reason == "state_missing_timeout":
                    pid_for_kill = (
                        wrapper_pid
                        if wrapper_pid is not None
                        else _read_pidfile(nas_unit_dir)
                    )
                    final = self._handle_state_missing_timeout(
                        nas_unit_dir, machine, expected_tokens,
                        pid_for_kill, _on_phase, _on_poll, run_id,
                    )
                    if final is None:
                        # Quarantine path taken inside helper.
                        _quarantine_and_log(
                            "state_missing_timeout_unverified",
                            "see prior log line for verify outcome",
                        )
                        return
                elif reason == "heartbeat_stale_alive":
                    # Delta 1: worker alive but heartbeat dead.  Verify + kill.
                    pid_for_kill = (
                        final.get("worker_pgid")
                        or final.get("worker_pid")
                        or wrapper_pid
                    )
                    outcome, detail = verify_and_kill_worker(
                        machine, pid_for_kill, expected_tokens,
                        c.cfg.global_cfg,
                    )
                    if outcome == VERIFY_INCONCLUSIVE:
                        _quarantine_and_log(
                            "heartbeat_stale_alive_unverified",
                            detail,
                        )
                        return
                    final = {
                        **final,
                        # Coordinator verified the worker dead/killed; carry
                        # the source so the enqueue below stamps it (v5 D2).
                        "terminal_source": _source_from_verify(outcome),
                        "error_message": (
                            (final.get("error_message") or "")
                            + f" — cleanup={detail}"
                        ),
                    }

            elapsed = time.monotonic() - t0

            # 7. Apply terminal state to DB (observed terminal OR verified-cleanup
            # synthetic terminal).  F1 + F3: INFO + WARN log lines.
            success = bool(final.get("success"))
            status = (ImagingRunStatus.SUCCESS if success
                      else ImagingRunStatus.FAILED)
            err = final.get("error_message")
            log.info(
                "dispatch: MARK_DONE %s run_id=%d machine=%s elapsed=%.1fs "
                "error=%s",
                status, run_id, machine, elapsed, err,
            )
            if not success:
                log.warning(
                    "slot %s unit run_id=%d failed: %s",
                    self.slot_id, run_id, err,
                )
            prov = final.get("provenance") or {}
            c.db_writer.q.put({
                "op": "MARK_DONE",
                "run_id": run_id,
                "status": status,
                "retcode": 0 if success else 1,
                "finished_at": final.get("finished_at") or now_iso(),
                "duration_sec": float(elapsed),
                "output_fits": final.get("output_fits"),
                "spw_selection": (json.dumps(final["spw_selection"])
                                  if final.get("spw_selection") else None),
                "field_selection": (json.dumps(final["field_selection"])
                                    if final.get("field_selection") else None),
                "error_message": err,
                "job_json_path": prov.get("job_json"),
                # v5: source set explicitly by poll/verify paths, carried
                # through; default unknown for anything that didn't stamp.
                "terminal_source": final.get("terminal_source", TS_UNKNOWN),
            })
            empty = c.scheduler.mark_terminal(
                machine, unit.gous_uid, run_id, success=success,
            )
            terminal_recorded = True
            if empty:
                # Possible cleanup opportunity — let the cleaner sweep
                log.debug("(machine=%s, gous=%s) drained", machine, unit.gous_uid)
        finally:
            # Blocking 1 + 2: only enter the recovery path if NEITHER a
            # terminal nor a deliberate quarantine has been recorded.
            # ``quarantine_recorded`` covers the case where a control-flow
            # path (e.g. ssh_launch_failed_no_pidfile) intentionally left
            # the row ACTIVE on a quarantined host and returned — without
            # this guard the finally would re-verify and possibly MARK_DONE
            # the row, undoing the quarantine.
            if not terminal_recorded and not quarantine_recorded:
                # Delta 2: split crash/finally into pre-launch (safe
                # MARK_DONE) vs post-launch (verify-or-quarantine).
                # Pre-launch: no worker ever started — mark FAILED.
                # Post-launch: worker may be alive — quarantine.
                if not worker_possibly_launched:
                    log.exception(
                        "MachineSlot crashed pre-launch: machine=%s gous=%s "
                        "run_id=%d; marking FAILED",
                        machine, unit.gous_uid, run_id,
                    )
                    # v5 Delta 11 wording: only MARK_DONE if we have a
                    # run_id allocated, and only mark_terminal if
                    # mark_inflight was actually called.  This avoids a
                    # spurious mark_terminal for an unregistered pair if
                    # the pre-launch crash happened between INSERT_QUEUED
                    # and mark_inflight (e.g. manifest write failure).
                    if run_id:
                        try:
                            # Pre-launch crash: the worker likely never
                            # started, but we did NOT prove it dead → untrusted.
                            _emit_mark_done_failed(
                                "MachineSlot crashed pre-launch",
                                terminal_source=TS_UNKNOWN,
                            )
                        except Exception:
                            log.exception(
                                "failed to emit MARK_DONE for run_id=%d",
                                run_id,
                            )
                        if mark_inflight_called:
                            try:
                                c.scheduler.mark_terminal(
                                    machine, unit.gous_uid, run_id,
                                    success=False,
                                )
                            except Exception:
                                log.exception(
                                    "failed to mark_terminal for machine=%s "
                                    "gous=%s run_id=%d",
                                    machine, unit.gous_uid, run_id,
                                )
                else:
                    log.exception(
                        "MachineSlot crashed POST-launch: machine=%s gous=%s "
                        "run_id=%d; worker may be alive",
                        machine, unit.gous_uid, run_id,
                    )
                    # If we have a wrapper_pid in this scope, try to verify+kill.
                    pid_for_kill: Optional[int] = wrapper_pid
                    if pid_for_kill is None and nas_unit_dir is not None:
                        pid_for_kill = _read_pidfile(nas_unit_dir)
                    if pid_for_kill is not None:
                        outcome, detail = verify_and_kill_worker(
                            machine, pid_for_kill, expected_tokens,
                            c.cfg.global_cfg,
                        )
                        if outcome != VERIFY_INCONCLUSIVE:
                            # Verified dead — safe to MARK_DONE FAILED.
                            try:
                                _emit_mark_done_failed(
                                    f"MachineSlot crashed post-launch; "
                                    f"cleanup={detail}",
                                    terminal_source=_source_from_verify(outcome),
                                )
                            except Exception:
                                log.exception(
                                    "failed to emit MARK_DONE for run_id=%d",
                                    run_id,
                                )
                            try:
                                c.scheduler.mark_terminal(
                                    machine, unit.gous_uid, run_id,
                                    success=False,
                                )
                            except Exception:
                                log.exception(
                                    "failed to mark_terminal for machine=%s "
                                    "gous=%s run_id=%d",
                                    machine, unit.gous_uid, run_id,
                                )
                            return
                    # Inconclusive → quarantine, leave row active.
                    _quarantine_and_log(
                        "post_launch_crash_unverified",
                        "MachineSlot crashed after launch_detached returned",
                    )

    def _handle_state_missing_timeout(
        self,
        nas_unit_dir: Path,
        machine: str,
        expected_tokens: list[str],
        wrapper_pid: Optional[int],
        on_phase_change,
        on_poll,
        run_id: int,
    ) -> Optional[dict]:
        """F2.b: handle the state_missing_timeout synthetic terminal.

        Returns the final terminal state if we have one to MARK_DONE,
        or None if we've already quarantined the host and the caller
        must return without MARK_DONE.
        """
        c = self.ctx
        g = c.cfg.global_cfg
        state_path = nas_unit_dir / "state.json"
        # Re-read state.json once.
        cur_state: dict = {}
        if state_path.exists():
            try:
                cur_state = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                cur_state = {}
        phase = cur_state.get("phase")
        if phase in ("publishing", "done", "running"):
            log.info(
                "dispatch: state_appeared_after_timeout machine=%s run_id=%d "
                "phase=%s — resuming polling within late_state_grace_sec=%d",
                machine, run_id, phase, g.late_state_grace_sec,
            )
            late_started = time.monotonic()
            # Bounded re-poll: build a tweaked GlobalCfg with the late
            # grace as the new state_appeared_timeout AND constrain the
            # total time spent here.
            while True:
                # Quick poll loop reusing poll_state_until_terminal.
                try:
                    late_final = poll_state_until_terminal(
                        machine, nas_unit_dir, g=g,
                        expected_tokens=expected_tokens,
                        on_phase_change=on_phase_change,
                        on_poll=on_poll,
                    )
                except Exception:
                    log.exception(
                        "late-state re-poll crashed run_id=%d", run_id,
                    )
                    late_final = None
                # Observed terminal → return it.
                if _is_observed_terminal(late_final):
                    return late_final
                # Another synthetic — check the grace budget.
                if (time.monotonic() - late_started) >= g.late_state_grace_sec:
                    log.warning(
                        "dispatch: late_state_grace_expired machine=%s "
                        "run_id=%d — quarantine",
                        machine, run_id,
                    )
                    # F2.h: grace expiry → quarantine, NEVER MARK_DONE.
                    return None
                # Otherwise loop and re-poll.
        # phase is None / starting / staging — proceed to verify+kill.
        outcome, detail = verify_and_kill_worker(
            machine, wrapper_pid, expected_tokens, g,
        )
        log.info(
            "dispatch: state_missing_timeout verify run_id=%d machine=%s "
            "outcome=%s detail=%s",
            run_id, machine, outcome, detail,
        )
        if outcome == VERIFY_INCONCLUSIVE:
            return None
        # Cleanup verified — synthesize a FAILED terminal so the normal
        # MARK_DONE flow proceeds with the typed error_message.  Carry the
        # verify outcome as the source (v5 D2).
        return {
            "phase": "failed",
            "success": False,
            "reason": "state_missing_timeout",
            "terminal_source": _source_from_verify(outcome),
            "error_message": (
                f"state.json never appeared within "
                f"{g.state_appeared_timeout_sec}s; cleanup={detail}"
            ),
        }


# ---------------------------------------------------------------------------
# Top-level dispatch context + entry point
# ---------------------------------------------------------------------------

class LiveCountWatchdog(threading.Thread):
    """Periodically log live thread counts so a wedge is visible in logs.

    Without this, a hung dispatcher just goes silent.  We emit a single
    INFO line on a fixed cadence with adoption / fixed-slot / dynamic-slot
    live counts so it is obvious whether the orchestrator is making
    progress.  No mutation, no force-cleanup — purely observational.
    """

    def __init__(
        self,
        ctx: "DispatchContext",
        adoption_threads: list,
        slots: list,
        interval_sec: int = 300,
        reapers: Optional[list] = None,
    ):
        super().__init__(name="watchdog", daemon=True)
        self.ctx = ctx
        self.adoption_threads = adoption_threads
        self.slots = slots
        self.interval_sec = int(interval_sec)
        self.reapers = reapers or []
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            try:
                live_a = sum(1 for t in self.adoption_threads if t.is_alive())
                live_s = sum(1 for s in self.slots if s.is_alive())
                with self.ctx.dynamic_slots_lock:
                    live_d = sum(1 for s in self.ctx.dynamic_slots if s.is_alive())
                with self.ctx.scheduler.lock:
                    q_n = len(self.ctx.scheduler.quarantined_machines)
                # Aggregate reaper counters across all reapers so the
                # previously-silent inconclusive-skip path is observable.
                r_db = r_ssh = r_skip = 0
                for r in self.reapers:
                    snap = r.counters.snapshot()
                    r_db += snap["tokens_reaped_db"]
                    r_ssh += snap["tokens_reaped_ssh"]
                    r_skip += snap["tokens_skipped_inconclusive"]
                log.info(
                    "watchdog: adoption=%d slots=%d dynamic=%d quarantined=%d "
                    "tokens_reaped_db=%d tokens_reaped_ssh=%d "
                    "tokens_skipped_inconclusive=%d",
                    live_a, live_s, live_d, q_n, r_db, r_ssh, r_skip,
                )
            except Exception:
                log.exception("watchdog tick failed")


class AdoptionPoller(threading.Thread):
    """Poll an adopted (already-running) worker until it terminates.

    Adopted workers were launched by a prior dispatch whose coordinator
    died; their state.json + heartbeat are still being updated on NAS.
    We resume polling, push the terminal event to the DB writer, and
    register the unit in the scheduler's seen_pairs/in_flight so the
    cleaner will sweep its /raid/ leftovers like any normal unit.
    """

    def __init__(
        self,
        adopted: dict,                      # dict from reconcile_prior
        ctx: "DispatchContext",
        machine_cfg: Optional[MachineCfg] = None,
    ):
        super().__init__(
            name=f"adopt-{adopted['run_id']}", daemon=True,
        )
        self.adopted = adopted
        self.ctx = ctx
        # ``machine_cfg`` is the new-dispatch MachineCfg for the host
        # this adopted unit runs on, used to size the successor-slot
        # spawn after termination.  Optional because reconciliation
        # may surface a machine that is not in the current cfg
        # (machines.json change between runs); in that case we still
        # poll the worker to terminal but skip successor spawning.
        self.machine_cfg = machine_cfg

    def run(self) -> None:
        c = self.ctx
        run_id = int(self.adopted["run_id"])
        machine = self.adopted["machine"]
        unit_dir = Path(self.adopted["unit_dir"])
        state = self.adopted["state"]
        gous_uid = state.get("gous_uid") or self._gous_from_db(run_id)
        prior_did = (
            self.adopted.get("prior_dispatch_id")
            or state.get("dispatch_id")
        )
        expected_tokens = self.adopted.get("expected_tokens") or [
            f"--dispatch-id {prior_did or ''}",
            f"--run-id {run_id}",
        ]

        if not gous_uid:
            log.warning(
                "adoption: run_id=%d machine=%s prior_dispatch=%s has no "
                "resolvable gous_uid; scheduler slot accounting and "
                "/raid cleanup will be skipped for this run",
                run_id, machine, prior_did,
            )

        # Account adopted unit in the scheduler so the cleaner reaches it.
        # Tag with the *prior* dispatch_id so cleanup hits the right
        # /raid/d_<prior_id>/... tree, NOT the new dispatch's.  Wrap
        # everything from here in try/finally so a failure during the
        # poll-setup (closures, timing, etc.) still releases the slot
        # we just reserved.
        if gous_uid:
            c.scheduler.mark_inflight(
                machine, gous_uid, run_id, dispatch_id=prior_did,
            )

        final: Optional[dict] = None
        crashed_exc: Optional[BaseException] = None
        quarantine_pending = False
        t0 = time.monotonic()
        try:
            last_db_hb_state = {"t": 0.0}
            hb_throttle = max(
                5.0, c.cfg.global_cfg.heartbeat_interval_sec / 2,
            )

            def _on_poll(state):
                now = time.monotonic()
                if now - last_db_hb_state["t"] < hb_throttle:
                    return
                last_db_hb_state["t"] = now
                c.db_writer.q.put({
                    "op": "HEARTBEAT", "run_id": run_id,
                    "ts": now_iso(),
                })

            try:
                final = poll_state_until_terminal(
                    machine, unit_dir,
                    g=c.cfg.global_cfg,
                    expected_tokens=expected_tokens,
                    on_poll=_on_poll,
                )
            except Exception as e:
                log.exception("adoption poller crashed for run_id=%d", run_id)
                crashed_exc = e

            # Delta 8: synthetic-terminal final → verify-or-quarantine
            # before MARK_DONE (mirror _dispatch_one's F2.b/F2.h flow).
            quarantine_pending = False
            if (
                final is not None
                and not _is_observed_terminal(final)
                and final.get("phase") == "failed"
            ):
                # Look up wrapper_pid from state.json's worker_pgid or
                # pidfile so we can run the verify+kill flow.
                pid_for_kill = (
                    final.get("worker_pgid")
                    or final.get("worker_pid")
                    or _read_pidfile(unit_dir)
                )
                outcome, detail = verify_and_kill_worker(
                    machine, pid_for_kill, expected_tokens,
                    c.cfg.global_cfg,
                )
                log.info(
                    "adoption: synthetic terminal verify run_id=%d "
                    "machine=%s reason=%s outcome=%s detail=%s",
                    run_id, machine, final.get("reason"), outcome, detail,
                )
                if outcome == VERIFY_INCONCLUSIVE:
                    quarantine_pending = True
                else:
                    final = {
                        **final,
                        # Coordinator-verified death (v4 map #6); carry source.
                        "terminal_source": _source_from_verify(outcome),
                        "error_message": (
                            (final.get("error_message") or "")
                            + f" — cleanup={detail}"
                        ),
                    }
        finally:
            elapsed = time.monotonic() - t0
            success = bool(final and final.get("success"))
            prov = (final or {}).get("provenance") or {}

            if quarantine_pending or crashed_exc is not None:
                # Adoption cleanup is unverified.  Quarantine the host
                # (Delta 8) and leave the run row ACTIVE so the next
                # dispatch's reconcile_prior can recover it.  Crashes
                # also fall through here because we can't confirm the
                # worker terminated.
                c.scheduler.quarantine(
                    machine, reason="adoption_cleanup_unverified",
                )
                log.warning(
                    "adoption: QUARANTINE machine=%s run_id=%d "
                    "(crashed=%s, quarantine_pending=%s) — leaving row "
                    "ACTIVE for next reconcile_prior",
                    machine, run_id, crashed_exc is not None,
                    quarantine_pending,
                )
            else:
                err_msg = (final or {}).get("error_message")
                status = (ImagingRunStatus.SUCCESS if success
                          else ImagingRunStatus.FAILED)
                log.info(
                    "adoption: MARK_DONE %s run_id=%d machine=%s "
                    "elapsed=%.1fs error=%s",
                    status, run_id, machine, elapsed, err_msg,
                )
                if not success:
                    log.warning(
                        "adoption: run_id=%d on %s failed: %s",
                        run_id, machine, err_msg,
                    )
                c.db_writer.q.put({
                    "op": "MARK_DONE",
                    "run_id": run_id,
                    "status": status,
                    "retcode": 0 if success else 1,
                    "finished_at": (final or {}).get("finished_at") or now_iso(),
                    "duration_sec": float(elapsed),
                    "output_fits": (final or {}).get("output_fits"),
                    "spw_selection": (json.dumps((final or {})["spw_selection"])
                                      if (final or {}).get("spw_selection") else None),
                    "field_selection": (json.dumps((final or {})["field_selection"])
                                        if (final or {}).get("field_selection") else None),
                    "error_message": err_msg,
                    "job_json_path": prov.get("job_json"),
                    # Observed terminal → worker_observed (carried by poll);
                    # synthetic-verified → mapped above; default unknown.
                    "terminal_source": (final or {}).get(
                        "terminal_source", TS_UNKNOWN,
                    ),
                })
                if gous_uid:
                    c.scheduler.mark_terminal(
                        machine, gous_uid, run_id, success=success,
                    )
            # Successor-slot spawn: take the now-freed slot capacity
            # and dispatch a fresh MachineSlot to consume queued units
            # on this machine.  Capped by the monotonic
            # ``successors_spawned`` counter so we never exceed the
            # machine's slot count, regardless of how many adoptions
            # terminate on the same host.
            #
            # F4 + Delta 8: only spawn if the terminal was OBSERVED (not
            # synthesized by our timeout, not on a quarantined host, not
            # on poll crash).  Otherwise the worker may still be alive
            # and a successor would oversubscribe.
            m_cfg = self.machine_cfg
            to_start: Optional[MachineSlot] = None
            if (
                m_cfg is not None
                and _is_observed_terminal(final)
                and not c.scheduler.is_quarantined(machine)
            ):
                with c.dynamic_slots_lock:
                    if not c.new_launch_machines_ready.is_set():
                        # Preflight hasn't run yet — queue for drain.
                        c.pending_successors[m_cfg.name] = (
                            c.pending_successors.get(m_cfg.name, 0) + 1
                        )
                    elif m_cfg.name in c.new_launch_machines_names:
                        spawned = c.successors_spawned.get(m_cfg.name, 0)
                        if spawned < m_cfg.slots:
                            c.successors_spawned[m_cfg.name] = spawned + 1
                            slot = MachineSlot(
                                f"{m_cfg.name}#post-adopt-{run_id}",
                                m_cfg, c,
                            )
                            c.dynamic_slots.append(slot)
                            to_start = slot
            if to_start is not None:
                try:
                    to_start.start()
                except Exception:
                    log.exception(
                        "failed to start successor MachineSlot for %s "
                        "(run_id=%d); rolling back counter",
                        m_cfg.name if m_cfg else "?", run_id,
                    )
                    with c.dynamic_slots_lock:
                        c.successors_spawned[m_cfg.name] = max(
                            0, c.successors_spawned.get(m_cfg.name, 0) - 1,
                        )
                        try:
                            c.dynamic_slots.remove(to_start)
                        except ValueError:
                            pass

    def _gous_from_db(self, run_id: int) -> Optional[str]:
        try:
            with self.ctx.db_manager.connect() as con:
                row = ImagingRunsQueries.get_by_id(con, run_id)
            return row.get("gous_uid") if row else None
        except Exception:
            return None


@dataclass
class DispatchContext:
    cfg: MachinesConfig
    dispatch_id: str
    dispatch_dir: Path                    # <base-dir>/imaging/dispatch/<dispatch_id>/
    publish_dir: Path
    tokens_dir: Path
    db_writer: DBWriter
    db_manager: DatabaseManager
    scheduler: SchedulerState
    transfer_method: str
    publish_policy: str
    deconvolver: str
    scales: list[int]
    gous_inputs: dict[str, list[dict]]
    # Successor-slot machinery (see AdoptionPoller / preflight drain).
    # When an adopted unit completes, the AdoptionPoller spawns one
    # replacement MachineSlot so the freed slot capacity is used.  All
    # mutation happens under ``dynamic_slots_lock``.
    dynamic_slots: list = field(default_factory=list)
    dynamic_slots_lock: threading.Lock = field(default_factory=threading.Lock)
    # ``pending_successors[machine_name] = count`` records adoption
    # finishes that happened *before* preflight completed.  The drain
    # block after preflight consumes these.
    pending_successors: dict[str, int] = field(default_factory=dict)
    # Monotonic per-machine cap counter: total successor MachineSlots
    # ever spawned for this machine.  Used as the cap (never decremented)
    # so increment-under-lock is sufficient — no is_alive() race.
    successors_spawned: dict[str, int] = field(default_factory=dict)
    # Set once preflight has populated ``new_launch_machines_names``.
    # Adoption pollers that finish before this point queue themselves
    # in ``pending_successors`` instead of spawning immediately.
    new_launch_machines_ready: threading.Event = field(default_factory=threading.Event)
    new_launch_machines_names: set[str] = field(default_factory=set)


_du_cache: dict[str, int] = {}


def _du_bytes(path: str) -> int:
    """Cached ``du -sb`` for *path*.  Returns 0 on error."""
    if path in _du_cache:
        return _du_cache[path]
    try:
        proc = subprocess.run(
            ["du", "-sb", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120,
        )
        if proc.returncode == 0 and proc.stdout:
            n = int(proc.stdout.split()[0])
        else:
            n = 0
    except (subprocess.SubprocessError, ValueError):
        n = 0
    _du_cache[path] = n
    return n


def _estimate_max_gous_gb(by_gous: dict[str, list[ImagingUnit]]) -> float:
    """Return the largest single-GOUS staged-input size in GB."""
    if not by_gous:
        return 0.0
    sizes_gb: list[float] = []
    for units in by_gous.values():
        seen: set[str] = set()
        total = 0
        for u in units:
            for p in list(u.vis_tm) + list(u.vis_sm):
                if p and p not in seen:
                    seen.add(p)
                    total += _du_bytes(p)
            if u.sdimage and u.sdimage not in seen:
                seen.add(u.sdimage)
                try:
                    total += os.path.getsize(u.sdimage)
                except OSError:
                    pass
        sizes_gb.append(total / 1e9)
    return max(sizes_gb) if sizes_gb else 0.0


def _git_meta(repo_path: Path) -> tuple[Optional[str], bool]:
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5,
        )
        commit = head.stdout.strip() if head.returncode == 0 else None
        dirty = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5,
        )
        is_dirty = bool(dirty.stdout.strip())
        return commit, is_dirty
    except (subprocess.SubprocessError, FileNotFoundError):
        return None, False


def run_dispatch(
    *,
    base_dir: Path,
    publish_dir: Path,
    cfg: MachinesConfig,
    db_manager: DatabaseManager,
    selection_filters: SelectionFilters,
    obs_csv: Path,
    data_dir: Path,
    transfer_method: str = "tar",
    publish_policy: str = "fail_if_exists",
    deconvolver: str = "multiscale",
    scales: Optional[list[int]] = None,
    cli_args: str = "",
    abandon_prior: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run one full dispatch.  Returns a summary dict."""
    if scales is None:
        scales = [0, 5, 10, 15, 20]

    g = cfg.global_cfg

    # 1. Dispatcher lock
    lock = DispatcherLock(base_dir / "imaging" / "dispatch" / ".dispatcher.lock")
    lock.acquire()
    # ``dispatch_log_handler`` is created later (after ``dispatch_dir`` is
    # known) and removed in the outer ``finally`` so we never leak a
    # handler back into the root logger for the next invocation.
    dispatch_log_handler: Optional[logging.Handler] = None
    try:
        # 2. Reconciliation — runs BEFORE selection so terminal results
        # discovered here are visible to success_exists() and the
        # active-row filter the selection uses.  Skipped under dry-run
        # because reconcile_prior() mutates the DB (terminal applies,
        # mark-failed for orphaned RUNNING rows, etc.); a read-only
        # dry-run preview must not have side effects.
        if dry_run:
            log.info("dry-run: skipping reconciliation (DB-mutating)")
            adoptable = []
        else:
            log.info("reconciling prior dispatches")
            adoptable = reconcile_prior(
                db_manager, base_dir, g, abandon=abandon_prior,
            )
            if adoptable:
                log.info(
                    "adopting %d in-flight unit(s) from prior dispatches",
                    len(adoptable),
                )

        # 2b. Selection — uses fresh DB state from reconciliation.
        log.info("selecting units to dispatch")
        selection = select_units_to_image(
            db_manager, obs_csv, data_dir, selection_filters,
        )
        log.info(
            "selection: ready=%d not_ready=%d skipped_done=%d "
            "skipped_active=%d skipped_filter=%d",
            len(selection.ready), len(selection.not_ready),
            selection.skipped_already_done, selection.skipped_active,
            selection.skipped_filtered_out,
        )

        if dry_run:
            return {
                "dispatch_id": None,
                "to_run": len(selection.ready),
                "skipped_already_done": selection.skipped_already_done,
                "skipped_active": selection.skipped_active,
                "skipped_filtered_out": selection.skipped_filtered_out,
                "adopted": len(adoptable or []),
                "machines": list(cfg.machines.keys()),
                "dry_run": True,
            }

        # 3. New dispatch identity + dir
        dispatch_id = (
            f"d_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        )
        dispatch_dir = base_dir / "imaging" / "dispatch" / dispatch_id
        dispatch_dir.mkdir(parents=True, exist_ok=True)
        tokens_dir = dispatch_dir / "staging_tokens"
        tokens_dir.mkdir(exist_ok=True)

        # 3b. Per-dispatch file log handler.  Captures everything the
        # orchestrator logs for the rest of this run into
        # ``<dispatch_dir>/dispatch.log`` so future operators can
        # diagnose hangs/wedges without relying on the launch-time
        # stdout redirect surviving.  Removed in the outer ``finally``.
        dispatch_log_handler = logging.FileHandler(
            dispatch_dir / "dispatch.log"
        )
        dispatch_log_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logging.getLogger().addHandler(dispatch_log_handler)

        # 4. Dispatches row
        commit, is_dirty = _git_meta(Path(cfg.repo_path))
        with db_manager.connect() as con:
            DispatchesQueries.insert(
                con,
                dispatch_id=dispatch_id,
                coordinator_host=socket.gethostname(),
                coordinator_pid=os.getpid(),
                git_commit=commit,
                git_dirty=is_dirty,
                machines_json=json.dumps({
                    name: {"raid": m.raid, "slots": m.slots, "nproc": m.nproc}
                    for name, m in cfg.machines.items()
                }),
                cli_args=cli_args,
            )
            con.commit()

        # 5. Group by GOUS, compute union of inputs
        ready = selection.ready
        if not ready and not adoptable:
            log.info("nothing to dispatch (no ready units, no adoptions)")
            with db_manager.connect() as con:
                DispatchesQueries.mark_terminal(con, dispatch_id, DispatchState.DONE)
                con.commit()
            return {
                "dispatch_id": dispatch_id,
                "to_run": 0,
                "skipped_already_done": selection.skipped_already_done,
                "skipped_active": selection.skipped_active,
                "adopted": 0,
            }

        by_gous: dict[str, list[ImagingUnit]] = {}
        for u in ready:
            by_gous.setdefault(u.gous_uid, []).append(u)
        gous_inputs = {g_uid: union_inputs_for_gous(us)
                       for g_uid, us in by_gous.items()}

        # 6. Spawn DB writer + scheduler.  These exist for both adoption
        # pollers (which have running workers from prior dispatches) AND
        # new slot threads, so they must come up BEFORE preflight.
        db_writer = DBWriter(db_manager, dispatch_id)
        db_writer.start()

        scheduler = SchedulerState(queue=list(ready))
        # Delta 5: seed quarantined hosts surfaced by reconcile_prior
        # (e.g. prior-dispatch run with no state/no pidfile, or
        # inconclusive verify+kill).  Must happen BEFORE preflight so
        # quarantined hosts are skipped end-to-end.
        if isinstance(adoptable, ReconcileResult):
            qhosts = adoptable.quarantined_hosts
        else:
            qhosts = set()
        if qhosts:
            scheduler.quarantine_all(qhosts)
            log.warning(
                "dispatch: seeded %d quarantined host(s) from reconcile: %s",
                len(qhosts), sorted(qhosts),
            )
        ctx = DispatchContext(
            cfg=cfg, dispatch_id=dispatch_id, dispatch_dir=dispatch_dir,
            publish_dir=publish_dir, tokens_dir=tokens_dir,
            db_writer=db_writer, db_manager=db_manager, scheduler=scheduler,
            transfer_method=transfer_method, publish_policy=publish_policy,
            deconvolver=deconvolver, scales=scales,
            gous_inputs=gous_inputs,
        )

        # 7. Cleaner — runs regardless of preflight outcome since adopted
        # workers also need their (machine, GOUS) directories cleaned up.
        cleaner = GousCleaner(scheduler, cfg, dispatch_id, g)
        cleaner.start()

        # 8. Token reapers — one for the new dispatch, plus one for each
        # *prior* dispatch we adopted from.  Without the prior reapers,
        # tokens held by crashed workers in old dispatches stay forever
        # and adopted live workers may starve waiting for them.
        # Hostname-resolver snapshot for the current dispatch: the keys are
        # exactly machines.json's short names, the form with valid host keys.
        cur_machines_snapshot = {
            name: {"raid": m.raid} for name, m in cfg.machines.items()
        }
        reapers: list[TokenReaper] = []
        new_reaper = TokenReaper(
            tokens_dir, g,
            expected_tokens=[f"--dispatch-id {dispatch_id}"],
            db_manager=db_manager,
            machines=cur_machines_snapshot,
        )
        new_reaper.start()
        reapers.append(new_reaper)

        prior_dispatch_ids = sorted({
            a.get("prior_dispatch_id")
            for a in (adoptable or [])
            if a.get("prior_dispatch_id")
        })
        for prior_did in prior_dispatch_ids:
            prior_tokens = (
                base_dir / "imaging" / "dispatch" / prior_did / "staging_tokens"
            )
            if not prior_tokens.exists():
                continue
            # Resolve hostnames against THAT dispatch's machines_json
            # snapshot, not the current cfg (keys can differ between runs).
            prior_machines = None
            try:
                with db_manager.connect() as con:
                    prow = DispatchesQueries.get(con, prior_did)
                if prow and prow.get("machines_json"):
                    prior_machines = json.loads(prow["machines_json"])
            except Exception:
                log.debug(
                    "could not load machines_json for prior dispatch %s",
                    prior_did, exc_info=True,
                )
            r = TokenReaper(
                prior_tokens, g,
                expected_tokens=[f"--dispatch-id {prior_did}"],
                db_manager=db_manager,
                machines=prior_machines,
            )
            r.start()
            reapers.append(r)
            log.info("watching prior dispatch staging_tokens: %s", prior_tokens)

        # 9. Adoption pollers — start BEFORE machine preflight so a
        # totally-unreachable cluster (or simply a different machine set
        # in the new run) does not block adopted workers from being
        # finalised in the DB.
        adopt_per_machine: dict[str, int] = {}
        adoption_threads: list[AdoptionPoller] = []
        for a in (adoptable or []):
            m_name = a["machine"]
            adopt_per_machine[m_name] = adopt_per_machine.get(m_name, 0) + 1
            # Pass the current-cfg MachineCfg if the adopted unit's host
            # is still in machines.json; otherwise None disables the
            # successor-spawn path for this poller (the unit still runs
            # to terminal under polling).
            t = AdoptionPoller(a, ctx, machine_cfg=cfg.machines.get(m_name))
            t.start()
            adoption_threads.append(t)

        # 10. Machine preflight — only gates NEW launches.  Adopted units
        # already running on prior-dispatch /raid/ paths don't care
        # whether the new machine config preflights cleanly.  We keep
        # ``cfg.machines`` as the full known set so the cleaner can
        # reach machines that host adopted-only units; preflight just
        # produces a separate ``new_launch_machines`` view.
        new_launch_machines: dict[str, MachineCfg] = {}
        if ready:
            max_gous_gb = _estimate_max_gous_gb(by_gous)
            nas_probe = str(publish_dir.parent if publish_dir.parent.exists()
                            else publish_dir)
            for name, m in cfg.machines.items():
                # Delta 5: a host quarantined by reconcile_prior never
                # gets a preflight or a slot.
                if scheduler.is_quarantined(name):
                    log.warning(
                        "skipping preflight for %s: quarantined by "
                        "reconcile_prior",
                        name,
                    )
                    continue
                required_gb = max_gous_gb + 50 + (m.slots * max_gous_gb)
                ok, details = ssh_preflight_machine(
                    name, m, cfg,
                    required_gb=required_gb,
                    nas_check_path=nas_probe,
                    timeout=g.ssh_timeout_sec,
                )
                if ok:
                    new_launch_machines[name] = m
                    log.info("preflight ok: %s (free=%s GB)",
                             name, details.get("free_gb"))
                else:
                    log.warning(
                        "excluding %s from new launches: %s",
                        name, details.get("error") or details,
                    )
            if not new_launch_machines:
                log.error(
                    "no machines passed preflight; "
                    "skipping new launches but adoptions will still complete"
                )

        # 10b. Drain any successor-spawn requests that adoption pollers
        # queued while preflight was running, then mark preflight done
        # so subsequent pollers spawn successors directly.  All under
        # ``dynamic_slots_lock`` so an in-flight poller cannot race the
        # cap (monotonic ``successors_spawned`` counter is incremented
        # under the same lock).
        drained_successor_slots: list[MachineSlot] = []
        with ctx.dynamic_slots_lock:
            ctx.new_launch_machines_names = set(new_launch_machines.keys())
            for m_name, count in list(ctx.pending_successors.items()):
                if m_name in ctx.new_launch_machines_names:
                    m_cfg = cfg.machines[m_name]
                    spawned = ctx.successors_spawned.get(m_name, 0)
                    remaining_cap = max(0, m_cfg.slots - spawned)
                    n = min(count, remaining_cap)
                    for i in range(n):
                        ctx.successors_spawned[m_name] = (
                            ctx.successors_spawned.get(m_name, 0) + 1
                        )
                        slot = MachineSlot(
                            f"{m_name}#post-adopt-drain-{i}", m_cfg, ctx,
                        )
                        ctx.dynamic_slots.append(slot)
                        drained_successor_slots.append(slot)
                ctx.pending_successors.pop(m_name, None)
            ctx.new_launch_machines_ready.set()
        # Start drained successors OUTSIDE the lock.  See comment in
        # AdoptionPoller.run() for the rationale.
        for slot in drained_successor_slots:
            try:
                slot.start()
            except Exception:
                log.exception(
                    "failed to start drained successor MachineSlot %s",
                    slot.name,
                )
                with ctx.dynamic_slots_lock:
                    name = slot.machine.name
                    ctx.successors_spawned[name] = max(
                        0, ctx.successors_spawned.get(name, 0) - 1,
                    )
                    try:
                        ctx.dynamic_slots.remove(slot)
                    except ValueError:
                        pass

        # 11. Slots — only on machines that passed preflight; account for
        # adopted units already running on each.
        slots: list[MachineSlot] = []
        for m in new_launch_machines.values():
            adopted_n = adopt_per_machine.get(m.name, 0)
            available = max(0, m.slots - adopted_n)
            if adopted_n:
                log.info(
                    "machine %s: %d adopted unit(s); spawning %d new slot(s)",
                    m.name, adopted_n, available,
                )
            for i in range(available):
                slot = MachineSlot(f"{m.name}#{i}", m, ctx)
                slot.start()
                slots.append(slot)

        # 11b. Observability watchdog — periodic live-counts so a wedge
        # is visible in the dispatch log rather than silent.
        watchdog = LiveCountWatchdog(ctx, adoption_threads, slots, reapers=reapers)
        watchdog.start()

        # 12. Wait for original slots, then alternate-check adoption
        # pollers and dynamic (successor) slots until both quiet.  Order
        # matters: check adoption FIRST each iteration so a successor
        # that an adoption thread is about to append cannot be missed.
        # Adoption threads are the only producers of ``dynamic_slots``;
        # once they are all dead, no new dynamic slots can appear, so
        # the post-adoption snapshot is final.
        for s in slots:
            s.join()
        while True:
            any_adoption_alive = any(
                t.is_alive() for t in adoption_threads
            )
            if any_adoption_alive:
                # Brief join to make progress without burning CPU; we
                # re-enter the loop to pick up any successor slots an
                # adoption thread appended in the interim.
                for t in adoption_threads:
                    t.join(timeout=2.0)
                continue
            # No adoption alive → no new dynamics can be added.  Snapshot
            # under lock and join whatever is there.
            with ctx.dynamic_slots_lock:
                snap = list(ctx.dynamic_slots)
            any_dynamic_alive = any(s.is_alive() for s in snap)
            if not any_dynamic_alive:
                break
            for s in snap:
                s.join(timeout=2.0)

        # 13. Drain the DB writer queue BEFORE the final cleanup sweep so
        # MARK_DONE events have committed before the cleaner queries
        # SUCCESS rows.  q.join() returns when every put() has been
        # task_done()'d by the writer.
        db_writer.q.join()

        # 14. Final cleanup sweep (idempotent rm -rf)
        cleaner.force_run()

        # 15. Stop background threads
        for r in reapers:
            r.stop()
        for r in reapers:
            r.join(timeout=5)
        cleaner.stop(); cleaner.join(timeout=5)
        watchdog.stop(); watchdog.join(timeout=5)

        # 16. End-of-run /raid/ sweep — backstop for paths the periodic
        # cleaner might have missed.  Iterate the union of:
        #   - new dispatch on each new-launch machine
        #   - per-pair (machine, dispatch_id) recorded by the scheduler
        # so adopted prior-dispatch /raid/ trees get cleaned too.
        sweep_targets: set[tuple[str, str]] = set()
        for m_name in new_launch_machines:
            sweep_targets.add((m_name, dispatch_id))
        with scheduler.lock:
            for (pair_machine, _gous), pair_did in scheduler.pair_dispatch_id.items():
                if pair_did:
                    sweep_targets.add((pair_machine, pair_did))
        for m_name, did in sorted(sweep_targets):
            m = cfg.machines.get(m_name)
            if m is None:
                continue
            # v3 F2.d: never rm -rf on a quarantined host (a worker may
            # still be alive and reading inputs).
            if scheduler.is_quarantined(m_name):
                log.info(
                    "end-of-run sweep: skipping %s/%s (quarantined)",
                    m_name, did,
                )
                continue
            try:
                ssh_run(
                    m_name,
                    f"rm -rf -- {shlex.quote(f'{m.raid}/d_{did}/input')}",
                    timeout=30,
                )
            except subprocess.SubprocessError as exc:
                log.warning(
                    "end-of-run sweep failed on %s/%s: %s",
                    m_name, did, exc,
                )

        # 17. Stop DB writer + mark dispatch(es) done.
        db_writer.stop()
        db_writer.join(timeout=10)
        # v5 Delta 11: if any rows for THIS dispatch remain non-terminal
        # (e.g. left active by a quarantine), DO NOT mark the dispatch
        # DONE — otherwise reconcile_prior scans only RUNNING dispatches
        # and the orphan row becomes invisible.
        with db_manager.connect() as con:
            current_active = ImagingRunsQueries.list_running_for_dispatch(
                con, dispatch_id,
            )
        if current_active:
            unresolved_ids = sorted(r["id"] for r in current_active)
            with scheduler.lock:
                qhosts = sorted(scheduler.quarantined_machines)
            log.warning(
                "dispatch %s left RUNNING: %d unresolved run(s) %s, "
                "quarantined hosts: %s — next dispatch's reconcile_prior "
                "will recover",
                dispatch_id, len(unresolved_ids), unresolved_ids, qhosts,
            )
        else:
            with db_manager.connect() as con:
                DispatchesQueries.mark_terminal(
                    con, dispatch_id, DispatchState.DONE,
                )
                con.commit()
        with db_manager.connect() as con:
            # Prior dispatches: mark DONE if they have no surviving
            # non-terminal rows now that adoptions have completed.
            for prior_did in prior_dispatch_ids:
                still_active = ImagingRunsQueries.list_running_for_dispatch(
                    con, prior_did,
                )
                if not still_active:
                    DispatchesQueries.mark_terminal(
                        con, prior_did, DispatchState.DONE,
                    )
            con.commit()

        return {
            "dispatch_id": dispatch_id,
            "to_run": len(ready),
            "skipped_already_done": selection.skipped_already_done,
            "skipped_active": selection.skipped_active,
            "adopted": len(adoption_threads),
            "machines_used": list(new_launch_machines.keys()),
        }
    finally:
        if dispatch_log_handler is not None:
            try:
                logging.getLogger().removeHandler(dispatch_log_handler)
                dispatch_log_handler.close()
            except Exception:
                log.exception("failed to detach per-dispatch log handler")
        lock.release()
