"""Distributed-imaging staging primitives.

Pure helpers — no DB, no SSH, no logging side-effects beyond ``log``.
Used by :mod:`panta_rei.imaging.remote_worker` (workers) and
:mod:`panta_rei.imaging.dispatch` (coordinator's stale-token reaper).

Key primitives:

- ``stage_one(src, dst_root, *, method, bucket)`` — copy a single MS or TP
  product from NAS to /raid/ via tar/rsync/cp into a ``.partial`` temp dir,
  then ``os.rename`` to its final location.  The temp+rename pattern keeps
  the destination from ever being half-populated.
- ``acquire_staging_token`` / ``release_staging_token`` — NAS-based
  global concurrency limit, atomic via ``os.mkdir``.  Detached workers
  cannot rely on the coordinator to gate them, so we use the filesystem.
- ``acquire_stage_lock`` / ``release_stage_lock`` — per-(machine, GOUS)
  staging mutex, also via atomic ``os.mkdir`` rather than NFS flock
  (which is historically flaky).
- ``read_manifest`` / ``write_manifest`` — atomic JSON read/write for the
  ``.staging.manifest`` file that records which inputs of the GOUS union
  have been staged so far.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import logging
import os
import random
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed exceptions and module-level constants for bounded waits
# ---------------------------------------------------------------------------

class StaleLockTimeout(Exception):
    """``_MkdirLock`` could not acquire within ``wait_timeout_sec``.

    Deliberately NOT a ``TimeoutError`` (which is itself an ``OSError``)
    so that callers' existing ``except OSError`` clauses for filesystem
    races do not silently swallow the bounded-wait signal.
    """


class TokenAcquireTimeout(TimeoutError):
    """``acquire_staging_token`` could not acquire a slot in time.

    IS-A ``TimeoutError`` to preserve the existing ``acquire_staging_token``
    contract (callers and tests expect ``TimeoutError``).  A dedicated
    subclass lets ``stage_one``'s populate block discriminate this
    unit-fatal case from generic filesystem ``ETIMEDOUT``.
    """


_MKDIR_LOCK_DEFAULT_WAIT_SEC = 300.0
_DEFAULT_TOKEN_WAIT_TIMEOUT_SEC = 1800.0
_MKDIR_LOCK_PROGRESS_LOG_SEC = 60.0


def _read_proc_starttime(pid: int) -> Optional[int]:
    """Return field 22 of ``/proc/<pid>/stat`` (starttime in clock ticks).

    Returns ``None`` if the PID is dead or ``/proc`` is unreadable.
    Used to disambiguate PID reuse: the start-time is stable for the
    lifetime of a process and differs when the kernel hands the same
    numeric PID to a fresh process.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
        rparen = data.rfind(b")")
        if rparen < 0:
            return None
        fields = data[rparen + 2:].split()
        if len(fields) < 20:
            return None
        return int(fields[19])
    except (OSError, ValueError):
        return None


def _same_host(stored: str) -> bool:
    """True iff *stored* names the current host.

    Compares against ``socket.gethostname()`` and ``socket.getfqdn()`` in
    both short and FQDN forms.  Case-insensitive per RFC 952/1123.  Two
    truly-different hosts that share a short name would false-positive
    here; on this fleet hostnames are unique on the short form.
    """
    if not stored:
        return False
    stored_low = stored.strip().lower()
    own: set[str] = set()
    for name in (socket.gethostname(), socket.getfqdn()):
        name_low = (name or "").strip().lower()
        if not name_low:
            continue
        own.add(name_low)
        own.add(name_low.split(".", 1)[0])
    return stored_low in own or stored_low.split(".", 1)[0] in own


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, payload: dict) -> None:
    """Write *payload* as JSON to *path* via temp file + ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{random.randint(0, 1<<30)}")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(str(tmp), str(path))


def read_manifest(path: Path) -> dict:
    """Return the staging manifest dict, or a fresh skeleton if missing."""
    if not Path(path).exists():
        return {"version": 1, "expected": [], "completed": []}
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Atomic mkdir-based mutex (used for staging lock and tokens)
# ---------------------------------------------------------------------------

@dataclass
class _MkdirLock:
    """A simple lock implemented via atomic directory creation.

    NFS guarantees ``mkdir`` atomicity at the server side, which is the
    primitive we rely on.  If a previous holder crashed, the lock dir
    persists indefinitely — so on every contended attempt we read the
    holder's metadata and self-heal: if the holder is on the same host
    and its PID is dead (including same-numeric-PID-but-fresh-process
    via start-time disambiguation), we ``rmtree`` and retry.  This
    bounds recovery to a single host's /proc check (no SSH from the
    worker).

    Cross-host stale lock-dir recovery is NOT implemented today.  A
    stale lock-dir from a remote worker surfaces on the local worker
    as ``StaleLockTimeout`` after ``wait_timeout_sec``; callers then
    fall back (cache locks) or fail the unit (stage locks).  A
    coordinator-side reaper analogous to ``TokenReaper`` is a documented
    follow-up.

    There is a small live-lock race between ``mkdir`` and the
    ``holder.json`` write: a contender that observes the directory
    after mkdir but before the metadata is in place would otherwise
    declare the live lock stale.  We address this by writing
    ``holder.json`` atomically (temp + rename) and by giving any
    metadata-missing observation a short grace window before reclaiming.
    """
    dir_path: Path
    holder_meta: dict = field(default_factory=dict)
    acquired: bool = False
    # Grace window for a freshly-mkdir'd lock to publish its metadata.
    missing_metadata_grace_sec: float = 3.0
    # Bounded wait — None disables (test/legacy convenience).  Production
    # defaults to _MKDIR_LOCK_DEFAULT_WAIT_SEC.
    wait_timeout_sec: Optional[float] = _MKDIR_LOCK_DEFAULT_WAIT_SEC
    # Poll sleep range (uniform random); tests can pass tight values.
    poll_sleep: tuple[float, float] = (0.5, 1.5)

    def __enter__(self) -> "_MkdirLock":
        started = time.monotonic()
        last_progress_log_at = started
        deadline: Optional[float] = (
            started + self.wait_timeout_sec
            if self.wait_timeout_sec is not None else None
        )
        while True:
            # Bound EVERY iteration, not just the alive-True branch.
            # Reclaim-loop pathological cases (filesystem replaying the
            # stale lock, repeated metadata-missing observations) must
            # also surface as StaleLockTimeout rather than spin forever.
            now = time.monotonic()
            if deadline is not None and now > deadline:
                holder_meta = _read_holder_meta(self.dir_path)
                raise StaleLockTimeout(
                    f"could not acquire {self.dir_path} in "
                    f"{self.wait_timeout_sec:.0f}s "
                    f"(holder={holder_meta})"
                )
            try:
                self.dir_path.mkdir(parents=True, exist_ok=False)
                self.acquired = True
                # Identity fields LAST so a caller-supplied holder_meta
                # cannot override host/pid/starttime_ticks.
                meta = {
                    **(self.holder_meta or {}),
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "starttime_ticks": _read_proc_starttime(os.getpid()),
                }
                # Atomic publish of holder metadata so contenders never
                # see a half-written file.
                tmp = self.dir_path / ".holder.json.tmp"
                tmp.write_text(json.dumps(meta))
                os.replace(str(tmp), str(self.dir_path / "holder.json"))
                return self
            except FileExistsError:
                # Snapshot the holder fingerprint BEFORE the alive check
                # so the post-rename race-detection compares against the
                # same generation we judged stale.  holder.json already
                # encodes pid + starttime_ticks, so PID-reuse cannot
                # produce a false-positive match.
                fp_snapshot = _read_blob(self.dir_path / "holder.json")
                if not _holder_alive_locally(
                    self.dir_path,
                    missing_metadata_grace_sec=self.missing_metadata_grace_sec,
                ):
                    log.warning(
                        "stale stage lock detected at %s; reclaiming",
                        self.dir_path,
                    )
                    _atomic_reclaim(
                        self.dir_path,
                        expected_fingerprint=fp_snapshot,
                        read_fingerprint=lambda d: _read_blob(d / "holder.json"),
                        malformed_grace_sec=self.missing_metadata_grace_sec,
                    )
                    continue
                if now - last_progress_log_at >= _MKDIR_LOCK_PROGRESS_LOG_SEC:
                    holder_meta = _read_holder_meta(self.dir_path)
                    log.warning(
                        "still waiting for lock at %s; holder=%s waited=%.0fs",
                        self.dir_path, holder_meta, now - started,
                    )
                    last_progress_log_at = now
                time.sleep(random.uniform(*self.poll_sleep))

    def __exit__(self, *exc_info) -> None:
        if self.acquired:
            shutil.rmtree(self.dir_path, ignore_errors=True)


def _read_holder_meta(lock_dir: Path) -> Optional[dict]:
    """Best-effort read of ``holder.json`` for diagnostics; never raises."""
    try:
        return json.loads((Path(lock_dir) / "holder.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_reclaim(
    stale_dir: Path,
    *,
    expected_fingerprint: Optional[bytes],
    read_fingerprint,
    malformed_grace_sec: float = 3.0,
) -> bool:
    """Atomically reclaim *stale_dir* IFF its generation matches.

    Returns ``True`` iff we removed the generation we observed as
    stale.  Returns ``False`` if another reclaimer beat us, the
    directory vanished, or the post-rename generation does not match
    (in which case we restore it).

    The race-closing story:
    1. ``os.rename`` is atomic, so exactly one contender can move
       ``stale_dir`` for a given observation.  Losers see
       ``FileNotFoundError`` and return ``False``.
    2. After rename, the winner reads the post-rename fingerprint via
       *read_fingerprint*.  If it matches *expected_fingerprint*, the
       moved directory IS the stale generation — safe to ``rmtree``.
    3. If it does not match, a fresh holder won the lock between our
       fingerprint snapshot and our rename.  We reverse-rename to put
       the fresh holder back.  If the reverse-rename fails (a third
       contender ``mkdir``'d at ``stale_dir`` in the gap), we DO NOT
       ``rmtree`` the renamed copy — that would silently delete a
       valid holder while a different one owns the original path.
       The stranded ``.reclaim.<uuid>`` directory is logged loudly
       for operator follow-up; data integrity is preserved at the
       cost of a leaked directory.

    Fingerprint design:
    - Callers snapshot the bytes BEFORE the liveness check.
    - The fingerprint MUST be strong enough to disambiguate two
      different live processes (PID reuse: same numeric PID but
      different start-time).  For lock-dirs, ``holder.json`` already
      encodes ``pid`` + ``starttime_ticks``.  For tokens, the
      fingerprint reader concatenates ``pid`` and ``starttime_ticks``
      file contents.
    - When the stale generation legitimately has no fingerprint
      (mkdir won but holder published nothing — the malformed window),
      callers pass ``expected_fingerprint=None``.  In that case we
      fall back to an mtime-age check: only rmtree if the directory
      is older than *malformed_grace_sec* (mirrors
      ``_holder_alive_locally``'s grace logic) — a freshly-mkdir'd
      dir under us would have age < grace and be put back.

    Residual race: a fresh holder that mkdir's, writes the
    fingerprint, AND has the same fingerprint bytes as the stale gen
    (e.g., same pid+starttime — astronomically unlikely with
    start-time check) — would still match and be rmtree'd.  This
    is a microsecond-scale window we accept as a defense-in-depth
    backstop; the primary protection remains the bounded
    ``wait_timeout_sec`` plus the dispatch-side ``TokenReaper`` for
    tokens.
    """
    reclaim_path = stale_dir.with_name(
        stale_dir.name + f".reclaim.{uuid.uuid4().hex}"
    )
    try:
        os.rename(str(stale_dir), str(reclaim_path))
    except (FileNotFoundError, OSError):
        return False

    actual = read_fingerprint(reclaim_path)
    safe_to_remove = False
    if expected_fingerprint is not None and actual == expected_fingerprint:
        safe_to_remove = True
    elif expected_fingerprint is None and actual is None:
        # Both fingerprint-less — distinguish "still old stale lock"
        # from "fresh mkdir under us" via directory mtime age.
        try:
            age = time.time() - reclaim_path.stat().st_mtime
        except OSError:
            age = float("inf")
        if age >= malformed_grace_sec:
            safe_to_remove = True

    if safe_to_remove:
        shutil.rmtree(reclaim_path, ignore_errors=True)
        return True

    # Not safe — restore the path so the fresh holder (or whoever
    # legitimately owns it now) is not disrupted.  Use renameat2 with
    # RENAME_NOREPLACE: plain os.rename would clobber an empty third-
    # contender directory at stale_dir.  If renameat2 is unavailable
    # OR the destination exists, leave the .reclaim.<uuid> stranded
    # rather than risk overwriting a valid holder.
    if not _safe_restore(reclaim_path, stale_dir):
        log.error(
            "reclaim race at %s: moved an unexpected generation, "
            "could not safely restore (renameat2 unavailable or "
            "destination occupied). Stranded copy left at %s; "
            "manual investigation needed.",
            stale_dir, reclaim_path,
        )
    return False


def _read_blob(path: Path) -> Optional[bytes]:
    """Best-effort bytes-read; returns None on any OSError."""
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def _read_token_fingerprint(slot_dir: Path) -> Optional[bytes]:
    """Generation fingerprint for a staging token slot.

    Returns the canonical bytes ``<pid>|<starttime_ticks>`` only when
    BOTH files exist AND each parses as a non-negative integer (after
    stripping).  Returns ``None`` for absent/empty/malformed files so
    ``_atomic_reclaim`` falls back to the mtime-age path rather than
    match a partial generation by PID alone — a fresh holder mid-write
    can briefly expose a zero-length ``starttime_ticks`` file
    (``Path.write_text`` truncates before writing), which would
    otherwise let an old same-PID stale slot's fingerprint collide
    with the new generation's "pid|empty".
    """
    pid_bytes = _read_blob(slot_dir / "pid")
    st_bytes = _read_blob(slot_dir / "starttime_ticks")
    if pid_bytes is None or st_bytes is None:
        return None
    try:
        pid = int(pid_bytes.strip())
        st = int(st_bytes.strip())
    except (ValueError, AttributeError):
        return None
    if pid < 0 or st < 0:
        return None
    return f"{pid}|{st}".encode()


# renameat2 + RENAME_NOREPLACE: atomic rename that REFUSES to clobber
# an existing destination.  POSIX ``os.rename`` allows renaming a
# non-empty dir onto an empty dir (which would commandeer a third
# contender's freshly-mkdir'd slot during reclaim put-back).  Linux
# 3.15+ exposes ``renameat2`` for the no-replace semantic; on older
# kernels or non-Linux, ``_safe_restore`` falls back to "leave
# stranded" rather than risk overwriting a fresh holder.

_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_libc = None


def _load_libc():
    global _libc
    if _libc is not None:
        return _libc
    libc_name = ctypes.util.find_library("c") or "libc.so.6"
    try:
        _libc = ctypes.CDLL(libc_name, use_errno=True)
    except OSError:
        _libc = False  # sentinel: failed to load
    return _libc


def _safe_restore(reclaim_path: Path, original_path: Path) -> bool:
    """Try to rename *reclaim_path* back to *original_path* WITHOUT
    overwriting an existing destination.

    Returns ``True`` on a successful restore.  Returns ``False`` if
    the destination already exists OR renameat2 is unavailable on
    this platform.  Crucially, this NEVER clobbers a fresh empty
    directory that a third contender may have just created at
    *original_path* — closing the rename-over-empty-dir hazard that
    plain ``os.rename`` would expose.
    """
    libc = _load_libc()
    if libc is False or not hasattr(libc, "renameat2"):
        return False
    libc.renameat2.restype = ctypes.c_int
    libc.renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    ret = libc.renameat2(
        _AT_FDCWD, str(reclaim_path).encode(),
        _AT_FDCWD, str(original_path).encode(),
        _RENAME_NOREPLACE,
    )
    return ret == 0


def _holder_alive_locally(
    lock_dir: Path,
    missing_metadata_grace_sec: float = 3.0,
) -> bool:
    """Return True iff the lock's holder is on this host AND alive.

    Cross-host holders are reported as alive (no local check is
    possible).  Cross-host stale lock-dir recovery is not implemented
    today — callers rely on the bounded ``_MkdirLock`` wait to surface
    a typed ``StaleLockTimeout`` instead of looping forever.

    Missing ``holder.json`` could mean either (a) a crashed holder
    that never wrote metadata, or (b) a freshly-mkdir'd lock whose
    holder is in the brief window between ``mkdir()`` and the
    holder.json write.  To avoid stealing live locks, we wait up to
    ``missing_metadata_grace_sec`` for the file to appear before
    declaring stale.

    Same-host liveness: ``os.kill(pid, 0)`` plus start-time match.
    The start-time check disambiguates PID reuse: a fresh process at
    the same numeric PID has a different ``/proc/<pid>/stat`` field
    22.  Holders written by older versions (no ``starttime_ticks``
    field) fall back to PID-only liveness — graceful degradation.
    """
    holder_file = Path(lock_dir) / "holder.json"
    if not holder_file.exists():
        # Race: holder may be mid-write.  Give it a grace window before
        # declaring stale.
        deadline = time.monotonic() + max(0.0, missing_metadata_grace_sec)
        while time.monotonic() < deadline:
            time.sleep(0.05)
            if holder_file.exists():
                break
        if not holder_file.exists():
            return False
    try:
        meta = json.loads(holder_file.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    host = meta.get("host")
    pid = int(meta.get("pid", 0))
    starttime = meta.get("starttime_ticks")
    if not pid:
        return False
    if host and not _same_host(host):
        return True

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True

    if starttime is not None:
        current = _read_proc_starttime(pid)
        if current is None:
            # PID died between os.kill and stat read — stale.
            return False
        try:
            if int(current) != int(starttime):
                # PID was reused — original holder is gone.
                return False
        except (TypeError, ValueError):
            return True
    return True


def acquire_stage_lock(gous_dir: Path, holder_meta: dict | None = None) -> _MkdirLock:
    """Per-(machine, GOUS) mutex via ``mkdir`` on ``<gous_dir>/.stage.lock.d``.

    Callers should use this as a context manager:

    >>> with acquire_stage_lock(gous_dir, {"run_id": 42}):
    ...     ensure_staged(...)
    """
    lock_dir = Path(gous_dir) / ".stage.lock.d"
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    return _MkdirLock(dir_path=lock_dir, holder_meta=holder_meta or {})


# ---------------------------------------------------------------------------
# NAS staging tokens (global concurrency cap)
# ---------------------------------------------------------------------------

def _token_holder_alive(
    slot_dir: Path,
    malformed_grace_sec: float = 60.0,
) -> bool:
    """Return True iff the token slot's holder is on this host AND alive.

    Mirrors ``_holder_alive_locally`` but for the per-token metadata
    shape (``host``, ``pid``, ``holder``, ``acquired_at``, optional
    ``starttime_ticks``) used by :func:`acquire_staging_token`.

    Missing-metadata slots get a ``malformed_grace_sec`` window before
    being declared stale — mirrors the dispatch-side
    ``_TOKEN_MALFORMED_GRACE_SEC`` so an in-flight acquire's brief
    mkdir-then-write window is not stolen.

    Cross-host holders are treated as alive; the dispatch coordinator's
    ``TokenReaper`` performs cross-host cleanup via ``ssh_pid_alive``.
    """
    slot_dir = Path(slot_dir)
    host_file = slot_dir / "host"
    pid_file = slot_dir / "pid"

    # Missing host or pid: respect the grace window for in-flight
    # acquires that have mkdir'd the slot but not yet written metadata.
    if not host_file.exists() or not pid_file.exists():
        try:
            slot_mtime = slot_dir.stat().st_mtime
        except OSError:
            return False
        age = time.time() - slot_mtime
        if age < malformed_grace_sec:
            return True
        return False

    try:
        host = host_file.read_text().strip()
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    if not pid:
        return False
    if host and not _same_host(host):
        return True

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True

    st_file = slot_dir / "starttime_ticks"
    if st_file.exists():
        try:
            stored = int(st_file.read_text().strip())
        except (OSError, ValueError):
            return True
        current = _read_proc_starttime(pid)
        if current is None:
            return False
        if current != stored:
            return False
    return True


def acquire_staging_token(
    tokens_root: Path,
    n_slots: int,
    *,
    holder_id: str,
    poll_sleep: tuple[float, float] = (1.0, 3.0),
    timeout_sec: Optional[float] = _DEFAULT_TOKEN_WAIT_TIMEOUT_SEC,
) -> int:
    """Acquire one of *n_slots* tokens under *tokens_root*.

    Atomic primitive: ``os.mkdir(tokens_root/<i>)``.  Returns the index
    of the slot acquired.  ``timeout_sec`` bounds the wait — defaults to
    30 minutes; ``None`` opts out (legacy / test convenience).

    Raises :class:`TokenAcquireTimeout` (an alias of ``TimeoutError``)
    when the bound is exceeded.  Same-host stale slots whose PID is
    dead — including same-numeric-PID-but-fresh-process via
    ``starttime_ticks`` mismatch — are reclaimed and re-tried.
    Cross-host stale slots are left to the dispatch coordinator's
    ``TokenReaper``.
    """
    tokens_root = Path(tokens_root)
    tokens_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        # Bound EVERY iteration of the outer loop, not just the
        # all-slots-pinned path.  A pathological reclaim cycle (e.g.,
        # filesystem churn racing our rename) must also terminate.
        if timeout_sec is not None and time.monotonic() - started > timeout_sec:
            raise TokenAcquireTimeout(
                f"could not acquire staging token under {tokens_root} "
                f"in {timeout_sec}s"
            )
        for i in range(n_slots):
            slot = tokens_root / str(i)
            try:
                slot.mkdir()
            except FileExistsError:
                # Snapshot the generation fingerprint (pid +
                # starttime_ticks) BEFORE the alive check.  Using both
                # closes the PID-reuse race: a fresh holder with the
                # same numeric PID will have a different start time.
                fp_snapshot = _read_token_fingerprint(slot)
                if not _token_holder_alive(slot):
                    log.warning(
                        "stale staging token at %s; reclaiming", slot,
                    )
                    _atomic_reclaim(
                        slot,
                        expected_fingerprint=fp_snapshot,
                        read_fingerprint=_read_token_fingerprint,
                        # Mirror dispatch._TOKEN_MALFORMED_GRACE_SEC for
                        # the no-metadata window.
                        malformed_grace_sec=60.0,
                    )
                    try:
                        slot.mkdir()
                    except FileExistsError:
                        continue
                else:
                    continue
            try:
                # Identity-bearing files written; ``starttime_ticks``
                # last so consumers checking for its presence can
                # safely fall back to PID-only on old slots.
                (slot / "host").write_text(socket.gethostname())
                (slot / "pid").write_text(str(os.getpid()))
                (slot / "holder").write_text(holder_id)
                (slot / "acquired_at").write_text(str(time.time()))
                st = _read_proc_starttime(os.getpid())
                if st is not None:
                    (slot / "starttime_ticks").write_text(str(st))
            except OSError as exc:
                # Best-effort cleanup of OUR partial write; the slot
                # path may belong to us (we just mkdir'd it) so a
                # direct rmtree here is safe — no peer can have raced
                # to write into a slot we own.
                shutil.rmtree(slot, ignore_errors=True)
                raise RuntimeError(f"failed to write token metadata: {exc}")
            return i
        time.sleep(random.uniform(*poll_sleep))


def release_staging_token(tokens_root: Path, i: int) -> None:
    """Release the token at slot *i*.  Idempotent."""
    shutil.rmtree(Path(tokens_root) / str(i), ignore_errors=True)


def list_held_tokens(tokens_root: Path) -> list[dict]:
    """Return metadata for every held token under *tokens_root*.

    ``starttime_ticks`` is OPTIONAL — old tokens (pre-PID-reuse fix)
    are reported with ``starttime_ticks: None`` and MUST NOT be
    misclassified as malformed.  The dispatch ``TokenReaper`` reaps
    malformed slots after a grace window; treating old-but-valid slots
    as malformed would disrupt live cross-version dispatches.
    """
    tokens_root = Path(tokens_root)
    if not tokens_root.exists():
        return []
    out: list[dict] = []
    for slot in sorted(tokens_root.iterdir()):
        if not slot.is_dir():
            continue
        try:
            host = (slot / "host").read_text().strip()
            pid = int((slot / "pid").read_text().strip())
            holder = (slot / "holder").read_text().strip() if (slot / "holder").exists() else ""
            # starttime_ticks is back-compat optional — its own
            # try/except so a corrupt-but-not-missing value still
            # reports None instead of poisoning the slot record.
            starttime_ticks: Optional[int] = None
            st_file = slot / "starttime_ticks"
            if st_file.exists():
                try:
                    starttime_ticks = int(st_file.read_text().strip())
                except (OSError, ValueError):
                    starttime_ticks = None
            out.append({
                "slot": slot.name, "path": slot, "host": host, "pid": pid,
                "holder": holder, "starttime_ticks": starttime_ticks,
            })
        except (OSError, ValueError):
            out.append({"slot": slot.name, "path": slot, "host": None, "pid": None,
                        "holder": "", "starttime_ticks": None})
    return out


# ---------------------------------------------------------------------------
# Cross-dispatch staging cache
# ---------------------------------------------------------------------------
#
# Layout (per host)::
#
#     <machine.raid>/
#     ├── cache/
#     │   ├── <basename>.<sha1[:16]>/         ← the cache *entry* dir
#     │   │   ├── <basename>/                 ← staged MS or TP
#     │   │   ├── .cache.json                 ← sidecar (commit marker)
#     │   │   └── .populating/                ← present iff a worker is filling
#     │   ├── ...
#     │   └── .gc.lock.d/                     ← GC lock (per-host eviction mutex)
#     └── d_<dispatch_id>/
#         └── input/<gous>/ms/<basename>/     ← cp -al hard-links into cache entry
#
# Hit semantics: ``stat(src_path)`` on NAS, then sidecar match on
# ``(mtime_ns, recursive_size_bytes)``.  ``du -sb`` is required because
# CASA tables are directories — ``stat(dir).st_size`` is not recursive,
# and the root-dir mtime doesn't always propagate to nested edits.
#
# The cache key is ``<basename>.<sha1(src_path)[:16]>`` so that two MSs
# with identical basenames but different source roots cannot collide.
#
# Files inside a cached entry are chmod'd to 0o444 after population so
# that any unexpected CASA write through a hard-linked inode fails loudly
# rather than silently corrupting the cache.  Directories stay 0o755 so
# CASA can still drop lock-files alongside.

_CACHE_KEY_HASH_LEN = 16


def _cache_key(src_path: str | Path) -> str:
    """Stable cache-entry key: ``<basename>.<sha1(src_path)[:16]>``."""
    p = Path(src_path)
    digest = hashlib.sha1(str(p.resolve(strict=False)).encode("utf-8")).hexdigest()
    return f"{p.name}.{digest[:_CACHE_KEY_HASH_LEN]}"


_FINGERPRINT_VERSION = 2


def _compute_tree_sig(src_path: Path) -> str:
    """Recursive metadata-only signature of *src_path*.

    Walks the tree (no follow_symlinks), collects every entry, and
    hashes a sorted record of ``(relpath, kind, size, mtime_ns,
    ctime_ns, symlink_target)`` per entry.  Catches inner-file
    rewrites and nested temp+rename patterns that the root mtime +
    recursive size pair cannot see (proven by
    ``scripts/test_cache_invalidation.py`` — A/B/C scenarios).

    Cost: one stat per entry, one ``readlink`` per symlink.  Pure
    metadata; never reads file contents.  For a CASA MS with ~50–200
    inner files the walk is well under a second.

    For a single-file root (TP FITS), the signature reduces to the
    file's own ``(size, mtime_ns, ctime_ns)``.

    Symlink-root handling: if *src_path* IS itself a symlink, we
    resolve it before walking.  Staging dereferences the link
    (``cp -a``, ``tar -C <parent> <name>/`` and ``rsync -a`` all
    follow leaf symlinks), so the cached data reflects the target's
    content.  Fingerprinting the link itself would let the target
    change underneath us with no fingerprint change.  ``cache_key``
    hashes the resolved path, so two symlinks at *the same basename*
    pointing at the same target collide on one cache entry; symlinks
    with different basenames do not (their cache_key prefix differs)
    — that's not a correctness issue, just less aggressive dedup.
    """
    h = hashlib.sha256()
    src_path = Path(src_path)
    if src_path.is_symlink():
        src_path = src_path.resolve(strict=False)

    if not src_path.is_dir():
        st = src_path.lstat()
        rec = (f"|f|{st.st_size}|{st.st_mtime_ns}|"
               f"{st.st_ctime_ns}|\n")
        h.update(rec.encode("utf-8"))
        return h.hexdigest()

    # Directory root — gather every descendant entry, sort, hash.
    # Skip the root itself: its mtime_ns is already in the v2
    # fingerprint as a separate field, and including it here would
    # double-count.
    entries: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(
        src_path, followlinks=False,
    ):
        for name in dirnames + filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, str(src_path))
            entries.append((rel, full))
    entries.sort()

    for rel, full in entries:
        try:
            st = os.lstat(full)
        except OSError:
            # Entry vanished between os.walk and our lstat — record an
            # explicit deletion so any concurrent regen still produces
            # a different signature.
            h.update(f"{rel}|MISSING\n".encode("utf-8"))
            continue
        if os.path.islink(full):
            try:
                target = os.readlink(full)
            except OSError:
                target = ""
            kind = "l"
        elif os.path.isdir(full):
            target = ""
            kind = "d"
        else:
            target = ""
            kind = "f"
        rec = (f"{rel}|{kind}|{st.st_size}|{st.st_mtime_ns}|"
               f"{st.st_ctime_ns}|{target}\n")
        h.update(rec.encode("utf-8"))
    return h.hexdigest()


def _compute_fingerprint(src_path: Path) -> dict:
    """Return ``{'version': 2, 'mtime_ns', 'size_bytes', 'tree_sig'}``.

    - ``mtime_ns`` is the root's ``st_mtime_ns`` (cheap pre-screen).
    - ``size_bytes`` is recursive ``du -sb`` for directories (used for
      eviction sizing), or ``stat().st_size`` for files.
    - ``tree_sig`` is a recursive metadata-only hash that catches the
      inner-file rewrite + nested temp+rename patterns the v1
      ``(mtime_ns, size_bytes)`` pair could not detect.

    If *src_path* is itself a symlink, it's resolved first — staging
    dereferences leaf symlinks (``cp -a``, ``tar -C parent name/``,
    ``rsync -a`` all follow), so the fingerprint must track the
    target.  Without this, a swap of the link's target would leave
    fingerprint unchanged and serve stale cache.

    Raises ``OSError`` if ``du -sb`` fails on a directory — a zero size
    would silently commit a wrong fingerprint.
    """
    src_path = Path(src_path)
    if src_path.is_symlink():
        src_path = src_path.resolve(strict=False)
    st = src_path.stat()
    if src_path.is_dir():
        size_bytes = _du_bytes(src_path)
        if size_bytes <= 0:
            raise OSError(
                f"_du_bytes returned {size_bytes} for {src_path} — "
                f"refusing to commit cache fingerprint with a bad size",
            )
    else:
        size_bytes = int(st.st_size)
    return {
        "version": _FINGERPRINT_VERSION,
        "mtime_ns": int(st.st_mtime_ns),
        "size_bytes": int(size_bytes),
        "tree_sig": _compute_tree_sig(src_path),
    }


def _read_sidecar(entry_dir: Path) -> Optional[dict]:
    side = Path(entry_dir) / ".cache.json"
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_sidecar_atomic(entry_dir: Path, src_path: Path, fingerprint: dict) -> None:
    """Write ``.cache.json`` LAST (it's the entry's commit marker)."""
    payload = {
        "src_path": str(Path(src_path).resolve(strict=False)),
        "version": int(fingerprint.get("version", _FINGERPRINT_VERSION)),
        "mtime_ns": int(fingerprint["mtime_ns"]),
        "size_bytes": int(fingerprint["size_bytes"]),
        "tree_sig": fingerprint["tree_sig"],
        "staged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_json(Path(entry_dir) / ".cache.json", payload)


def cache_lookup(
    cache_root: Path,
    src_path: str | Path,
    *,
    ignore_populating: bool = False,
) -> Optional[Path]:
    """Return the staged subdir if the cache has a valid entry.

    Returns ``cache_root/<key>/<basename>`` on hit, ``None`` on miss.
    A miss covers: missing entry, missing sidecar, mtime mismatch, size
    mismatch, or (when ``ignore_populating=False``) an entry mid-populate
    (``.populating`` present).

    ``ignore_populating=True`` is for the *populator* to peek inside its
    own ``.populating`` lock and discover that a peer's commit landed
    just before our ``mkdir(.populating)`` won — in which case we should
    release our lock and return a hit rather than clobbering the entry.
    External callers never set this.
    """
    cache_root = Path(cache_root)
    src_path = Path(src_path)
    if not src_path.exists():
        return None
    entry_dir = cache_root / _cache_key(src_path)
    if not entry_dir.is_dir():
        return None
    if not ignore_populating and (entry_dir / ".populating").exists():
        return None
    side = _read_sidecar(entry_dir)
    if side is None:
        return None
    # v1 sidecars (no version, or version != current) are treated as
    # stale — the v1 fingerprint (mtime_ns + size_bytes) couldn't
    # detect inner-file rewrites or nested temp+rename, so we refuse
    # to trust those entries even if mtime + size happen to match.
    # See ``_compute_tree_sig`` for the v2 motivation.
    if int(side.get("version", 1)) != _FINGERPRINT_VERSION:
        return None
    if not side.get("tree_sig"):
        return None
    try:
        fp = _compute_fingerprint(src_path)
    except OSError:
        return None
    if (int(side.get("mtime_ns", -1)) != fp["mtime_ns"]
            or int(side.get("size_bytes", -1)) != fp["size_bytes"]
            or side.get("tree_sig") != fp["tree_sig"]):
        return None
    staged = entry_dir / src_path.name
    if not staged.exists():
        return None
    return staged


def _chmod_readonly_files(root: Path) -> None:
    """Set files under *root* to 0o444 and dirs to 0o755.

    Handles both file and directory roots: TP cubes are single FITS
    files, MSs are CASA-table directory trees.  Read-only files act as
    a canary against unexpected CASA writes through the hard-linked
    inode.  Directories stay writable so CASA can create lock-files
    inside.

    Raises ``OSError`` on any chmod failure — without read-only files,
    the canary is gone and an unexpected CASA write through the
    hard-link would silently corrupt the cache.  Better to refuse to
    commit the cache entry than ship a writable cache.

    Symlinks are skipped: ``os.chmod`` on a symlink follows the link
    and would alter the *target* — if a CASA table happens to contain
    a symlink to outside the entry, we must not modify the target.
    """
    root = Path(root)
    if root.is_file() and not root.is_symlink():
        os.chmod(root, 0o444)
        return
    if root.is_symlink():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        os.chmod(dirpath, 0o755)
        for f in filenames:
            full = os.path.join(dirpath, f)
            if os.path.islink(full):
                # Don't follow link — would chmod the target.
                continue
            os.chmod(full, 0o444)


def cache_link_into(cache_staged: Path, dst_path: Path) -> None:
    """Hard-link *cache_staged* into *dst_path*.

    Handles both directory sources (CASA tables) and file sources (TP
    FITS).  Uses ``cp -al`` (recursive hard-link) for directories;
    ``ln`` (single hard-link) for files.  Same-filesystem only — the
    cache lives on the same ``/raid`` as the dispatch input dir, so
    this is always satisfied.  Eviction-safe: hard-links keep the
    inode alive even if the cache entry is rmtree'd.

    Atomic: builds the destination as a sibling ``.partial`` then
    ``os.rename``.  On any failure the partial is removed and the
    caller can fall back to a NAS read.
    """
    cache_staged = Path(cache_staged)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return
    partial = dst_path.parent / f".{dst_path.name}.partial"
    if partial.exists():
        if partial.is_dir():
            shutil.rmtree(partial)
        else:
            partial.unlink()
    try:
        if cache_staged.is_dir():
            # ``cp -al SRC/. PARTIAL/`` — the trailing ``/.`` semantics
            # copy the *contents* of SRC into PARTIAL, preserving
            # recursive structure as hard-links.
            partial.mkdir()
            _run_check(["cp", "-al", str(cache_staged) + "/.", str(partial)])
        else:
            # Single-file hard-link (TP FITS).  ``os.link`` is atomic
            # and avoids subprocess overhead for the common case.
            os.link(str(cache_staged), str(partial))
        os.rename(str(partial), str(dst_path))
    except Exception:
        if partial.exists():
            if partial.is_dir():
                shutil.rmtree(partial, ignore_errors=True)
            else:
                try:
                    partial.unlink()
                except OSError:
                    pass
        raise


@dataclass
class _PopulateLockHandle:
    """Handle for a cache populate lock — release via ``__exit__``."""
    lock_dir: Path

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        shutil.rmtree(self.lock_dir, ignore_errors=True)


def acquire_cache_populate(
    cache_root: Path,
    src_path: str | Path,
    *,
    holder_meta: Optional[dict] = None,
    timeout_sec: float = 3600.0,
    poll_sleep: tuple[float, float] = (2.0, 5.0),
) -> tuple[str, Optional[_PopulateLockHandle]]:
    """Acquire the populator role for the cache entry of *src_path*.

    Returns one of:
      ``("populate", handle)`` — caller is the populator.  Use ``handle``
        as a context manager (it removes the lock dir on exit).
      ``("hit", None)`` — waited for another worker; cache is now valid.

    Raises ``TimeoutError`` if neither outcome occurs within
    ``timeout_sec`` (caller should fall back to NAS-direct staging).
    Raises :class:`StaleLockTimeout` if the inner GC lock acquisition
    exceeds its (capped) bound — this is NOT a ``TimeoutError`` /
    ``OSError`` subclass, so callers' ``except OSError`` filesystem-race
    handlers do not silently swallow it.

    Stale-lock recovery: if the populator's host matches ours and its PID
    is dead (PID reuse defended via ``starttime_ticks``), we ``rmtree``
    and retry.  Cross-host stale recovery is NOT implemented today; the
    bounded GC-lock wait surfaces it as ``StaleLockTimeout`` to the
    caller.
    """
    cache_root = Path(cache_root)
    src_path = Path(src_path)
    entry_dir = cache_root / _cache_key(src_path)
    populating = entry_dir / ".populating"
    deadline = time.monotonic() + timeout_sec
    cache_root.mkdir(parents=True, exist_ok=True)
    gc_lock_path = cache_root / ".gc.lock.d"

    while True:
        # Check FIRST whether a peer's populate already landed.  Without
        # this, after A finishes populating and removes .populating, B
        # (still in the wait loop) will succeed at ``mkdir(.populating)``
        # and re-stage from NAS even though the cache is now valid.
        hit = cache_lookup(cache_root, src_path)
        if hit is not None:
            return ("hit", None)

        # Inner GC-lock wait is capped by remaining outer budget so a
        # caller passing a small ``timeout_sec`` does not get a 300s
        # inner timeout overshoot.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"cache populate wait exceeded {timeout_sec:.0f}s for {src_path}"
            )
        inner_timeout = min(_MKDIR_LOCK_DEFAULT_WAIT_SEC, remaining)

        # Publish ``.populating`` *under the GC lock* so concurrent
        # ``cache_evict_until_free`` cannot rmtree our entry between
        # our mkdir and our holder.json write.
        try:
            with _MkdirLock(
                dir_path=gc_lock_path,
                holder_meta={"role": "publish_populate", **(holder_meta or {})},
                wait_timeout_sec=inner_timeout,
            ):
                try:
                    populating.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    publish_outcome = "exists"
                else:
                    # Identity fields LAST so caller-supplied
                    # holder_meta cannot override host/pid/starttime.
                    meta = {
                        **(holder_meta or {}),
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                        "starttime_ticks": _read_proc_starttime(os.getpid()),
                    }
                    tmp = populating / ".holder.json.tmp"
                    tmp.write_text(json.dumps(meta))
                    os.replace(str(tmp), str(populating / "holder.json"))
                    publish_outcome = "won"
        except OSError as exc:
            # GC lock acquisition itself failed (e.g. cache_root vanished
            # under us during stale-recovery); retry, but respect the
            # outer timeout so a persistent filesystem error eventually
            # surfaces as TimeoutError rather than spinning forever.
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"cache populate wait exceeded {timeout_sec:.0f}s "
                    f"for {src_path} (last error: {exc})"
                )
            time.sleep(random.uniform(0.1, 0.3))
            continue

        if publish_outcome == "won":
            # Post-publish recheck (ignoring our OWN .populating).
            # Closes the race where peer A finishes + removes its
            # .populating between our top-of-loop ``cache_lookup`` and
            # our publish.  Without this we'd clobber a fresh valid
            # entry and re-stage from NAS.
            hit = cache_lookup(cache_root, src_path, ignore_populating=True)
            if hit is not None:
                shutil.rmtree(populating, ignore_errors=True)
                return ("hit", None)
            return ("populate", _PopulateLockHandle(populating))

        # ``.populating`` already present — fall through to wait/stale/timeout.

        # Same-host stale recovery (cross-host is left to the bounded
        # outer ``timeout_sec`` — no coordinator reaper for lock dirs).
        fp_snapshot = _read_blob(populating / "holder.json")
        if not _holder_alive_locally(populating):
            log.warning("stale cache populate lock at %s; reclaiming", populating)
            _atomic_reclaim(
                populating,
                expected_fingerprint=fp_snapshot,
                read_fingerprint=lambda d: _read_blob(d / "holder.json"),
            )
            continue

        if time.monotonic() > deadline:
            raise TimeoutError(
                f"cache populate wait exceeded {timeout_sec:.0f}s for {src_path}"
            )
        time.sleep(random.uniform(*poll_sleep))


def _wipe(path: Path) -> None:
    """Remove a file or directory tree at *path*.  Idempotent."""
    if not Path(path).exists():
        return
    if Path(path).is_dir() and not Path(path).is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            Path(path).unlink()
        except OSError:
            pass


def _du_free_bytes(root: Path) -> int:
    """Bytes free under *root*'s filesystem (``shutil.disk_usage``)."""
    try:
        return shutil.disk_usage(str(root)).free
    except OSError:
        return 0


def cache_evict_until_free(
    cache_root: Path,
    target_free_bytes: int,
    *,
    skip_keys: Optional[set[str]] = None,
    wait_timeout_sec: float = _MKDIR_LOCK_DEFAULT_WAIT_SEC,
) -> int:
    """Evict oldest cache entries (by sidecar ``staged_at``) until at
    least *target_free_bytes* are free on the cache filesystem.

    Returns the number of entries evicted.  Skips entries whose key is
    in *skip_keys* (the entry currently being populated by the caller),
    plus any entry with a ``.populating`` lock.

    On GC-lock contention exceeding ``wait_timeout_sec``, logs a WARN
    and returns 0 (no eviction performed; caller proceeds without
    trimmed space).  This bounds the worst case when a peer worker's
    eviction stalls under load.

    Concurrency: a per-host GC mutex (mkdir-based) serialises eviction
    so two workers cannot rmtree the same entry concurrently or race
    a ``cp -al`` reader.
    """
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return 0
    skip_keys = set(skip_keys or ())
    gc_lock = cache_root / ".gc.lock.d"
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        gc_cm = _MkdirLock(
            dir_path=gc_lock,
            holder_meta={"role": "cache_gc"},
            wait_timeout_sec=wait_timeout_sec,
        )
        gc_cm.__enter__()
    except StaleLockTimeout as exc:
        log.warning(
            "cache_evict_until_free: GC lock timeout on %s after %.0fs: %s",
            gc_lock, wait_timeout_sec, exc,
        )
        return 0
    try:
        free = _du_free_bytes(cache_root)
        if free >= target_free_bytes:
            return 0
        # Collect candidates: (staged_at, entry_dir, size_bytes)
        cands = []
        for entry in cache_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name in skip_keys:
                continue
            if entry.name == ".gc.lock.d":
                continue
            if (entry / ".populating").exists():
                continue
            side = _read_sidecar(entry)
            if side is None:
                # Malformed: evict first to reclaim space
                cands.append(("", entry, 0))
                continue
            cands.append((side.get("staged_at", ""), entry,
                          int(side.get("size_bytes", 0))))
        cands.sort(key=lambda t: t[0])  # oldest first; "" sorts before iso dates

        evicted = 0
        for _, entry_dir, _size in cands:
            if _du_free_bytes(cache_root) >= target_free_bytes:
                break
            # Re-check ``.populating`` immediately before rmtree.  A
            # populator that raced our candidate-collection above will
            # have published its lock under our same ``.gc.lock.d``
            # mutex (see ``acquire_cache_populate``), so by the time
            # we get here either ``.populating`` was created before
            # we entered this lock (and we'll see it now) or the
            # populator is still waiting for us to release.  Either
            # way the recheck is sufficient to avoid deleting a live
            # populate in progress.
            if (entry_dir / ".populating").exists():
                continue
            log.info("cache evict: %s (free=%.1fGB target=%.1fGB)",
                     entry_dir.name,
                     _du_free_bytes(cache_root) / 1e9,
                     target_free_bytes / 1e9)
            shutil.rmtree(entry_dir, ignore_errors=True)
            evicted += 1
        return evicted
    finally:
        gc_cm.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# TokenLease — lazy NAS staging token (acquired only when actually reading
# from NAS, not for cache hits).  Idempotent release.
# ---------------------------------------------------------------------------

class TokenLease:
    """Acquire a NAS staging token on first use; release on close.

    The dispatcher used to acquire the token outside ``_ensure_staged_inputs``
    and hold it across all stages — which serialised even all-cache-hit
    runs behind the NAS gate.  This class lets ``stage_one`` acquire only
    when it actually does a NAS read, and reuse the token across multiple
    misses within a single worker.

    If ``tokens_dir is None`` (no NAS gate configured), all methods are
    no-ops.
    """

    def __init__(
        self,
        tokens_dir: Optional[Path],
        n_slots: int,
        *,
        holder_id: str,
        timeout_sec: float = _DEFAULT_TOKEN_WAIT_TIMEOUT_SEC,
    ) -> None:
        self.tokens_dir = Path(tokens_dir) if tokens_dir else None
        self.n_slots = int(n_slots)
        self.holder_id = holder_id
        self.timeout_sec = float(timeout_sec)
        self._idx: Optional[int] = None
        self._wait_total_sec = 0.0
        self._acquires = 0

    def acquire_if_needed(self) -> None:
        if self.tokens_dir is None or self._idx is not None:
            return
        started = time.monotonic()
        self._idx = acquire_staging_token(
            self.tokens_dir, self.n_slots, holder_id=self.holder_id,
            timeout_sec=self.timeout_sec,
        )
        waited = time.monotonic() - started
        self._wait_total_sec += waited
        self._acquires += 1
        if waited > 5:
            log.info(
                "acquired staging token %d after %.1fs wait (holder=%s)",
                self._idx, waited, self.holder_id,
            )

    def release(self) -> None:
        if self._idx is None or self.tokens_dir is None:
            return
        release_staging_token(self.tokens_dir, self._idx)
        self._idx = None

    def __enter__(self) -> "TokenLease":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()

    @property
    def stats(self) -> dict:
        return {
            "token_acquires": self._acquires,
            "token_wait_sec": round(self._wait_total_sec, 1),
        }


# ---------------------------------------------------------------------------
# Staging primitives
# ---------------------------------------------------------------------------

def stage_one(
    src: str,
    dst_root: Path,
    *,
    method: str = "tar",
    bucket: str = "ms",
    validate: bool = True,
    cache_root: Optional[Path] = None,
    cache_min_free_bytes: Optional[int] = None,
    token_lease: Optional["TokenLease"] = None,
    cache_populate_timeout_sec: float = 3600.0,
) -> tuple[Path, str]:
    """Stage a single source path (MS dir or TP FITS) into *dst_root*.

    Layout: ``<dst_root>/<bucket>/<basename(src)>``.

    Returns ``(staged_path, source)`` where ``source`` is one of:

    - ``"existing"``  — destination already present (no work)
    - ``"cache_hit"`` — hard-linked from cache; no NAS read
    - ``"cache_hit_after_wait"`` — cache hit after waiting for another
      worker's populate to complete
    - ``"cache_miss"`` — populated cache from NAS, then linked into dst
    - ``"nas_direct"`` — fallback: read from NAS into dst directly
      (cache disabled, populate timed out, or cache write failed)

    Atomicity: the destination is built in a sibling ``.partial``
    directory and ``os.rename``'d into place at the end.

    NAS access is gated by *token_lease* (if supplied).  The lease is
    acquired only on cache miss / NAS-direct paths — cache hits never
    touch the NAS gate, which is the whole point of the cache.

    Supported ``method`` values: ``"tar"``, ``"rsync"``, ``"cp"``.
    """
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"staging source does not exist: {src_path}")

    bucket_dir = Path(dst_root) / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    final = bucket_dir / src_path.name

    if final.exists():
        return final, "existing"

    # ---- 1. Cache hit fast path (no NAS, no token) ----
    if cache_root is not None:
        try:
            hit = cache_lookup(Path(cache_root), src_path)
        except OSError:
            hit = None
        if hit is not None:
            try:
                cache_link_into(hit, final)
                return final, "cache_hit"
            except Exception as exc:
                log.warning(
                    "cache_link_into failed for %s; falling through to populate: %s",
                    src_path.name, exc,
                )

    # ---- 2. Try to populate the cache (or wait for another worker) ----
    if cache_root is not None:
        try:
            outcome, lock = acquire_cache_populate(
                Path(cache_root), src_path,
                holder_meta={"role": "cache_populate"},
                timeout_sec=cache_populate_timeout_sec,
            )
        except (TimeoutError, StaleLockTimeout) as exc:
            log.warning(
                "cache populate gave up for %s (%s); falling back to NAS-direct",
                src_path.name, exc,
            )
            outcome, lock = "fallback", None

        if outcome == "hit":
            # Populator finished while we waited — link in.
            hit = cache_lookup(Path(cache_root), src_path)
            if hit is not None:
                try:
                    cache_link_into(hit, final)
                    return final, "cache_hit_after_wait"
                except Exception as exc:
                    log.warning(
                        "cache_link_into failed after wait for %s: %s — NAS-direct",
                        src_path.name, exc,
                    )

        if outcome == "populate":
            with lock:
                # NOTE: post-lock recheck (peer's commit landed just
                # before we won mkdir) is performed inside
                # ``acquire_cache_populate`` itself — it returns
                # ("hit", None) instead of ("populate", lock) in that
                # case, so we only reach here when there is genuinely
                # no valid entry to reuse.
                # stage_one-level populate deadline: we already committed
                # to bounded time when entering this branch, so cap any
                # downstream eviction wait by the same budget.
                populate_deadline = time.monotonic() + cache_populate_timeout_sec
                entry_dir = Path(cache_root) / _cache_key(src_path)
                entry_dir.mkdir(parents=True, exist_ok=True)
                cache_partial = entry_dir / f".{src_path.name}.partial"
                cache_final = entry_dir / src_path.name
                populate_ok = False
                try:
                    fp = _compute_fingerprint(src_path)
                    if cache_min_free_bytes is not None:
                        remaining = max(0.0, populate_deadline - time.monotonic())
                        cache_evict_until_free(
                            Path(cache_root),
                            cache_min_free_bytes + int(fp["size_bytes"]),
                            skip_keys={_cache_key(src_path)},
                            wait_timeout_sec=min(
                                _MKDIR_LOCK_DEFAULT_WAIT_SEC, remaining,
                            ),
                        )
                    _wipe(cache_partial)
                    if cache_final.exists():
                        # Stale (mtime/size mismatch landed us here) — wipe.
                        _wipe(cache_final)

                    if token_lease is not None:
                        token_lease.acquire_if_needed()
                    _do_nas_read(src_path, cache_partial, method)
                    os.rename(str(cache_partial), str(cache_final))
                    # Read-only canary BEFORE sidecar — if chmod fails
                    # we must not commit (sidecar is the commit marker).
                    _chmod_readonly_files(cache_final)
                    # Sidecar LAST — this is the entry's commit marker.
                    _write_sidecar_atomic(entry_dir, src_path, fp)
                    populate_ok = True
                except TokenAcquireTimeout:
                    # Token gate exhausted: NAS read is gated, no gate =
                    # no NAS read.  Surface as unit failure rather than
                    # silent NAS-direct retry (which would also fail).
                    raise
                except Exception as exc:
                    log.warning(
                        "cache populate failed for %s: %s — NAS-direct",
                        src_path.name, exc,
                    )
                finally:
                    if not populate_ok:
                        # Don't leak a (possibly very large) .partial
                        # under the cache.  cache_final is left alone:
                        # if it was rename'd before chmod/sidecar
                        # failed, removing it would also kill any
                        # peer's hard-linked dispatch dir.  An entry
                        # without sidecar is treated as a miss anyway.
                        _wipe(cache_partial)

                if populate_ok:
                    try:
                        cache_link_into(cache_final, final)
                        return final, "cache_miss"
                    except Exception as exc:
                        log.warning(
                            "cache_link_into post-populate failed for %s: %s — NAS-direct",
                            src_path.name, exc,
                        )

    # ---- 3. Fallback: NAS-direct stage into dst (no cache involvement) ----
    if token_lease is not None:
        token_lease.acquire_if_needed()
    partial = bucket_dir / f".{src_path.name}.partial"
    if partial.exists():
        if partial.is_dir():
            shutil.rmtree(partial)
        else:
            partial.unlink()
    _do_nas_read(src_path, partial, method)
    os.rename(str(partial), str(final))

    if validate:
        try:
            src_bytes = _du_bytes(src_path)
            dst_bytes = _du_bytes(final)
        except Exception:
            src_bytes = dst_bytes = None
        if (src_bytes and dst_bytes
                and abs(src_bytes - dst_bytes) > max(src_bytes * 0.001, 4096)):
            log.warning(
                "stage size mismatch for %s: src=%d dst=%d (%.2f%% diff)",
                src_path.name, src_bytes, dst_bytes,
                100.0 * abs(src_bytes - dst_bytes) / max(src_bytes, 1),
            )

    return final, "nas_direct"


def _do_nas_read(src_path: Path, partial_dst: Path, method: str) -> None:
    """Stage *src_path* into *partial_dst* using the requested method."""
    if src_path.is_file() or method == "cp":
        if src_path.is_dir():
            partial_dst.mkdir(parents=True, exist_ok=True)
            _run_check(["cp", "-a", str(src_path) + "/.", str(partial_dst)])
        else:
            partial_dst.parent.mkdir(parents=True, exist_ok=True)
            _run_check(["cp", "-p", str(src_path), str(partial_dst)])
    elif method == "tar":
        partial_dst.mkdir(parents=True, exist_ok=True)
        cmd = (
            f"tar -cf - -C {_shquote(str(src_path.parent))} "
            f"{_shquote(src_path.name)}/ "
            f"| tar -xf - -C {_shquote(str(partial_dst))} --strip-components=1"
        )
        _run_check(cmd, shell=True)
    elif method == "rsync":
        partial_dst.mkdir(parents=True, exist_ok=True)
        _run_check([
            "rsync", "-a", "--inplace", "--whole-file",
            str(src_path) + "/", str(partial_dst) + "/",
        ])
    else:
        raise ValueError(f"unknown transfer method: {method!r}")


def _du_bytes(path: Path) -> int:
    """Return ``du -sb <path>`` (recursive total size).  0 on error."""
    try:
        proc = subprocess.run(
            ["du", "-sb", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120,
        )
        if proc.returncode != 0:
            return 0
        return int(proc.stdout.split()[0])
    except (subprocess.SubprocessError, ValueError, IndexError):
        return 0


def _run_check(cmd, *, shell: bool = False) -> None:
    """``subprocess.run`` with check + helpful error tail on failure."""
    log.debug("stage exec: %s", cmd if shell else " ".join(cmd))
    proc = subprocess.run(
        cmd, shell=shell,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace",
    )
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-2000:]
        raise RuntimeError(
            f"staging command failed (rc={proc.returncode}): {tail}"
        )


def _shquote(s: str) -> str:
    """Minimal shell quoter for the `tar` pipe path arguments."""
    import shlex
    return shlex.quote(s)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_workdir(path: Path) -> None:
    """``rm -rf`` *path* if it exists.  Safe on missing dirs."""
    if Path(path).exists():
        shutil.rmtree(path, ignore_errors=True)
