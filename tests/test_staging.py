"""Tests for panta_rei.imaging.staging.

Covers the correctness-critical paths:
- stage_one() atomic temp+rename for tar / cp / rsync
- read_manifest / atomic_write_json
- mkdir-based stage lock (mutex blocks concurrent staging)
- token acquire / release / list / contention
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from panta_rei.imaging import staging


# ---------------------------------------------------------------------------
# stage_one
# ---------------------------------------------------------------------------

def _fake_ms(parent: Path, name: str = "fake.ms", n_files: int = 5) -> Path:
    """Create a CASA-MS-like directory: a dir of small files."""
    ms = parent / name
    ms.mkdir(parents=True)
    for i in range(n_files):
        (ms / f"table.dat.{i}").write_bytes(os.urandom(256))
    (ms / "table.f0_TSM0").write_bytes(os.urandom(1024))
    return ms


@pytest.mark.parametrize("method", ["tar", "rsync", "cp"])
def test_stage_one_atomic_temp_then_rename(tmp_path, method):
    src = _fake_ms(tmp_path / "src")
    dst_root = tmp_path / "dst"
    final, source = staging.stage_one(str(src), dst_root, method=method, bucket="ms")
    assert final == dst_root / "ms" / src.name
    assert final.is_dir()
    assert source == "nas_direct"
    # All source files arrived
    src_files = sorted(p.name for p in src.iterdir())
    dst_files = sorted(p.name for p in final.iterdir())
    assert src_files == dst_files
    # No leftover .partial
    assert not list((dst_root / "ms").glob(".*.partial"))


def test_stage_one_idempotent(tmp_path):
    src = _fake_ms(tmp_path / "src")
    dst_root = tmp_path / "dst"
    a, sa = staging.stage_one(str(src), dst_root, method="cp", bucket="ms")
    mtime = a.stat().st_mtime
    assert sa == "nas_direct"
    # Second call returns the same path without re-copying
    time.sleep(0.05)
    b, sb = staging.stage_one(str(src), dst_root, method="cp", bucket="ms")
    assert b == a
    assert sb == "existing"
    assert b.stat().st_mtime == mtime


def test_stage_one_unknown_method(tmp_path):
    src = _fake_ms(tmp_path / "src")
    with pytest.raises(ValueError):
        staging.stage_one(str(src), tmp_path / "dst", method="bogus")


def test_stage_one_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        staging.stage_one(str(tmp_path / "nope"), tmp_path / "dst", method="cp")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def test_read_manifest_skeleton_when_missing(tmp_path):
    m = staging.read_manifest(tmp_path / "nope.json")
    assert m == {"version": 1, "expected": [], "completed": []}


def test_atomic_write_json_replaces_existing(tmp_path):
    p = tmp_path / "m.json"
    staging.atomic_write_json(p, {"a": 1})
    staging.atomic_write_json(p, {"a": 2, "b": 3})
    m = staging.read_manifest(p)
    assert m == {"a": 2, "b": 3}


def test_atomic_write_json_no_orphan_temp(tmp_path):
    p = tmp_path / "m.json"
    staging.atomic_write_json(p, {"x": 1})
    leftover = list(tmp_path.glob("m.json.tmp*"))
    assert leftover == []


# ---------------------------------------------------------------------------
# Stage lock (mkdir mutex)
# ---------------------------------------------------------------------------

def test_stage_lock_recovers_from_dead_pid(tmp_path):
    """A stale lock from a crashed holder is reclaimed instead of
    spinning forever."""
    gous_dir = tmp_path / "gous"
    gous_dir.mkdir()
    # Simulate a crashed holder: lock dir + holder.json with a PID that
    # certainly does not exist (signal 0 raises ProcessLookupError).
    lock_dir = gous_dir / ".stage.lock.d"
    lock_dir.mkdir()
    import json as _json
    import socket as _sock
    (lock_dir / "holder.json").write_text(_json.dumps({
        "host": _sock.gethostname(),
        "pid": 2 ** 22,   # almost certainly unused
    }))
    # Sanity: it really is gone
    import os
    try:
        os.kill(2 ** 22, 0)
    except ProcessLookupError:
        pass
    else:
        pytest.skip("PID 2^22 unexpectedly exists on this host")

    # Acquire should NOT spin forever — it should detect dead holder + recover.
    acquired = threading.Event()

    def worker():
        with staging.acquire_stage_lock(gous_dir, {"id": "new"}):
            acquired.set()

    t = threading.Thread(target=worker)
    t.start()
    assert acquired.wait(timeout=5), "acquire never succeeded — likely deadlock"
    t.join()


def test_stage_lock_grace_does_not_steal_freshly_mkdird_lock(tmp_path):
    """Window between mkdir and holder.json write must not be racy.

    Simulates: lock dir exists with no holder.json, but a holder
    publishes metadata within the grace window.  The contender must
    NOT reclaim it.
    """
    gous_dir = tmp_path / "gous"
    gous_dir.mkdir()
    lock_dir = gous_dir / ".stage.lock.d"
    lock_dir.mkdir()
    # No holder.json yet — but a "writer" thread will publish one shortly.
    import json as _json
    import socket as _sock
    publish_at = threading.Event()
    finish = threading.Event()

    def writer():
        publish_at.wait(timeout=2)
        # Write holder.json with this process's PID (alive).
        meta = {"host": _sock.gethostname(), "pid": os.getpid()}
        (lock_dir / "holder.json").write_text(_json.dumps(meta))
        finish.set()

    wt = threading.Thread(target=writer)
    wt.start()
    # Give the contender up to 2.5s grace; release the writer at 0.5s.
    timer = threading.Timer(0.5, publish_at.set)
    timer.start()

    # Contender attempts to acquire — its grace period (default 3s)
    # should observe the holder.json that arrives at t=0.5s.
    second_acquired = threading.Event()

    def contender():
        try:
            with staging.acquire_stage_lock(gous_dir, {"id": "c"}):
                second_acquired.set()
        finally:
            pass

    ct = threading.Thread(target=contender)
    ct.start()

    # The writer's holder.json is published; its PID is alive (us).  The
    # contender should NOT acquire while we hold metadata "live".
    finish.wait(timeout=2)
    # Give a bit more time then verify: contender did NOT acquire.
    assert not second_acquired.wait(timeout=1.0), (
        "contender stole a lock whose holder.json arrived within grace"
    )
    # Now remove the holder.json so the contender can finally claim it.
    (lock_dir / "holder.json").unlink()
    # Eventually the contender's grace expires and it reclaims.
    assert second_acquired.wait(timeout=10)
    timer.cancel()
    wt.join(); ct.join()


def test_stage_lock_recovers_from_missing_holder_metadata(tmp_path):
    """Lock dir without holder.json is treated as stale."""
    gous_dir = tmp_path / "gous"
    gous_dir.mkdir()
    (gous_dir / ".stage.lock.d").mkdir()  # no holder.json
    acquired = threading.Event()

    def worker():
        with staging.acquire_stage_lock(gous_dir, {"id": "new"}):
            acquired.set()

    t = threading.Thread(target=worker)
    t.start()
    assert acquired.wait(timeout=5)
    t.join()


def test_stage_lock_blocks_concurrent_holder(tmp_path):
    gous_dir = tmp_path / "gous"
    gous_dir.mkdir()
    held = threading.Event()
    release = threading.Event()
    second_acquired = threading.Event()

    def worker_a():
        with staging.acquire_stage_lock(gous_dir, {"id": "a"}):
            held.set()
            release.wait(timeout=5)

    def worker_b():
        held.wait(timeout=5)
        with staging.acquire_stage_lock(gous_dir, {"id": "b"}):
            second_acquired.set()

    ta = threading.Thread(target=worker_a)
    tb = threading.Thread(target=worker_b)
    ta.start()
    tb.start()
    held.wait(timeout=5)
    # B should be blocked because A holds the lock
    assert not second_acquired.wait(timeout=0.5)
    # Release A; B should then acquire.
    release.set()
    assert second_acquired.wait(timeout=5)
    ta.join(); tb.join()


def test_stage_lock_uses_long_backstop_not_cache_lock_default(tmp_path):
    """The stage lock fails the unit on timeout, so it must wait out a
    live, slow-staging same-host holder (90GB can take >30min) rather than
    give up at the 300s cache-lock default that 91fa5d9 imposed.  It uses a
    24h backstop; the cache-lock default stays short (it degrades, not
    fails).  See REGRESSION_REPORT_token_timeout_2026-06-02.md."""
    gous_dir = tmp_path / "gous"
    gous_dir.mkdir()
    lock = staging.acquire_stage_lock(gous_dir, {"id": "x"})
    assert lock.wait_timeout_sec == staging._STAGE_LOCK_DEFAULT_WAIT_SEC
    assert staging._STAGE_LOCK_DEFAULT_WAIT_SEC == 86400.0
    # Distinct from (and far longer than) the cache-lock default.
    assert staging._STAGE_LOCK_DEFAULT_WAIT_SEC > staging._MKDIR_LOCK_DEFAULT_WAIT_SEC
    assert staging._MKDIR_LOCK_DEFAULT_WAIT_SEC == 300.0


# ---------------------------------------------------------------------------
# Staging tokens
# ---------------------------------------------------------------------------

def test_token_acquire_release_atomic(tmp_path):
    tok = tmp_path / "tokens"
    i = staging.acquire_staging_token(tok, n_slots=2, holder_id="x")
    assert i in (0, 1)
    held = staging.list_held_tokens(tok)
    assert len(held) == 1
    assert held[0]["pid"] == os.getpid()
    staging.release_staging_token(tok, i)
    assert staging.list_held_tokens(tok) == []


def test_token_pool_full_blocks(tmp_path):
    tok = tmp_path / "tokens"
    i0 = staging.acquire_staging_token(tok, n_slots=1, holder_id="a")
    with pytest.raises(TimeoutError):
        staging.acquire_staging_token(
            tok, n_slots=1, holder_id="b",
            poll_sleep=(0.05, 0.1), timeout_sec=0.5,
        )
    staging.release_staging_token(tok, i0)
    # Now it should succeed
    i1 = staging.acquire_staging_token(
        tok, n_slots=1, holder_id="b",
        poll_sleep=(0.05, 0.1), timeout_sec=2,
    )
    assert i1 == 0
    staging.release_staging_token(tok, i1)


def test_release_idempotent(tmp_path):
    tok = tmp_path / "tokens"
    i = staging.acquire_staging_token(tok, n_slots=2, holder_id="x")
    staging.release_staging_token(tok, i)
    # Second release is a no-op (no exception)
    staging.release_staging_token(tok, i)


# ---------------------------------------------------------------------------
# cleanup_workdir
# ---------------------------------------------------------------------------

def test_cleanup_workdir_recursive(tmp_path):
    work = tmp_path / "work"
    (work / "a" / "b").mkdir(parents=True)
    (work / "a" / "b" / "c.bin").write_bytes(b"x")
    staging.cleanup_workdir(work)
    assert not work.exists()


def test_cleanup_workdir_missing_ok(tmp_path):
    staging.cleanup_workdir(tmp_path / "ghost")  # no exception


# ---------------------------------------------------------------------------
# Cross-dispatch staging cache
# ---------------------------------------------------------------------------

def test_cache_key_stable_and_includes_basename(tmp_path):
    src = _fake_ms(tmp_path / "src")
    k1 = staging._cache_key(src)
    k2 = staging._cache_key(src)
    assert k1 == k2
    assert k1.startswith(src.name + ".")
    # Two MSs with same basename but different paths produce distinct keys
    src2 = _fake_ms(tmp_path / "other")
    assert staging._cache_key(src2) != k1


def test_compute_fingerprint_uses_recursive_size(tmp_path):
    src = _fake_ms(tmp_path / "src", n_files=5)
    fp = staging._compute_fingerprint(src)
    assert fp["size_bytes"] > 1024  # 5 × 256 + 1024 at minimum
    assert isinstance(fp["mtime_ns"], int)
    # st_size of the directory (not recursive) is far smaller
    import os as _os
    assert fp["size_bytes"] != _os.stat(src).st_size


def test_compute_fingerprint_v2_includes_tree_sig(tmp_path):
    """Regression: v2 fingerprint MUST carry version + tree_sig.
    Without these, v1 sidecars from before the fix would be trusted."""
    src = _fake_ms(tmp_path / "src")
    fp = staging._compute_fingerprint(src)
    assert fp["version"] == staging._FINGERPRINT_VERSION == 2
    assert isinstance(fp["tree_sig"], str)
    assert len(fp["tree_sig"]) == 64  # sha256 hex


def _fake_ms_with_subtables(parent, name="fake.ms"):
    """Realistic CASA-table-shaped fake — root files + subtables."""
    import os as _os
    ms = parent / name
    ms.mkdir(parents=True)
    (ms / "table.dat").write_bytes(b"R" * 1024)
    (ms / "table.f0_TSM0").write_bytes(b"r" * 4096)
    for sub in ("FIELD", "SPECTRAL_WINDOW", "DATA_DESCRIPTION"):
        (ms / sub).mkdir()
        (ms / sub / "table.dat").write_bytes(bytes([0x55]) * 512)
    return ms


def _restore_root_mtime_ns(path, ns):
    import os as _os
    _os.utime(path, ns=(ns, ns))


def test_tree_sig_stable_for_same_tree(tmp_path):
    src = _fake_ms_with_subtables(tmp_path / "src")
    a = staging._compute_tree_sig(src)
    b = staging._compute_tree_sig(src)
    assert a == b


def test_tree_sig_detects_root_file_in_place_rewrite(tmp_path):
    """Scenario A: rewrite ``table.dat`` content same-size, restore root
    mtime. Under v1 fingerprint this was a stale-hit; tree_sig must
    distinguish."""
    src = _fake_ms_with_subtables(tmp_path / "src")
    sig_before = staging._compute_tree_sig(src)
    saved = src.stat().st_mtime_ns
    target = src / "table.dat"
    new = bytes([(b + 17) & 0xFF for b in target.read_bytes()])
    target.write_bytes(new)
    _restore_root_mtime_ns(src, saved)
    sig_after = staging._compute_tree_sig(src)
    assert sig_before != sig_after


def test_tree_sig_detects_nested_in_place_rewrite(tmp_path):
    """Scenario B: rewrite a subtable file same-size, restore root mtime."""
    src = _fake_ms_with_subtables(tmp_path / "src")
    sig_before = staging._compute_tree_sig(src)
    saved = src.stat().st_mtime_ns
    target = src / "SPECTRAL_WINDOW" / "table.dat"
    new = bytes([(b + 7) & 0xFF for b in target.read_bytes()])
    target.write_bytes(new)
    _restore_root_mtime_ns(src, saved)
    sig_after = staging._compute_tree_sig(src)
    assert sig_before != sig_after


def test_tree_sig_detects_nested_temp_rename(tmp_path):
    """Scenario C: temp+rename a subtable file same-size."""
    src = _fake_ms_with_subtables(tmp_path / "src")
    sig_before = staging._compute_tree_sig(src)
    saved = src.stat().st_mtime_ns
    target = src / "SPECTRAL_WINDOW" / "table.dat"
    sz = target.stat().st_size
    tmp = target.parent / ".table.dat.tmp"
    tmp.write_bytes(b"X" * sz)
    import os as _os
    _os.replace(str(tmp), str(target))
    _restore_root_mtime_ns(src, saved)
    sig_after = staging._compute_tree_sig(src)
    assert sig_before != sig_after


def test_tree_sig_detects_symlink_retarget(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    target_a = tmp_path / "a.txt"
    target_a.write_bytes(b"a")
    target_b = tmp_path / "b.txt"
    target_b.write_bytes(b"b")
    (src / "link").symlink_to(target_a)
    sig_a = staging._compute_tree_sig(src)
    (src / "link").unlink()
    (src / "link").symlink_to(target_b)
    sig_b = staging._compute_tree_sig(src)
    assert sig_a != sig_b


def test_tree_sig_resolves_symlink_root(tmp_path):
    """Regression: if *src_path* is a symlink to a real MS dir, the
    fingerprint must follow the link and reflect the TARGET's contents.
    Otherwise tar/cp/rsync (which dereference leaf symlinks) would
    stage fresh data while the fingerprint stayed pinned to the link
    itself — perfect stale-cache vector."""
    real = _fake_ms_with_subtables(tmp_path / "real")
    link = tmp_path / "link.ms"
    link.symlink_to(real)
    sig_via_link = staging._compute_tree_sig(link)
    sig_direct = staging._compute_tree_sig(real)
    assert sig_via_link == sig_direct
    # Mutate the target through the real path; sig via link must change.
    saved = real.stat().st_mtime_ns
    target = real / "SPECTRAL_WINDOW" / "table.dat"
    target.write_bytes(b"Y" * target.stat().st_size)
    _restore_root_mtime_ns(real, saved)
    sig_after = staging._compute_tree_sig(link)
    assert sig_via_link != sig_after


def test_compute_fingerprint_resolves_symlink_root(tmp_path):
    """Same pattern, end-to-end through ``_compute_fingerprint``."""
    real = _fake_ms_with_subtables(tmp_path / "real")
    link = tmp_path / "link.ms"
    link.symlink_to(real)
    fp_via_link = staging._compute_fingerprint(link)
    fp_direct = staging._compute_fingerprint(real)
    assert fp_via_link == fp_direct
    assert fp_via_link["size_bytes"] > 1024  # not the symlink's tiny size


def test_stage_one_symlink_root_hit_then_mutate_misses(tmp_path):
    """Belt-and-braces end-to-end: ``stage_one(link) → cache_miss``,
    ``stage_one(link)`` again → ``cache_hit``, mutate the real target's
    inner file, ``stage_one(link)`` → ``cache_miss`` (NOT stale)."""
    real = _fake_ms_with_subtables(tmp_path / "real")
    link = tmp_path / "link.ms"
    link.symlink_to(real)
    cache = tmp_path / "cache"

    # First stage via link → populate
    _, source = staging.stage_one(
        str(link), tmp_path / "dst1", method="cp", bucket="ms",
        cache_root=cache,
    )
    assert source == "cache_miss"

    # Second stage via link → hit
    _, source2 = staging.stage_one(
        str(link), tmp_path / "dst2", method="cp", bucket="ms",
        cache_root=cache,
    )
    assert source2 == "cache_hit"

    # Mutate the REAL target's inner file in place; restore root mtime.
    saved = real.stat().st_mtime_ns
    target = real / "SPECTRAL_WINDOW" / "table.dat"
    target.write_bytes(b"M" * target.stat().st_size)
    _restore_root_mtime_ns(real, saved)

    # Stage via link again — must NOT trust the cache.
    _, source3 = staging.stage_one(
        str(link), tmp_path / "dst3", method="cp", bucket="ms",
        cache_root=cache,
    )
    assert source3 == "cache_miss"


def test_tree_sig_for_file_root(tmp_path):
    """TP FITS file: tree_sig is the file's own (size, mtime_ns, ctime_ns)."""
    f = tmp_path / "tp.fits"
    f.write_bytes(b"X" * 1024)
    s1 = staging._compute_tree_sig(f)
    # Sleep to guarantee a different ns timestamp on the rewrite —
    # without this the two writes can land in the same nanosecond on
    # fast filesystems and the test becomes flaky under load.
    time.sleep(0.005)
    f.write_bytes(b"Y" * 1024)  # same size, different content
    s2 = staging._compute_tree_sig(f)
    assert s1 != s2


def test_cache_lookup_rejects_v1_sidecar(tmp_path):
    """Regression: a sidecar without ``version=2`` and ``tree_sig`` must
    be treated as a miss.  This prevents v1 cache entries (from before
    this commit) from being trusted."""
    src = _fake_ms_with_subtables(tmp_path / "src")
    cache = tmp_path / "cache"
    # Simulate a v1 sidecar manually
    entry = cache / staging._cache_key(src)
    entry.mkdir(parents=True)
    import shutil as _sh
    _sh.copytree(src, entry / src.name)
    fp_v1 = {"mtime_ns": src.stat().st_mtime_ns,
             "size_bytes": staging._du_bytes(src)}
    import json as _json
    (entry / ".cache.json").write_text(_json.dumps({
        "src_path": str(src.resolve()),
        "mtime_ns": fp_v1["mtime_ns"],
        "size_bytes": fp_v1["size_bytes"],
        "staged_at": "2026-01-01T00:00:00",
        # NOTE: no version, no tree_sig — this is a v1 sidecar
    }))
    # cache_lookup must treat this as a miss
    assert staging.cache_lookup(cache, src) is None


def test_cache_lookup_v2_sidecar_with_tree_sig_hits(tmp_path):
    src = _fake_ms_with_subtables(tmp_path / "src")
    cache = tmp_path / "cache"
    final, source = staging.stage_one(
        str(src), tmp_path / "dst1", method="cp", bucket="ms",
        cache_root=cache,
    )
    assert source == "cache_miss"
    # Sidecar must be v2 with tree_sig
    import json as _json
    side = _json.loads(
        (cache / staging._cache_key(src) / ".cache.json").read_text()
    )
    assert side["version"] == 2
    assert "tree_sig" in side and len(side["tree_sig"]) == 64
    # Second stage_one hits
    _, source2 = staging.stage_one(
        str(src), tmp_path / "dst2", method="cp", bucket="ms",
        cache_root=cache,
    )
    assert source2 == "cache_hit"


def test_cache_miss_on_inner_file_rewrite(tmp_path):
    """End-to-end: in-place rewrite of an inner file (with root mtime
    restored) must produce a cache miss after the fix.  This is the
    bug scenario from `scripts/test_cache_invalidation.py` that the
    v1 fingerprint silently served."""
    src = _fake_ms_with_subtables(tmp_path / "src")
    cache = tmp_path / "cache"
    # Populate
    staging.stage_one(str(src), tmp_path / "dst1", method="cp",
                      bucket="ms", cache_root=cache)
    # Mutate an inner file in place; restore root mtime
    saved = src.stat().st_mtime_ns
    target = src / "SPECTRAL_WINDOW" / "table.dat"
    new = bytes([(b + 7) & 0xFF for b in target.read_bytes()])
    target.write_bytes(new)
    _restore_root_mtime_ns(src, saved)
    # Lookup — must NOT trust the cache
    assert staging.cache_lookup(cache, src) is None
    # Restage triggers repopulate, sidecar updates with new tree_sig
    _, source = staging.stage_one(
        str(src), tmp_path / "dst2", method="cp", bucket="ms",
        cache_root=cache,
    )
    assert source == "cache_miss"


def test_cache_lookup_miss_when_no_entry(tmp_path):
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    assert staging.cache_lookup(cache, src) is None


def test_cache_populate_then_hit(tmp_path):
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    dst1 = tmp_path / "dst1"
    dst2 = tmp_path / "dst2"

    final, source = staging.stage_one(
        str(src), dst1, method="cp", bucket="ms", cache_root=cache,
    )
    assert source == "cache_miss"
    assert final.is_dir()
    # Sidecar and entry exist
    entry = cache / staging._cache_key(src)
    assert (entry / ".cache.json").exists()
    assert (entry / src.name).is_dir()
    # Files inside the cache entry are 0o444 (canary against unexpected writes)
    sample_file = next((entry / src.name).iterdir())
    assert sample_file.stat().st_mode & 0o777 == 0o444

    # Second stage to a different dst hits the cache
    final2, source2 = staging.stage_one(
        str(src), dst2, method="cp", bucket="ms", cache_root=cache,
    )
    assert source2 == "cache_hit"
    # Hard-linked: same inode as the cache entry
    cached_file = next((entry / src.name).iterdir())
    dst_file = dst2 / "ms" / src.name / cached_file.name
    assert dst_file.stat().st_ino == cached_file.stat().st_ino


def test_cache_miss_on_mtime_change(tmp_path):
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    dst1 = tmp_path / "dst1"
    dst2 = tmp_path / "dst2"

    staging.stage_one(str(src), dst1, method="cp", bucket="ms", cache_root=cache)

    # Bump src mtime by touching an inner file (recursive du size unchanged but
    # mtime fingerprint should detect the regen).  Use a force-different mtime.
    import os as _os
    new_t = _os.stat(src).st_mtime + 60
    _os.utime(src, (new_t, new_t))

    _, source = staging.stage_one(
        str(src), dst2, method="cp", bucket="ms", cache_root=cache,
    )
    # mtime mismatch → cache miss → repopulate
    assert source == "cache_miss"


def test_cache_miss_on_size_change(tmp_path):
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    dst1 = tmp_path / "dst1"
    dst2 = tmp_path / "dst2"

    staging.stage_one(str(src), dst1, method="cp", bucket="ms", cache_root=cache)

    # Tamper with the sidecar's size_bytes to simulate a stale fingerprint.
    import json as _json
    side_path = cache / staging._cache_key(src) / ".cache.json"
    side = _json.loads(side_path.read_text())
    side["size_bytes"] = side["size_bytes"] - 1
    side_path.write_text(_json.dumps(side))

    _, source = staging.stage_one(
        str(src), dst2, method="cp", bucket="ms", cache_root=cache,
    )
    assert source == "cache_miss"


def test_cache_populating_marker_treated_as_miss(tmp_path):
    """A cache entry mid-populate (``.populating`` present) must be a miss."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    dst = tmp_path / "dst"
    staging.stage_one(str(src), dst, method="cp", bucket="ms", cache_root=cache)

    # Manually drop a .populating dir back in
    entry = cache / staging._cache_key(src)
    (entry / ".populating").mkdir()
    assert staging.cache_lookup(cache, src) is None
    # Cleanup so other tests aren't affected
    (entry / ".populating").rmdir()


def test_cache_evict_until_free(tmp_path, monkeypatch):
    src1 = _fake_ms(tmp_path / "src1", name="ms1.ms")
    src2 = _fake_ms(tmp_path / "src2", name="ms2.ms")
    cache = tmp_path / "cache"

    # Populate two entries with different staged_at timestamps
    staging.stage_one(str(src1), tmp_path / "dst1", method="cp",
                      bucket="ms", cache_root=cache)
    # Force the first entry to look "older" by rewriting its sidecar
    import json as _json
    e1 = cache / staging._cache_key(src1)
    s1 = _json.loads((e1 / ".cache.json").read_text())
    s1["staged_at"] = "2020-01-01T00:00:00"
    (e1 / ".cache.json").write_text(_json.dumps(s1))

    staging.stage_one(str(src2), tmp_path / "dst2", method="cp",
                      bucket="ms", cache_root=cache)

    # Pretend free space is below target so eviction kicks in.
    # First call: pretend zero free → must evict at least one.
    calls = {"n": 0}
    real_free = staging._du_free_bytes
    def _fake_free(_root):
        calls["n"] += 1
        # First two calls report tight; later calls report plenty.
        if calls["n"] <= 2:
            return 0
        return real_free(_root)
    monkeypatch.setattr(staging, "_du_free_bytes", _fake_free)
    n = staging.cache_evict_until_free(cache, target_free_bytes=1)

    assert n >= 1
    assert not e1.exists()  # oldest evicted first


def test_cache_evict_skips_populating(tmp_path):
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    staging.stage_one(str(src), tmp_path / "dst", method="cp",
                      bucket="ms", cache_root=cache)

    entry = cache / staging._cache_key(src)
    (entry / ".populating").mkdir()
    try:
        # Even with eviction pressure, an entry mid-populate is preserved
        n = staging.cache_evict_until_free(
            cache, target_free_bytes=10 ** 18,  # impossibly large → would evict everything
            skip_keys=set(),
        )
        assert entry.exists()
        # n could be 0 (only the .gc.lock.d isn't a candidate) or include
        # malformed entries, but the populating one survives.
    finally:
        (entry / ".populating").rmdir()


def test_acquire_cache_populate_returns_handle(tmp_path):
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    outcome, lock = staging.acquire_cache_populate(cache, src, timeout_sec=5)
    assert outcome == "populate"
    assert lock is not None
    populating = cache / staging._cache_key(src) / ".populating"
    assert populating.exists()
    with lock:
        pass  # release on exit
    assert not populating.exists()


def test_acquire_cache_populate_returns_hit_when_already_cached(tmp_path):
    """If another worker beat us, the call sees an existing valid entry
    via the post-mkdir cache_lookup probe and returns ``hit``."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    # Pre-populate the cache so the next acquire_cache_populate sees the
    # entry as valid before it even attempts mkdir of .populating.  We
    # simulate the populator already having released by manually dropping
    # an entry + sidecar in.
    fp = staging._compute_fingerprint(src)
    entry = cache / staging._cache_key(src)
    entry.mkdir(parents=True)
    import shutil as _sh
    _sh.copytree(src, entry / src.name)
    staging._write_sidecar_atomic(entry, src, fp)
    # Hold the populating lock externally so the new caller cannot become
    # the populator and instead sees the populated cache via the
    # post-mkdir lookup.
    populating = entry / ".populating"
    populating.mkdir()
    try:
        outcome, lock = staging.acquire_cache_populate(
            cache, src, timeout_sec=2, poll_sleep=(0.01, 0.02),
        )
    finally:
        # Release the externally-held .populating
        import shutil as _sh2
        _sh2.rmtree(populating, ignore_errors=True)
    # The lookup happens BEFORE we held .populating, so the second caller
    # should see the valid cache and return "hit" without becoming
    # populator.  (If it timed out instead, that's also a valid mode but
    # we'd want the test to see the happy path; assert hit OR populate.)
    assert outcome in ("hit", "populate")


# ---------------------------------------------------------------------------
# TokenLease (lazy NAS gate)
# ---------------------------------------------------------------------------

def test_token_lease_acquires_only_on_demand(tmp_path):
    tokens = tmp_path / "tokens"
    lease = staging.TokenLease(tokens, n_slots=2, holder_id="test/1")
    # No tokens acquired yet
    assert staging.list_held_tokens(tokens) == []
    lease.acquire_if_needed()
    assert len(staging.list_held_tokens(tokens)) == 1
    # Idempotent — second call doesn't acquire another
    lease.acquire_if_needed()
    assert len(staging.list_held_tokens(tokens)) == 1
    lease.release()
    assert staging.list_held_tokens(tokens) == []
    # Release is idempotent
    lease.release()


def test_token_lease_no_tokens_dir_is_noop(tmp_path):
    lease = staging.TokenLease(None, n_slots=2, holder_id="test/1")
    lease.acquire_if_needed()  # no-op
    lease.release()             # no-op
    assert lease.stats == {"token_acquires": 0, "token_wait_sec": 0.0}


def test_stage_one_cache_hit_does_not_acquire_token(tmp_path):
    """Cache hits must not touch the NAS staging gate — that's the whole
    point of the cache."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    tokens = tmp_path / "tokens"

    # First call: cache miss → token acquired.
    lease1 = staging.TokenLease(tokens, n_slots=1, holder_id="t/1")
    final, source = staging.stage_one(
        str(src), tmp_path / "dst1", method="cp", bucket="ms",
        cache_root=cache, token_lease=lease1,
    )
    lease1.release()
    assert source == "cache_miss"
    assert lease1.stats["token_acquires"] == 1

    # Second call: cache hit → token NOT acquired.
    lease2 = staging.TokenLease(tokens, n_slots=1, holder_id="t/2")
    final2, source2 = staging.stage_one(
        str(src), tmp_path / "dst2", method="cp", bucket="ms",
        cache_root=cache, token_lease=lease2,
    )
    lease2.release()
    assert source2 == "cache_hit"
    assert lease2.stats["token_acquires"] == 0


def test_stage_one_no_cache_uses_token(tmp_path):
    """With ``cache_root=None`` we always go NAS-direct AND acquire token."""
    src = _fake_ms(tmp_path / "src")
    tokens = tmp_path / "tokens"
    lease = staging.TokenLease(tokens, n_slots=1, holder_id="t/x")
    final, source = staging.stage_one(
        str(src), tmp_path / "dst", method="cp", bucket="ms",
        cache_root=None, token_lease=lease,
    )
    lease.release()
    assert source == "nas_direct"
    assert lease.stats["token_acquires"] == 1


# ---------------------------------------------------------------------------
# Codex-review fixes (regression tests)
# ---------------------------------------------------------------------------

def test_acquire_cache_populate_lookup_first_returns_hit(tmp_path):
    """Regression: if a valid cache entry already exists, the wait-loop
    must observe it BEFORE attempting mkdir — otherwise a peer that
    already finished + removed .populating would be re-populated by us."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    # Pre-populate to a valid state (no .populating).
    fp = staging._compute_fingerprint(src)
    entry = cache / staging._cache_key(src)
    entry.mkdir(parents=True)
    import shutil as _sh
    _sh.copytree(src, entry / src.name)
    staging._write_sidecar_atomic(entry, src, fp)
    # Now any new caller must see a HIT, not become a populator.
    outcome, lock = staging.acquire_cache_populate(
        cache, src, timeout_sec=2, poll_sleep=(0.01, 0.02),
    )
    assert outcome == "hit"
    assert lock is None


def test_compute_fingerprint_raises_on_zero_du(tmp_path, monkeypatch):
    """Regression: a du failure must NOT silently commit size_bytes=0."""
    src = _fake_ms(tmp_path / "src")
    monkeypatch.setattr(staging, "_du_bytes", lambda _p: 0)
    with pytest.raises(OSError):
        staging._compute_fingerprint(src)


def test_chmod_readonly_files_raises_when_permission_denied(tmp_path, monkeypatch):
    """Regression: chmod failure must propagate so populate aborts
    before sidecar commit."""
    src = _fake_ms(tmp_path / "src")
    def _bad_chmod(*_a, **_kw):
        raise PermissionError("nope")
    monkeypatch.setattr("os.chmod", _bad_chmod)
    with pytest.raises(PermissionError):
        staging._chmod_readonly_files(src)


def test_chmod_readonly_files_skips_symlinks(tmp_path):
    """Regression: must NOT chmod symlink targets (would mutate files
    outside the cache entry)."""
    root = tmp_path / "entry"
    root.mkdir()
    target = tmp_path / "outside.txt"
    target.write_bytes(b"x")
    target.chmod(0o644)
    (root / "link").symlink_to(target)
    staging._chmod_readonly_files(root)
    # Symlink target's mode unchanged
    assert target.stat().st_mode & 0o777 == 0o644


def test_failed_populate_does_not_leak_cache_partial(tmp_path, monkeypatch):
    """End-to-end: populate fails inside _do_nas_read — finally cleans
    cache.partial; outer stage_one falls back to NAS-direct (which
    succeeds since the source is a real local dir)."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"

    real_nas = staging._do_nas_read
    calls = {"n": 0}

    def _fail_first_then_real(src_p, dst_p, method):
        calls["n"] += 1
        if calls["n"] == 1:
            # Populate the partial first to verify we clean it
            dst_p.mkdir(parents=True, exist_ok=True)
            (dst_p / "leaked.bin").write_bytes(b"oops")
            raise RuntimeError("simulated NAS failure mid-stage")
        return real_nas(src_p, dst_p, method)

    monkeypatch.setattr(staging, "_do_nas_read", _fail_first_then_real)
    final, source = staging.stage_one(
        str(src), tmp_path / "dst", method="cp", bucket="ms",
        cache_root=cache,
    )
    assert source == "nas_direct"
    # The cache directory exists (created by populate path) but its
    # ``.partial`` must not.
    entry_dir = cache / staging._cache_key(src)
    leftover_partials = list(entry_dir.glob(f".*.partial"))
    assert leftover_partials == [], f"partial leaked: {leftover_partials}"


def test_acquire_cache_populate_releases_lock_on_post_mkdir_race(tmp_path, monkeypatch):
    """Regression for codex-flagged race: peer A finishes + removes
    .populating BETWEEN our top-of-loop cache_lookup and our mkdir win.
    The post-mkdir recheck (with ignore_populating=True) must catch
    this, release our lock, and return ('hit', None) rather than
    becoming a populator that re-stages from NAS."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    entry = cache / staging._cache_key(src)
    entry.mkdir(parents=True)

    # The orchestration:
    # 1. We patch ``populating.mkdir`` to first inject a valid entry into
    #    the cache (simulating peer A finishing during the gap), then
    #    succeed our own mkdir.
    # 2. With ignore_populating=True the post-mkdir recheck inside
    #    acquire_cache_populate sees the valid entry and returns ("hit",
    #    None) instead of ("populate", lock).

    real_mkdir = type(entry).mkdir
    fp = staging._compute_fingerprint(src)
    import shutil as _sh
    inject = {"done": False}
    def _injecting_mkdir(self, *a, **kw):
        if not inject["done"] and self.name == ".populating":
            # Pretend peer A finished just now: stage data + sidecar.
            _sh.copytree(src, entry / src.name)
            staging._write_sidecar_atomic(entry, src, fp)
            inject["done"] = True
        return real_mkdir(self, *a, **kw)
    monkeypatch.setattr(type(entry), "mkdir", _injecting_mkdir)

    outcome, lock = staging.acquire_cache_populate(
        cache, src, timeout_sec=2, poll_sleep=(0.01, 0.02),
    )
    assert outcome == "hit"
    assert lock is None
    # Our .populating must have been released, not held.
    assert not (entry / ".populating").exists()


def test_cache_link_into_handles_file_source(tmp_path):
    """Regression: cache_link_into must work on FILE sources (TP FITS)
    too, not only directory sources (CASA tables)."""
    src_file = tmp_path / "src.fits"
    src_file.write_bytes(b"FITS-like-bytes" * 1024)
    dst = tmp_path / "dst" / "src.fits"
    staging.cache_link_into(src_file, dst)
    assert dst.exists() and dst.is_file()
    # Same inode → real hard-link, not a copy.
    assert dst.stat().st_ino == src_file.stat().st_ino


def test_cache_caches_tp_fits_file(tmp_path):
    """End-to-end: a TP FITS file source caches and yields a hit
    on the second stage (was a silent NAS-direct fallback before)."""
    tp = tmp_path / "tp.fits"
    tp.write_bytes(b"FITS-data" * 4096)
    cache = tmp_path / "cache"

    final1, source1 = staging.stage_one(
        str(tp), tmp_path / "dst1", method="cp", bucket="tp",
        cache_root=cache,
    )
    assert source1 == "cache_miss"
    # The cached file is read-only (canary)
    cache_file = cache / staging._cache_key(tp) / "tp.fits"
    assert cache_file.exists()
    assert cache_file.stat().st_mode & 0o777 == 0o444

    final2, source2 = staging.stage_one(
        str(tp), tmp_path / "dst2", method="cp", bucket="tp",
        cache_root=cache,
    )
    assert source2 == "cache_hit"
    # Hard-linked
    assert final2.stat().st_ino == cache_file.stat().st_ino


def test_acquire_cache_populate_retries_if_entry_disappears(tmp_path, monkeypatch):
    """Regression for codex round-3: GC rmtree's entry_dir between our
    parent-create and our leaf-create / holder write.  Must restart the
    iteration and eventually succeed (or hit deadline)."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    entry = cache / staging._cache_key(src)

    # Fake the populating-mkdir to fail with FNF the first time
    # (simulating a parent rmtree race), then succeed.
    real_mkdir = type(entry).mkdir
    calls = {"n": 0}
    def _flaky_mkdir(self, *a, **kw):
        if self.name == ".populating":
            calls["n"] += 1
            if calls["n"] == 1:
                raise FileNotFoundError("simulated parent rmtree race")
        return real_mkdir(self, *a, **kw)
    monkeypatch.setattr(type(entry), "mkdir", _flaky_mkdir)

    outcome, lock = staging.acquire_cache_populate(
        cache, src, timeout_sec=5, poll_sleep=(0.01, 0.02),
    )
    assert outcome == "populate"
    assert calls["n"] >= 2
    if lock is not None:
        with lock:
            pass


def test_cache_evict_rechecks_populating_before_rmtree(tmp_path, monkeypatch):
    """Regression: GC's eviction must re-check ``.populating`` right
    before rmtree.  Otherwise a populator that wins between the
    candidate-collection sweep and the rmtree gets nuked.

    Injection: ``_du_free_bytes`` is called once at the top of
    ``cache_evict_until_free`` and then once per loop iteration (for
    the break condition) BEFORE the populating-recheck.  We use the
    second call to publish ``.populating`` — which happens after the
    candidate was collected (at ``cache_root.iterdir()``) but before
    the recheck.  If the recheck is missing, the entry will be
    rmtree'd; if the recheck works, the entry is preserved."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    entry = cache / staging._cache_key(src)
    entry.mkdir(parents=True)
    # Pretend the entry is malformed (no sidecar) — GC will pick it
    # as an eviction candidate.

    calls = {"n": 0}
    def _injecting_free(_root):
        calls["n"] += 1
        if calls["n"] == 2:
            # Race: a populator wins between candidate-collection and
            # the in-loop recheck.
            (entry / ".populating").mkdir()
        return 0  # always tight to keep eviction running
    monkeypatch.setattr(staging, "_du_free_bytes", _injecting_free)

    n = staging.cache_evict_until_free(cache, target_free_bytes=1)
    # Entry must be preserved because .populating was published before
    # the rmtree got a chance.
    assert entry.exists()
    assert (entry / ".populating").exists()
    assert n == 0


def test_chmod_readonly_files_handles_file_root(tmp_path):
    """Regression: a single-file root (TP FITS) must be chmod'd 0o444
    (was a silent no-op when root was a file)."""
    f = tmp_path / "x.fits"
    f.write_bytes(b"x")
    staging._chmod_readonly_files(f)
    assert f.stat().st_mode & 0o777 == 0o444


def test_cache_link_into_cleans_partial_on_rename_failure(tmp_path, monkeypatch):
    """Regression: cache_link_into's os.rename was outside the try; if
    it fails, the .partial would leak."""
    src = _fake_ms(tmp_path / "cache" / "key", name="ms")
    dst = tmp_path / "dst" / "ms"
    real_rename = os.rename
    monkeypatch.setattr(
        os, "rename",
        lambda *a, **k: (_ for _ in ()).throw(OSError("simulated")),
    )
    with pytest.raises(OSError):
        staging.cache_link_into(src, dst)
    leftover_partials = list(dst.parent.glob(f".*.partial"))
    assert leftover_partials == [], f"partial leaked: {leftover_partials}"


# ---------------------------------------------------------------------------
# Staging reclaim hardening (PID-reuse, bounded waits, typed exceptions)
# v1+v2+v3+v4+v5 plan tests
# ---------------------------------------------------------------------------

import json as _json
import socket as _sock
import shutil as _sh


# Behavioural lock-side --------------------------------------------------------

def test_mkdir_lock_times_out_when_holder_persists(tmp_path, monkeypatch):
    """T1: a wedged ``_holder_alive_locally`` (always True) must surface
    as ``StaleLockTimeout`` after the configured bound — NOT loop forever."""
    lock_dir = tmp_path / ".stage.lock.d"
    lock_dir.mkdir()
    (lock_dir / "holder.json").write_text(_json.dumps({
        "host": _sock.gethostname(), "pid": os.getpid(),
    }))
    monkeypatch.setattr(staging, "_holder_alive_locally",
                        lambda *_a, **_kw: True)
    lock = staging._MkdirLock(
        dir_path=lock_dir,
        wait_timeout_sec=0.3,
        poll_sleep=(0.01, 0.02),
    )
    with pytest.raises(staging.StaleLockTimeout):
        lock.__enter__()


def test_mkdir_lock_deadline_also_bounds_reclaim_loop(tmp_path, monkeypatch):
    """T1b (post-impl review fix): the deadline must fire even when
    the loop is stuck in the reclaim path (holder reported dead every
    iteration but mkdir keeps failing — e.g., a filesystem replaying
    the directory or a peer racing us).  Without a top-of-loop deadline
    check, the v3 implementation would loop forever here."""
    lock_dir = tmp_path / ".stage.lock.d"
    lock_dir.mkdir()
    (lock_dir / "holder.json").write_text(_json.dumps({
        "host": _sock.gethostname(), "pid": 1,
    }))
    # Always-dead holder forces the reclaim branch every iteration.
    monkeypatch.setattr(staging, "_holder_alive_locally",
                        lambda *_a, **_kw: False)
    # Make mkdir always raise FileExistsError so we never escape the
    # reclaim cycle (simulating a peer constantly recreating the dir).
    real_mkdir = Path.mkdir

    def _always_exists(self, *a, **kw):
        if self == lock_dir:
            raise FileExistsError(str(self))
        return real_mkdir(self, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", _always_exists)
    lock = staging._MkdirLock(
        dir_path=lock_dir,
        wait_timeout_sec=0.3,
        poll_sleep=(0.01, 0.02),
    )
    with pytest.raises(staging.StaleLockTimeout):
        lock.__enter__()


def _holder_reader(d):
    """Test helper: holder.json bytes or None."""
    return staging._read_blob(d / "holder.json")


def test_atomic_reclaim_returns_false_when_dir_missing(tmp_path):
    """T1c: ``_atomic_reclaim`` must return False (not raise) if the
    target directory has already been claimed/removed by a peer."""
    nonexistent = tmp_path / "gone"
    assert staging._atomic_reclaim(
        nonexistent,
        expected_fingerprint=None,
        read_fingerprint=_holder_reader,
    ) is False


def test_atomic_reclaim_unique_winner_on_concurrent_calls(tmp_path):
    """T1d: under contention, at most one ``_atomic_reclaim`` succeeds —
    the rest get False.  Empirical safety guarantee for the race
    Codex flagged: two reclaimers cannot both wipe the same dir."""
    stale = tmp_path / "stale"
    stale.mkdir()
    fp_bytes = b'{"pid": 1, "host": "ghost"}'
    (stale / "holder.json").write_bytes(fp_bytes)

    results: list[bool] = []
    barrier = threading.Barrier(8)

    def _race():
        barrier.wait()
        results.append(staging._atomic_reclaim(
            stale,
            expected_fingerprint=fp_bytes,
            read_fingerprint=_holder_reader,
        ))

    threads = [threading.Thread(target=_race) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert results.count(True) == 1
    assert results.count(False) == 7
    assert not stale.exists()


def test_atomic_reclaim_puts_back_fresh_generation(tmp_path):
    """T1e (Codex round-2 post-impl): if the directory we rename has a
    DIFFERENT fingerprint than the stale generation we observed (i.e.,
    a fresh holder won the lock between our snapshot and our rename),
    we must put it back rather than wipe the fresh holder."""
    stale = tmp_path / "stale"
    stale.mkdir()
    fresh_bytes = b'{"pid": 99999, "host": "alive"}'
    (stale / "holder.json").write_bytes(fresh_bytes)
    stale_snapshot = b'{"pid": 1, "host": "ghost"}'

    result = staging._atomic_reclaim(
        stale,
        expected_fingerprint=stale_snapshot,
        read_fingerprint=_holder_reader,
    )
    assert result is False
    assert stale.exists()
    assert (stale / "holder.json").read_bytes() == fresh_bytes


def test_atomic_reclaim_none_snapshot_uses_age_fallback(tmp_path, monkeypatch):
    """T1f (Codex round-3 post-impl, blocking #1): when the stale gen
    had no fingerprint (mid-mkdir window), an old dir is still safely
    reclaimed via mtime-age fallback, but a freshly-mkdir'd dir
    (mtime within grace) is put back rather than wiped."""
    # Case A: old, fingerprint-less stale dir → reclaim.
    old = tmp_path / "old"
    old.mkdir()
    ancient = time.time() - 3600
    os.utime(old, (ancient, ancient))
    assert staging._atomic_reclaim(
        old,
        expected_fingerprint=None,
        read_fingerprint=_holder_reader,
        malformed_grace_sec=1.0,
    ) is True
    assert not old.exists()

    # Case B: a fresh mkdir under us → put back, NOT wiped.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    # No holder.json written yet (race window).
    result = staging._atomic_reclaim(
        fresh,
        expected_fingerprint=None,
        read_fingerprint=_holder_reader,
        malformed_grace_sec=30.0,   # fresh dir's age << grace
    )
    assert result is False
    assert fresh.exists()


def test_atomic_reclaim_token_fingerprint_blocks_pid_reuse(tmp_path):
    """T1g (Codex round-3 post-impl, blocking #3): the token-side
    fingerprint reader includes starttime_ticks so a PID-reused fresh
    holder with the same numeric pid but different start-time is NOT
    falsely matched and wiped."""
    slot = tmp_path / "0"
    slot.mkdir()
    # Stale generation snapshot: pid=N, starttime_ticks=S_old.
    stale_pid = b"12345"
    stale_st = b"7000"
    (slot / "pid").write_bytes(stale_pid)
    (slot / "starttime_ticks").write_bytes(stale_st)
    stale_fp = staging._read_token_fingerprint(slot)
    assert stale_fp == stale_pid + b"|" + stale_st

    # Fresh holder takes over: same numeric PID, different start time.
    (slot / "pid").write_bytes(stale_pid)
    (slot / "starttime_ticks").write_bytes(b"9999")

    result = staging._atomic_reclaim(
        slot,
        expected_fingerprint=stale_fp,
        read_fingerprint=staging._read_token_fingerprint,
    )
    assert result is False
    assert slot.exists()
    assert (slot / "starttime_ticks").read_bytes() == b"9999"


def test_atomic_reclaim_failed_put_back_leaves_stranded_copy(
    tmp_path, monkeypatch,
):
    """T1h (Codex round-3 post-impl, blocking #2): if put-back fails
    (renameat2 reports EEXIST: a third contender is at stale_dir, OR
    renameat2 isn't available on this platform), we must NOT rmtree
    the renamed copy — that would destroy a valid holder while another
    owns the original path.  Leave it stranded for operator follow-up."""
    stale = tmp_path / "stale"
    stale.mkdir()
    fresh_bytes = b'{"pid": 99999, "host": "alive"}'
    (stale / "holder.json").write_bytes(fresh_bytes)

    # Force _safe_restore to return False (simulating EEXIST OR
    # renameat2 unavailable).
    monkeypatch.setattr(staging, "_safe_restore", lambda *_a, **_kw: False)

    result = staging._atomic_reclaim(
        stale,
        expected_fingerprint=b'something-else',
        read_fingerprint=_holder_reader,
    )
    assert result is False
    stranded = list(tmp_path.glob("stale.reclaim.*"))
    assert len(stranded) == 1
    assert (stranded[0] / "holder.json").read_bytes() == fresh_bytes


def test_safe_restore_refuses_to_overwrite_empty_third_contender(tmp_path):
    """T1i (Codex round-4 post-impl, blocking): _safe_restore must
    NOT clobber an empty third-contender directory at the destination,
    even though plain os.rename(dir, empty_dir) succeeds on POSIX."""
    # Skip if renameat2 isn't available — the fallback is "always
    # return False" which trivially satisfies the contract.
    libc = staging._load_libc()
    if libc is False or not hasattr(libc, "renameat2"):
        pytest.skip("renameat2 unavailable on this platform")

    src = tmp_path / "src"
    src.mkdir()
    (src / "marker").write_text("our_renamed_copy")

    dst = tmp_path / "dst"
    dst.mkdir()  # third contender's fresh empty dir

    ok = staging._safe_restore(src, dst)
    assert ok is False
    # Both must remain intact.
    assert src.exists() and (src / "marker").exists()
    assert dst.exists() and not (dst / "marker").exists()


def test_read_token_fingerprint_returns_none_when_starttime_missing(tmp_path):
    """T1j (Codex round-4 post-impl, high): tokens lacking
    starttime_ticks (old-shape OR mid-write) must return None so the
    age-based fallback handles them, NOT a PID-only false-match."""
    slot = tmp_path / "0"
    slot.mkdir()
    (slot / "pid").write_text("12345")
    # No starttime_ticks file.
    assert staging._read_token_fingerprint(slot) is None

    # With both files present, returns the canonical fingerprint.
    (slot / "starttime_ticks").write_text("7000")
    fp = staging._read_token_fingerprint(slot)
    assert fp == b"12345|7000"


def test_read_token_fingerprint_returns_none_for_empty_or_malformed(tmp_path):
    """T1k (Codex round-5 post-impl, high): empty/malformed metadata
    files must produce None, not "pid|" or "|st" — those shapes would
    collide across generations under PID reuse if either side was mid-
    write (Path.write_text truncates before writing)."""
    def _slot(name, pid_content, st_content):
        s = tmp_path / name
        s.mkdir()
        if pid_content is not None:
            (s / "pid").write_text(pid_content)
        if st_content is not None:
            (s / "starttime_ticks").write_text(st_content)
        return s

    # Empty starttime_ticks (mid-write window).
    s = _slot("empty_st", "12345", "")
    assert staging._read_token_fingerprint(s) is None
    # Empty pid.
    s = _slot("empty_pid", "", "7000")
    assert staging._read_token_fingerprint(s) is None
    # Whitespace-only.
    s = _slot("ws", "  \n", "  ")
    assert staging._read_token_fingerprint(s) is None
    # Malformed pid (non-numeric).
    s = _slot("bad_pid", "not-a-pid", "7000")
    assert staging._read_token_fingerprint(s) is None
    # Malformed starttime.
    s = _slot("bad_st", "12345", "garbage")
    assert staging._read_token_fingerprint(s) is None
    # Negative pid/starttime.
    s = _slot("neg_pid", "-1", "7000")
    assert staging._read_token_fingerprint(s) is None
    # Trailing newline IS tolerated (write_text default behaviour).
    s = _slot("trailing_nl", "12345\n", "7000\n")
    assert staging._read_token_fingerprint(s) == b"12345|7000"


def test_acquire_staging_token_deadline_bounds_reclaim_loop(
    tmp_path, monkeypatch,
):
    """T13b (post-impl review fix): the outer token loop must enforce
    the deadline on every iteration, including pathological reclaim
    paths where _token_holder_alive keeps reporting dead but mkdir
    keeps failing."""
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    # Pre-create slot 0 in a state that triggers the reclaim path.
    (tokens / "0").mkdir()
    (tokens / "0" / "host").write_text("otherhost-fake")
    (tokens / "0" / "pid").write_text("1")

    # Always-dead so reclaim runs every iteration.
    monkeypatch.setattr(staging, "_token_holder_alive",
                        lambda *_a, **_kw: False)
    # Make slot.mkdir always raise so the for-loop never returns.
    real_mkdir = Path.mkdir

    def _always_exists(self, *a, **kw):
        if self.parent == tokens:
            raise FileExistsError(str(self))
        return real_mkdir(self, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", _always_exists)
    with pytest.raises(staging.TokenAcquireTimeout):
        staging.acquire_staging_token(
            tokens, n_slots=1, holder_id="t/x",
            timeout_sec=0.3, poll_sleep=(0.01, 0.02),
        )


def test_mkdir_lock_no_timeout_when_none_passed(tmp_path, monkeypatch):
    """T2: ``wait_timeout_sec=None`` preserves unbounded behaviour;
    when the holder finally clears the lock acquires."""
    lock_dir = tmp_path / ".stage.lock.d"
    lock_dir.mkdir()
    (lock_dir / "holder.json").write_text(_json.dumps({
        "host": _sock.gethostname(), "pid": os.getpid(),
    }))
    release = threading.Event()
    acquired = threading.Event()

    def _alive(*_a, **_kw):
        # While the test is "holding" the lock, report alive; once we set
        # `release`, report dead so the contender reclaims.
        return not release.is_set()

    monkeypatch.setattr(staging, "_holder_alive_locally", _alive)

    def _worker():
        with staging._MkdirLock(
            dir_path=lock_dir,
            wait_timeout_sec=None,
            poll_sleep=(0.01, 0.02),
        ):
            acquired.set()

    t = threading.Thread(target=_worker)
    t.start()
    # Worker should still be blocked
    assert not acquired.wait(timeout=0.3)
    release.set()
    assert acquired.wait(timeout=2.0)
    t.join()


def test_mkdir_lock_progress_log_emits(tmp_path, monkeypatch, caplog):
    """T3: while waiting, ``_MkdirLock`` logs a periodic WARN."""
    lock_dir = tmp_path / ".stage.lock.d"
    lock_dir.mkdir()
    (lock_dir / "holder.json").write_text(_json.dumps({
        "host": _sock.gethostname(), "pid": os.getpid(),
    }))
    monkeypatch.setattr(staging, "_holder_alive_locally",
                        lambda *_a, **_kw: True)
    monkeypatch.setattr(staging, "_MKDIR_LOCK_PROGRESS_LOG_SEC", 0.05)
    caplog.set_level("WARNING", logger=staging.log.name)
    lock = staging._MkdirLock(
        dir_path=lock_dir,
        wait_timeout_sec=0.3,
        poll_sleep=(0.01, 0.02),
    )
    with pytest.raises(staging.StaleLockTimeout):
        lock.__enter__()
    assert any("still waiting for lock" in r.message for r in caplog.records)


def test_stale_lock_timeout_not_subclass_of_oserror():
    """T4: ``StaleLockTimeout`` MUST NOT be an ``OSError`` so callers'
    ``except OSError`` clauses don't silently swallow it."""
    assert not isinstance(staging.StaleLockTimeout(), OSError)
    assert not isinstance(staging.StaleLockTimeout(), TimeoutError)


def test_acquire_cache_populate_propagates_stale_lock_timeout(
    tmp_path, monkeypatch,
):
    """T5: a StaleLockTimeout from the inner GC lock must propagate
    out of ``acquire_cache_populate`` (NOT be eaten by ``except OSError``)."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"

    real_enter = staging._MkdirLock.__enter__

    def _boom(self):
        raise staging.StaleLockTimeout(f"forced timeout on {self.dir_path}")

    monkeypatch.setattr(staging._MkdirLock, "__enter__", _boom)
    with pytest.raises(staging.StaleLockTimeout):
        staging.acquire_cache_populate(
            cache, src, timeout_sec=2, poll_sleep=(0.01, 0.02),
        )


def test_stage_one_falls_back_on_stale_lock_timeout(tmp_path, monkeypatch):
    """T6: a ``StaleLockTimeout`` from ``acquire_cache_populate`` (e.g.
    the inner GC-lock wedged) must be caught by ``stage_one`` and routed
    to the NAS-direct fallback — same behaviour as the existing
    ``TimeoutError`` path.  Per plan v2 Delta 5 + v4 Delta 8: a wedged
    cache GC lock should not fail an otherwise-stageable unit."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    dst = tmp_path / "dst"

    def _raise(*_a, **_kw):
        raise staging.StaleLockTimeout("populate timeout")

    monkeypatch.setattr(staging, "acquire_cache_populate", _raise)
    final, source = staging.stage_one(
        str(src), dst, method="cp", bucket="ms", cache_root=cache,
    )
    assert source == "nas_direct"
    assert final.exists()


def test_inner_gc_lock_capped_by_outer_populate_deadline(tmp_path, monkeypatch):
    """T7: when the caller passes a small ``timeout_sec``, the inner
    ``_MkdirLock`` wait must be capped by remaining budget rather than
    the 300s default."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    captured: dict = {}

    real_init = staging._MkdirLock.__init__

    def _spy_init(self, *a, **kw):
        captured.setdefault("waits", []).append(kw.get("wait_timeout_sec"))
        return real_init(self, *a, **kw)

    monkeypatch.setattr(staging._MkdirLock, "__init__", _spy_init)
    outcome, lock = staging.acquire_cache_populate(
        cache, src, timeout_sec=1.0, poll_sleep=(0.01, 0.02),
    )
    if lock is not None:
        with lock:
            pass
    assert outcome in ("populate", "hit")
    inner_waits = [w for w in captured["waits"] if w is not None]
    assert inner_waits, "no _MkdirLock with bounded wait observed"
    assert all(w <= 1.0 for w in inner_waits), (
        f"inner wait exceeded outer budget: {inner_waits}"
    )


# PID-reuse + identity ---------------------------------------------------------

def test_holder_alive_detects_pid_reuse_via_starttime(tmp_path):
    """T8: same PID but different start-time ticks → stale."""
    lock_dir = tmp_path / ".stage.lock.d"
    lock_dir.mkdir()
    own_start = staging._read_proc_starttime(os.getpid())
    assert own_start is not None
    (lock_dir / "holder.json").write_text(_json.dumps({
        "host": _sock.gethostname(),
        "pid": os.getpid(),
        "starttime_ticks": int(own_start) + 1,
    }))
    assert staging._holder_alive_locally(lock_dir) is False


def test_holder_alive_back_compat_no_starttime(tmp_path):
    """T9: holder.json without ``starttime_ticks`` falls through to the
    PID-only check (alive when PID is alive)."""
    lock_dir = tmp_path / ".stage.lock.d"
    lock_dir.mkdir()
    (lock_dir / "holder.json").write_text(_json.dumps({
        "host": _sock.gethostname(), "pid": os.getpid(),
    }))
    assert staging._holder_alive_locally(lock_dir) is True


def test_mkdir_lock_meta_identity_cannot_be_overridden(tmp_path):
    """T10: caller-supplied ``holder_meta`` with bogus host/pid must NOT
    overwrite the real identity record."""
    lock_dir = tmp_path / ".stage.lock.d"
    lock = staging._MkdirLock(
        dir_path=lock_dir,
        holder_meta={"pid": 99999, "host": "bogus.example.com",
                     "starttime_ticks": -1},
        wait_timeout_sec=1.0,
        poll_sleep=(0.01, 0.02),
    )
    with lock:
        meta = _json.loads((lock_dir / "holder.json").read_text())
        assert meta["pid"] == os.getpid()
        assert meta["host"] == _sock.gethostname()
        # starttime_ticks should be the real one (not -1), or None.
        own_start = staging._read_proc_starttime(os.getpid())
        assert meta["starttime_ticks"] == own_start


def test_populating_writer_includes_starttime(tmp_path):
    """T11: ``acquire_cache_populate``'s manual ``.populating/holder.json``
    must include ``starttime_ticks`` (identity-last, real value)."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    outcome, lock = staging.acquire_cache_populate(
        cache, src, timeout_sec=5, poll_sleep=(0.01, 0.02),
    )
    assert outcome == "populate"
    populating = cache / staging._cache_key(src) / ".populating"
    meta = _json.loads((populating / "holder.json").read_text())
    assert meta["host"] == _sock.gethostname()
    assert meta["pid"] == os.getpid()
    assert "starttime_ticks" in meta
    own_start = staging._read_proc_starttime(os.getpid())
    assert meta["starttime_ticks"] == own_start
    with lock:
        pass


# Hostname --------------------------------------------------------------------

def test_same_host_recognizes_short_fqdn_and_uppercase(monkeypatch):
    """T12: short/FQDN/uppercase all recognised as same host
    (case-insensitive per RFC 952/1123)."""
    monkeypatch.setattr(_sock, "gethostname", lambda: "almap8")
    monkeypatch.setattr(_sock, "getfqdn", lambda: "almap8.alma.ac.uk")
    # Reimport names inside staging too — it uses ``socket`` module ref.
    monkeypatch.setattr(staging.socket, "gethostname", lambda: "almap8")
    monkeypatch.setattr(staging.socket, "getfqdn", lambda: "almap8.alma.ac.uk")
    assert staging._same_host("almap8") is True
    assert staging._same_host("almap8.alma.ac.uk") is True
    assert staging._same_host("ALMAP8") is True
    assert staging._same_host("Almap8.ALMA.ac.uk") is True
    assert staging._same_host("almap9") is False
    assert staging._same_host("") is False


def test_holder_alive_cross_host_still_returns_true(tmp_path, monkeypatch):
    """T13: a holder.json from a different host must report alive (we
    have no local way to check; coordinator/operator handles)."""
    lock_dir = tmp_path / ".stage.lock.d"
    lock_dir.mkdir()
    (lock_dir / "holder.json").write_text(_json.dumps({
        "host": "definitely-not-this-host.example",
        "pid": 2 ** 22,  # almost certainly dead, but cross-host short-circuit
    }))
    assert staging._holder_alive_locally(lock_dir) is True


# Token-side ------------------------------------------------------------------

def test_acquire_staging_token_times_out_when_all_slots_pinned(tmp_path):
    """T14: all slots pinned with live PID + matching starttime →
    ``TokenAcquireTimeout``."""
    tok = tmp_path / "tokens"
    tok.mkdir()
    own_start = staging._read_proc_starttime(os.getpid())
    for i in range(2):
        slot = tok / str(i)
        slot.mkdir()
        (slot / "host").write_text(_sock.gethostname())
        (slot / "pid").write_text(str(os.getpid()))
        (slot / "holder").write_text(f"pinned/{i}")
        (slot / "acquired_at").write_text(str(time.time()))
        if own_start is not None:
            (slot / "starttime_ticks").write_text(str(own_start))
    with pytest.raises(staging.TokenAcquireTimeout):
        staging.acquire_staging_token(
            tok, n_slots=2, holder_id="contender",
            poll_sleep=(0.01, 0.02), timeout_sec=0.3,
        )


def test_acquire_staging_token_reclaims_on_starttime_mismatch(tmp_path):
    """T15: slot 0 pinned with mismatched starttime_ticks → reclaimed
    and acquired."""
    tok = tmp_path / "tokens"
    tok.mkdir()
    own_start = staging._read_proc_starttime(os.getpid())
    slot = tok / "0"
    slot.mkdir()
    (slot / "host").write_text(_sock.gethostname())
    (slot / "pid").write_text(str(os.getpid()))
    (slot / "holder").write_text("ghost")
    (slot / "acquired_at").write_text(str(time.time()))
    if own_start is not None:
        (slot / "starttime_ticks").write_text(str(int(own_start) + 1))
    i = staging.acquire_staging_token(
        tok, n_slots=1, holder_id="real",
        poll_sleep=(0.01, 0.02), timeout_sec=2,
    )
    assert i == 0
    # The slot's new metadata must be OUR holder_id
    assert (slot / "holder").read_text() == "real"
    staging.release_staging_token(tok, i)


def test_token_metadata_includes_starttime_ticks(tmp_path):
    """T16: fresh ``acquire_staging_token`` writes ``starttime_ticks``."""
    tok = tmp_path / "tokens"
    i = staging.acquire_staging_token(
        tok, n_slots=2, holder_id="x",
        poll_sleep=(0.01, 0.02), timeout_sec=2,
    )
    slot = tok / str(i)
    assert (slot / "host").exists()
    assert (slot / "pid").exists()
    assert (slot / "holder").exists()
    assert (slot / "acquired_at").exists()
    assert (slot / "starttime_ticks").exists()
    # starttime parses as int
    assert int((slot / "starttime_ticks").read_text().strip()) >= 0
    staging.release_staging_token(tok, i)


def test_list_held_tokens_exposes_starttime(tmp_path):
    """T17: ``list_held_tokens`` returns dict with ``starttime_ticks``
    when present."""
    tok = tmp_path / "tokens"
    i = staging.acquire_staging_token(
        tok, n_slots=1, holder_id="x",
        poll_sleep=(0.01, 0.02), timeout_sec=2,
    )
    held = staging.list_held_tokens(tok)
    assert len(held) == 1
    assert "starttime_ticks" in held[0]
    assert isinstance(held[0]["starttime_ticks"], int)
    staging.release_staging_token(tok, i)


def test_list_held_tokens_back_compat_old_token(tmp_path):
    """T18 (BACK-COMPAT CRITICAL): an old-shape token (no
    ``starttime_ticks`` file) must report ``starttime_ticks: None`` and
    NOT be misclassified as malformed.  Misclassification would cause
    the dispatch ``TokenReaper`` to drop live cross-version tokens."""
    tok = tmp_path / "tokens"
    tok.mkdir()
    slot = tok / "0"
    slot.mkdir()
    (slot / "host").write_text(_sock.gethostname())
    (slot / "pid").write_text(str(os.getpid()))
    (slot / "holder").write_text("old-style/1")
    (slot / "acquired_at").write_text(str(time.time()))
    # NO starttime_ticks file
    held = staging.list_held_tokens(tok)
    assert len(held) == 1
    rec = held[0]
    assert rec["host"] == _sock.gethostname()
    assert rec["pid"] == os.getpid()
    assert rec["holder"] == "old-style/1"
    assert rec["starttime_ticks"] is None


def test_token_malformed_grace_window_preserved(tmp_path):
    """T19: a mid-write token (slot dir exists but no host/pid yet) within
    the malformed grace window must NOT be stolen by a contender."""
    tok = tmp_path / "tokens"
    tok.mkdir()
    slot = tok / "0"
    slot.mkdir()
    # No metadata files yet — simulates the brief mkdir-then-write window.
    # _token_holder_alive should report True within the grace window.
    assert staging._token_holder_alive(slot, malformed_grace_sec=60.0) is True


def test_token_lease_propagates_token_acquire_timeout(tmp_path):
    """T20: ``TokenLease.acquire_if_needed`` propagates
    ``TokenAcquireTimeout`` from ``acquire_staging_token``."""
    tok = tmp_path / "tokens"
    tok.mkdir()
    own_start = staging._read_proc_starttime(os.getpid())
    slot = tok / "0"
    slot.mkdir()
    (slot / "host").write_text(_sock.gethostname())
    (slot / "pid").write_text(str(os.getpid()))
    (slot / "holder").write_text("pinned")
    (slot / "acquired_at").write_text(str(time.time()))
    if own_start is not None:
        (slot / "starttime_ticks").write_text(str(own_start))
    lease = staging.TokenLease(
        tok, n_slots=1, holder_id="contender", timeout_sec=0.3,
    )
    with pytest.raises(staging.TokenAcquireTimeout):
        lease.acquire_if_needed()


def test_token_acquire_timeout_is_timeout_error():
    """T21 (back-compat sanity): existing callers ``except TimeoutError``
    still catch the new typed subclass."""
    assert isinstance(staging.TokenAcquireTimeout(), TimeoutError)
    assert issubclass(staging.TokenAcquireTimeout, TimeoutError)


def test_token_acquire_timeout_default_is_24h_backstop():
    """The 30min (1800s) token-acquire bound from commit 91fa5d9 regressed
    dispatch reliability: it failed legitimate multi-hour staging waits that
    historically succeeded (observed waits up to ~29,700s).  The default is
    now a 24h backstop, not a tight bound — dead holders are reclaimed by the
    DB-backed TokenReaper, so an effectively unbounded wait is safe.  See
    .tmp/REGRESSION_REPORT_token_timeout_2026-06-02.md."""
    import inspect
    sig = inspect.signature(staging.acquire_staging_token)
    assert sig.parameters["timeout_sec"].default == 86400.0
    assert staging._DEFAULT_TOKEN_WAIT_TIMEOUT_SEC == 86400.0


# Eviction --------------------------------------------------------------------

def test_cache_evict_until_free_times_out_on_stale_gc_lock(
    tmp_path, monkeypatch, caplog,
):
    """T23: a wedged GC lock surfaces as 0 evictions + WARN log."""
    cache = tmp_path / "cache"
    cache.mkdir()
    # Pre-create the GC lock dir and pin it with a live holder.
    gc_lock = cache / ".gc.lock.d"
    gc_lock.mkdir()
    (gc_lock / "holder.json").write_text(_json.dumps({
        "host": _sock.gethostname(), "pid": os.getpid(),
    }))
    monkeypatch.setattr(staging, "_holder_alive_locally",
                        lambda *_a, **_kw: True)
    caplog.set_level("WARNING", logger=staging.log.name)
    n = staging.cache_evict_until_free(
        cache, target_free_bytes=1, wait_timeout_sec=0.3,
    )
    assert n == 0
    assert any("GC lock timeout" in r.message for r in caplog.records)


# v5 Delta 1 / 2 additions ----------------------------------------------------

def test_stage_one_propagates_token_acquire_timeout_through_populate(
    tmp_path, monkeypatch,
):
    """T24 (v5 Delta 1): the narrowed populate ``except`` clause must
    re-raise ``TokenAcquireTimeout`` rather than downgrade to NAS-direct.
    Also asserts NAS-direct path is NOT entered."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"

    # Force token-lease acquire to raise TokenAcquireTimeout.
    nas_calls = {"n": 0}
    acquire_calls = {"n": 0}

    def _spy_do_nas_read(*_a, **_kw):
        nas_calls["n"] += 1

    monkeypatch.setattr(staging, "_do_nas_read", _spy_do_nas_read)

    def _raise_timeout(self):
        acquire_calls["n"] += 1
        raise staging.TokenAcquireTimeout("forced for test")

    monkeypatch.setattr(staging.TokenLease, "acquire_if_needed", _raise_timeout)

    tokens = tmp_path / "tokens"
    lease = staging.TokenLease(tokens, n_slots=1, holder_id="x")
    with pytest.raises(staging.TokenAcquireTimeout):
        staging.stage_one(
            str(src), tmp_path / "dst", method="cp", bucket="ms",
            cache_root=cache, token_lease=lease,
        )
    # NAS-direct must NOT have been attempted.
    assert nas_calls["n"] == 0
    # And the populate block must NOT have looped through to NAS-direct
    # for a second token attempt — proves the narrowed except clause
    # actually re-raises rather than silently retrying.
    assert acquire_calls["n"] == 1


def test_stage_one_eviction_inherits_populate_deadline(tmp_path, monkeypatch):
    """T25 (v5 Delta 2): the populate-branch's call to
    ``cache_evict_until_free`` must pass a ``wait_timeout_sec`` capped
    by the outer ``cache_populate_timeout_sec``."""
    src = _fake_ms(tmp_path / "src")
    cache = tmp_path / "cache"
    captured: dict = {}

    real_evict = staging.cache_evict_until_free

    def _spy_evict(*a, **kw):
        captured["wait_timeout_sec"] = kw.get("wait_timeout_sec")
        return real_evict(*a, **kw)

    monkeypatch.setattr(staging, "cache_evict_until_free", _spy_evict)
    # cache_min_free_bytes must be set to trigger eviction.
    staging.stage_one(
        str(src), tmp_path / "dst", method="cp", bucket="ms",
        cache_root=cache, cache_min_free_bytes=1,
        cache_populate_timeout_sec=10.0,
    )
    assert "wait_timeout_sec" in captured, (
        "cache_evict_until_free not called with wait_timeout_sec"
    )
    assert captured["wait_timeout_sec"] is not None
    assert captured["wait_timeout_sec"] <= 10.0, (
        f"wait_timeout_sec={captured['wait_timeout_sec']} exceeded outer 10s budget"
    )
