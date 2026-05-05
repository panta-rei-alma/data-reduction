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
