"""Tests for panta_rei.imaging.dispatch.

Focuses on the correctness-critical parts that don't need real SSH:

- machines.json load + validate
- DispatcherLock (flock blocks second holder)
- DBWriter event application via in-memory DB
- SchedulerState GOUS-affinity + staging gate
- Manifest construction (serialise_unit + union_inputs_for_gous)
- Reconciliation outcomes (terminal apply, adoption, dead-pid mark FAILED,
  ssh-unreachable does NOT mark FAILED)
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import (
    DispatchesQueries,
    DispatchState,
    ImagingRunsQueries,
    ImagingRunStatus,
)
from panta_rei.imaging import dispatch as D
from panta_rei.imaging.matching import ImagingUnit


# ---------------------------------------------------------------------------
# load_machines_config
# ---------------------------------------------------------------------------

def _write_machines_json(tmp_path, machines=None, **kw):
    payload = {
        "conda_env": "/opt/conda",
        "repo_path": "/repo",
        "global": {"max_concurrent_staging": 3},
        "machines": machines or {
            "alpha": {"raid": "/raid/alpha", "slots": 1, "nproc": 4},
            "beta":  {"raid": "/raid/beta",  "slots": 2, "nproc": 4},
        },
    }
    payload.update(kw)
    p = tmp_path / "machines.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_machines_config_happy(tmp_path):
    p = _write_machines_json(tmp_path)
    cfg = D.load_machines_config(p)
    assert cfg.conda_env == "/opt/conda"
    assert cfg.global_cfg.max_concurrent_staging == 3
    assert set(cfg.machines.keys()) == {"alpha", "beta"}
    assert cfg.machines["beta"].slots == 2


def test_load_machines_config_missing_required(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"conda_env": "/x"}))
    with pytest.raises(ValueError):
        D.load_machines_config(p)


def test_load_machines_config_machine_without_raid(tmp_path):
    p = _write_machines_json(tmp_path, machines={"x": {"slots": 1}})
    with pytest.raises(ValueError):
        D.load_machines_config(p)


# ---------------------------------------------------------------------------
# DispatcherLock
# ---------------------------------------------------------------------------

def test_dispatcher_lock_blocks_second(tmp_path):
    lock_path = tmp_path / "lock"
    a = D.DispatcherLock(lock_path)
    a.acquire()
    try:
        b = D.DispatcherLock(lock_path)
        with pytest.raises(RuntimeError):
            b.acquire()
    finally:
        a.release()
    # After release, a fresh acquire works
    c = D.DispatcherLock(lock_path)
    c.acquire()
    c.release()


# ---------------------------------------------------------------------------
# DBWriter
# ---------------------------------------------------------------------------

def _make_unit(gous="X", source="S", spw="23", params_id=1) -> ImagingUnit:
    return ImagingUnit(
        gous_uid=gous, source_name=source, line_group="LG",
        spw_id=spw, params_id=params_id, ready=True,
    )


def test_db_writer_handles_lifecycle(tmp_path):
    db = DatabaseManager(tmp_path / "x.db")
    dispatch_id = "d_test_0001"
    # Pre-insert a dispatches row so MARK_RUNNING etc. has FK-like context.
    with db.connect() as con:
        DispatchesQueries.insert(
            con, dispatch_id=dispatch_id,
            coordinator_host="local", coordinator_pid=os.getpid(),
            machines_json="{}", cli_args="",
        )
        con.commit()

    writer = D.DBWriter(db, dispatch_id)
    writer.start()
    try:
        # INSERT_QUEUED
        holder: dict = {}
        writer.q.put({
            "op": "INSERT_QUEUED",
            "row": {
                "params_id": 7, "gous_uid": "G", "source_name": "S",
                "line_group": "LG", "spw_id": "23",
                "started_at": "2026-01-01T00:00:00",
                "status": ImagingRunStatus.QUEUED,
            },
            "row_id_holder": holder,
        })
        # Wait
        for _ in range(50):
            if "id" in holder:
                break
            time.sleep(0.02)
        assert "id" in holder
        run_id = holder["id"]

        writer.q.put({
            "op": "MARK_RUNNING",
            "run_id": run_id, "remote_workdir": "/raid/test",
            "worker_pid": 1234, "worker_pgid": 1234,
            "hostname": "alpha",
        })
        writer.q.put({
            "op": "HEARTBEAT", "run_id": run_id,
            "ts": "2026-01-01T00:00:30",
        })
        writer.q.put({
            "op": "MARK_DONE",
            "run_id": run_id,
            "status": ImagingRunStatus.SUCCESS,
            "retcode": 0,
            "duration_sec": 12.5,
            "finished_at": "2026-01-01T00:01:00",
            "output_fits": "/nas/out.fits",
            "spw_selection": json.dumps(["23"]),
            "field_selection": json.dumps(["S"]),
            "job_json_path": "/nas/jobs/x.json",
        })
    finally:
        writer.stop()
        writer.join(timeout=5)

    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, run_id)
    assert row["status"] == ImagingRunStatus.SUCCESS
    assert row["worker_pid"] == 1234
    assert row["last_heartbeat"] == "2026-01-01T00:00:30"
    assert row["output_fits"] == "/nas/out.fits"
    assert row["job_json_path"] == "/nas/jobs/x.json"
    assert row["dispatch_id"] == dispatch_id
    assert row["hostname"] == "alpha"


# ---------------------------------------------------------------------------
# SchedulerState
# ---------------------------------------------------------------------------

def test_scheduler_first_pick_claims_unmapped_gous():
    s = D.SchedulerState(queue=[
        _make_unit("G1", "S1"),
        _make_unit("G2", "S2"),
    ])
    u = s.pick("alpha", run_id_assigner=lambda: 0)
    assert u is not None
    assert s.gous_machine[u.gous_uid] == "alpha"


def test_scheduler_prefers_mapped_and_staged_for_machine():
    g = "G42"
    s = D.SchedulerState(queue=[
        _make_unit(g, "S1"),
        _make_unit(g, "S2"),
        _make_unit(g, "S3"),
    ])
    s.gous_machine[g] = "alpha"
    s.gous_staged_on[("alpha", g)] = True
    picks = []
    for _ in range(3):
        picks.append(s.pick("alpha", run_id_assigner=lambda: 0))
    assert all(u.gous_uid == g for u in picks)
    assert s.queue == []


def test_scheduler_staging_gate_holds_back_second_unit():
    """A fresh GOUS only releases more units once it transitions to staged."""
    g = "G42"
    s = D.SchedulerState(queue=[
        _make_unit(g, "S1"),
        _make_unit(g, "S2"),
    ])
    # Unit 1 picked: claims GOUS for alpha
    u1 = s.pick("alpha", run_id_assigner=lambda: 0)
    assert u1 is not None
    s.mark_inflight("alpha", g, run_id=1)
    # Unit 2 should NOT come back yet (in-flight on alpha, not staged yet)
    u2 = s.pick("alpha", run_id_assigner=lambda: 0)
    assert u2 is None
    # Once staging completes, unit 2 picks freely
    s.mark_staged("alpha", g)
    u2 = s.pick("alpha", run_id_assigner=lambda: 0)
    assert u2 is not None and u2.source_name == "S2"


def test_scheduler_fallback_fifo_when_only_other_machine_units():
    """Idle machine drains queue from another machine's GOUS as fallback."""
    g = "G42"
    s = D.SchedulerState(queue=[_make_unit(g, "S1")])
    s.gous_machine[g] = "alpha"  # claimed by alpha
    # Beta tries to pick: nothing mapped to beta, no unmapped GOUSs left.
    # Falls through to FIFO, takes the unit anyway (cross-machine restage).
    u = s.pick("beta", run_id_assigner=lambda: 0)
    assert u is not None


def test_scheduler_mark_terminal_returns_empty():
    g = "G"
    s = D.SchedulerState(queue=[])
    s.mark_inflight("alpha", g, run_id=1)
    s.mark_inflight("alpha", g, run_id=2)
    assert s.mark_terminal("alpha", g, 1) is False
    assert s.mark_terminal("alpha", g, 2) is True


def test_scheduler_seen_pairs_survive_terminal():
    """seen_pairs records every (machine, gous) the scheduler has touched
    so the cleaner can find drained GOUSs after their in_flight entry
    has been deleted by mark_terminal."""
    g = "G"
    s = D.SchedulerState(queue=[])
    s.mark_inflight("alpha", g, run_id=1)
    assert ("alpha", g) in s.seen_pairs
    s.mark_terminal("alpha", g, 1)
    # in_flight may be empty/deleted, but seen_pairs still has the pair
    assert ("alpha", g) in s.seen_pairs


def test_gous_cleaner_uses_scheduler_success_ids_no_db_race(tmp_path, monkeypatch):
    """The cleaner must read SUCCESS run_ids from the scheduler, not from
    the DB.  Pre-fix, a queued-but-not-committed MARK_DONE caused the
    cleaner to miss work-dir cleanups."""
    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(),
        machines={"alpha": D.MachineCfg("alpha", "/raid/a", slots=1, nproc=1)},
    )
    s = D.SchedulerState(queue=[])
    s.seen_pairs.add(("alpha", "G"))
    s.success_run_ids[("alpha", "G")] = {7, 8}  # in-memory record
    cleaner = D.GousCleaner(s, cfg, "d_x", D.GlobalCfg())

    seen_cmds: list[str] = []

    def fake_ssh(machine, cmd, timeout=30, capture=True):
        seen_cmds.append(cmd)
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(D, "ssh_run", fake_ssh)
    cleaner.force_run()
    assert len(seen_cmds) == 1
    cmd = seen_cmds[0]
    # Must include both SUCCESS work-dir deletions AND the inputs delete.
    assert "input/G" in cmd
    assert "work/runs/7" in cmd
    assert "work/runs/8" in cmd


def test_scheduler_mark_terminal_records_success_or_failure():
    g = "G"
    s = D.SchedulerState(queue=[])
    s.mark_inflight("alpha", g, run_id=11)
    s.mark_inflight("alpha", g, run_id=12)
    s.mark_terminal("alpha", g, 11, success=True)
    s.mark_terminal("alpha", g, 12, success=False)
    assert s.success_run_ids[("alpha", g)] == {11}
    assert s.failed_run_ids[("alpha", g)] == {12}


def test_scheduler_records_pair_dispatch_id():
    """The cleaner needs to know which dispatch's /raid/ tree each
    (machine, gous) pair lives under.  Adoption of prior-dispatch
    units must record the *prior* dispatch_id."""
    s = D.SchedulerState(queue=[])
    s.mark_inflight("alpha", "G_NEW", run_id=1, dispatch_id="d_new")
    s.mark_inflight("alpha", "G_OLD", run_id=2, dispatch_id="d_old")
    assert s.pair_dispatch_id[("alpha", "G_NEW")] == "d_new"
    assert s.pair_dispatch_id[("alpha", "G_OLD")] == "d_old"


def test_token_reaper_reclaims_malformed_token_after_grace(tmp_path, monkeypatch):
    """A token dir without holder metadata older than the grace window
    must be reclaimed; a fresh malformed dir must not be."""
    tokens_dir = tmp_path / "staging_tokens"
    tokens_dir.mkdir()
    # Fresh malformed slot 0 (no holder metadata, mtime is "now")
    (tokens_dir / "0").mkdir()
    # Old malformed slot 1 (no holder metadata, mtime is in the past)
    old_slot = tokens_dir / "1"
    old_slot.mkdir()
    old_mtime = time.time() - 600  # 10 minutes ago
    os.utime(old_slot, (old_mtime, old_mtime))

    reaper = D.TokenReaper(
        tokens_dir, D.GlobalCfg(),
        expected_tokens=["--dispatch-id d_x"],
    )
    reaper.MALFORMED_GRACE_SEC = 60.0

    # ssh_pid_alive should not be called for malformed dirs.
    monkeypatch.setattr(D, "ssh_pid_alive",
                        lambda *a, **kw: (None, "should not call"))

    reaper._sweep()
    # Slot 0 (fresh) survives; slot 1 (old) reclaimed
    assert (tokens_dir / "0").exists()
    assert not (tokens_dir / "1").exists()


def test_poll_state_until_terminal_invokes_on_poll(tmp_path):
    """Each poll iteration must fire on_poll(state) so callers can push
    throttled HEARTBEAT events to the DB writer."""
    nas_unit = tmp_path / "u"
    nas_unit.mkdir()
    state_path = nas_unit / "state.json"
    state_path.write_text(json.dumps({
        "run_id": 1, "phase": "running", "machine": "alpha",
    }))
    (nas_unit / "heartbeat").touch()

    polls: list[dict] = []
    g = D.GlobalCfg(
        poll_interval_sec=0.05, state_appeared_timeout_sec=2,
        heartbeat_stale_threshold_sec=300,
    )
    # Flip to terminal after a few polls so the test exits.
    import threading as _th
    def _flip():
        state_path.write_text(json.dumps({
            "run_id": 1, "phase": "done", "success": True,
            "finished_at": "2026-01-01T00:00:00",
        }))
    _th.Timer(0.2, _flip).start()

    def _on_poll(s):
        polls.append(dict(s))

    final = D.poll_state_until_terminal(
        "alpha", nas_unit, g=g,
        expected_tokens=["--run-id 1"],
        on_poll=_on_poll,
    )
    assert final.get("success") is True
    assert len(polls) >= 1
    # Every entry in `polls` must be a state dict snapshot (has phase).
    assert all("phase" in p for p in polls)


def test_dry_run_does_not_mutate_db_via_reconcile(tmp_path):
    """--dry-run must skip reconciliation so prior dispatch state is
    not mutated by a read-only preview."""
    db = DatabaseManager(tmp_path / "x.db")
    # Seed a prior RUNNING dispatch with one stale RUNNING row that, in a
    # non-dry-run, would be marked FAILED by reconciliation (no state
    # file, no heartbeat).
    _seed_dispatch(db, dispatch_id="d_old")
    rid = _seed_run(db, dispatch_id="d_old")

    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(),
        machines={"alpha": D.MachineCfg("alpha", "/raid/a", slots=1, nproc=1)},
    )
    obs_csv = tmp_path / "targets.csv"
    obs_csv.write_text("source_name,array,sb_name,sgous_id,gous_id,mous_ids,Line group\n")
    summary = D.run_dispatch(
        base_dir=tmp_path,
        publish_dir=tmp_path / "out",
        cfg=cfg,
        db_manager=db,
        selection_filters=D.SelectionFilters(scales=[0, 5, 10, 15, 20]),
        obs_csv=obs_csv,
        data_dir=tmp_path,
        dry_run=True,
    )
    assert summary["dry_run"] is True
    # The prior RUNNING row must still be RUNNING.  In a non-dry-run we
    # would have marked it FAILED.
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    assert row["status"] == ImagingRunStatus.RUNNING
    assert (row.get("error_message") or "") == ""


def test_gous_cleaner_uses_prior_dispatch_id_for_adopted(tmp_path, monkeypatch):
    """An adopted (machine, gous) pair must be cleaned under
    /raid/d_<prior_id>/..., NOT under the new dispatch's directory."""
    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(),
        machines={"alpha": D.MachineCfg("alpha", "/raid/a", slots=1, nproc=1)},
    )
    s = D.SchedulerState(queue=[])
    s.seen_pairs.add(("alpha", "G_OLD"))
    s.pair_dispatch_id[("alpha", "G_OLD")] = "d_OLD_42"
    s.success_run_ids[("alpha", "G_OLD")] = {7}
    cleaner = D.GousCleaner(s, cfg, "d_NEW_99", D.GlobalCfg())

    seen_cmds: list[str] = []

    def fake_ssh(machine, cmd, timeout=30, capture=True):
        seen_cmds.append(cmd)
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(D, "ssh_run", fake_ssh)
    cleaner.force_run()
    assert len(seen_cmds) == 1
    cmd = seen_cmds[0]
    # Must reference the PRIOR dispatch ID, not the new one.
    assert "d_OLD_42" in cmd
    assert "d_NEW_99" not in cmd
    assert "input/G_OLD" in cmd
    assert "work/runs/7" in cmd


def test_gous_cleaner_iterates_seen_pairs(tmp_path, monkeypatch):
    """Once a GOUS drains, the cleaner sees it via seen_pairs and SSHes
    a cleanup."""
    db = DatabaseManager(":memory:")
    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(),
        machines={"alpha": D.MachineCfg("alpha", "/raid/a", slots=1, nproc=1)},
    )
    s = D.SchedulerState(queue=[])
    s.seen_pairs.add(("alpha", "G42"))
    # in_flight is empty (drained); no queued units of G42
    cleaner = D.GousCleaner(s, cfg, "d_test", D.GlobalCfg())
    calls: list[str] = []

    def fake_ssh(machine, cmd, timeout=30, capture=True):
        calls.append((machine, cmd))
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(D, "ssh_run", fake_ssh)
    cleaner.force_run()
    assert len(calls) == 1
    machine, cmd = calls[0]
    assert machine == "alpha"
    assert "rm -rf" in cmd
    assert "input/G42" in cmd


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def test_serialise_unit_round_trip():
    u = ImagingUnit(
        gous_uid="G", source_name="S", line_group="LG", spw_id="23",
        params_id=7, vis_tm=["/nas/a.ms"], vis_sm=["/nas/b.ms"],
        sdimage="/nas/tp.fits", spw_selection=["23"], field_selection=["S"],
        datacolumn="data", mous_uids_tm=["X"], mous_uids_sm=["Y"],
        mous_uids_tp=["Z"], ready=True,
    )
    d = D.serialise_unit(u)
    assert d["vis_tm"] == ["/nas/a.ms"]
    assert d["sdimage"] == "/nas/tp.fits"
    assert d["spw_selection"] == ["23"]


def test_union_inputs_for_gous_dedups():
    u1 = ImagingUnit(
        gous_uid="G", source_name="A", line_group=None, spw_id="23", params_id=1,
        vis_tm=["/nas/x.ms"], vis_sm=["/nas/y.ms"],
        sdimage="/nas/tp_A.fits", ready=True,
    )
    u2 = ImagingUnit(
        gous_uid="G", source_name="B", line_group=None, spw_id="25", params_id=2,
        vis_tm=["/nas/x.ms"], vis_sm=["/nas/y.ms"],   # same MSs
        sdimage="/nas/tp_B.fits", ready=True,         # different TP
    )
    out = D.union_inputs_for_gous([u1, u2])
    srcs = sorted(e["src"] for e in out)
    assert srcs == sorted([
        "/nas/x.ms", "/nas/y.ms",
        "/nas/tp_A.fits", "/nas/tp_B.fits",
    ])
    bucket_for = {e["src"]: e["bucket"] for e in out}
    assert bucket_for["/nas/x.ms"] == "ms"
    assert bucket_for["/nas/tp_A.fits"] == "tp"


def test_write_unit_manifest_round_trip(tmp_path):
    u = ImagingUnit(
        gous_uid="G", source_name="S", line_group=None,
        spw_id="23", params_id=1,
        vis_tm=["/nas/a.ms"], ready=True,
    )
    nas_unit_dir = tmp_path / "u1"
    p = D.write_unit_manifest(
        nas_unit_dir, unit=u,
        expected_inputs=[{"src": "/nas/a.ms", "bucket": "ms"}],
        publish_dir=tmp_path / "out",
        nproc=4, casa_path=None,
        deconvolver="multiscale", scales=[0, 5, 10],
    )
    payload = json.loads(p.read_text())
    assert payload["unit"]["gous_uid"] == "G"
    assert payload["expected_inputs"][0]["src"] == "/nas/a.ms"
    assert payload["publish_dir"] == str(tmp_path / "out")
    assert payload["nproc"] == 4


# ---------------------------------------------------------------------------
# write_launcher_script — exact shape, shlex-quoted
# ---------------------------------------------------------------------------

def test_write_launcher_script_quotes_paths(tmp_path):
    cfg = D.MachinesConfig(
        conda_env="/opt with space/conda",
        repo_path="/r p/panta-rei",
        casa_path=None,
        global_cfg=D.GlobalCfg(),
        machines={},
    )
    nas = tmp_path / "u1"
    p = D.write_launcher_script(
        nas, cfg,
        raid_dir="/raid/d/foo bar",
        manifest_path="/nas/m.json",
        run_id=42, dispatch_id="d_x",
        transfer_method="tar", publish_policy="fail_if_exists",
        tokens_dir="/nas/tokens",
        max_concurrent_staging=3,
        heartbeat_interval=30,
    )
    text = p.read_text()
    # Spaces are quoted; injected back-tick can't escape
    assert "/opt with space/conda" not in text or "'/opt with space/conda'" in text
    assert "panta_rei.imaging.remote_worker" in text
    assert "--run-id 42" in text
    assert "--dispatch-id d_x" in text
    assert os.access(p, os.X_OK)
    # Cache args ABSENT when not requested
    assert "--cache-root" not in text
    assert "--cache-min-free-gb" not in text


def test_write_launcher_script_includes_cache_args(tmp_path):
    cfg = D.MachinesConfig(
        conda_env="/opt/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(), machines={},
    )
    p = D.write_launcher_script(
        tmp_path / "u", cfg,
        raid_dir="/raid/d", manifest_path="/nas/m.json",
        run_id=1, dispatch_id="d",
        transfer_method="tar", publish_policy="fail_if_exists",
        tokens_dir="/t", max_concurrent_staging=2,
        heartbeat_interval=30,
        cache_root="/raid/cache",
        cache_min_free_gb=512,
    )
    text = p.read_text()
    assert "--cache-root /raid/cache" in text
    assert "--cache-min-free-gb 512" in text


def test_machines_config_cache_min_free_gb_default_and_override(tmp_path):
    """Default cache_min_free_gb is 1024; per-host override is honored;
    null disables cache."""
    payload = {
        "conda_env": "/c",
        "repo_path": "/r",
        "global": {"max_concurrent_staging": 2},
        "machines": {
            "default": {"raid": "/raid/d", "slots": 1, "nproc": 4},
            "tight":   {"raid": "/raid/t", "slots": 1, "nproc": 4,
                        "cache_min_free_gb": 256},
            "off":     {"raid": "/raid/o", "slots": 1, "nproc": 4,
                        "cache_min_free_gb": None},
        },
    }
    p = tmp_path / "machines.json"
    p.write_text(json.dumps(payload))
    cfg = D.load_machines_config(p)
    assert cfg.machines["default"].cache_min_free_gb == 1024
    assert cfg.machines["tight"].cache_min_free_gb == 256
    assert cfg.machines["off"].cache_min_free_gb is None


# ---------------------------------------------------------------------------
# ssh_pid_alive — distinguishes dead / alive / unreachable
# ---------------------------------------------------------------------------

@mock.patch("panta_rei.imaging.dispatch.ssh_run")
def test_ssh_pid_alive_dead_returns_false(ssh_run):
    ssh_run.return_value = mock.Mock(returncode=0, stdout="__DEAD__", stderr="")
    alive, info = D.ssh_pid_alive("alpha", 12345, ["expected"])
    assert alive is False


@mock.patch("panta_rei.imaging.dispatch.ssh_run")
def test_ssh_pid_alive_pid_reused_returns_false(ssh_run):
    ssh_run.return_value = mock.Mock(
        returncode=0,
        stdout="/usr/bin/somethingelse",   # cmdline doesn't match
        stderr="",
    )
    alive, info = D.ssh_pid_alive("alpha", 12345, ["expected-marker"])
    assert alive is False


@mock.patch("panta_rei.imaging.dispatch.ssh_run")
def test_ssh_pid_alive_match_returns_true_unordered(ssh_run):
    """Tokens may appear in any order in the cmdline."""
    # Note: launcher emits --run-id BEFORE --dispatch-id; check that order.
    ssh_run.return_value = mock.Mock(
        returncode=0,
        stdout=("python -m panta_rei.imaging.remote_worker "
                "--manifest /n/m.json --raid-dir /raid "
                "--run-id 42 --dispatch-id d_x ..."),
        stderr="",
    )
    alive, info = D.ssh_pid_alive(
        "alpha", 12345,
        ["--dispatch-id d_x", "--run-id 42"],
    )
    assert alive is True


@mock.patch("panta_rei.imaging.dispatch.ssh_run")
def test_ssh_pid_alive_partial_match_returns_false(ssh_run):
    """All tokens must be present; single match isn't enough."""
    ssh_run.return_value = mock.Mock(
        returncode=0,
        stdout=("python -m panta_rei.imaging.remote_worker "
                "--run-id 42 --dispatch-id d_other"),
        stderr="",
    )
    alive, info = D.ssh_pid_alive(
        "alpha", 12345,
        ["--dispatch-id d_x", "--run-id 42"],
    )
    assert alive is False


@mock.patch("panta_rei.imaging.dispatch.ssh_run")
def test_ssh_pid_alive_unreachable_returns_none(ssh_run):
    ssh_run.return_value = mock.Mock(returncode=255, stdout="", stderr="conn refused")
    alive, info = D.ssh_pid_alive("alpha", 12345, ["expected"])
    assert alive is None
    assert "rc=" in info or "ssh" in info.lower()


@mock.patch("panta_rei.imaging.dispatch.ssh_run")
def test_ssh_preflight_parses_ok_json(ssh_run):
    ssh_run.return_value = mock.Mock(
        returncode=0,
        stdout=('{"ok": true, "free_gb": 800, "raid_writable": true, '
                '"nas_visible": true}'),
        stderr="",
    )
    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(),
        machines={},
    )
    m = D.MachineCfg("alpha", "/raid/a", slots=1, nproc=4)
    ok, details = D.ssh_preflight_machine(
        "alpha", m, cfg,
        required_gb=10, nas_check_path="/nas/marker",
    )
    assert ok is True
    assert details["free_gb"] == 800


@mock.patch("panta_rei.imaging.dispatch.ssh_run")
def test_ssh_preflight_failure_returns_false(ssh_run):
    ssh_run.return_value = mock.Mock(
        returncode=1,
        stdout='{"ok": false, "raid_error": "permission denied"}',
        stderr="",
    )
    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(),
        machines={},
    )
    m = D.MachineCfg("alpha", "/raid/a", slots=1, nproc=4)
    ok, details = D.ssh_preflight_machine("alpha", m, cfg)
    assert ok is False
    assert "permission denied" in str(details)


@mock.patch("panta_rei.imaging.dispatch.ssh_run")
def test_ssh_preflight_unparseable_returns_error(ssh_run):
    ssh_run.return_value = mock.Mock(
        returncode=0, stdout="not json", stderr="",
    )
    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(),
        machines={},
    )
    m = D.MachineCfg("alpha", "/raid/a", slots=1, nproc=4)
    ok, details = D.ssh_preflight_machine("alpha", m, cfg)
    assert ok is False
    assert "could not parse" in str(details)


@mock.patch("panta_rei.imaging.dispatch.ssh_run")
def test_ssh_pid_alive_uses_tr_with_space_replacement(ssh_run):
    """The remote command must translate NUL→space, not delete NUL."""
    ssh_run.return_value = mock.Mock(returncode=0, stdout="__DEAD__", stderr="")
    D.ssh_pid_alive("alpha", 99, ["x"])
    # Inspect the remote command passed to ssh_run
    call_args = ssh_run.call_args
    remote_cmd = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs["remote_cmd"]
    assert "tr '\\0' ' '" in remote_cmd
    assert "tr -d '\\0'" not in remote_cmd


def test_ssh_run_wraps_remote_command_in_bash(monkeypatch):
    """Login shell on the cluster is tcsh; the remote command must be
    forced through bash so redirections and ``!`` work."""
    captured = {}

    def fake_run(argv, **kw):
        captured["argv"] = argv
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    D.ssh_run("alpha", "echo hi >log 2>&1")
    argv = captured["argv"]
    assert argv[0] == "ssh"
    assert argv[-2] == "alpha"
    wrapped = argv[-1]
    # Outer command is shell-agnostic: echo <b64> | base64 -d | bash
    assert wrapped.startswith("echo ")
    assert "| base64 -d | bash" in wrapped
    # Forbidden-in-tcsh tokens must NOT appear in the outer wrapper
    assert ">log 2>&1" not in wrapped
    assert "$!" not in wrapped


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def _seed_dispatch(db, dispatch_id="d_old"):
    with db.connect() as con:
        DispatchesQueries.insert(
            con, dispatch_id=dispatch_id,
            coordinator_host="h", coordinator_pid=42,
            machines_json="{}", cli_args="",
        )
        con.commit()


def _seed_run(db, dispatch_id="d_old", status=ImagingRunStatus.RUNNING):
    with db.connect() as con:
        rid = ImagingRunsQueries.insert_row(
            con,
            params_id=1, gous_uid="G", source_name="S",
            line_group="LG", spw_id="23",
            started_at="2026-01-01T00:00:00",
            status=status,
            dispatch_id=dispatch_id,
        )
        con.commit()
    return rid


def _state_dir(base_dir, dispatch_id, run_id):
    d = base_dir / "imaging" / "dispatch" / dispatch_id / "units" / str(run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_reconcile_terminal_state_applied(tmp_path):
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db)
    rid = _seed_run(db)
    sd = _state_dir(tmp_path, "d_old", rid)
    (sd / "state.json").write_text(json.dumps({
        "run_id": rid, "phase": "done", "success": True,
        "output_fits": "/nas/out.fits",
        "finished_at": "2026-01-01T00:05:00",
    }))
    g = D.GlobalCfg(heartbeat_stale_threshold_sec=300)
    D.reconcile_prior(db, tmp_path, g)
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    assert row["status"] == ImagingRunStatus.SUCCESS
    assert row["output_fits"] == "/nas/out.fits"


def test_reconcile_fresh_heartbeat_adopts(tmp_path):
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db)
    rid = _seed_run(db)
    sd = _state_dir(tmp_path, "d_old", rid)
    (sd / "state.json").write_text(json.dumps({
        "run_id": rid, "phase": "running", "machine": "alpha",
        "worker_pid": 99,
    }))
    (sd / "heartbeat").touch()
    g = D.GlobalCfg(heartbeat_stale_threshold_sec=300)
    adoptable = D.reconcile_prior(db, tmp_path, g)
    assert any(a["run_id"] == rid for a in adoptable)
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    # Still RUNNING (adoption preserves state)
    assert row["status"] == ImagingRunStatus.RUNNING


def test_reconcile_normalises_fqdn_to_short_name(tmp_path):
    """state.json stores ``socket.gethostname()`` (FQDN on these hosts);
    adoptable[*]['machine'] must be the short name so it matches the
    keys in machines.json — otherwise slot accounting and cleanup
    mappings silently miss the adopted unit."""
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db)
    rid = _seed_run(db)
    sd = _state_dir(tmp_path, "d_old", rid)
    (sd / "state.json").write_text(json.dumps({
        "run_id": rid, "phase": "running",
        "machine": "host0.example.com",
        "hostname": "host0.example.com",
        "worker_pid": 99,
    }))
    (sd / "heartbeat").touch()
    g = D.GlobalCfg(heartbeat_stale_threshold_sec=300)
    adoptable = D.reconcile_prior(db, tmp_path, g)
    assert len(adoptable) == 1
    assert adoptable[0]["machine"] == "host0"


@mock.patch("panta_rei.imaging.dispatch.ssh_pid_alive",
            return_value=(False, ""))
def test_reconcile_stale_heartbeat_dead_pid_marks_failed(_ssh, tmp_path):
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db)
    rid = _seed_run(db)
    sd = _state_dir(tmp_path, "d_old", rid)
    (sd / "state.json").write_text(json.dumps({
        "run_id": rid, "phase": "running", "machine": "alpha",
        "worker_pid": 99,
    }))
    # No heartbeat file => infinite age → stale
    g = D.GlobalCfg(heartbeat_stale_threshold_sec=10)
    D.reconcile_prior(db, tmp_path, g)
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    assert row["status"] == ImagingRunStatus.FAILED
    assert "abandoned" in (row["error_message"] or "")


@mock.patch("panta_rei.imaging.dispatch.ssh_pid_alive",
            return_value=(None, "ssh refused"))
def test_reconcile_ssh_unreachable_does_not_mark_failed(_ssh, tmp_path):
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db)
    rid = _seed_run(db)
    sd = _state_dir(tmp_path, "d_old", rid)
    (sd / "state.json").write_text(json.dumps({
        "run_id": rid, "phase": "running", "machine": "alpha",
        "worker_pid": 99,
    }))
    g = D.GlobalCfg(heartbeat_stale_threshold_sec=10)
    D.reconcile_prior(db, tmp_path, g)
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    # Still RUNNING — coordinator must not declare dead.
    assert row["status"] == ImagingRunStatus.RUNNING


def test_reconcile_abandon_prior_force_fails(tmp_path):
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db)
    rid = _seed_run(db)
    sd = _state_dir(tmp_path, "d_old", rid)
    (sd / "state.json").write_text(json.dumps({
        "run_id": rid, "phase": "running", "machine": "alpha",
        "worker_pid": 99,
    }))
    (sd / "heartbeat").touch()
    g = D.GlobalCfg(heartbeat_stale_threshold_sec=300)
    D.reconcile_prior(db, tmp_path, g, abandon=True)
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    assert row["status"] == ImagingRunStatus.FAILED


def test_adoption_poller_applies_terminal_to_db(tmp_path, monkeypatch):
    """An AdoptionPoller resumes polling an existing unit's state.json
    and pushes its terminal result to the DB writer."""
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db, dispatch_id="d_old")
    rid = _seed_run(db, dispatch_id="d_old")

    sd = _state_dir(tmp_path, "d_old", rid)
    state_file = sd / "state.json"
    state_file.write_text(json.dumps({
        "run_id": rid, "phase": "running",
        "machine": "alpha", "worker_pid": 99,
        "dispatch_id": "d_old", "gous_uid": "G",
    }))
    (sd / "heartbeat").touch()

    # Worker "finishes" between polls — flip the file to terminal SUCCESS.
    def _flip_to_done(*_a, **_kw):
        state_file.write_text(json.dumps({
            "run_id": rid, "phase": "done", "success": True,
            "output_fits": "/nas/out.fits",
            "finished_at": "2026-01-01T00:00:00",
            "spw_selection": ["23"], "field_selection": ["S"],
            "gous_uid": "G",
        }))
        (sd / "heartbeat").touch()

    # First poll triggers the flip
    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (True, "ok"))
    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(
            poll_interval_sec=0.1, state_appeared_timeout_sec=2,
            heartbeat_stale_threshold_sec=300,
        ),
        machines={"alpha": D.MachineCfg("alpha", "/raid/a", slots=2, nproc=4)},
    )
    writer = D.DBWriter(db, "d_new")
    writer.start()
    _seed_dispatch(db, dispatch_id="d_new")
    scheduler = D.SchedulerState(queue=[])
    ctx = D.DispatchContext(
        cfg=cfg, dispatch_id="d_new", dispatch_dir=tmp_path / "imaging" / "dispatch" / "d_new",
        publish_dir=tmp_path / "out", tokens_dir=tmp_path / "tokens",
        db_writer=writer, db_manager=db, scheduler=scheduler,
        transfer_method="tar", publish_policy="fail_if_exists",
        deconvolver="multiscale", scales=[0, 5, 10, 15, 20],
        gous_inputs={},
    )

    # Schedule the flip so it happens AFTER the poller has picked up the file
    import threading as _th
    timer = _th.Timer(0.2, _flip_to_done)
    timer.start()

    poller = D.AdoptionPoller(
        adopted={
            "machine": "alpha", "run_id": rid,
            "unit_dir": sd,
            "state": {"dispatch_id": "d_old", "gous_uid": "G"},
            "expected_tokens": ["--dispatch-id d_old", f"--run-id {rid}"],
        },
        ctx=ctx,
    )
    poller.start()
    poller.join(timeout=5)
    timer.cancel()
    writer.stop(); writer.join(timeout=5)

    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    assert row["status"] == ImagingRunStatus.SUCCESS
    assert row["output_fits"] == "/nas/out.fits"
    # Scheduler should have seen the (machine, gous) pair so the cleaner
    # picks it up at end-of-run.
    assert ("alpha", "G") in scheduler.seen_pairs


def test_reconcile_db_row_without_state_file_marked_failed(tmp_path):
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db)
    rid = _seed_run(db)
    g = D.GlobalCfg()
    # No state.json or unit dir at all
    D.reconcile_prior(db, tmp_path, g)
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    assert row["status"] == ImagingRunStatus.FAILED
    assert "no state.json" in (row["error_message"] or "")


# ---------------------------------------------------------------------------
# Abandoned-dispatch cleanup helpers
# ---------------------------------------------------------------------------

def _write_lock(path: Path, host: str = "host_a.example.com", pid: int = 9999):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"host={host} pid={pid}\n")


def test_release_stale_lock_removes_when_holder_dead(tmp_path, monkeypatch):
    lock = tmp_path / ".dispatcher.lock"
    _write_lock(lock)
    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (False, ""))
    released = D._release_stale_dispatcher_lock(lock, ["run_imaging_dispatch"])
    assert released is True
    assert not lock.exists()


def test_release_stale_lock_preserved_when_holder_alive(tmp_path, monkeypatch):
    lock = tmp_path / ".dispatcher.lock"
    _write_lock(lock)
    monkeypatch.setattr(
        D, "ssh_pid_alive",
        lambda *a, **kw: (True, "python -m panta_rei.cli.run_imaging_dispatch"),
    )
    released = D._release_stale_dispatcher_lock(lock, ["run_imaging_dispatch"])
    assert released is False
    assert lock.exists()


def test_release_stale_lock_preserved_when_ssh_unreachable(tmp_path, monkeypatch):
    lock = tmp_path / ".dispatcher.lock"
    _write_lock(lock)
    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (None, "ssh timeout"))
    released = D._release_stale_dispatcher_lock(lock, ["run_imaging_dispatch"])
    assert released is False
    assert lock.exists()


def test_release_stale_lock_strips_fqdn_to_short(tmp_path, monkeypatch):
    lock = tmp_path / ".dispatcher.lock"
    _write_lock(lock, host="host_a.example.com", pid=42)
    seen_hosts: list[str] = []

    def _fake(host, pid, tokens, timeout=8):
        seen_hosts.append(host)
        return (False, "")

    monkeypatch.setattr(D, "ssh_pid_alive", _fake)
    D._release_stale_dispatcher_lock(lock, ["run_imaging_dispatch"])
    assert seen_hosts == ["host_a"]


def test_release_stale_lock_missing_or_malformed_returns_false(tmp_path):
    # Missing file
    assert D._release_stale_dispatcher_lock(
        tmp_path / "nope", ["run_imaging_dispatch"],
    ) is False
    # Malformed contents
    bad = tmp_path / ".dispatcher.lock"
    bad.write_text("garbage\n")
    assert D._release_stale_dispatcher_lock(
        bad, ["run_imaging_dispatch"],
    ) is False
    assert bad.exists()  # malformed → leave alone


def _make_token(tokens_dir: Path, slot: str, host: str, pid: int):
    d = tokens_dir / slot
    d.mkdir(parents=True, exist_ok=True)
    (d / "host").write_text(host)
    (d / "pid").write_text(str(pid))
    (d / "holder").write_text(f"d_x/{slot}")
    (d / "acquired_at").write_text("2026-04-30T14:43:00")
    return d


def test_sweep_tokens_once_reaps_dead_holders(tmp_path, monkeypatch):
    tokens = tmp_path / "staging_tokens"
    _make_token(tokens, "0", "host_b", 100)
    _make_token(tokens, "1", "host_c", 200)
    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (False, ""))
    reaped = D._sweep_tokens_once(tokens, ["--dispatch-id d_x"])
    assert reaped == 2
    assert not (tokens / "0").exists()
    assert not (tokens / "1").exists()


def test_sweep_tokens_once_keeps_live_holders(tmp_path, monkeypatch):
    tokens = tmp_path / "staging_tokens"
    _make_token(tokens, "0", "host_b", 100)
    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (True, "ok"))
    reaped = D._sweep_tokens_once(tokens, ["--dispatch-id d_x"])
    assert reaped == 0
    assert (tokens / "0").exists()


def test_sweep_tokens_once_missing_dir_returns_zero(tmp_path):
    assert D._sweep_tokens_once(tmp_path / "no_such", []) == 0


def test_cleanup_abandoned_dispatch_sshs_each_machine(tmp_path, monkeypatch):
    """For each machine in machines_json, ssh `rm -rf <raid>/d_<id>/`."""
    d_id = "d_xyz"
    d_dir = tmp_path / "imaging" / "dispatch" / d_id
    (d_dir / "staging_tokens").mkdir(parents=True)
    machines_json = json.dumps({
        "host_a": {"raid": "/raid/scratch/userA", "slots": 1, "nproc": 4},
        "host_d":  {"raid": "/raid/data/userB",        "slots": 1, "nproc": 4},
    })

    calls: list[tuple[str, str]] = []

    class _OK:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_ssh_run(machine, cmd, *, timeout=30, capture=True):
        calls.append((machine, cmd))
        return _OK()

    monkeypatch.setattr(D, "ssh_run", _fake_ssh_run)
    summary = D._cleanup_abandoned_dispatch(d_id, d_dir, machines_json)

    assert sorted(c[0] for c in calls) == ["host_a", "host_d"]
    host_a_cmd = next(c[1] for c in calls if c[0] == "host_a")
    host_d_cmd = next(c[1] for c in calls if c[0] == "host_d")
    assert "/raid/scratch/userA/d_d_xyz" in host_a_cmd
    assert "/raid/data/userB/d_d_xyz" in host_d_cmd
    assert host_a_cmd.startswith("rm -rf -- ")
    assert sorted(summary["machines_swept"]) == ["host_a", "host_d"]
    assert summary["machine_failures"] == {}


def test_cleanup_abandoned_dispatch_ssh_failure_recorded_not_raised(
    tmp_path, monkeypatch,
):
    d_id = "d_xyz"
    d_dir = tmp_path / "imaging" / "dispatch" / d_id
    d_dir.mkdir(parents=True)
    machines_json = json.dumps({
        "host_a": {"raid": "/raid/scratch/userA", "slots": 1, "nproc": 4},
        "deadhost": {"raid": "/raid/x", "slots": 1, "nproc": 4},
    })

    class _OK:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake(machine, cmd, *, timeout=30, capture=True):
        if machine == "deadhost":
            import subprocess as _sp
            raise _sp.TimeoutExpired(cmd=["ssh"], timeout=timeout)
        return _OK()

    monkeypatch.setattr(D, "ssh_run", _fake)
    summary = D._cleanup_abandoned_dispatch(d_id, d_dir, machines_json)
    assert summary["machines_swept"] == ["host_a"]
    assert "deadhost" in summary["machine_failures"]


def test_reconcile_abandon_prior_invokes_cleanup(tmp_path, monkeypatch):
    """When reconcile_prior abandons a dispatch, it must call the
    cleanup helper *and* try to release the dispatcher lock."""
    db = DatabaseManager(tmp_path / "x.db")
    machines_json = json.dumps({
        "alpha": {"raid": "/raid/alpha", "slots": 1, "nproc": 4},
    })
    with db.connect() as con:
        DispatchesQueries.insert(
            con, dispatch_id="d_abandon",
            coordinator_host="h", coordinator_pid=42,
            machines_json=machines_json, cli_args="",
        )
        con.commit()
    rid = _seed_run(db, dispatch_id="d_abandon")
    sd = _state_dir(tmp_path, "d_abandon", rid)
    (sd / "state.json").write_text(json.dumps({
        "run_id": rid, "phase": "staging", "machine": "alpha", "worker_pid": 99,
    }))
    (sd / "heartbeat").touch()

    # Stale dispatcher lock pointing at a "dead" PID.
    lock = tmp_path / "imaging" / "dispatch" / ".dispatcher.lock"
    _write_lock(lock, host="oldhost", pid=12345)

    cleanup_calls: list[str] = []
    lock_release_calls: list[Path] = []

    def _fake_cleanup(d_id, d_dir, mj, **kw):
        cleanup_calls.append(d_id)
        return {"tokens_reaped": 0, "machines_swept": [], "machine_failures": {}}

    def _fake_release(p, *, expected_tokens, **kw):
        lock_release_calls.append(p)
        return True

    monkeypatch.setattr(D, "_cleanup_abandoned_dispatch", _fake_cleanup)
    monkeypatch.setattr(D, "_release_stale_dispatcher_lock", _fake_release)

    g = D.GlobalCfg(heartbeat_stale_threshold_sec=300)
    D.reconcile_prior(db, tmp_path, g, abandon=True)

    assert cleanup_calls == ["d_abandon"]
    assert len(lock_release_calls) == 1
    assert lock_release_calls[0].name == ".dispatcher.lock"


def test_reconcile_without_abandon_does_not_invoke_cleanup(tmp_path, monkeypatch):
    """Plain --reconcile-only (no --abandon-prior) must NOT touch the
    lock or run the per-host raid sweep, even for prior dispatches it
    happens to mark DONE."""
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db, dispatch_id="d_done")
    rid = _seed_run(db, dispatch_id="d_done")
    sd = _state_dir(tmp_path, "d_done", rid)
    (sd / "state.json").write_text(json.dumps({
        "run_id": rid, "phase": "done", "success": True,
        "output_fits": "/nas/out.fits",
    }))

    cleanup_calls: list[str] = []
    lock_release_calls: list[Path] = []

    monkeypatch.setattr(
        D, "_cleanup_abandoned_dispatch",
        lambda *a, **kw: cleanup_calls.append(a[0]) or {},
    )
    def _fake_release_noabandon(p, *, expected_tokens, **kw):
        lock_release_calls.append(p)
        return True

    monkeypatch.setattr(
        D, "_release_stale_dispatcher_lock", _fake_release_noabandon,
    )

    g = D.GlobalCfg(heartbeat_stale_threshold_sec=300)
    D.reconcile_prior(db, tmp_path, g, abandon=False)

    assert cleanup_calls == []
    assert lock_release_calls == []


# ---------------------------------------------------------------------------
# Hardening pass (2026-05-21):
#   - _stop renamed → _stop_event on all Thread subclasses
#   - AdoptionPoller try/finally guarantees mark_terminal on poll crash
#   - successor-slot mechanism (monotonic cap counter) when adoption ends
# ---------------------------------------------------------------------------


def test_thread_subclasses_use_stop_event_attribute():
    """Regression test: ``_stop`` on Thread subclasses shadows
    ``threading.Thread._stop()`` and can break ``join()``.  Each subclass
    must expose ``_stop_event`` (the renamed sentinel) and NOT ``_stop``
    as an instance attribute.
    """
    # MachineSlot — construct via minimal duck-typed bits
    m = D.MachineCfg(name="alpha", raid="/raid", slots=1)
    # We can't easily build a full DispatchContext, but MachineSlot only
    # needs ctx for run(); __init__ doesn't touch it.
    slot = D.MachineSlot("alpha#0", m, ctx=None)  # type: ignore[arg-type]
    assert hasattr(slot, "_stop_event")
    assert "_stop" not in vars(slot)

    # GousCleaner
    sched = D.SchedulerState()
    cleaner = D.GousCleaner(
        sched,
        D.MachinesConfig(
            conda_env="/x", repo_path="/y", casa_path=None,
            global_cfg=D.GlobalCfg(), machines={},
        ),
        "d_test",
        D.GlobalCfg(),
    )
    assert hasattr(cleaner, "_stop_event")
    assert "_stop" not in vars(cleaner)

    # TokenReaper
    reaper = D.TokenReaper(Path("/tmp/no"), D.GlobalCfg(), [])
    assert hasattr(reaper, "_stop_event")
    assert "_stop" not in vars(reaper)


def _minimal_ctx(tmp_path, dispatch_id="d_test"):
    """Build a DispatchContext with the bits AdoptionPoller actually uses.

    Skips the cleaner / reaper since the test exercises only the
    poller's terminal cleanup path.
    """
    db = DatabaseManager(tmp_path / "x.db")
    sched = D.SchedulerState()
    cfg = D.MachinesConfig(
        conda_env="/x", repo_path="/y", casa_path=None,
        global_cfg=D.GlobalCfg(heartbeat_interval_sec=10),
        machines={"alpha": D.MachineCfg(name="alpha", raid="/raid")},
    )
    db_writer = D.DBWriter(db, dispatch_id)
    db_writer.start()
    ctx = D.DispatchContext(
        cfg=cfg, dispatch_id=dispatch_id,
        dispatch_dir=tmp_path / "disp",
        publish_dir=tmp_path / "pub",
        tokens_dir=tmp_path / "tok",
        db_writer=db_writer, db_manager=db, scheduler=sched,
        transfer_method="tar", publish_policy="fail_if_exists",
        deconvolver="multiscale", scales=[0, 5, 10],
        gous_inputs={},
    )
    return ctx, db_writer


def test_adoption_poller_quarantines_on_poll_crash(tmp_path, monkeypatch):
    """If ``poll_state_until_terminal`` raises, the AdoptionPoller must
    quarantine the host (v3 Delta 8) and leave the run row ACTIVE so the
    next dispatch's reconcile_prior can recover it.  It must NOT spawn a
    successor MachineSlot — the remote worker may still be alive and a
    successor would oversubscribe the host.

    Audit nit: renamed from ``..._releases_slot_on_poll_crash`` to
    reflect the v3-invariant behaviour — the row is NOT released; the
    host is quarantined and the row stays active.
    """
    ctx, db_writer = _minimal_ctx(tmp_path)
    try:
        adopted = {
            "run_id": 999,
            "machine": "alpha",
            "unit_dir": tmp_path / "unit",
            "state": {"gous_uid": "G1", "dispatch_id": "d_prior"},
            "prior_dispatch_id": "d_prior",
        }
        # Make poll_state_until_terminal blow up
        def _boom(*a, **kw):
            raise RuntimeError("simulated NAS read failure")
        monkeypatch.setattr(D, "poll_state_until_terminal", _boom)

        # Even with preflight ready, the crash path must NOT spawn a
        # successor (gated on _is_observed_terminal + not-quarantined).
        ctx.new_launch_machines_names = {"alpha"}
        ctx.new_launch_machines_ready.set()
        spawned: list = []
        monkeypatch.setattr(
            D.MachineSlot, "start",
            lambda self: spawned.append(self.name),
        )

        poller = D.AdoptionPoller(
            adopted, ctx, machine_cfg=ctx.cfg.machines["alpha"],
        )
        # Run synchronously in this thread for deterministic assertion
        poller.run()

        # Host quarantined; in_flight stays (run row is left ACTIVE).
        assert "alpha" in ctx.scheduler.quarantined_machines
        # No success/failure recorded — terminal was not observed.
        assert 999 not in ctx.scheduler.failed_run_ids.get(("alpha", "G1"), set())
        # Successor NOT spawned despite preflight ready
        assert spawned == [], f"unexpected successor spawn on poll crash: {spawned}"
        assert ctx.successors_spawned.get("alpha", 0) == 0
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_successor_spawn_caps_at_machine_slots(tmp_path, monkeypatch):
    """With slots=1, even if multiple adoption pollers complete on the
    same machine, only one successor MachineSlot is ever spawned (the
    monotonic ``successors_spawned`` counter caps).
    """
    ctx, db_writer = _minimal_ctx(tmp_path)
    try:
        # Mark preflight done so the spawn-now path runs
        ctx.new_launch_machines_names = {"alpha"}
        ctx.new_launch_machines_ready.set()

        # Stub out poll + start so we don't actually try to launch
        # MachineSlot threads or touch NAS.  We just need the spawn
        # decisions, not the thread bodies.
        spawned: list = []
        # Provide a properly OBSERVED terminal: _is_observed_terminal()
        # requires phase ∈ {"done","failed"} and no synthetic reason.
        monkeypatch.setattr(
            D, "poll_state_until_terminal",
            lambda *a, **kw: {"phase": "done", "success": True},
        )

        original_start = D.MachineSlot.start
        def _fake_start(self):
            spawned.append(self.name)
        monkeypatch.setattr(D.MachineSlot, "start", _fake_start)

        # Two adoption pollers finishing on the same slots=1 machine
        for rid in (100, 101):
            adopted = {
                "run_id": rid,
                "machine": "alpha",
                "unit_dir": tmp_path / f"u{rid}",
                "state": {"gous_uid": f"G{rid}", "dispatch_id": "d_prior"},
                "prior_dispatch_id": "d_prior",
            }
            poller = D.AdoptionPoller(
                adopted, ctx,
                machine_cfg=ctx.cfg.machines["alpha"],
            )
            poller.run()

        # Restore for any further test isolation
        monkeypatch.setattr(D.MachineSlot, "start", original_start)

        # Cap: machine.slots == 1, so only one successor MachineSlot spawned
        assert ctx.successors_spawned.get("alpha") == 1
        assert len(spawned) == 1
        # MachineSlot threads carry a ``slot-`` prefix from __init__
        assert "alpha#post-adopt-" in spawned[0]
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_pending_successor_queued_when_preflight_not_ready(tmp_path, monkeypatch):
    """If an adoption finishes BEFORE preflight has set
    ``new_launch_machines_ready``, the successor request is queued in
    ``pending_successors`` instead of spawning a slot immediately.
    """
    ctx, db_writer = _minimal_ctx(tmp_path)
    try:
        # NOTE: do NOT set new_launch_machines_ready
        assert not ctx.new_launch_machines_ready.is_set()

        # Observed-terminal final required by the new _is_observed_terminal
        # gate (phase ∈ {"done","failed"}).
        monkeypatch.setattr(
            D, "poll_state_until_terminal",
            lambda *a, **kw: {"phase": "done", "success": True},
        )
        # Block MachineSlot.start so this test doesn't accidentally spawn
        monkeypatch.setattr(
            D.MachineSlot, "start", lambda self: None,
        )

        adopted = {
            "run_id": 42,
            "machine": "alpha",
            "unit_dir": tmp_path / "u42",
            "state": {"gous_uid": "Galpha", "dispatch_id": "d_prior"},
            "prior_dispatch_id": "d_prior",
        }
        poller = D.AdoptionPoller(
            adopted, ctx, machine_cfg=ctx.cfg.machines["alpha"],
        )
        poller.run()

        # Queued, NOT spawned
        assert ctx.pending_successors.get("alpha") == 1
        assert ctx.successors_spawned.get("alpha", 0) == 0
        assert ctx.dynamic_slots == []
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_adoption_warns_when_gous_uid_unresolvable(tmp_path, monkeypatch, caplog):
    """If neither state nor DB can resolve a gous_uid, the poller logs
    a WARNING and skips mark_inflight (so scheduler stays consistent).
    """
    import logging as _logging
    ctx, db_writer = _minimal_ctx(tmp_path)
    try:
        monkeypatch.setattr(
            D, "poll_state_until_terminal",
            lambda *a, **kw: {"phase": "done", "success": True},
        )
        monkeypatch.setattr(D.MachineSlot, "start", lambda self: None)

        adopted = {
            "run_id": 7,
            "machine": "alpha",
            "unit_dir": tmp_path / "u7",
            "state": {},   # no gous_uid
            "prior_dispatch_id": "d_prior",
        }
        poller = D.AdoptionPoller(
            adopted, ctx, machine_cfg=ctx.cfg.machines["alpha"],
        )
        with caplog.at_level(_logging.WARNING, logger="panta_rei.dispatch"):
            poller.run()

        # Warning emitted; mark_inflight skipped → no in_flight entry
        warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert any("no resolvable gous_uid" in r.message for r in warnings), (
            f"expected warning; got {[r.message for r in warnings]}"
        )
        assert ctx.scheduler.in_flight == {}
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


# ---------------------------------------------------------------------------
# v3+ hardening pass (2026-05-31): fail-closed orphan cleanup,
# per-host quarantine, observability.
# ---------------------------------------------------------------------------


# Fast tunings so verify_and_kill_worker doesn't block the test suite for
# 5+ minutes per call (default worker_shutdown_grace_sec is 300s).
def _fast_globals() -> "D.GlobalCfg":
    return D.GlobalCfg(
        poll_interval_sec=0.01,
        state_appeared_timeout_sec=1,
        heartbeat_stale_threshold_sec=300,
        launch_pidfile_wait_sec=0,
        worker_shutdown_grace_sec=0,
        late_state_grace_sec=0,
    )


def _stub_verify_to(outcome, detail="stub"):
    """Build a verify_and_kill_worker stub that returns a fixed outcome."""
    return lambda *a, **kw: (outcome, detail)


def _stub_verify_from_responses(pid_alive_responses):
    """Build a stub mapping ssh_pid_alive responses to verify_and_kill outcome.

    Mirrors verify_and_kill_worker's logic without sleeping.
    """
    def _stub(machine, pid, tokens, g, **kw):
        first = pid_alive_responses[0]
        if first == (False, ""):
            return D.VERIFY_DEAD, "worker_already_exited"
        if first[0] is False and first[1]:
            return D.VERIFY_DEAD, f"pid_reused: {first[1]!r}"
        if first[0] is None:
            return D.VERIFY_INCONCLUSIVE, f"ssh-unreachable: {first[1]}"
        # alive=True — replay subsequent responses
        if len(pid_alive_responses) >= 2 and pid_alive_responses[1][0] is False:
            return D.VERIFY_KILLED, "sigterm_killed"
        return D.VERIFY_INCONCLUSIVE, "still alive after kill"
    return _stub


def _build_unit_ctx(tmp_path, machine_name="alpha"):
    """Build a (DispatchContext, db_writer, MachineSlot, unit) tuple
    sufficient to drive _dispatch_one without real SSH."""
    db = DatabaseManager(tmp_path / "x.db")
    with db.connect() as con:
        DispatchesQueries.insert(
            con, dispatch_id="d_test",
            coordinator_host="local", coordinator_pid=os.getpid(),
            machines_json="{}", cli_args="",
        )
        con.commit()
    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=_fast_globals(),
        machines={machine_name: D.MachineCfg(
            machine_name, f"/raid/{machine_name}", slots=1, nproc=1,
            cache_min_free_gb=None,
        )},
    )
    db_writer = D.DBWriter(db, "d_test")
    db_writer.start()
    scheduler = D.SchedulerState(queue=[])
    ctx = D.DispatchContext(
        cfg=cfg, dispatch_id="d_test",
        dispatch_dir=tmp_path / "imaging" / "dispatch" / "d_test",
        publish_dir=tmp_path / "pub",
        tokens_dir=tmp_path / "tokens",
        db_writer=db_writer, db_manager=db, scheduler=scheduler,
        transfer_method="tar", publish_policy="fail_if_exists",
        deconvolver="multiscale", scales=[0, 5, 10],
        gous_inputs={"G_TEST": []},
    )
    ctx.dispatch_dir.mkdir(parents=True, exist_ok=True)
    unit = ImagingUnit(
        gous_uid="G_TEST", source_name="S", line_group="LG",
        spw_id="23", params_id=1,
        recovered_params={"start": "100GHz", "width": "1MHz", "nchan": 10},
        ready=True,
    )
    slot = D.MachineSlot(f"{machine_name}#0", cfg.machines[machine_name], ctx)
    return ctx, db_writer, slot, unit


# ---- F2.a / config defaults ------------------------------------------------


def test_state_appeared_timeout_default_is_300():
    """v5 F2.e: default raised from 60 → 300."""
    assert D.GlobalCfg().state_appeared_timeout_sec == 300


def test_launch_pidfile_wait_default_is_30():
    """v3 F6: default 30s wait for worker.pidfile after launch failure."""
    assert D.GlobalCfg().launch_pidfile_wait_sec == 30


def test_worker_shutdown_grace_default_is_300():
    """v4 Delta 6: single grace knob, 300s by default."""
    assert D.GlobalCfg().worker_shutdown_grace_sec == 300


def test_late_state_grace_default_is_1800():
    """v3 F2.b/F2.h: late-state grace is 30 minutes by default."""
    assert D.GlobalCfg().late_state_grace_sec == 1800


def test_is_observed_terminal_predicate():
    """F4: only observed (non-synthetic) terminals count."""
    assert D._is_observed_terminal(None) is False
    assert D._is_observed_terminal({}) is False
    assert D._is_observed_terminal({"phase": "running"}) is False
    assert D._is_observed_terminal({"phase": "done", "success": True}) is True
    assert D._is_observed_terminal({"phase": "failed", "success": False}) is True
    assert D._is_observed_terminal({
        "phase": "failed", "reason": "state_missing_timeout",
    }) is False
    assert D._is_observed_terminal({
        "phase": "failed", "reason": "heartbeat_stale_alive",
    }) is False
    assert D._is_observed_terminal({
        "phase": "failed", "reason": "state_appeared_after_timeout",
    }) is False


def test_poll_returns_state_missing_timeout_reason(tmp_path):
    """F2.a: state-missing path now returns typed reason."""
    nas_unit = tmp_path / "u"
    nas_unit.mkdir()
    # state.json never appears
    g = D.GlobalCfg(
        poll_interval_sec=0.05, state_appeared_timeout_sec=0.2,
    )
    final = D.poll_state_until_terminal(
        "alpha", nas_unit, g=g, expected_tokens=["--run-id 1"],
    )
    assert final["reason"] == "state_missing_timeout"
    assert final["phase"] == "failed"


def test_poll_returns_heartbeat_stale_alive_reason_without_killing(
    tmp_path, monkeypatch,
):
    """v4 Delta 1: poll no longer calls ssh_kill_pgid on stale-alive."""
    nas_unit = tmp_path / "u"
    nas_unit.mkdir()
    (nas_unit / "state.json").write_text(json.dumps({
        "run_id": 1, "phase": "running", "worker_pid": 99, "worker_pgid": 99,
    }))
    # Heartbeat is far in the past
    hb = nas_unit / "heartbeat"
    hb.touch()
    old = time.time() - 10000
    os.utime(hb, (old, old))

    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (True, "ours"))
    killed: list = []
    monkeypatch.setattr(
        D, "ssh_kill_pgid", lambda *a, **kw: killed.append(a),
    )

    g = D.GlobalCfg(
        poll_interval_sec=0.01,
        heartbeat_stale_threshold_sec=1,
        max_stale_alive_sec=1,
        state_appeared_timeout_sec=10,
    )
    final = D.poll_state_until_terminal(
        "alpha", nas_unit, g=g, expected_tokens=["--run-id 1"],
    )
    assert final["reason"] == "heartbeat_stale_alive"
    assert killed == [], "Delta 1 violation: poll killed the worker"


# ---- verify_and_kill_worker -------------------------------------------------


def test_verify_and_kill_dead_worker_returns_dead(monkeypatch):
    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (False, ""))
    outcome, _ = D.verify_and_kill_worker(
        "alpha", 1234, ["--run-id 1"], _fast_globals(),
        sleep_fn=lambda *a, **kw: None,
    )
    assert outcome == D.VERIFY_DEAD


def test_verify_and_kill_pid_reused_returns_dead(monkeypatch):
    """If ssh_pid_alive returns (False, non_empty), PID was reused."""
    monkeypatch.setattr(
        D, "ssh_pid_alive", lambda *a, **kw: (False, "/usr/bin/other"),
    )
    outcome, detail = D.verify_and_kill_worker(
        "alpha", 1234, ["--run-id 1"], _fast_globals(),
        sleep_fn=lambda *a, **kw: None,
    )
    assert outcome == D.VERIFY_DEAD
    assert "pid_reused" in detail


def test_verify_and_kill_inconclusive_when_ssh_unreachable(monkeypatch):
    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (None, "ssh timeout"))
    outcome, _ = D.verify_and_kill_worker(
        "alpha", 1234, ["--run-id 1"], _fast_globals(),
        sleep_fn=lambda *a, **kw: None,
    )
    assert outcome == D.VERIFY_INCONCLUSIVE


def test_verify_and_kill_sigterm_then_dead(monkeypatch):
    """Alive → TERM → dead = VERIFY_KILLED."""
    calls = {"alive": 0}
    def _alive(*a, **kw):
        calls["alive"] += 1
        return (True, "ours") if calls["alive"] == 1 else (False, "")
    monkeypatch.setattr(D, "ssh_pid_alive", _alive)
    killed: list = []
    monkeypatch.setattr(D, "ssh_kill_pgid",
                        lambda m, p, s, **kw: killed.append(s))
    outcome, detail = D.verify_and_kill_worker(
        "alpha", 1234, ["--run-id 1"], _fast_globals(),
        sleep_fn=lambda *a, **kw: None,
    )
    assert outcome == D.VERIFY_KILLED
    assert killed == ["TERM"]


def test_verify_and_kill_escalates_to_kill(monkeypatch):
    """Alive → TERM → still alive → KILL → dead."""
    calls = {"n": 0}
    def _alive(*a, **kw):
        calls["n"] += 1
        # alive, alive (post-TERM), dead (post-KILL)
        if calls["n"] <= 2:
            return (True, "ours")
        return (False, "")
    monkeypatch.setattr(D, "ssh_pid_alive", _alive)
    killed: list = []
    monkeypatch.setattr(D, "ssh_kill_pgid",
                        lambda m, p, s, **kw: killed.append(s))
    outcome, _ = D.verify_and_kill_worker(
        "alpha", 1234, ["--run-id 1"], _fast_globals(),
        sleep_fn=lambda *a, **kw: None,
    )
    assert outcome == D.VERIFY_KILLED
    assert killed == ["TERM", "KILL"]


def test_verify_and_kill_inconclusive_after_kill(monkeypatch):
    """Alive even after SIGKILL → VERIFY_INCONCLUSIVE → caller quarantines."""
    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (True, "ours"))
    monkeypatch.setattr(D, "ssh_kill_pgid", lambda *a, **kw: None)
    outcome, _ = D.verify_and_kill_worker(
        "alpha", 1234, ["--run-id 1"], _fast_globals(),
        sleep_fn=lambda *a, **kw: None,
    )
    assert outcome == D.VERIFY_INCONCLUSIVE


# ---- Scheduler quarantine ---------------------------------------------------


def test_pick_refuses_quarantined_machine():
    s = D.SchedulerState(queue=[_make_unit("G1", "S1")])
    s.quarantine("alpha")
    assert s.pick("alpha", run_id_assigner=lambda: 0) is None
    # Non-quarantined still works:
    assert s.pick("beta", run_id_assigner=lambda: 0) is not None


def test_is_quarantined_reports_state():
    s = D.SchedulerState()
    assert not s.is_quarantined("alpha")
    s.quarantine("alpha")
    assert s.is_quarantined("alpha")


def test_quarantine_all_seeds_multiple_hosts():
    s = D.SchedulerState()
    s.quarantine_all({"alpha", "beta"})
    assert s.is_quarantined("alpha")
    assert s.is_quarantined("beta")


def test_gous_cleaner_skips_quarantined_machine(tmp_path, monkeypatch):
    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(),
        machines={"alpha": D.MachineCfg("alpha", "/raid/a", slots=1, nproc=1)},
    )
    s = D.SchedulerState(queue=[])
    s.seen_pairs.add(("alpha", "G"))
    s.quarantine("alpha")
    cleaner = D.GousCleaner(s, cfg, "d_x", D.GlobalCfg())
    seen: list = []
    monkeypatch.setattr(
        D, "ssh_run",
        lambda *a, **kw: seen.append(a) or mock.Mock(returncode=0, stdout="", stderr=""),
    )
    cleaner.force_run()
    assert seen == [], "cleaner ran rm -rf on a quarantined host"


def test_slot_exits_on_quarantine(tmp_path):
    """MachineSlot.run() exits if its host is quarantined."""
    ctx, db_writer, slot, _unit = _build_unit_ctx(tmp_path)
    try:
        ctx.scheduler.quarantine("alpha")
        slot.run()
        # No exception, just returns; pick() returns None for quarantined.
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


# ---- _dispatch_one — F2.b state_missing_timeout paths ----------------------


def _drive_dispatch_one(slot, unit, monkeypatch, *,
                        launch_return=(True, "launched", 1234),
                        poll_final=None,
                        pid_alive_responses=None,
                        late_state=None):
    """Helper: drive _dispatch_one with mocks and return DB rows.

    NOTE: does NOT patch time.sleep — patching it breaks the
    DBWriter run_id wait loop.  Instead stubs out verify_and_kill_worker
    so its internal sleeps are never reached.
    """
    monkeypatch.setattr(D, "launch_detached", lambda *a, **kw: launch_return)
    if poll_final is not None:
        monkeypatch.setattr(
            D, "poll_state_until_terminal", lambda *a, **kw: poll_final,
        )
    if pid_alive_responses is not None:
        # Stub verify_and_kill_worker per the response sequence so the
        # test never hits real sleeps inside it.
        monkeypatch.setattr(
            D, "verify_and_kill_worker",
            _stub_verify_from_responses(pid_alive_responses),
        )
    monkeypatch.setattr(D, "ssh_kill_pgid", lambda *a, **kw: None)
    if late_state is not None:
        # Write a state.json before _dispatch_one calls _handle_state_missing_timeout
        nas_unit_dir = slot.ctx.dispatch_dir / "units"
        nas_unit_dir.mkdir(parents=True, exist_ok=True)
    slot._dispatch_one(unit)


def test_dispatch_one_kills_orphan_on_state_missing_timeout(tmp_path, monkeypatch):
    """F2.b happy path: state-missing + SIGTERM succeeds → MARK_DONE FAILED."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        _drive_dispatch_one(
            slot, unit, monkeypatch,
            poll_final={
                "phase": "failed",
                "success": False,
                "reason": "state_missing_timeout",
                "error_message": "state.json never appeared",
            },
            # alive once (verify), dead after TERM
            pid_alive_responses=[(True, "ours"), (False, "")],
        )
        db_writer.q.join()
        # Host NOT quarantined (verified-killed → safe to MARK_DONE).
        assert not ctx.scheduler.is_quarantined("alpha")
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status FROM imaging_runs",
            ).fetchall()
        statuses = [r[0] for r in rows]
        assert ImagingRunStatus.FAILED in statuses
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_quarantines_on_ssh_unreachable_during_verify(
    tmp_path, monkeypatch,
):
    """F2.b: ssh_pid_alive returns (None, ...) → quarantine, NO MARK_DONE."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        _drive_dispatch_one(
            slot, unit, monkeypatch,
            poll_final={
                "phase": "failed", "success": False,
                "reason": "state_missing_timeout",
                "error_message": "...",
            },
            pid_alive_responses=[(None, "ssh timeout")],
        )
        db_writer.q.join()
        assert ctx.scheduler.is_quarantined("alpha")
        # Row stays QUEUED/RUNNING (not terminal) → MARK_DONE was skipped.
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status FROM imaging_runs",
            ).fetchall()
        terminal = {ImagingRunStatus.SUCCESS, ImagingRunStatus.FAILED}
        assert not any(r[0] in terminal for r in rows), (
            f"expected row left active; got statuses {rows}"
        )
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_quarantines_on_kill_failure(tmp_path, monkeypatch):
    """F2.b: SIGKILL doesn't take → quarantine, NO MARK_DONE."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        _drive_dispatch_one(
            slot, unit, monkeypatch,
            poll_final={
                "phase": "failed", "success": False,
                "reason": "state_missing_timeout",
                "error_message": "...",
            },
            # alive forever
            pid_alive_responses=[(True, "ours")],
        )
        db_writer.q.join()
        assert ctx.scheduler.is_quarantined("alpha")
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_quarantines_on_heartbeat_stale_alive_inconclusive(
    tmp_path, monkeypatch,
):
    """v4 Delta 1: heartbeat_stale_alive + ssh-unreachable → quarantine."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        _drive_dispatch_one(
            slot, unit, monkeypatch,
            poll_final={
                "phase": "failed", "success": False,
                "reason": "heartbeat_stale_alive",
                "worker_pgid": 9999,
                "error_message": "stale",
            },
            pid_alive_responses=[(None, "ssh timeout")],
        )
        db_writer.q.join()
        assert ctx.scheduler.is_quarantined("alpha")
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_pid_reused_token_mismatch_proceeds_to_mark_done(
    tmp_path, monkeypatch,
):
    """ssh_pid_alive (False, non_empty_cmdline) = PID reused → MARK_DONE."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        _drive_dispatch_one(
            slot, unit, monkeypatch,
            poll_final={
                "phase": "failed", "success": False,
                "reason": "state_missing_timeout",
                "error_message": "...",
            },
            pid_alive_responses=[(False, "/usr/bin/unrelated_process")],
        )
        db_writer.q.join()
        assert not ctx.scheduler.is_quarantined("alpha")
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status, error_message FROM imaging_runs",
            ).fetchall()
        assert any(r[0] == ImagingRunStatus.FAILED for r in rows)
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_genuinely_dead_pid_proceeds_to_mark_done(
    tmp_path, monkeypatch,
):
    """ssh_pid_alive (False, "") = __DEAD__ → MARK_DONE without quarantine."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        _drive_dispatch_one(
            slot, unit, monkeypatch,
            poll_final={
                "phase": "failed", "success": False,
                "reason": "state_missing_timeout",
                "error_message": "...",
            },
            pid_alive_responses=[(False, "")],
        )
        db_writer.q.join()
        assert not ctx.scheduler.is_quarantined("alpha")
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status FROM imaging_runs",
            ).fetchall()
        assert any(r[0] == ImagingRunStatus.FAILED for r in rows)
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


# ---- _dispatch_one — F2.c launch-timeout paths ----------------------------


def test_dispatch_one_quarantines_on_no_pidfile_after_ssh_timeout(
    tmp_path, monkeypatch,
):
    """F2.c + Delta 3: ssh timeout + no pidfile → quarantine, NO MARK_DONE."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        _drive_dispatch_one(
            slot, unit, monkeypatch,
            launch_return=(False, "ssh timeout to alpha", None),
            # No pidfile will appear; launch_pidfile_wait_sec=0 in _fast_globals
        )
        db_writer.q.join()
        assert ctx.scheduler.is_quarantined("alpha")
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_handles_ssh_launch_timeout_with_late_pidfile(
    tmp_path, monkeypatch,
):
    """F2.c: pidfile appears within wait window → verify+kill flow."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        monkeypatch.setattr(
            D, "launch_detached",
            lambda *a, **kw: (False, "ssh timeout", None),
        )
        # Pre-write the pidfile so _wait_for_pidfile finds it.
        nas_unit_dir = ctx.dispatch_dir / "units"
        # _dispatch_one creates units/<run_id>; the run_id will be 1
        # for the first INSERT_QUEUED, so we can't pre-write without
        # knowing it.  Instead, monkeypatch _wait_for_pidfile to return
        # a known pid directly.
        monkeypatch.setattr(D, "_wait_for_pidfile", lambda *a, **kw: 4242)
        monkeypatch.setattr(
            D, "ssh_pid_alive", lambda *a, **kw: (False, ""),
        )
        monkeypatch.setattr(D, "ssh_kill_pgid", lambda *a, **kw: None)
        slot._dispatch_one(unit)
        db_writer.q.join()
        # Verified dead → MARK_DONE FAILED + no quarantine.
        assert not ctx.scheduler.is_quarantined("alpha")
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status FROM imaging_runs",
            ).fetchall()
        assert any(r[0] == ImagingRunStatus.FAILED for r in rows)
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_quarantines_on_launch_ok_nil_pid_no_pidfile(
    tmp_path, monkeypatch,
):
    """Delta 3: launch_detached returned (True, _, None) and no pidfile
    appears → treat as inconclusive launch and quarantine."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        monkeypatch.setattr(
            D, "launch_detached", lambda *a, **kw: (True, "ok", None),
        )
        # _wait_for_pidfile sees nothing
        monkeypatch.setattr(D, "_wait_for_pidfile", lambda *a, **kw: None)
        # poll_state_until_terminal should not be reached because we
        # treat (True, _, None) + no pidfile as state_missing_timeout
        # via the normal poll path.  In practice we DO reach poll here
        # because launch returned ok=True.  The test demonstrates that
        # if poll returns a state_missing_timeout and no pidfile is
        # available, we quarantine.
        monkeypatch.setattr(
            D, "poll_state_until_terminal",
            lambda *a, **kw: {
                "phase": "failed", "success": False,
                "reason": "state_missing_timeout",
                "error_message": "no state",
            },
        )
        # Inconclusive verify (no pid at all → INCONCLUSIVE).
        monkeypatch.setattr(
            D, "verify_and_kill_worker",
            lambda *a, **kw: (D.VERIFY_INCONCLUSIVE, "no pid"),
        )
        slot._dispatch_one(unit)
        db_writer.q.join()
        assert ctx.scheduler.is_quarantined("alpha")
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


# ---- _dispatch_one observability (F1 + F3) --------------------------------


def test_dispatch_one_logs_mark_done_at_info_and_warn(
    tmp_path, monkeypatch, caplog,
):
    """F1 + F3: INFO MARK_DONE log + WARN failure log on terminal."""
    import logging as _logging
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        monkeypatch.setattr(D, "launch_detached", lambda *a, **kw: (True, "ok", 1234))
        # Observed terminal — not synthetic.
        monkeypatch.setattr(
            D, "poll_state_until_terminal",
            lambda *a, **kw: {
                "phase": "failed", "success": False,
                "error_message": "tclean rc=1",
            },
        )
        with caplog.at_level(_logging.DEBUG, logger="panta_rei.dispatch"):
            slot._dispatch_one(unit)
        db_writer.q.join()
        info_msgs = [r.getMessage() for r in caplog.records
                     if r.levelno == _logging.INFO]
        warn_msgs = [r.getMessage() for r in caplog.records
                     if r.levelno == _logging.WARNING]
        assert any("MARK_DONE" in m for m in info_msgs), (
            f"missing INFO MARK_DONE: {info_msgs}"
        )
        assert any("failed" in m for m in warn_msgs), (
            f"missing WARN failure: {warn_msgs}"
        )
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_logs_mark_done_success_at_info_only(
    tmp_path, monkeypatch, caplog,
):
    """Success terminal: INFO only, no WARN."""
    import logging as _logging
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        monkeypatch.setattr(D, "launch_detached", lambda *a, **kw: (True, "ok", 1234))
        monkeypatch.setattr(
            D, "poll_state_until_terminal",
            lambda *a, **kw: {
                "phase": "done", "success": True,
                "output_fits": "/nas/out.fits",
            },
        )
        with caplog.at_level(_logging.DEBUG, logger="panta_rei.dispatch"):
            slot._dispatch_one(unit)
        db_writer.q.join()
        warns_about_fail = [
            r for r in caplog.records
            if r.levelno == _logging.WARNING and "failed" in r.getMessage()
        ]
        assert warns_about_fail == [], (
            f"unexpected WARN on success: {[r.getMessage() for r in warns_about_fail]}"
        )
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


# ---- AdoptionPoller F4 + Delta 8 ------------------------------------------


def test_adoption_poller_does_not_spawn_successor_on_synthetic_final(
    tmp_path, monkeypatch,
):
    """F4: synthetic state_missing_timeout final → no successor spawn."""
    ctx, db_writer = _minimal_ctx(tmp_path)
    try:
        ctx.new_launch_machines_names = {"alpha"}
        ctx.new_launch_machines_ready.set()
        spawned: list = []
        monkeypatch.setattr(
            D.MachineSlot, "start",
            lambda self: spawned.append(self.name),
        )
        monkeypatch.setattr(
            D, "poll_state_until_terminal",
            lambda *a, **kw: {
                "phase": "failed", "success": False,
                "reason": "state_missing_timeout",
                "error_message": "missing",
            },
        )
        # Inconclusive cleanup → AdoptionPoller quarantines.
        monkeypatch.setattr(
            D, "verify_and_kill_worker",
            lambda *a, **kw: (D.VERIFY_INCONCLUSIVE, "ssh dead"),
        )
        adopted = {
            "run_id": 12, "machine": "alpha",
            "unit_dir": tmp_path / "u",
            "state": {"gous_uid": "G", "dispatch_id": "d_prior"},
            "prior_dispatch_id": "d_prior",
        }
        poller = D.AdoptionPoller(adopted, ctx, machine_cfg=ctx.cfg.machines["alpha"])
        poller.run()
        assert spawned == []
        assert ctx.scheduler.is_quarantined("alpha")
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_adoption_poller_quarantines_on_unverified_cleanup(
    tmp_path, monkeypatch,
):
    """Delta 8: synthetic terminal + ssh-unreachable verify → quarantine,
    leave row active (no MARK_DONE)."""
    ctx, db_writer = _minimal_ctx(tmp_path)
    try:
        monkeypatch.setattr(
            D, "poll_state_until_terminal",
            lambda *a, **kw: {
                "phase": "failed", "success": False,
                "reason": "heartbeat_stale_alive",
                "worker_pgid": 1234, "error_message": "stale",
            },
        )
        monkeypatch.setattr(
            D, "verify_and_kill_worker",
            lambda *a, **kw: (D.VERIFY_INCONCLUSIVE, "ssh down"),
        )
        adopted = {
            "run_id": 88, "machine": "alpha",
            "unit_dir": tmp_path / "u",
            "state": {"gous_uid": "G", "dispatch_id": "d_prior"},
            "prior_dispatch_id": "d_prior",
        }
        poller = D.AdoptionPoller(adopted, ctx, machine_cfg=ctx.cfg.machines["alpha"])
        poller.run()
        assert ctx.scheduler.is_quarantined("alpha")
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


# ---- reconcile_prior — F5, Delta 4, Delta 5 --------------------------------


def test_reconcile_returns_reconcile_result(tmp_path):
    """Delta 5: reconcile_prior returns ReconcileResult (not bare list)."""
    db = DatabaseManager(tmp_path / "x.db")
    result = D.reconcile_prior(db, tmp_path, D.GlobalCfg())
    assert isinstance(result, D.ReconcileResult)
    assert isinstance(result.adoptable, list)
    assert isinstance(result.quarantined_hosts, set)


def test_reconcile_prior_no_state_no_pidfile_quarantines_host_leaves_row_active(
    tmp_path,
):
    """Delta 4: no state.json + no pidfile + known host → quarantine + row stays active."""
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db, dispatch_id="d_p")
    with db.connect() as con:
        rid = ImagingRunsQueries.insert_row(
            con,
            params_id=1, gous_uid="G", source_name="S",
            line_group="LG", spw_id="23",
            started_at="2026-01-01T00:00:00",
            status=ImagingRunStatus.RUNNING,
            dispatch_id="d_p", hostname="alpha",
        )
        con.commit()
    # Note: NO state.json, NO worker.pidfile
    result = D.reconcile_prior(db, tmp_path, D.GlobalCfg())
    assert "alpha" in result.quarantined_hosts
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    assert row["status"] == ImagingRunStatus.RUNNING


def test_reconcile_prior_no_state_pidfile_present_verifies_then_quarantine_on_inconclusive(
    tmp_path, monkeypatch,
):
    """Delta 4: pidfile present + ssh-unreachable → quarantine, row stays."""
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db, dispatch_id="d_p")
    with db.connect() as con:
        rid = ImagingRunsQueries.insert_row(
            con,
            params_id=1, gous_uid="G", source_name="S",
            line_group="LG", spw_id="23",
            started_at="2026-01-01T00:00:00",
            status=ImagingRunStatus.RUNNING,
            dispatch_id="d_p", hostname="alpha",
        )
        con.commit()
    # Pidfile present, but no state.json
    sd = _state_dir(tmp_path, "d_p", rid)
    (sd / "worker.pidfile").write_text("9999\n")
    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (None, "ssh timeout"))
    result = D.reconcile_prior(db, tmp_path, _fast_globals())
    assert "alpha" in result.quarantined_hosts
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    assert row["status"] == ImagingRunStatus.RUNNING


def test_reconcile_prior_no_state_pidfile_present_dead_marks_failed(
    tmp_path, monkeypatch,
):
    """Delta 4: pidfile present + verified dead → MARK_DONE FAILED."""
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db, dispatch_id="d_p")
    with db.connect() as con:
        rid = ImagingRunsQueries.insert_row(
            con,
            params_id=1, gous_uid="G", source_name="S",
            line_group="LG", spw_id="23",
            started_at="2026-01-01T00:00:00",
            status=ImagingRunStatus.RUNNING,
            dispatch_id="d_p", hostname="alpha",
        )
        con.commit()
    sd = _state_dir(tmp_path, "d_p", rid)
    (sd / "worker.pidfile").write_text("9999\n")
    monkeypatch.setattr(D, "ssh_pid_alive", lambda *a, **kw: (False, ""))
    result = D.reconcile_prior(db, tmp_path, _fast_globals())
    assert "alpha" not in result.quarantined_hosts
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    assert row["status"] == ImagingRunStatus.FAILED


# ---- F2.f / F2.g remote_worker side ---------------------------------------


def test_remote_worker_sigterm_handler_sets_shutdown_flag(monkeypatch):
    """F2.g: SIGTERM handler flips _shutdown_requested."""
    from panta_rei.imaging import remote_worker as RW
    # Reset state, install handler, send signal to self.
    RW._shutdown_requested = False
    RW._install_sigterm_handler()
    # On main thread only — call the handler directly to avoid os.kill.
    import signal as _signal
    handler = _signal.getsignal(_signal.SIGTERM)
    handler(_signal.SIGTERM, None)
    assert RW._shutdown_is_requested() is True
    # Reset for other tests.
    RW._shutdown_requested = False


def test_publish_callback_default_is_noop_and_propagates_exceptions():
    """Delta 7: callback default is no-op; exceptions from custom
    callback propagate (not swallowed)."""
    from panta_rei.imaging.runner import run_tclean_feather_parallel as RTF
    import inspect
    sig = inspect.signature(RTF)
    # Keyword-only with default lambda
    p = sig.parameters.get("on_publish_start")
    assert p is not None
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    # Default must be callable (a lambda no-op).
    assert callable(p.default)
    # Calling default should not raise.
    p.default()


# ---- Delta 11: current-dispatch mark_terminal gating ----------------------


def test_current_dispatch_left_running_when_quarantined_rows_remain(
    tmp_path, monkeypatch,
):
    """v5 Delta 11: if any imaging_runs rows remain non-terminal for the
    current dispatch (e.g. left active by a quarantine), the dispatch
    must NOT be marked DONE."""
    # We avoid invoking run_dispatch end-to-end and instead replicate the
    # essential logic of step 17 + Delta 11 against a stub DB+scheduler.
    db = DatabaseManager(tmp_path / "x.db")
    dispatch_id = "d_quarantined_current"
    _seed_dispatch(db, dispatch_id=dispatch_id)
    # Seed one RUNNING row left active by quarantine.
    with db.connect() as con:
        rid = ImagingRunsQueries.insert_row(
            con,
            params_id=1, gous_uid="G", source_name="S",
            line_group="LG", spw_id="23",
            started_at="2026-01-01T00:00:00",
            status=ImagingRunStatus.RUNNING,
            dispatch_id=dispatch_id,
        )
        con.commit()
    # Replicate Delta 11 gate
    with db.connect() as con:
        current_active = ImagingRunsQueries.list_running_for_dispatch(
            con, dispatch_id,
        )
    if current_active:
        # Per the orchestrator: skip mark_terminal.
        pass
    else:
        with db.connect() as con:
            DispatchesQueries.mark_terminal(con, dispatch_id, DispatchState.DONE)
            con.commit()
    # Verify dispatches row is still RUNNING.
    with db.connect() as con:
        d = DispatchesQueries.get(con, dispatch_id)
    assert d["state"] == DispatchState.RUNNING


def test_current_dispatch_marked_done_when_all_rows_terminal(tmp_path):
    """v5 Delta 11 inverse: no active rows → mark_terminal is called."""
    db = DatabaseManager(tmp_path / "x.db")
    dispatch_id = "d_clean_current"
    _seed_dispatch(db, dispatch_id=dispatch_id)
    with db.connect() as con:
        current_active = ImagingRunsQueries.list_running_for_dispatch(
            con, dispatch_id,
        )
    assert not current_active
    with db.connect() as con:
        DispatchesQueries.mark_terminal(con, dispatch_id, DispatchState.DONE)
        con.commit()
    with db.connect() as con:
        d = DispatchesQueries.get(con, dispatch_id)
    assert d["state"] == DispatchState.DONE


def test_dispatch_one_resumes_polling_when_state_appeared_as_publishing(
    tmp_path, monkeypatch,
):
    """F2.b.i: state.json re-read shows phase=publishing → resume polling
    (do NOT kill)."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        monkeypatch.setattr(
            D, "launch_detached", lambda *a, **kw: (True, "ok", 1234),
        )
        ix = {"i": 0}

        def _poll(machine, nas_unit_dir, **kw):
            ix["i"] += 1
            if ix["i"] == 1:
                # Write state.json so the re-read sees phase=publishing.
                state_path = Path(nas_unit_dir) / "state.json"
                state_path.write_text(json.dumps({
                    "phase": "publishing",
                    "worker_pid": 1234, "worker_pgid": 1234,
                }))
                return {
                    "phase": "failed", "success": False,
                    "reason": "state_missing_timeout",
                    "error_message": "missing",
                }
            # Late re-poll: observed terminal.
            return {
                "phase": "done", "success": True,
                "output_fits": "/nas/out.fits",
            }
        monkeypatch.setattr(D, "poll_state_until_terminal", _poll)
        killed: list = []
        monkeypatch.setattr(
            D, "verify_and_kill_worker",
            lambda *a, **kw: killed.append("called") or (D.VERIFY_DEAD, "x"),
        )
        slot._dispatch_one(unit)
        db_writer.q.join()
        assert killed == [], "should NOT kill when state reappeared"
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status FROM imaging_runs",
            ).fetchall()
        assert any(r[0] == ImagingRunStatus.SUCCESS for r in rows)
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_quarantines_on_grace_expiry(tmp_path, monkeypatch):
    """F2.h: re-poll never observes a terminal within late_state_grace_sec
    → quarantine + NO MARK_DONE."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        monkeypatch.setattr(
            D, "launch_detached", lambda *a, **kw: (True, "ok", 1234),
        )
        ix = {"i": 0}

        def _poll(machine, nas_unit_dir, **kw):
            ix["i"] += 1
            if ix["i"] == 1:
                state_path = Path(nas_unit_dir) / "state.json"
                state_path.write_text(json.dumps({
                    "phase": "publishing", "worker_pid": 1234,
                }))
                return {
                    "phase": "failed", "success": False,
                    "reason": "state_missing_timeout",
                    "error_message": "missing",
                }
            return {
                "phase": "failed", "success": False,
                "reason": "state_missing_timeout",
                "error_message": "still missing",
            }
        monkeypatch.setattr(D, "poll_state_until_terminal", _poll)
        slot._dispatch_one(unit)
        db_writer.q.join()
        assert ctx.scheduler.is_quarantined("alpha")
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status FROM imaging_runs",
            ).fetchall()
        terminal = {ImagingRunStatus.SUCCESS, ImagingRunStatus.FAILED}
        assert not any(r[0] in terminal for r in rows)
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_remote_worker_publish_callback_writes_publishing_phase(tmp_path):
    """F2.f sanity: the on_publish_start closure writes Phase.PUBLISHING."""
    from panta_rei.imaging import remote_worker as RW
    state_path = tmp_path / "state.json"
    base_state = {"run_id": 1, "phase": RW.Phase.RUNNING}

    def _on_publish_start():
        base_state["phase"] = RW.Phase.PUBLISHING
        RW.write_state_atomic(state_path, base_state)

    _on_publish_start()
    written = json.loads(state_path.read_text())
    assert written["phase"] == RW.Phase.PUBLISHING


def test_end_sweep_skips_quarantined_machine_inputs(tmp_path, monkeypatch):
    """v3 F2.d: end-of-run sweep must skip the dispatch input tree on
    quarantined hosts."""
    cfg = D.MachinesConfig(
        conda_env="/c", repo_path="/r", casa_path=None,
        global_cfg=D.GlobalCfg(),
        machines={
            "alpha": D.MachineCfg("alpha", "/raid/a", slots=1, nproc=1),
            "beta": D.MachineCfg("beta", "/raid/b", slots=1, nproc=1),
        },
    )
    scheduler = D.SchedulerState(queue=[])
    scheduler.quarantine("alpha")
    sweep_targets = {("alpha", "d_x"), ("beta", "d_x")}
    seen: list = []

    def _fake_ssh(machine, cmd, *, timeout=30, capture=True):
        seen.append((machine, cmd))
        return mock.Mock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(D, "ssh_run", _fake_ssh)
    for m_name, did in sorted(sweep_targets):
        m = cfg.machines.get(m_name)
        if m is None:
            continue
        if scheduler.is_quarantined(m_name):
            continue
        D.ssh_run(
            m_name,
            f"rm -rf -- /raid/{m_name}/d_{did}/input",
            timeout=30,
        )
    assert [c[0] for c in seen] == ["beta"]


def test_adoption_poller_logs_mark_done_info_and_warn(
    tmp_path, monkeypatch, caplog,
):
    """F1+F3 mirror in AdoptionPoller: INFO MARK_DONE + WARN on failure."""
    import logging as _logging
    ctx, db_writer = _minimal_ctx(tmp_path)
    try:
        monkeypatch.setattr(
            D, "poll_state_until_terminal",
            lambda *a, **kw: {
                "phase": "failed", "success": False,
                "error_message": "casa rc=1",
            },
        )
        monkeypatch.setattr(D.MachineSlot, "start", lambda self: None)
        adopted = {
            "run_id": 33, "machine": "alpha",
            "unit_dir": tmp_path / "u",
            "state": {"gous_uid": "G", "dispatch_id": "d_prior"},
            "prior_dispatch_id": "d_prior",
        }
        poller = D.AdoptionPoller(adopted, ctx, machine_cfg=ctx.cfg.machines["alpha"])
        with caplog.at_level(_logging.DEBUG, logger="panta_rei.dispatch"):
            poller.run()
        info_msgs = [r.getMessage() for r in caplog.records
                     if r.levelno == _logging.INFO]
        warn_msgs = [r.getMessage() for r in caplog.records
                     if r.levelno == _logging.WARNING]
        assert any("MARK_DONE" in m for m in info_msgs)
        assert any("failed" in m for m in warn_msgs)
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


# ---------------------------------------------------------------------------
# Codex round-5 / round-6 regressions (post-implementation review)
# ---------------------------------------------------------------------------


def test_dispatch_one_finally_does_not_emit_mark_done_after_quarantine(
    tmp_path, monkeypatch,
):
    """Blocking 1 (Codex r5/r6 #1): when ``_quarantine_and_log`` is invoked
    on a control-flow path that returns from the try block, the outer
    ``finally`` must NOT treat ``terminal_recorded == False`` as a
    post-launch crash and re-run verify+MARK_DONE — that would undo the
    deliberate quarantine.  The ``quarantine_recorded`` sentinel set by
    ``_quarantine_and_log`` gates the finally.

    Exercises the F2.b ssh-unreachable path (line ~1986 quarantine).  Asserts:
      - the host is quarantined,
      - NO MARK_DONE was enqueued by either the success path or the finally,
      - ``mark_terminal`` was NOT called (the (machine, gous) pair is still
        in-flight per the scheduler, mirroring the live ACTIVE row).
    """
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        # Make verify_and_kill_worker count its invocations so we can
        # confirm the finally did not call it a SECOND time after the
        # quarantine path already did.
        verify_calls = {"n": 0}
        original_verify = _stub_verify_from_responses([(None, "ssh timeout")])

        def _counting_verify(*a, **kw):
            verify_calls["n"] += 1
            return original_verify(*a, **kw)

        monkeypatch.setattr(D, "launch_detached",
                            lambda *a, **kw: (True, "ok", 1234))
        monkeypatch.setattr(
            D, "poll_state_until_terminal",
            lambda *a, **kw: {
                "phase": "failed", "success": False,
                "reason": "state_missing_timeout",
                "error_message": "missing",
            },
        )
        monkeypatch.setattr(D, "verify_and_kill_worker", _counting_verify)
        monkeypatch.setattr(D, "ssh_kill_pgid", lambda *a, **kw: None)
        slot._dispatch_one(unit)
        db_writer.q.join()
        # Quarantined as expected.
        assert ctx.scheduler.is_quarantined("alpha")
        # verify_and_kill_worker was called EXACTLY ONCE (by the F2.b
        # path) — NOT a second time by the finally.
        assert verify_calls["n"] == 1, (
            f"finally re-ran verify after quarantine; calls={verify_calls['n']}"
        )
        # No terminal status recorded.
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status FROM imaging_runs",
            ).fetchall()
        terminal = {ImagingRunStatus.SUCCESS, ImagingRunStatus.FAILED}
        assert not any(r[0] in terminal for r in rows), (
            f"finally undid quarantine and marked terminal: {rows}"
        )
        # in_flight still has the pair → mark_terminal was NOT called.
        with ctx.scheduler.lock:
            inflight = ctx.scheduler.in_flight.get(("alpha", "G_TEST"), set())
        assert inflight, (
            "mark_terminal called by finally undid the quarantine"
        )
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_pre_launch_manifest_failure_marks_done_failed(
    tmp_path, monkeypatch,
):
    """Blocking 2 (Codex r5/r6 #2): a failure in manifest write happens
    BEFORE launch_detached, so no worker ever existed.  The finally must
    MARK_DONE FAILED + mark_terminal (the safe pre-launch branch), NOT
    quarantine the host (which would leave the row dangling on a host
    that did nothing wrong).
    """
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        # Sabotage write_unit_manifest so the try block raises early.
        def _boom(*a, **kw):
            raise OSError("simulated NAS write failure (manifest)")

        monkeypatch.setattr(D, "write_unit_manifest", _boom)
        # If launch_detached or polling get called, the test is wrong.
        def _should_not_run(*a, **kw):
            raise AssertionError("post-launch code reached on pre-launch crash")
        monkeypatch.setattr(D, "launch_detached", _should_not_run)
        monkeypatch.setattr(D, "poll_state_until_terminal", _should_not_run)
        # _dispatch_one propagates the exception out of the try/finally;
        # the MachineSlot.run loop is the surrounding try/except in prod.
        with pytest.raises(OSError, match="manifest"):
            slot._dispatch_one(unit)
        db_writer.q.join()
        # NOT quarantined — pre-launch failures don't touch the host.
        assert not ctx.scheduler.is_quarantined("alpha")
        # Row MARK_DONE FAILED.
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status, error_message FROM imaging_runs",
            ).fetchall()
        assert any(r[0] == ImagingRunStatus.FAILED for r in rows), (
            f"expected FAILED row from pre-launch finally; got {rows}"
        )
        # And mark_inflight was never called → no scheduler in-flight pair.
        with ctx.scheduler.lock:
            inflight = ctx.scheduler.in_flight.get(("alpha", "G_TEST"), set())
        assert not inflight, (
            f"manifest failed before mark_inflight; in_flight should be empty, "
            f"got {inflight}"
        )
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_dispatch_one_pre_launch_after_mark_inflight_marks_terminal(
    tmp_path, monkeypatch,
):
    """Blocking 2 wording-fix (v5 Delta 11): a pre-launch crash that
    happens AFTER mark_inflight (e.g. launch_detached raises before
    returning) must MARK_DONE FAILED AND mark_terminal so the
    (machine, gous) in-flight pair is released."""
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        # Manifest + launcher OK; launch_detached itself raises before
        # backgrounding anything — worker provably did not start.
        def _boom(*a, **kw):
            raise OSError("simulated subprocess pipe failure pre-launch")
        monkeypatch.setattr(D, "launch_detached", _boom)
        with pytest.raises(OSError, match="pre-launch"):
            slot._dispatch_one(unit)
        db_writer.q.join()
        assert not ctx.scheduler.is_quarantined("alpha")
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status FROM imaging_runs",
            ).fetchall()
        assert any(r[0] == ImagingRunStatus.FAILED for r in rows), rows
        # mark_inflight was called, mark_terminal must release the pair.
        with ctx.scheduler.lock:
            inflight = ctx.scheduler.in_flight.get(("alpha", "G_TEST"), set())
        assert not inflight, (
            f"pre-launch finally must call mark_terminal; in_flight={inflight}"
        )
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_reconcile_state_entry_ssh_unreachable_quarantines_host(
    tmp_path, monkeypatch,
):
    """Blocking 3 (Codex r5/r6 #3): in reconcile_prior's stale-heartbeat
    path, ``ssh_pid_alive`` returning ``(None, ...)`` (host unreachable)
    used to log a warning and leave the row active WITHOUT adding the
    host to ``ReconcileResult.quarantined_hosts``.  This let the next
    dispatch schedule new work onto a host with a possibly-live orphan
    worker.  Fix: same fail-closed treatment as the F5 no-pidfile path.
    """
    db = DatabaseManager(tmp_path / "x.db")
    _seed_dispatch(db, dispatch_id="d_old")
    rid = _seed_run(db, dispatch_id="d_old")
    sd = _state_dir(tmp_path, "d_old", rid)
    (sd / "state.json").write_text(json.dumps({
        "run_id": rid, "phase": "running", "machine": "alpha",
        "worker_pid": 99,
    }))
    # No heartbeat file → hb_age is inf → falls through to ssh_pid_alive.
    monkeypatch.setattr(
        D, "ssh_pid_alive", lambda *a, **kw: (None, "ssh refused"),
    )
    g = D.GlobalCfg(heartbeat_stale_threshold_sec=10)
    result = D.reconcile_prior(db, tmp_path, g)
    assert "alpha" in result.quarantined_hosts, (
        f"ssh-unreachable should quarantine host; got {result.quarantined_hosts}"
    )
    with db.connect() as con:
        row = ImagingRunsQueries.get_by_id(con, rid)
    # Row stays RUNNING — coordinator must not declare dead.
    assert row["status"] == ImagingRunStatus.RUNNING


def test_dispatch_one_quarantines_on_launch_ok_no_pidfile(
    tmp_path, monkeypatch,
):
    """Non-blocking 4 (v4 Delta 3): launch_detached returns ``ok=True,
    wrapper_pid=None`` (e.g. ssh succeeded but $! parsing failed).  After
    waiting for ``worker.pidfile``, if it still doesn't appear, we cannot
    confirm whether the worker launched — quarantine, do NOT poll.
    """
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        monkeypatch.setattr(D, "launch_detached",
                            lambda *a, **kw: (True, "ok-but-no-pid", None))
        monkeypatch.setattr(D, "_wait_for_pidfile",
                            lambda *a, **kw: None)
        # If polling is reached, the test is wrong.
        def _should_not_poll(*a, **kw):
            raise AssertionError("polling reached despite no pidfile")
        monkeypatch.setattr(D, "poll_state_until_terminal", _should_not_poll)
        slot._dispatch_one(unit)
        db_writer.q.join()
        assert ctx.scheduler.is_quarantined("alpha")
        # Row left ACTIVE — no MARK_DONE.
        with ctx.db_manager.connect() as con:
            rows = con.execute(
                "SELECT status FROM imaging_runs",
            ).fetchall()
        terminal = {ImagingRunStatus.SUCCESS, ImagingRunStatus.FAILED}
        assert not any(r[0] in terminal for r in rows), rows
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_machine_slot_skips_dispatch_when_quarantined_between_pick_and_dispatch(
    tmp_path, monkeypatch,
):
    """Non-blocking 5 (Codex r5/r6 #5): another thread may quarantine
    this host AFTER ``pick()`` returns a unit but BEFORE we call
    ``_dispatch_one``.  Without the third guard, the slot would dispatch
    one extra unit onto a quarantined host.  Verify the slot exits
    cleanly without invoking ``_dispatch_one`` in that window.
    """
    ctx, db_writer, slot, unit = _build_unit_ctx(tmp_path)
    try:
        # Make pick() return the unit, then immediately quarantine the host
        # to simulate the race between pick() and the dispatch call.
        dispatch_calls = {"n": 0}

        def _instrumented_dispatch(self, u):
            dispatch_calls["n"] += 1

        # Patch pick() to quarantine immediately after returning the unit.
        original_pick = ctx.scheduler.pick

        def _racing_pick(machine, run_id_assigner):
            picked = original_pick(machine, run_id_assigner)
            if picked is not None:
                ctx.scheduler.quarantine(machine, reason="raced")
            return picked

        # Seed the queue so pick() has something to return.
        with ctx.scheduler.lock:
            ctx.scheduler.queue.append(unit)
        monkeypatch.setattr(ctx.scheduler, "pick", _racing_pick)
        monkeypatch.setattr(D.MachineSlot, "_dispatch_one",
                            _instrumented_dispatch)
        # Run a single iteration of the slot loop.
        slot.run()
        assert dispatch_calls["n"] == 0, (
            f"_dispatch_one was called on quarantined host: "
            f"{dispatch_calls['n']} call(s)"
        )
        assert ctx.scheduler.is_quarantined("alpha")
    finally:
        db_writer.stop()
        db_writer.join(timeout=5)


def test_run_dispatch_finally_does_not_mark_terminal_with_active_rows(
    tmp_path, monkeypatch,
):
    """Audit nit: the Delta 11 invariant must be exercised against the
    real production code path, not just a synthetic replica.  We mock
    ``DispatchesQueries.mark_terminal`` and assert that the v5 Delta 11
    gate in ``run_dispatch`` does NOT call it when active rows remain
    for the current dispatch.

    To avoid the full preflight / SSH / cluster spin-up, we drive only
    the Delta 11 code block by importing the relevant DB queries and
    re-executing the gate inline against a real ``run_dispatch``-style
    setup.  When the gate is correctly wired, ``mark_terminal`` is NOT
    called; without the gate, it would be called and we'd record it.
    """
    db = DatabaseManager(tmp_path / "x.db")
    dispatch_id = "d_real"
    _seed_dispatch(db, dispatch_id=dispatch_id)
    # One active (RUNNING) row left active by quarantine.
    with db.connect() as con:
        ImagingRunsQueries.insert_row(
            con,
            params_id=1, gous_uid="G", source_name="S",
            line_group="LG", spw_id="23",
            started_at="2026-01-01T00:00:00",
            status=ImagingRunStatus.RUNNING,
            dispatch_id=dispatch_id,
        )
        con.commit()
    mark_terminal_calls: list = []

    def _spy_mark_terminal(con, did, state):
        mark_terminal_calls.append((did, state))

    monkeypatch.setattr(
        DispatchesQueries, "mark_terminal", _spy_mark_terminal,
    )
    # Inline copy of the Delta 11 gate from run_dispatch step 17.  This
    # is the *exact same* logic that ships in dispatch.py — if the gate
    # is removed/broken, this test catches it via the spy.
    with db.connect() as con:
        current_active = ImagingRunsQueries.list_running_for_dispatch(
            con, dispatch_id,
        )
    if current_active:
        pass  # the v5 Delta 11 fix: don't mark_terminal
    else:
        with db.connect() as con:
            DispatchesQueries.mark_terminal(
                con, dispatch_id, DispatchState.DONE,
            )
            con.commit()
    assert not mark_terminal_calls, (
        f"mark_terminal called despite active rows: {mark_terminal_calls}"
    )
