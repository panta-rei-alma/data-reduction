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
