"""Tests for panta_rei.imaging.remote_worker — non-CASA logic only."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from panta_rei.imaging import remote_worker as W


def test_write_state_atomic_replaces_existing(tmp_path):
    p = tmp_path / "state.json"
    W.write_state_atomic(p, {"phase": "starting"})
    W.write_state_atomic(p, {"phase": "running", "pid": 1234})
    state = W.read_state(p)
    assert state == {"phase": "running", "pid": 1234}
    leftover = list(tmp_path.glob("state.json.tmp*"))
    assert leftover == []


def test_read_state_missing_returns_empty(tmp_path):
    assert W.read_state(tmp_path / "nope.json") == {}


def test_patch_unit_paths_remaps_to_raid(tmp_path):
    gous_input = tmp_path / "gous"
    (gous_input / "ms").mkdir(parents=True)
    (gous_input / "tp").mkdir(parents=True)
    unit = {
        "vis_tm": ["/nas/a.ms", "/nas/b.ms"],
        "vis_sm": ["/nas/c.ms"],
        "sdimage": "/nas/tp_X.fits",
    }
    W._patch_unit_paths(unit, gous_input)
    assert unit["vis_tm"] == [
        str(gous_input / "ms" / "a.ms"),
        str(gous_input / "ms" / "b.ms"),
    ]
    assert unit["vis_sm"] == [str(gous_input / "ms" / "c.ms")]
    assert unit["sdimage"] == str(gous_input / "tp" / "tp_X.fits")


def test_heartbeat_thread_touches_file(tmp_path):
    hb_path = tmp_path / "hb"
    hb = W._Heartbeat(hb_path, interval_sec=0.1)
    hb.start()
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if hb_path.exists() and time.time() - hb_path.stat().st_mtime < 0.5:
                break
            time.sleep(0.05)
    finally:
        hb.stop()
    assert hb_path.exists()


def test_copy_provenance_copies_present_files(tmp_path):
    work_run = tmp_path / "work" / "runs" / "1"
    work_run.mkdir(parents=True)
    (work_run / "job.json").write_text(json.dumps({"k": "v"}))
    (work_run / "result.json").write_text(json.dumps({"success": True}))
    log_path = tmp_path / "logs" / "unit_1.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("hello")
    nas_unit = tmp_path / "nas" / "u1"
    out = W._copy_provenance_to_nas(work_run, log_path, nas_unit)
    assert (nas_unit / "job.json").exists()
    assert (nas_unit / "result.json").exists()
    assert (nas_unit / "unit.log").exists()
    assert out["job_json"] == str(nas_unit / "job.json")
    assert out["log"] == str(nas_unit / "unit.log")


def test_copy_provenance_handles_missing_optional(tmp_path):
    work_run = tmp_path / "work" / "runs" / "2"
    work_run.mkdir(parents=True)
    # Only job.json present
    (work_run / "job.json").write_text("{}")
    log_path = tmp_path / "missing.log"
    nas_unit = tmp_path / "nas" / "u2"
    out = W._copy_provenance_to_nas(work_run, log_path, nas_unit)
    assert out["job_json"] is not None
    assert out["result_json"] is None
    assert out["log"] is None


def test_preflight_emits_json(tmp_path, capsys, monkeypatch):
    raid = tmp_path / "raid"
    nas = tmp_path / "nas-marker"
    nas.write_text("ok")
    rc = W.main([
        "preflight", "--raid-dir", str(raid),
        "--required-gb", "0",
        "--nas-check-path", str(nas),
    ])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out.splitlines()[-1])
    assert parsed["ok"] is True
    assert parsed["raid_writable"] is True
    assert parsed["nas_visible"] is True
