"""Tests for panta_rei.imaging.remote_worker — non-CASA logic only."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

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
    (work_run / "tclean_summary.json").write_text(json.dumps({"iterdone": 5}))
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
    assert out["tclean_summary_json"] == str(nas_unit / "tclean_summary.json")
    assert (nas_unit / "tclean_summary.json").exists()


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
    assert out["tclean_summary_json"] is None
    assert out["log"] is None


def test_terminal_provenance_survives_post_casa_copy_failure(tmp_path, monkeypatch):
    """CASA wrote result.json, then the NAS copy failed: the crash path
    still records the cleaning CASA version; the strict path re-raises."""
    work_run = tmp_path / "work" / "runs" / "3"
    work_run.mkdir(parents=True)
    (work_run / "result.json").write_text(
        json.dumps({"success": True, "casa_version": "6.6.6.17"})
    )
    log_path = tmp_path / "unit.log"
    log_path.write_text("log")

    def broken_copy(*_a, **_kw):
        raise OSError("NAS gone")

    monkeypatch.setattr(W.shutil, "copy2", broken_copy)
    fields = W._terminal_provenance(work_run, log_path, tmp_path / "nas",
                                    best_effort=True)
    assert fields == {"casa_version": "6.6.6.17"}
    with pytest.raises(OSError):
        W._terminal_provenance(work_run, log_path, tmp_path / "nas")


def test_terminal_provenance_happy_path(tmp_path):
    work_run = tmp_path / "work" / "runs" / "4"
    work_run.mkdir(parents=True)
    (work_run / "result.json").write_text(json.dumps({"casa_version": "6.6.6.17"}))
    fields = W._terminal_provenance(work_run, tmp_path / "none.log", tmp_path / "nas")
    assert fields["casa_version"] == "6.6.6.17"
    assert fields["provenance"]["result_json"] == str(tmp_path / "nas" / "result.json")


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
