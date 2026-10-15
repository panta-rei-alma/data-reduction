"""Tests for the thin panta-rei-continuum-dispatch wrapper."""

from __future__ import annotations

from unittest import mock

from panta_rei.cli import run_continuum_dispatch


def test_injects_product_type_cont():
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    with mock.patch.object(run_continuum_dispatch, "_dispatch_main", fake_main):
        rc = run_continuum_dispatch.main(["--base-dir", "/b", "--limit", "5"])

    assert rc == 0
    assert captured["argv"][:2] == ["--product-type", "cont"]
    assert "--base-dir" in captured["argv"] and "5" in captured["argv"]


def test_slots_override_mutates_each_machine(tmp_path, monkeypatch):
    """--slots overrides slots-per-machine without touching machines.json."""
    import json as _json
    from panta_rei.cli import run_imaging_dispatch as rid

    mj = tmp_path / "machines.json"
    mj.write_text(_json.dumps({
        "conda_env": "/e", "repo_path": "/r", "casa_path": "/c",
        "global": {}, "machines": {
            "m1": {"raid": "/raid", "slots": 1, "nproc": 4},
            "m2": {"raid": "/raid", "slots": 1, "nproc": 4},
        },
    }))
    captured = {}

    def fake_run_dispatch(**kw):
        captured["slots"] = {n: m.slots for n, m in kw["cfg"].machines.items()}
        return {"to_run": 0}

    monkeypatch.setattr(rid, "run_dispatch", fake_run_dispatch)
    monkeypatch.setattr(rid.DatabaseManager, "__init__", lambda self, *a, **k: None)
    rid.main([
        "--base-dir", str(tmp_path), "--output-dir", str(tmp_path),
        "--machines-config", str(mj), "--db", str(tmp_path / "x.db"),
        "--slots", "2",
    ])
    assert captured["slots"] == {"m1": 2, "m2": 2}


def test_does_not_double_inject_when_already_present():
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    with mock.patch.object(run_continuum_dispatch, "_dispatch_main", fake_main):
        run_continuum_dispatch.main(["--product-type", "cont", "--base-dir", "/b"])

    assert captured["argv"].count("--product-type") == 1


def test_rejects_explicit_line_product_type():
    called = []
    with mock.patch.object(
        run_continuum_dispatch, "_dispatch_main", lambda argv: called.append(argv)
    ):
        assert run_continuum_dispatch.main(["--product-type", "line"]) == 2
        assert run_continuum_dispatch.main(["--product-type=line"]) == 2
    assert called == []


def test_accepts_equals_form_cont():
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    with mock.patch.object(run_continuum_dispatch, "_dispatch_main", fake_main):
        assert run_continuum_dispatch.main(["--product-type=cont"]) == 0
    assert captured["argv"] == ["--product-type=cont"]
