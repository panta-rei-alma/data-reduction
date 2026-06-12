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


def test_does_not_double_inject_when_already_present():
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    with mock.patch.object(run_continuum_dispatch, "_dispatch_main", fake_main):
        run_continuum_dispatch.main(["--product-type", "cont", "--base-dir", "/b"])

    assert captured["argv"].count("--product-type") == 1
