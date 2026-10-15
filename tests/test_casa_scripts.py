"""Tests for the standalone CASA scripts (casatasks/casatools faked)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

CASA_DIR = Path(__file__).resolve().parent.parent / "panta_rei" / "casa"


def _load_script(name, monkeypatch, tclean=None):
    """Import ``panta_rei/casa/<name>.py`` against fake CASA modules."""
    casatasks = types.ModuleType("casatasks")
    for task in ("tclean", "feather", "exportfits", "importfits", "imregrid", "imtrans"):
        setattr(casatasks, task, lambda **_kw: None)
    if tclean is not None:
        casatasks.tclean = tclean

    def exportfits(imagename, fitsimage, overwrite=False):
        Path(fitsimage).write_text("fits")

    casatasks.exportfits = exportfits
    casatools = types.ModuleType("casatools")
    casatools.image = object
    casatools.version_string = lambda: "6.6.6.17"
    monkeypatch.setitem(sys.modules, "casatasks", casatasks)
    monkeypatch.setitem(sys.modules, "casatools", casatools)

    spec = importlib.util.spec_from_file_location(
        f"_casa_script_{name}", CASA_DIR / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestJsonSafe:
    def test_numpy_and_unserialisable_values(self, monkeypatch):
        script = _load_script("tclean_feather", monkeypatch)
        out = script.json_safe({
            "iterdone": np.int64(42),
            "peakres": np.float32(0.5),
            "summaryminor": {0: {0: {"peakRes": np.array([1.0, 2.0])}}},
            "flags": (True, np.bool_(False)),
            "odd": object(),
        })
        json.dumps(out)  # must serialise
        assert out["iterdone"] == 42
        assert out["summaryminor"]["0"]["0"]["peakRes"] == [1.0, 2.0]
        assert out["flags"] == [True, False]
        assert out["odd"].startswith("<object")


class TestTcleanContScript:
    def _job(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        imagename = str(run_dir / "img")
        job = {
            "run_dir": str(run_dir),
            "params": {"imagename": imagename},
            "canonical_paths": {"tt0": str(tmp_path / "pub" / "x.cont.tt0.pbcor.fits")},
            "aux_products": [],
        }
        job_path = run_dir / "job.json"
        job_path.write_text(json.dumps(job))
        return run_dir, imagename, job_path

    def test_records_tclean_return_and_casa_version(self, tmp_path, monkeypatch):
        run_dir, imagename, job_path = self._job(tmp_path)

        def fake_tclean(**params):
            Path(f"{params['imagename']}.image.tt0.pbcor").mkdir()
            return {
                "stopcode": np.int64(2), "stopDescription": "Reached global threshold",
                "iterdone": np.int64(1234), "nmajordone": np.int32(5),
                "summaryminor": {0: {0: {0: {"iterDone": np.array([10, 20])}}}},
            }

        script = _load_script("tclean_cont", monkeypatch, tclean=fake_tclean)
        monkeypatch.setattr(sys, "argv", ["tclean_cont.py", str(job_path)])
        assert script.main() == 0

        result = json.loads((run_dir / "result.json").read_text())
        assert result["success"] is True
        assert result["casa_version"] == "6.6.6.17"
        assert result["tclean_summary"] == {
            "stopcode": 2, "stopDescription": "Reached global threshold",
            "iterdone": 1234, "nmajordone": 5,
        }
        sidecar = json.loads((run_dir / "tclean_summary.json").read_text())
        assert sidecar["summaryminor"]["0"]["0"]["0"]["iterDone"] == [10, 20]
        assert result["tclean_summary_json"] == str(run_dir / "tclean_summary.json")

    def test_resume_skips_summary(self, tmp_path, monkeypatch):
        run_dir, imagename, job_path = self._job(tmp_path)
        Path(f"{imagename}.image.tt0.pbcor").mkdir()  # products already exist
        script = _load_script("tclean_cont", monkeypatch)
        monkeypatch.setattr(sys, "argv", ["tclean_cont.py", str(job_path)])
        assert script.main() == 0
        result = json.loads((run_dir / "result.json").read_text())
        assert result["tclean_summary"] is None
        assert result["casa_version"] == "6.6.6.17"
        assert not (run_dir / "tclean_summary.json").exists()
