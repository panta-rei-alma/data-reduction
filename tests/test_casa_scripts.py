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


def _load_script(name, monkeypatch, tclean=None, version="6.6.6.17",
                 exportfits=None):
    """Import ``panta_rei/casa/<name>.py`` against fake CASA modules."""
    casatasks = types.ModuleType("casatasks")
    for task in ("tclean", "feather", "exportfits", "importfits", "imregrid", "imtrans"):
        setattr(casatasks, task, lambda **_kw: None)
    if tclean is not None:
        casatasks.tclean = tclean

    def _exportfits(imagename, fitsimage, overwrite=False):
        Path(fitsimage).write_text("fits")

    casatasks.exportfits = exportfits if exportfits is not None else _exportfits
    casatools = types.ModuleType("casatools")
    casatools.image = object
    casatools.version_string = lambda: version
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
        assert result["export_casa_version"] == "6.6.6.17"
        assert result["tclean_summary"] == {
            "stopcode": 2, "stopDescription": "Reached global threshold",
            "iterdone": 1234, "nmajordone": 5,
        }
        record = json.loads((run_dir / "tclean_summary.json").read_text())
        assert record["completed"] is True
        assert record["casa_version"] == "6.6.6.17"
        assert record["summary"]["summaryminor"]["0"]["0"]["0"]["iterDone"] == [10, 20]
        assert result["tclean_summary_json"] == str(run_dir / "tclean_summary.json")

    def test_resume_without_prior_provenance(self, tmp_path, monkeypatch):
        run_dir, imagename, job_path = self._job(tmp_path)
        Path(f"{imagename}.image.tt0.pbcor").mkdir()  # products already exist
        script = _load_script("tclean_cont", monkeypatch)
        monkeypatch.setattr(sys, "argv", ["tclean_cont.py", str(job_path)])
        assert script.main() == 0
        result = json.loads((run_dir / "result.json").read_text())
        assert result["tclean_summary"] is None
        assert result["casa_version"] == "unknown"
        assert result["export_casa_version"] == "6.6.6.17"
        assert not (run_dir / "tclean_summary.json").exists()

    def _run(self, job_path, monkeypatch, **load):
        script = _load_script("tclean_cont", monkeypatch, **load)
        monkeypatch.setattr(sys, "argv", ["tclean_cont.py", str(job_path)])
        return script, script.main()

    def _result(self, run_dir):
        return json.loads((run_dir / "result.json").read_text())

    def test_resume_ignores_legacy_result_json_stamp(self, tmp_path, monkeypatch):
        """A (cleaned, no record) → B (legacy export-only resume stamped
        casa_version=B) → C resume: cleaning version stays unknown."""
        run_dir, imagename, job_path = self._job(tmp_path)
        Path(f"{imagename}.image.tt0.pbcor").mkdir()
        (run_dir / "result.json").write_text(json.dumps(
            {"success": True, "casa_version": "B", "tclean_summary": {"iterdone": 7}}
        ))
        _, rc = self._run(job_path, monkeypatch, version="C")
        assert rc == 0
        result = self._result(run_dir)
        assert result["casa_version"] == "unknown"
        assert result["export_casa_version"] == "C"
        assert result["tclean_summary"] is None

    def test_resume_takes_version_from_completed_record(self, tmp_path, monkeypatch):
        """A cleans (record), B resumes, C resumes: always A, with A's headline."""
        run_dir, imagename, job_path = self._job(tmp_path)

        def fake_tclean(**params):
            Path(f"{params['imagename']}.image.tt0.pbcor").mkdir()
            return {"stopcode": 2, "stopDescription": "threshold",
                    "iterdone": 123, "nmajordone": 4}

        _, rc = self._run(job_path, monkeypatch, tclean=fake_tclean, version="A")
        assert rc == 0 and self._result(run_dir)["casa_version"] == "A"
        for version in ("B", "C"):
            _, rc = self._run(job_path, monkeypatch, version=version)
            assert rc == 0
            result = self._result(run_dir)
            assert result["casa_version"] == "A"
            assert result["export_casa_version"] == version
            assert result["tclean_summary"]["iterdone"] == 123

    def test_failed_fresh_tclean_does_not_inherit_old_headline(self, tmp_path, monkeypatch):
        run_dir, imagename, job_path = self._job(tmp_path)
        (run_dir / "tclean_summary.json").write_text(json.dumps({
            "completed": True, "casa_version": "A",
            "summary": {"stopcode": 2, "iterdone": 123},
        }))

        def failing_tclean(**_params):
            raise RuntimeError("tclean blew up")

        _, rc = self._run(job_path, monkeypatch, tclean=failing_tclean, version="B")
        assert rc == 1
        record = json.loads((run_dir / "tclean_summary.json").read_text())
        assert record == {"completed": False, "casa_version": "B", "summary": None}
        result = self._result(run_dir)
        assert result["casa_version"] == "B"
        assert result["tclean_summary"] is None

        # A later resume over whatever image exists cannot claim A's run.
        Path(f"{imagename}.image.tt0.pbcor").mkdir()
        _, rc = self._run(job_path, monkeypatch, version="C")
        result = self._result(run_dir)
        assert result["casa_version"] == "unknown"
        assert result["tclean_summary"] is None

    def test_record_written_before_export_failure(self, tmp_path, monkeypatch):
        """N1: tclean succeeds, export fails — the cleaning record survives
        and a resume attributes the image to the CASA that cleaned it."""
        run_dir, imagename, job_path = self._job(tmp_path)

        def fake_tclean(**params):
            Path(f"{params['imagename']}.image.tt0.pbcor").mkdir()
            return {"stopcode": 1, "iterdone": 50}

        def broken_export(**_kw):
            raise OSError("disk full")

        _, rc = self._run(job_path, monkeypatch, tclean=fake_tclean,
                          version="A", exportfits=broken_export)
        assert rc == 1
        assert json.loads((run_dir / "tclean_summary.json").read_text())["completed"] is True
        _, rc = self._run(job_path, monkeypatch, version="B")
        assert rc == 0
        result = self._result(run_dir)
        assert result["casa_version"] == "A"
        assert result["tclean_summary"]["iterdone"] == 50

    def test_feather_script_shares_record_helpers(self, tmp_path, monkeypatch):
        run_dir, _imagename, _job_path = self._job(tmp_path)
        script = _load_script("tclean_feather", monkeypatch, version="B")
        result = {"casa_version": "unknown"}
        (run_dir / "result.json").write_text(json.dumps({"casa_version": "B"}))
        assert script.apply_cleaning_record(run_dir, result) is False
        assert result["casa_version"] == "unknown"

        calls = []
        script.run_tclean_recorded(
            lambda **kw: calls.append(kw) or {"iterdone": 9},
            {"imagename": "x"}, run_dir, result, "B",
        )
        assert result["casa_version"] == "B"
        assert result["tclean_summary"]["iterdone"] == 9
