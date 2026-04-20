"""Tests for panta_rei.imaging.runner — imaging execution engine."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from panta_rei.imaging.matching import ImagingUnit
from panta_rei.imaging.runner import (
    FIXED_PARAMS,
    FIXED_TCLEAN_PARAMS,
    build_sdintimaging_params,
    build_tclean_params,
    cleanup_intermediates,
    get_casa_version,
    run_sdintimaging,
    run_tclean_feather,
    run_tclean_feather_parallel,
    run_trusted_preflight,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_unit():
    """A minimal ImagingUnit for testing."""
    return ImagingUnit(
        gous_uid="X3833_X64b9",
        source_name="SRC1",
        line_group="N2H+",
        spw_id="23",
        params_id=1,
        recovered_params={
            "imsize": [480, 450],
            "cell": ["0.42arcsec"],
            "phasecenter": "J2000 07:27:07.5 -18:55:30",
            "specmode": "cube",
            "nchan": 956,
            "start": "86.05GHz",
            "width": "244.140625kHz",
            "outframe": "LSRK",
            "veltype": "radio",
            "weighting": "briggsbwtaper",
            "robust": 0.5,
            "niter": 30000,
            "threshold": "0.276Jy",
            "pblimit": 0.2,
            "usemask": "auto-multithresh",
            "sidelobethreshold": 2.0,
            "noisethreshold": 4.25,
            "lownoisethreshold": 1.5,
            "negativethreshold": 0.0,
            "minbeamfrac": 0.3,
            "growiterations": 75,
        },
        vis_tm=["/data/tm1.ms", "/data/tm2.ms"],
        vis_sm=["/data/sm1.ms"],
        sdimage="/data/tp.fits",
        mous_uids_tm=["m1"],
        mous_uids_sm=["m2"],
        mous_uids_tp=["m3"],
        tp_freq_min=86.0e9,
        tp_freq_max=86.3e9,
        ready=True,
    )


# ---------------------------------------------------------------------------
# build_sdintimaging_params
# ---------------------------------------------------------------------------

class TestBuildParams:
    def test_has_fixed_params(self, basic_unit):
        params = build_sdintimaging_params(basic_unit, imagename="/out/test")
        assert params["gridder"] == "mosaic"
        assert params["pbcor"] is True
        assert params["stokes"] == "I"
        assert params["dishdia"] == 12.0
        assert params["usedata"] == "sdint"

    def test_vis_combines_tm_and_sm(self, basic_unit):
        params = build_sdintimaging_params(basic_unit, imagename="/out/test")
        assert params["vis"] == ["/data/tm1.ms", "/data/tm2.ms", "/data/sm1.ms"]

    def test_overrides(self, basic_unit):
        params = build_sdintimaging_params(
            basic_unit,
            imagename="/out/test",
            sdgain=0.5,
            deconvolver="hogbom",
            scales=[0, 10],
        )
        assert params["sdgain"] == 0.5
        assert params["deconvolver"] == "hogbom"
        assert params["scales"] == [0, 10]

    def test_recovered_params_used(self, basic_unit):
        params = build_sdintimaging_params(basic_unit, imagename="/out/test")
        assert params["imsize"] == [480, 450]
        assert params["nchan"] == 956
        assert params["robust"] == 0.5
        assert params["phasecenter"] == "J2000 07:27:07.5 -18:55:30"

    def test_fixed_params_override_recovered(self, basic_unit):
        """Fixed params like gridder should override anything in recovered."""
        basic_unit.recovered_params["gridder"] = "standard"
        params = build_sdintimaging_params(basic_unit, imagename="/out/test")
        assert params["gridder"] == "mosaic"  # Fixed, not recovered


# ---------------------------------------------------------------------------
# cleanup_intermediates
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_removes_intermediates(self, tmp_path):
        imagename = str(tmp_path / "test_image")
        # Create expected products
        for suffix in [".joint.cube.image.pbcor", ".joint.cube.pb",
                       ".joint.cube.psf", ".joint.cube.residual",
                       ".joint.cube.model", ".joint.cube.sumwt",
                       ".joint.cube.weight", ".cf"]:
            (tmp_path / f"test_image{suffix}").mkdir()

        removed = cleanup_intermediates(imagename)
        assert removed == 6  # psf, residual, model, sumwt, weight, cf

        # Keep products should still exist
        assert (tmp_path / "test_image.joint.cube.image.pbcor").exists()
        assert (tmp_path / "test_image.joint.cube.pb").exists()

        # Deleted products should be gone
        assert not (tmp_path / "test_image.joint.cube.psf").exists()
        assert not (tmp_path / "test_image.cf").exists()

    def test_keep_all(self, tmp_path):
        imagename = str(tmp_path / "test_image")
        (tmp_path / "test_image.joint.cube.psf").mkdir()
        (tmp_path / "test_image.cf").mkdir()

        removed = cleanup_intermediates(imagename, keep_all=True)
        assert removed == 0
        assert (tmp_path / "test_image.joint.cube.psf").exists()
        assert (tmp_path / "test_image.cf").exists()

    def test_no_products_to_clean(self, tmp_path):
        imagename = str(tmp_path / "test_image")
        removed = cleanup_intermediates(imagename)
        assert removed == 0


# ---------------------------------------------------------------------------
# get_casa_version
# ---------------------------------------------------------------------------

class TestCasaVersion:
    def test_returns_unknown_when_casatools_unavailable(self):
        with patch.dict("sys.modules", {"casatools": None}):
            # This should not raise, just return 'unknown'
            version = get_casa_version()
            assert isinstance(version, str)


# ---------------------------------------------------------------------------
# Trusted preflight (mocked casatools)
# ---------------------------------------------------------------------------

class TestTrustedPreflight:
    def test_preflight_populates_selections(self, basic_unit):
        """Test that trusted preflight populates spw_selection, field_selection, datacolumn."""
        mock_casatools = MagicMock()

        # Mock msmetadata
        mock_msmd = MagicMock()
        import numpy as np
        # Target SPWs: 1 and 3 have OBSERVE_TARGET intent
        mock_msmd.spwsforintent.return_value = np.array([1, 3])
        # SPW 1 center ~86.15 GHz matches the recovered params
        mock_msmd.chanfreqs.side_effect = lambda spw_id: {
            1: np.linspace(86.0e9, 86.3e9, 960),  # This should match
            3: np.linspace(90e9, 90.2e9, 960),
        }.get(spw_id, np.array([]))
        mock_msmd.fieldsforname.return_value = [0]
        mock_casatools.msmetadata.return_value = mock_msmd

        # Mock table
        mock_tb = MagicMock()
        mock_tb.colnames.return_value = ["DATA", "CORRECTED_DATA", "FLAG"]
        mock_casatools.table.return_value = mock_tb

        with patch("panta_rei.imaging.runner._ensure_casatools", return_value=mock_casatools):
            ok, msg = run_trusted_preflight(basic_unit)

        assert ok, msg
        assert len(basic_unit.spw_selection) == 3  # 2 TM + 1 SM
        assert all(s == "1" for s in basic_unit.spw_selection)  # SPW 1 matched
        assert len(basic_unit.field_selection) == 3
        assert basic_unit.datacolumn == "corrected"

    def test_preflight_fails_on_missing_spw(self, basic_unit):
        """If no SPW matches in an MS, preflight should fail."""
        mock_casatools = MagicMock()
        mock_msmd = MagicMock()
        import numpy as np
        mock_msmd.spwsforintent.return_value = np.array([0, 1])
        # No frequency match — both SPWs are far from target
        mock_msmd.chanfreqs.side_effect = lambda spw_id: np.linspace(200e9, 200.1e9, 960)
        mock_msmd.fieldsforname.return_value = [0]  # Field exists
        mock_casatools.msmetadata.return_value = mock_msmd

        with patch("panta_rei.imaging.runner._ensure_casatools", return_value=mock_casatools):
            ok, msg = run_trusted_preflight(basic_unit)

        assert not ok
        assert "No matching SPW" in msg

    def test_preflight_fails_on_missing_field(self, basic_unit):
        """If field name doesn't resolve in any MS, preflight should fail."""
        mock_casatools = MagicMock()
        mock_msmd = MagicMock()
        import numpy as np
        mock_msmd.spwsforintent.return_value = np.array([1])
        mock_msmd.chanfreqs.side_effect = lambda spw_id: np.linspace(86.0e9, 86.3e9, 960)
        mock_msmd.fieldsforname.return_value = []  # Field not found in any MS
        mock_casatools.msmetadata.return_value = mock_msmd

        with patch("panta_rei.imaging.runner._ensure_casatools", return_value=mock_casatools):
            ok, msg = run_trusted_preflight(basic_unit)

        assert not ok
        assert "not found in any MS" in msg

    def test_preflight_always_uses_corrected(self, basic_unit):
        """Even if no CORRECTED_DATA, datacolumn should be 'corrected' (CASA falls back to data)."""
        mock_casatools = MagicMock()
        mock_msmd = MagicMock()
        import numpy as np
        mock_msmd.spwsforintent.return_value = np.array([1])
        mock_msmd.chanfreqs.side_effect = lambda spw_id: np.linspace(86.0e9, 86.3e9, 960)
        mock_msmd.fieldsforname.return_value = [0]
        mock_casatools.msmetadata.return_value = mock_msmd

        mock_tb = MagicMock()
        mock_tb.colnames.return_value = ["DATA", "FLAG"]  # No CORRECTED_DATA
        mock_casatools.table.return_value = mock_tb

        with patch("panta_rei.imaging.runner._ensure_casatools", return_value=mock_casatools):
            ok, msg = run_trusted_preflight(basic_unit)

        assert ok
        assert basic_unit.datacolumn == "corrected"

    def test_preflight_handles_mixed_datacolumn(self, basic_unit):
        """Mixed CORRECTED_DATA / DATA across MSs should succeed with corrected."""
        mock_casatools = MagicMock()
        mock_msmd = MagicMock()
        import numpy as np
        mock_msmd.spwsforintent.return_value = np.array([1])
        mock_msmd.chanfreqs.side_effect = lambda spw_id: np.linspace(86.0e9, 86.3e9, 960)
        mock_msmd.fieldsforname.return_value = [0]
        mock_casatools.msmetadata.return_value = mock_msmd

        # First two MS have CORRECTED_DATA, third does not
        call_count = [0]
        def fake_colnames():
            call_count[0] += 1
            if call_count[0] <= 2:
                return ["DATA", "CORRECTED_DATA", "FLAG"]
            return ["DATA", "FLAG"]

        mock_tb = MagicMock()
        mock_tb.colnames.side_effect = fake_colnames
        mock_casatools.table.return_value = mock_tb

        with patch("panta_rei.imaging.runner._ensure_casatools", return_value=mock_casatools):
            ok, msg = run_trusted_preflight(basic_unit)

        assert ok
        assert basic_unit.datacolumn == "corrected"

    def test_preflight_fails_without_freq_params(self):
        """If recovered params lack start/width/nchan, preflight should fail."""
        unit = ImagingUnit(
            gous_uid="g1", source_name="SRC1", line_group="N2H+",
            spw_id="23", params_id=1,
            recovered_params={},  # No frequency info
            vis_tm=["/data/tm.ms"],
            vis_sm=["/data/sm.ms"],
            ready=True,
        )
        # No need to mock casatools — it should fail before calling them
        ok, msg = run_trusted_preflight(unit)
        assert not ok
        assert "Cannot compute TM frequency" in msg


# ---------------------------------------------------------------------------
# Full run_sdintimaging (mocked CASA)
# ---------------------------------------------------------------------------

class TestRunSdintimaging:
    def test_dry_run(self, basic_unit, tmp_path):
        """Dry run should succeed without calling casatasks."""
        with patch("panta_rei.imaging.runner.run_trusted_preflight", return_value=(True, "OK")):
            success, msg, output = run_sdintimaging(
                unit=basic_unit,
                output_dir=tmp_path / "output",
                dry_run=True,
            )
        assert success
        assert msg == "dry-run"
        assert output is not None

    def test_full_run_mocked(self, basic_unit, tmp_path):
        """Full run with all CASA functions mocked."""
        output_dir = tmp_path / "output"

        # Mock the entire chain
        mock_sdintimaging = MagicMock()
        mock_exportfits = MagicMock()
        mock_importfits = MagicMock()

        # Create the pbcor directory that export expects
        def fake_sdintimaging(**kwargs):
            imagename = kwargs["imagename"]
            Path(f"{imagename}.joint.cube.image.pbcor").mkdir(parents=True)
            Path(f"{imagename}.joint.cube.pb").mkdir(parents=True)
            Path(f"{imagename}.joint.cube.psf").mkdir(parents=True)

        mock_sdintimaging.side_effect = fake_sdintimaging

        with patch("panta_rei.imaging.runner.run_trusted_preflight", return_value=(True, "OK")), \
             patch("panta_rei.imaging.runner._ensure_casatasks", return_value=(mock_sdintimaging, mock_exportfits, mock_importfits)), \
             patch("panta_rei.imaging.runner.import_tp_to_casa_image", return_value="/data/tp.image"), \
             patch("panta_rei.imaging.runner.get_casa_version", return_value="6.6.6"):

            success, msg, output_fits = run_sdintimaging(
                unit=basic_unit,
                output_dir=output_dir,
                sdgain=1.0,
                deconvolver="multiscale",
                scales=[0, 5, 10, 15, 20],
                keep_intermediates=False,
            )

        assert success, msg
        assert output_fits is not None
        assert "12m7mTP" in output_fits
        assert ".pbcor.fits" in output_fits

        # Verify sdintimaging was called
        mock_sdintimaging.assert_called_once()
        call_kwargs = mock_sdintimaging.call_args[1]
        assert call_kwargs["gridder"] == "mosaic"
        assert call_kwargs["usedata"] == "sdint"
        assert call_kwargs["sdimage"] == "/data/tp.image"

        # Verify job spec JSON was written
        jobs_dir = output_dir / "jobs"
        assert jobs_dir.exists()
        job_files = list(jobs_dir.glob("*.json"))
        assert len(job_files) == 1
        job_spec = json.loads(job_files[0].read_text())
        assert job_spec["overrides"]["sdgain"] == 1.0
        assert job_spec["casa_version"] == "6.6.6"

    def test_handles_sdintimaging_failure(self, basic_unit, tmp_path):
        """If sdintimaging raises, should return failure gracefully."""
        output_dir = tmp_path / "output"

        mock_sdintimaging = MagicMock(side_effect=RuntimeError("CASA error"))
        mock_exportfits = MagicMock()
        mock_importfits = MagicMock()

        with patch("panta_rei.imaging.runner.run_trusted_preflight", return_value=(True, "OK")), \
             patch("panta_rei.imaging.runner._ensure_casatasks", return_value=(mock_sdintimaging, mock_exportfits, mock_importfits)), \
             patch("panta_rei.imaging.runner.import_tp_to_casa_image", return_value="/data/tp.image"), \
             patch("panta_rei.imaging.runner.get_casa_version", return_value="6.6.6"):

            with pytest.raises(RuntimeError, match="CASA error"):
                run_sdintimaging(
                    unit=basic_unit,
                    output_dir=output_dir,
                )


# ---------------------------------------------------------------------------
# build_tclean_params
# ---------------------------------------------------------------------------

class TestBuildTcleanParams:
    def test_has_fixed_tclean_params(self, basic_unit):
        params = build_tclean_params(basic_unit, imagename="/out/test")
        assert params["gridder"] == "mosaic"
        assert params["pbcor"] is True
        assert params["stokes"] == "I"

    def test_no_sdintimaging_params(self, basic_unit):
        """tclean params should NOT contain sdintimaging-specific keys."""
        params = build_tclean_params(basic_unit, imagename="/out/test")
        assert "sdimage" not in params
        assert "sdgain" not in params
        assert "sdpsf" not in params
        assert "usedata" not in params
        assert "dishdia" not in params

    def test_vis_combines_tm_and_sm(self, basic_unit):
        params = build_tclean_params(basic_unit, imagename="/out/test")
        assert params["vis"] == ["/data/tm1.ms", "/data/tm2.ms", "/data/sm1.ms"]

    def test_parallel_flag(self, basic_unit):
        params = build_tclean_params(basic_unit, imagename="/out/test", parallel=True)
        assert params["parallel"] is True

        params2 = build_tclean_params(basic_unit, imagename="/out/test", parallel=False)
        assert params2["parallel"] is False

    def test_overrides(self, basic_unit):
        params = build_tclean_params(
            basic_unit, imagename="/out/test",
            deconvolver="hogbom", scales=[0, 10],
        )
        assert params["deconvolver"] == "hogbom"
        assert params["scales"] == [0, 10]

    def test_recovered_params_used(self, basic_unit):
        params = build_tclean_params(basic_unit, imagename="/out/test")
        assert params["imsize"] == [480, 450]
        assert params["nchan"] == 956
        assert params["robust"] == 0.5
        assert params["threshold"] == "0.276Jy"


# ---------------------------------------------------------------------------
# run_tclean_feather (mocked CASA)
# ---------------------------------------------------------------------------

class TestRunTcleanFeather:
    def test_dry_run(self, basic_unit, tmp_path):
        """Dry run should succeed without calling casatasks."""
        with patch("panta_rei.imaging.runner.run_trusted_preflight", return_value=(True, "OK")):
            success, msg, output = run_tclean_feather(
                unit=basic_unit,
                output_dir=tmp_path / "output",
                row_id=99,
                dry_run=True,
            )
        assert success
        assert msg == "dry-run"
        assert output is not None
        assert "12m7mTP" in output

    def test_full_run_mocked(self, basic_unit, tmp_path):
        """Full tclean+feather run with all CASA functions mocked."""
        output_dir = tmp_path / "output"

        mock_tclean = MagicMock()
        mock_feather = MagicMock()
        mock_importfits = MagicMock()
        mock_imregrid = MagicMock()

        # exportfits creates the output FITS file
        def fake_exportfits(imagename, fitsimage, **kwargs):
            Path(fitsimage).parent.mkdir(parents=True, exist_ok=True)
            Path(fitsimage).write_text("fake fits")

        mock_exportfits = MagicMock(side_effect=fake_exportfits)

        # tclean creates .image.pbcor and .image directories
        def fake_tclean(**kwargs):
            imagename = kwargs["imagename"]
            Path(f"{imagename}.image.pbcor").mkdir(parents=True)
            Path(f"{imagename}.image").mkdir(parents=True)
            Path(f"{imagename}.pb").mkdir(parents=True)
            Path(f"{imagename}.psf").mkdir(parents=True)

        mock_tclean.side_effect = fake_tclean

        # importfits creates the TP image
        def fake_importfits(fitsimage, imagename, **kwargs):
            Path(imagename).mkdir(parents=True, exist_ok=True)

        mock_importfits.side_effect = fake_importfits

        # imregrid creates the regridded TP
        def fake_imregrid(imagename, output, **kwargs):
            Path(output).mkdir(parents=True, exist_ok=True)

        mock_imregrid.side_effect = fake_imregrid

        # feather creates the feathered image
        def fake_feather(imagename, highres, lowres):
            Path(imagename).mkdir(parents=True, exist_ok=True)

        mock_feather.side_effect = fake_feather

        # Mock prepare_tp_for_feather to return a fake regridded TP path
        def fake_prepare_tp(tp_fits_path, tclean_image, run_dir):
            regridded = Path(run_dir) / "tp" / "tp_regridded.image"
            regridded.mkdir(parents=True, exist_ok=True)
            return str(regridded)

        with patch("panta_rei.imaging.runner.run_trusted_preflight", return_value=(True, "OK")), \
             patch("panta_rei.imaging.runner._ensure_casatasks_tclean", return_value=(mock_tclean, mock_feather, mock_exportfits, mock_importfits)), \
             patch("panta_rei.imaging.runner.prepare_tp_for_feather", side_effect=fake_prepare_tp), \
             patch("panta_rei.imaging.runner.get_casa_version", return_value="6.6.6"):

            success, msg, output_fits = run_tclean_feather(
                unit=basic_unit,
                output_dir=output_dir,
                row_id=42,
            )

        assert success, msg
        assert output_fits is not None
        assert "12m7mTP" in output_fits

        # Verify tclean was called (not sdintimaging)
        mock_tclean.assert_called_once()
        call_kwargs = mock_tclean.call_args[1]
        assert call_kwargs["gridder"] == "mosaic"
        assert "sdimage" not in call_kwargs
        assert "sdgain" not in call_kwargs

        # Verify feather was called
        mock_feather.assert_called_once()
        feather_kwargs = mock_feather.call_args
        assert "pbcor" in feather_kwargs[1]["highres"]

        # Verify exportfits was called (at least twice: tclean + feathered)
        assert mock_exportfits.call_count >= 2
        # Verify no dropdeg in any exportfits call
        for call in mock_exportfits.call_args_list:
            if call[1]:
                assert call[1].get("dropdeg") is not True

        # Verify per-run work directory was created
        run_dir = output_dir / "runs" / "42"
        assert run_dir.exists()

        # Verify job spec JSON
        job_json = run_dir / "job.json"
        assert job_json.exists()
        spec = json.loads(job_json.read_text())
        assert spec["method"] == "tclean_feather"
        assert "sdgain" not in spec["overrides"]
        assert spec["canonical_paths"]["feathered"].endswith(".fits")
        assert spec["canonical_paths"]["tclean_only"].endswith(".fits")
        assert "12m7mTP" in spec["canonical_paths"]["feathered"]
        assert "12m7m." in spec["canonical_paths"]["tclean_only"]

    def test_preflight_failure(self, basic_unit, tmp_path):
        """If trusted preflight fails, should return failure."""
        with patch("panta_rei.imaging.runner.run_trusted_preflight", return_value=(False, "no SPW")):
            success, msg, output = run_tclean_feather(
                unit=basic_unit,
                output_dir=tmp_path / "output",
                row_id=1,
            )
        assert not success
        assert "preflight" in msg.lower()


# ---------------------------------------------------------------------------
# Parallel tclean+feather (mocked subprocess)
# ---------------------------------------------------------------------------

class TestRunTcleanFeatherParallel:
    def test_dry_run(self, basic_unit, tmp_path):
        """Dry run should succeed without launching subprocess."""
        casa_path = tmp_path / "casa"
        (casa_path / "bin").mkdir(parents=True)
        (casa_path / "bin" / "mpicasa").touch()
        (casa_path / "bin" / "casa").touch()

        with patch("panta_rei.imaging.runner.run_trusted_preflight", return_value=(True, "OK")):
            success, msg, output = run_tclean_feather_parallel(
                unit=basic_unit,
                output_dir=tmp_path / "output",
                row_id=99,
                casa_path=str(casa_path),
                dry_run=True,
            )
        assert success
        assert msg == "dry-run"

    def test_missing_casa_path(self, basic_unit, tmp_path):
        """Should fail if CASA_PATH is not provided."""
        success, msg, _ = run_tclean_feather_parallel(
            unit=basic_unit,
            output_dir=tmp_path / "output",
            row_id=1,
            casa_path=None,
        )
        assert not success
        assert "CASA_PATH" in msg

    def test_subprocess_success(self, basic_unit, tmp_path):
        """Successful subprocess should publish canonical FITS."""
        casa_path = tmp_path / "casa"
        (casa_path / "bin").mkdir(parents=True)
        (casa_path / "bin" / "mpicasa").touch()
        (casa_path / "bin" / "casa").touch()

        output_dir = tmp_path / "output"

        # Mock subprocess to write result.json and fake FITS
        def fake_subprocess_run(cmd, **kwargs):
            # Extract run_dir from the job.json written by the function
            job_json = Path(cmd[-1])
            spec = json.loads(job_json.read_text())
            run_dir = Path(spec["run_dir"])

            # Create fake output FITS in run_dir
            feathered_name = Path(spec["canonical_paths"]["feathered"]).name
            tclean_name = Path(spec["canonical_paths"]["tclean_only"]).name
            (run_dir / feathered_name).write_text("fake feathered")
            (run_dir / tclean_name).write_text("fake tclean")

            # Write result.json
            result = {
                "success": True,
                "feathered_fits": str(run_dir / feathered_name),
                "tclean_fits": str(run_dir / tclean_name),
                "error_message": None,
            }
            (run_dir / "result.json").write_text(json.dumps(result))

            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        with patch("panta_rei.imaging.runner.run_trusted_preflight", return_value=(True, "OK")), \
             patch("panta_rei.imaging.runner.get_casa_version", return_value="6.6.6"), \
             patch("subprocess.run", side_effect=fake_subprocess_run) as mock_sub:

            success, msg, output = run_tclean_feather_parallel(
                unit=basic_unit,
                output_dir=output_dir,
                row_id=77,
                nproc=4,
                casa_path=str(casa_path),
            )

        assert success, msg
        assert "12m7mTP" in output

        # Verify subprocess was called with mpicasa
        mock_sub.assert_called_once()
        cmd = mock_sub.call_args[0][0]
        assert "mpicasa" in cmd[0]
        assert "-n" in cmd
        assert "4" in cmd
        assert any("tclean_feather.py" in c for c in cmd)

        # Verify canonical FITS were published
        assert Path(output).exists()
        assert "fake feathered" in Path(output).read_text()
