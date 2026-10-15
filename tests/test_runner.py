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
    FIXED_TCLEAN_CONT_PARAMS,
    FIXED_TCLEAN_PARAMS,
    _publish,
    _publish_pair,
    build_sdintimaging_params,
    build_tclean_cont_params,
    build_tclean_params,
    cleanup_intermediates,
    get_casa_version,
    run_continuum_trusted_preflight,
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


class TestExecutingCasaVersion:
    def test_reads_version_from_result_json(self, tmp_path):
        from panta_rei.imaging.runner import executing_casa_version
        (tmp_path / "result.json").write_text(
            json.dumps({"success": True, "casa_version": "6.6.6.17"})
        )
        assert executing_casa_version(tmp_path) == "6.6.6.17"

    def test_none_when_missing_unknown_or_corrupt(self, tmp_path):
        from panta_rei.imaging.runner import executing_casa_version
        assert executing_casa_version(tmp_path) is None
        (tmp_path / "result.json").write_text(json.dumps({"casa_version": "unknown"}))
        assert executing_casa_version(tmp_path) is None
        (tmp_path / "result.json").write_text("{not json")
        assert executing_casa_version(tmp_path) is None


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

    def test_preflight_fails_when_no_eb_has_spw(self, basic_unit):
        """If every EB lacks a matching science SPW, preflight should fail
        with a clear no-usable-SPW message (not 'field not found')."""
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
        assert "usable" in msg and "SPW" in msg

    def test_preflight_skips_ebs_without_spw(self, basic_unit):
        """An EB that has the field but no matching science SPW (e.g. one
        whose science SPW was fully flagged upstream and lost its
        OBSERVE_TARGET intent) must be dropped from the joint imaging
        input rather than failing the whole unit, as long as at least
        one TM and one SM EB still survive."""
        import numpy as np
        mock_casatools = MagicMock()
        mock_msmd = MagicMock()

        # Per-MS behaviour: tm1.ms is the "buggy" EB whose science SPW
        # is missing from OBSERVE_TARGET.  All other MSes have it.
        path_state = {"current": None}

        def open_path(p):
            path_state["current"] = Path(p).name

        def spws_side(_):
            if path_state["current"] == "tm1.ms":
                return np.array([])  # no science SPW flagged here
            return np.array([1])

        def chanfreqs_side(spw_id):
            return np.linspace(86.0e9, 86.3e9, 960)

        mock_msmd.open.side_effect = open_path
        mock_msmd.spwsforintent.side_effect = spws_side
        mock_msmd.chanfreqs.side_effect = chanfreqs_side
        mock_msmd.fieldsforname.return_value = [0]
        mock_casatools.msmetadata.return_value = mock_msmd

        mock_tb = MagicMock()
        mock_tb.colnames.return_value = ["DATA", "CORRECTED_DATA", "FLAG"]
        mock_casatools.table.return_value = mock_tb

        with patch("panta_rei.imaging.runner._ensure_casatools", return_value=mock_casatools):
            ok, msg = run_trusted_preflight(basic_unit)

        assert ok, msg
        # tm1.ms dropped; tm2.ms and sm1.ms survive.
        assert basic_unit.vis_tm == ["/data/tm2.ms"]
        assert basic_unit.vis_sm == ["/data/sm1.ms"]
        assert basic_unit.spw_selection == ["1", "1"]
        assert basic_unit.field_selection == ["SRC1", "SRC1"]

    def test_preflight_fails_when_no_tm_eb_has_spw(self, basic_unit):
        """If skip-on-miss leaves zero TM EBs, the unit must fail with
        a clear TM-specific message — joint imaging needs both arrays."""
        import numpy as np
        mock_casatools = MagicMock()
        mock_msmd = MagicMock()

        path_state = {"current": None}

        def open_path(p):
            path_state["current"] = Path(p).name

        def spws_side(_):
            # Both TM EBs lack the science SPW; SM EB has it.
            if path_state["current"].startswith("tm"):
                return np.array([])
            return np.array([1])

        mock_msmd.open.side_effect = open_path
        mock_msmd.spwsforintent.side_effect = spws_side
        mock_msmd.chanfreqs.side_effect = lambda _: np.linspace(86.0e9, 86.3e9, 960)
        mock_msmd.fieldsforname.return_value = [0]
        mock_casatools.msmetadata.return_value = mock_msmd

        mock_tb = MagicMock()
        mock_tb.colnames.return_value = ["DATA", "CORRECTED_DATA", "FLAG"]
        mock_casatools.table.return_value = mock_tb

        with patch("panta_rei.imaging.runner._ensure_casatools", return_value=mock_casatools):
            ok, msg = run_trusted_preflight(basic_unit)

        assert not ok
        assert "No TM EB" in msg

    def test_preflight_fails_when_no_sm_eb_has_spw(self, basic_unit):
        """Symmetric SM-side check."""
        import numpy as np
        mock_casatools = MagicMock()
        mock_msmd = MagicMock()

        path_state = {"current": None}

        def open_path(p):
            path_state["current"] = Path(p).name

        def spws_side(_):
            if path_state["current"].startswith("sm"):
                return np.array([])
            return np.array([1])

        mock_msmd.open.side_effect = open_path
        mock_msmd.spwsforintent.side_effect = spws_side
        mock_msmd.chanfreqs.side_effect = lambda _: np.linspace(86.0e9, 86.3e9, 960)
        mock_msmd.fieldsforname.return_value = [0]
        mock_casatools.msmetadata.return_value = mock_msmd

        mock_tb = MagicMock()
        mock_tb.colnames.return_value = ["DATA", "CORRECTED_DATA", "FLAG"]
        mock_casatools.table.return_value = mock_tb

        with patch("panta_rei.imaging.runner._ensure_casatools", return_value=mock_casatools):
            ok, msg = run_trusted_preflight(basic_unit)

        assert not ok
        assert "No SM EB" in msg

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
        # Field skips collapse to the same "no MS has both field and SPW"
        # error since neither survives the per-EB filter.
        assert "SRC1" in msg

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
        assert params["restoringbeam"] == "common"

    def test_common_beam_overrides_recovered(self, basic_unit):
        """Every line cube gets one common restoring beam, whatever QA2 used."""
        basic_unit.recovered_params["restoringbeam"] = ["2arcsec", "1arcsec", "0deg"]
        params = build_tclean_params(basic_unit, imagename="/out/test")
        assert params["restoringbeam"] == "common"

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

    def test_qa2_dirty_seed_gets_positive_budget(self, basic_unit, caplog):
        """A real QA2 dirty iter0 seed (niter=0, threshold='0.0mJy',
        nsigma=0.0) gets a positive niter and nsigma=2 stopping depth."""
        from panta_rei.imaging.runner import clean_depth_fallback
        basic_unit.recovered_params.update({
            "niter": 0, "threshold": "0.0mJy", "nsigma": 0.0,
            "qa2_dirty_only": True, "fastnoise": False,
            "minpercentchange": 1.0, "dogrowprune": True,
            "imagename": "x.spw27.cube.regcal.I.iter0",
        })
        with caplog.at_level("WARNING", logger="panta_rei.imaging.runner"):
            params = build_tclean_params(basic_unit, imagename="/out/test")
        assert params["niter"] == 30000
        assert params["nsigma"] == 2.0
        assert params["threshold"] == "0.0mJy"
        assert params["fastnoise"] is False
        assert "qa2_dirty_only" not in params
        assert "dirty-only" in caplog.text
        assert clean_depth_fallback(basic_unit.recovered_params, 30000) == {
            "niter": 30000, "nsigma": 2.0,
        }

    def test_dirty_seed_without_marker_still_covered(self, basic_unit):
        """Pre-marker sidecars only have niter=0."""
        basic_unit.recovered_params.update({"niter": 0, "threshold": "0.0mJy"})
        params = build_tclean_params(basic_unit, "/o")
        assert params["niter"] == 30000 and params["nsigma"] == 2.0

    def test_dirty_seed_with_positive_threshold_keeps_it(self, basic_unit):
        basic_unit.recovered_params.update({"niter": 0, "threshold": "0.01Jy",
                                            "nsigma": 0.0})
        params = build_tclean_params(basic_unit, "/o")
        assert params["niter"] == 30000
        assert params["nsigma"] == 0.0 and params["threshold"] == "0.01Jy"

    def test_cleaned_seed_has_no_fallback(self, basic_unit):
        from panta_rei.imaging.runner import clean_depth_fallback
        basic_unit.recovered_params.update({"threshold": "0.0mJy", "nsigma": 0.0})
        assert clean_depth_fallback(basic_unit.recovered_params, 30000) == {}
        assert build_tclean_params(basic_unit, "/o")["nsigma"] == 0.0

    def test_positive_niter_unchanged(self, basic_unit):
        basic_unit.recovered_params["niter"] = 4360
        assert build_tclean_params(basic_unit, "/o")["niter"] == 4360

    def test_recovered_controls_forwarded(self, basic_unit):
        basic_unit.recovered_params.update({
            "cyclefactor": 3.0, "fastnoise": False, "minpercentchange": 1.0,
            "dogrowprune": True, "nsigma": 0.0, "mosweight": True,
            "perchanweightdensity": True, "smoothfactor": 1.0,
        })
        params = build_tclean_params(basic_unit, "/o")
        assert params["cyclefactor"] == 3.0
        assert params["fastnoise"] is False
        assert params["minpercentchange"] == 1.0
        assert params["dogrowprune"] is True
        assert params["nsigma"] == 0.0
        assert params["mosweight"] is True
        assert params["perchanweightdensity"] is True
        assert params["smoothfactor"] == 1.0

    def test_absent_controls_left_to_casa(self, basic_unit):
        params = build_tclean_params(basic_unit, "/o")
        for key in ("cyclefactor", "fastnoise", "minpercentchange",
                    "minpsffraction", "maxpsffraction", "cutthreshold",
                    "mosweight", "perchanweightdensity", "nsigma"):
            assert key not in params


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

    def test_subprocess_success_publishes_aux_products(self, basic_unit, tmp_path):
        """When the CASA script reports aux_fits, the runner publishes
        each under the aux/ sub-directory using the canonical aux paths
        passed in the job spec."""
        casa_path = tmp_path / "casa"
        (casa_path / "bin").mkdir(parents=True)
        (casa_path / "bin" / "mpicasa").touch()
        (casa_path / "bin" / "casa").touch()

        output_dir = tmp_path / "output"
        observed_spec = {}

        def fake_subprocess_run(cmd, **kwargs):
            job_json = Path(cmd[-1])
            spec = json.loads(job_json.read_text())
            observed_spec.update(spec)
            run_dir = Path(spec["run_dir"])

            feathered_name = Path(spec["canonical_paths"]["feathered"]).name
            tclean_name = Path(spec["canonical_paths"]["tclean_only"]).name
            aux_paths = spec["canonical_paths"]["aux"]
            (run_dir / feathered_name).write_text("F")
            (run_dir / tclean_name).write_text("T")

            aux_fits: dict[str, str] = {}
            # Simulate CASA exporting all 3 aux products successfully
            for kind, canonical in aux_paths.items():
                local = run_dir / Path(canonical).name
                local.write_text(f"AUX-{kind}")
                aux_fits[kind] = str(local)

            (run_dir / "result.json").write_text(json.dumps({
                "success": True,
                "feathered_fits": str(run_dir / feathered_name),
                "tclean_fits": str(run_dir / tclean_name),
                "aux_fits": aux_fits,
                "error_message": None,
            }))
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        with patch("panta_rei.imaging.runner.run_trusted_preflight", return_value=(True, "OK")), \
             patch("panta_rei.imaging.runner.get_casa_version", return_value="6.6.6"), \
             patch("subprocess.run", side_effect=fake_subprocess_run):
            success, msg, output = run_tclean_feather_parallel(
                unit=basic_unit,
                output_dir=output_dir,
                row_id=77,
                nproc=4,
                casa_path=str(casa_path),
            )

        assert success, msg
        # job spec should advertise the 3 default aux products + canonical
        # paths for each
        assert observed_spec["aux_products"] == ["mask", "residual", "pb"]
        assert observed_spec["clean_depth_fallback"] is None  # cleaned seed
        # Aux products live under output_dir/aux/group.../, separate from
        # the feathered cube which stays in output_dir/group.../.
        feathered_parent = Path(output).parent
        for kind in ("mask", "residual", "pb"):
            canonical = Path(observed_spec["canonical_paths"]["aux"][kind])
            assert canonical.parent != feathered_parent
            assert canonical.parent.parent.name == "aux"
            assert canonical.parent.name == feathered_parent.name
            assert canonical.name.endswith(f".cube.{kind}.fits")
            assert "12m7m." in canonical.name and "12m7mTP" not in canonical.name
            assert canonical.exists()
            assert canonical.read_text() == f"AUX-{kind}"

    def test_subprocess_success_aux_missing_does_not_fail(self, basic_unit, tmp_path):
        """If the CASA script reports no aux_fits, the unit still
        succeeds — aux is best-effort."""
        casa_path = tmp_path / "casa"
        (casa_path / "bin").mkdir(parents=True)
        (casa_path / "bin" / "mpicasa").touch()
        (casa_path / "bin" / "casa").touch()
        output_dir = tmp_path / "output"

        def fake_subprocess_run(cmd, **kwargs):
            spec = json.loads(Path(cmd[-1]).read_text())
            run_dir = Path(spec["run_dir"])
            feathered_name = Path(spec["canonical_paths"]["feathered"]).name
            tclean_name = Path(spec["canonical_paths"]["tclean_only"]).name
            (run_dir / feathered_name).write_text("F")
            (run_dir / tclean_name).write_text("T")
            (run_dir / "result.json").write_text(json.dumps({
                "success": True,
                "feathered_fits": str(run_dir / feathered_name),
                "tclean_fits": str(run_dir / tclean_name),
                # No aux_fits key at all — older CASA scripts won't have it
                "error_message": None,
            }))
            mr = MagicMock(); mr.returncode = 0
            return mr

        with patch("panta_rei.imaging.runner.run_trusted_preflight", return_value=(True, "OK")), \
             patch("panta_rei.imaging.runner.get_casa_version", return_value="6.6.6"), \
             patch("subprocess.run", side_effect=fake_subprocess_run):
            success, msg, output = run_tclean_feather_parallel(
                unit=basic_unit,
                output_dir=output_dir,
                row_id=78,
                nproc=4,
                casa_path=str(casa_path),
            )
        assert success, msg
        assert Path(output).exists()
        # No aux files published
        for kind in ("mask", "residual", "pb"):
            assert not list(Path(output).parent.glob(f"*.cube.{kind}.fits"))


# ---------------------------------------------------------------------------
# _publish + _publish_pair
# ---------------------------------------------------------------------------

class TestPublish:
    def test_publish_overwrite_replaces_existing(self, tmp_path):
        src = tmp_path / "src.fits"
        src.write_text("new")
        dst = tmp_path / "dst.fits"
        dst.write_text("old")
        _publish(str(src), str(dst), policy="overwrite")
        assert dst.read_text() == "new"

    def test_publish_fail_if_exists_raises(self, tmp_path):
        src = tmp_path / "src.fits"
        src.write_text("new")
        dst = tmp_path / "dst.fits"
        dst.write_text("old")
        with pytest.raises(FileExistsError):
            _publish(str(src), str(dst), policy="fail_if_exists")
        # Pre-existing dst is untouched
        assert dst.read_text() == "old"

    def test_publish_fail_if_exists_succeeds_when_missing(self, tmp_path):
        src = tmp_path / "src.fits"
        src.write_text("new")
        dst = tmp_path / "out" / "dst.fits"
        _publish(str(src), str(dst), policy="fail_if_exists")
        assert dst.read_text() == "new"


class TestPublishPair:
    def test_pair_both_succeed(self, tmp_path):
        a_src = tmp_path / "a.fits"; a_src.write_text("A")
        b_src = tmp_path / "b.fits"; b_src.write_text("B")
        a_dst = tmp_path / "out" / "A.fits"
        b_dst = tmp_path / "out" / "B.fits"
        ok, msg = _publish_pair(
            [(str(a_src), str(a_dst)), (str(b_src), str(b_dst))],
            policy="fail_if_exists",
        )
        assert ok
        assert a_dst.read_text() == "A"
        assert b_dst.read_text() == "B"

    def test_pair_second_fails_rolls_back_first(self, tmp_path):
        """If the second publish fails, the first must be unlinked
        so retries see a clean canonical area."""
        a_src = tmp_path / "a.fits"; a_src.write_text("A")
        b_src = tmp_path / "ghost_b.fits"  # missing → second publish raises
        a_dst = tmp_path / "out" / "A.fits"
        b_dst = tmp_path / "out" / "B.fits"

        ok, msg = _publish_pair(
            [(str(a_src), str(a_dst)), (str(b_src), str(b_dst))],
            policy="fail_if_exists",
        )
        assert not ok
        assert "rolled back 1" in msg
        # First publish was rolled back
        assert not a_dst.exists()
        assert not b_dst.exists()

    def test_pair_fail_if_exists_collision_refused_up_front(self, tmp_path):
        a_src = tmp_path / "a.fits"; a_src.write_text("A")
        b_src = tmp_path / "b.fits"; b_src.write_text("B")
        a_dst = tmp_path / "out" / "A.fits"
        b_dst = tmp_path / "out" / "B.fits"
        b_dst.parent.mkdir(parents=True, exist_ok=True)
        b_dst.write_text("preexisting")
        ok, msg = _publish_pair(
            [(str(a_src), str(a_dst)), (str(b_src), str(b_dst))],
            policy="fail_if_exists",
        )
        assert not ok and "collision" in msg
        assert not a_dst.exists()  # nothing was published
        assert b_dst.read_text() == "preexisting"

    def test_pair_first_fails_no_publishes(self, tmp_path):
        a_src = tmp_path / "a.fits"; a_src.write_text("A")
        b_src = tmp_path / "b.fits"; b_src.write_text("B")
        a_dst = tmp_path / "out" / "A.fits"
        b_dst = tmp_path / "out" / "B.fits"
        a_dst.parent.mkdir(parents=True, exist_ok=True)
        a_dst.write_text("preexisting")  # forces failure on first
        ok, msg = _publish_pair(
            [(str(a_src), str(a_dst)), (str(b_src), str(b_dst))],
            policy="fail_if_exists",
        )
        assert not ok
        assert not b_dst.exists()
        assert a_dst.read_text() == "preexisting"

    def test_pair_overwrite_rollback_restores_prior_canonical(self, tmp_path):
        """Under policy=overwrite, a failed second publish must restore
        the prior canonical FITS that the first publish replaced —
        otherwise --re-run can permanently destroy existing data."""
        a_src = tmp_path / "a.fits"; a_src.write_text("A_NEW")
        a_dst = tmp_path / "out" / "A.fits"
        a_dst.parent.mkdir(parents=True, exist_ok=True)
        a_dst.write_text("A_PRIOR_CANONICAL")
        # b_src does not exist, so the second _publish raises on copy
        b_src = tmp_path / "ghost_b.fits"
        b_dst = tmp_path / "out" / "B.fits"
        b_dst.write_text("B_PRIOR_CANONICAL")

        ok, msg = _publish_pair(
            [(str(a_src), str(a_dst)), (str(b_src), str(b_dst))],
            policy="overwrite",
        )
        assert not ok, msg
        # Both prior canonicals must be intact
        assert a_dst.read_text() == "A_PRIOR_CANONICAL"
        assert b_dst.read_text() == "B_PRIOR_CANONICAL"
        # No leftover backup files
        assert not list(a_dst.parent.glob("*.bak.*"))

    def test_pair_overwrite_retires_stale_destination(self, tmp_path):
        a_src = tmp_path / "a.fits"; a_src.write_text("TT0_NEW")
        a_dst = tmp_path / "out" / "tt0.fits"
        a_dst.parent.mkdir(parents=True)
        a_dst.write_text("TT0_OLD")
        stale = tmp_path / "out" / "alpha.fits"
        stale.write_text("ALPHA_OLD")
        absent = tmp_path / "out" / "tt1.fits"  # retire of a missing file: no-op
        ok, msg = _publish_pair(
            [(str(a_src), str(a_dst))], policy="overwrite",
            retire=(str(stale), str(absent)),
        )
        assert ok, msg
        assert a_dst.read_text() == "TT0_NEW"
        assert not stale.exists() and not absent.exists()
        assert not list(a_dst.parent.glob("*.bak.*"))

    def test_pair_overwrite_rollback_restores_retired(self, tmp_path):
        a_src = tmp_path / "a.fits"; a_src.write_text("TT0_NEW")
        a_dst = tmp_path / "out" / "tt0.fits"
        a_dst.parent.mkdir(parents=True)
        a_dst.write_text("TT0_OLD")
        stale = tmp_path / "out" / "alpha.fits"
        stale.write_text("ALPHA_OLD")
        ghost = tmp_path / "ghost.fits"  # missing src → second publish fails
        ok, _msg = _publish_pair(
            [(str(a_src), str(a_dst)), (str(ghost), str(tmp_path / "out" / "x.fits"))],
            policy="overwrite", retire=(str(stale),),
        )
        assert not ok
        assert a_dst.read_text() == "TT0_OLD"
        assert stale.read_text() == "ALPHA_OLD"
        assert not list(a_dst.parent.glob("*.bak.*"))

    def test_pair_fail_if_exists_refuses_stale_retire_candidate(self, tmp_path):
        """No-overwrite policy: an old optional product from another
        generation is a collision, not something to publish beside."""
        a_src = tmp_path / "a.fits"; a_src.write_text("A")
        tt0 = tmp_path / "out" / "tt0.fits"
        stale = tmp_path / "out" / "alpha.fits"
        stale.parent.mkdir(parents=True)
        stale.write_text("ALPHA_OLD")
        ok, msg = _publish_pair(
            [(str(a_src), str(tt0))],
            policy="fail_if_exists", retire=(str(stale),),
        )
        assert not ok
        assert "collision" in msg and "alpha.fits" in msg
        assert not tt0.exists()
        assert stale.read_text() == "ALPHA_OLD"

    def test_pair_overwrite_success_drops_backups(self, tmp_path):
        a_src = tmp_path / "a.fits"; a_src.write_text("A_NEW")
        b_src = tmp_path / "b.fits"; b_src.write_text("B_NEW")
        a_dst = tmp_path / "out" / "A.fits"
        b_dst = tmp_path / "out" / "B.fits"
        a_dst.parent.mkdir(parents=True, exist_ok=True)
        a_dst.write_text("A_PRIOR")
        b_dst.write_text("B_PRIOR")
        ok, msg = _publish_pair(
            [(str(a_src), str(a_dst)), (str(b_src), str(b_dst))],
            policy="overwrite",
        )
        assert ok
        assert a_dst.read_text() == "A_NEW"
        assert b_dst.read_text() == "B_NEW"
        assert not list(a_dst.parent.glob("*.bak.*"))


# ---------------------------------------------------------------------------
# Continuum: params + per-EB trusted preflight
# ---------------------------------------------------------------------------

@pytest.fixture
def cont_unit():
    """A minimal continuum ImagingUnit (product_type='cont')."""
    return ImagingUnit(
        gous_uid="X3833_X64b9",
        source_name="SRC1",
        line_group="N2H+",
        spw_id="cont",
        params_id=1,
        product_type="cont",
        recovered_params={
            "spw": ["23:86.0~86.2GHz,25:87.0~87.2GHz"],
            "imsize": [160, 160],
            "cell": ["1.9arcsec"],
            "phasecenter": "ICRS 17:01:55.4 -041.09.30",
            "weighting": "briggs",
            "robust": 0.5,
            "niter": 20000,
            "threshold": "0.00311Jy",
            "mosweight": True,
            "perchanweightdensity": False,
            "usemask": "auto-multithresh",
        },
        vis_tm=["/data/tm1.ms", "/data/tm2.ms"],
        vis_sm=["/data/sm1.ms"],
        mous_uids_tm=["m1"],
        mous_uids_sm=["m2"],
        ready=True,
    )


def _cont_casatools(has_corrected=True):
    """Mock casatools for the per-EB preflight: only the datacolumn probe is
    needed (spw selection comes from the recovered spw_by_eb map, not casa)."""
    mock = MagicMock()
    tb = MagicMock()
    tb.colnames.return_value = ["DATA", "FLAG"] + (
        ["CORRECTED_DATA"] if has_corrected else []
    )
    mock.table.return_value = tb
    return mock


# cont_unit fixture vis are /data/{tm1,tm2,sm1}.ms → eb uid is the basename
# (no _targets suffix in the test names): "tm1.ms", "tm2.ms", "sm1.ms".
_EB_MAP = {
    "tm1.ms": "25:90.60~90.70GHz",
    "tm2.ms": "25:90.61~90.71GHz",   # sibling EB, its own (offset) selection
    "sm1.ms": "24:90.59~90.69GHz",   # 7m: its own spw id + frame
}


class TestBuildContParams:
    def test_fixed_continuum_overrides(self, cont_unit):
        cont_unit.spw_selection = ["10:86.0~86.2GHz", "10:86.0~86.2GHz", "5:..."]
        cont_unit.field_selection = ["SRC1", "SRC1", "SRC1"]
        cont_unit.datacolumn = "corrected"
        p = build_tclean_cont_params(cont_unit, imagename="/out/test")
        assert p["specmode"] == "mfs"
        assert p["gridder"] == "mosaic"
        assert p["deconvolver"] == "mtmfs"
        assert p["nterms"] == 2
        assert p["pbcor"] is True
        assert p["stokes"] == "I"
        assert p["restoringbeam"] == "common"

    def test_vis_and_recovered_geometry(self, cont_unit):
        p = build_tclean_cont_params(cont_unit, imagename="/out/test")
        assert p["vis"] == ["/data/tm1.ms", "/data/tm2.ms", "/data/sm1.ms"]
        assert p["imsize"] == [160, 160]
        assert p["cell"] == ["1.9arcsec"]
        assert p["mosweight"] is True

    def test_overrides_cannot_beat_fixed(self, cont_unit):
        cont_unit.recovered_params["deconvolver"] = "hogbom"
        cont_unit.recovered_params["specmode"] = "cube"
        p = build_tclean_cont_params(cont_unit, imagename="/out/test")
        assert p["deconvolver"] == "mtmfs"
        assert p["specmode"] == "mfs"

    def test_recovered_controls_forwarded(self, cont_unit):
        cont_unit.recovered_params.update({"cyclefactor": 3.0, "fastnoise": False})
        p = build_tclean_cont_params(cont_unit, "/o")
        assert p["cyclefactor"] == 3.0
        assert p["fastnoise"] is False
        assert "minpercentchange" not in p

    def test_zero_niter_gets_positive_budget(self, cont_unit):
        cont_unit.recovered_params["niter"] = 0
        p = build_tclean_cont_params(cont_unit, "/o")
        assert p["niter"] == 20000
        assert "nsigma" not in p  # threshold 0.00311Jy is a real depth

    def test_scales_default_and_override(self, cont_unit):
        assert build_tclean_cont_params(cont_unit, "/o")["scales"] == [0]
        assert build_tclean_cont_params(
            cont_unit, "/o", scales=[0, 5, 10]
        )["scales"] == [0, 5, 10]


class TestRunTcleanContParallelPublish:
    def test_reimage_without_alpha_retires_old_alpha(self, cont_unit, tmp_path):
        """Overwrite re-image whose alpha export failed: the old alpha must
        not survive beside the new tt0 (one generation per unit)."""
        from panta_rei.imaging.runner import run_tclean_cont_parallel
        casa_path = tmp_path / "casa"
        (casa_path / "bin").mkdir(parents=True)
        (casa_path / "bin" / "mpicasa").touch()
        (casa_path / "bin" / "casa").touch()
        output_dir = tmp_path / "output"
        seen = {}

        def fake_subprocess_run(cmd, **kwargs):
            spec = json.loads(Path(cmd[-1]).read_text())
            run_dir = Path(spec["run_dir"])
            canonical = spec["canonical_paths"]
            seen.update(canonical)
            # Previous generation already published tt0, tt1, alpha, mask.
            for key in ("tt0", "tt1", "alpha"):
                Path(canonical[key]).parent.mkdir(parents=True, exist_ok=True)
                Path(canonical[key]).write_text(f"OLD-{key}")
            Path(canonical["aux"]["mask"]).parent.mkdir(parents=True, exist_ok=True)
            Path(canonical["aux"]["mask"]).write_text("OLD-mask")
            # New generation: tt0 + tt1 only (alpha and mask exports failed).
            local = {}
            for key in ("tt0", "tt1"):
                local[key] = run_dir / Path(canonical[key]).name
                local[key].write_text(f"NEW-{key}")
            (run_dir / "result.json").write_text(json.dumps({
                "success": True,
                "tt0_fits": str(local["tt0"]),
                "tt1_fits": str(local["tt1"]),
                "alpha_fits": None,
                "aux_fits": {},
            }))
            mr = MagicMock(); mr.returncode = 0
            return mr

        with patch("panta_rei.imaging.runner.run_continuum_trusted_preflight",
                   return_value=(True, "OK")), \
             patch("subprocess.run", side_effect=fake_subprocess_run):
            ok, msg, out = run_tclean_cont_parallel(
                unit=cont_unit, output_dir=output_dir, row_id=5,
                casa_path=str(casa_path), publish_policy="overwrite",
            )
        assert ok, msg
        assert Path(seen["tt0"]).read_text() == "NEW-tt0"
        assert Path(seen["tt1"]).read_text() == "NEW-tt1"
        assert not Path(seen["alpha"]).exists()
        assert not Path(seen["aux"]["mask"]).exists()


class TestRunTcleanContReducedAux:
    def test_reduced_aux_overwrite_retires_unrequested_old_aux(self, cont_unit, tmp_path):
        """Full-aux publish, then an overwrite with a reduced aux list: the
        old psf/tt0_image/tt1_image must not survive beside the new tt0."""
        from panta_rei.imaging.runner import run_tclean_cont_parallel
        casa_path = tmp_path / "casa"
        (casa_path / "bin").mkdir(parents=True)
        (casa_path / "bin" / "mpicasa").touch()
        (casa_path / "bin" / "casa").touch()
        output_dir = tmp_path / "output"
        specs = []

        def fake_subprocess_run(cmd, **kwargs):
            spec = json.loads(Path(cmd[-1]).read_text())
            specs.append(spec)
            run_dir = Path(spec["run_dir"])
            canonical = spec["canonical_paths"]
            tag = f"GEN{len(specs)}"
            out = {}
            for key in ("tt0", "tt1", "alpha"):
                out[key] = run_dir / Path(canonical[key]).name
                out[key].write_text(f"{tag}-{key}")
            aux = {}
            for kind, dst in canonical["aux"].items():
                aux[kind] = str(run_dir / Path(dst).name)
                Path(aux[kind]).write_text(f"{tag}-{kind}")
            (run_dir / "result.json").write_text(json.dumps({
                "success": True, "tt0_fits": str(out["tt0"]),
                "tt1_fits": str(out["tt1"]), "alpha_fits": str(out["alpha"]),
                "aux_fits": aux,
            }))
            mr = MagicMock(); mr.returncode = 0
            return mr

        with patch("panta_rei.imaging.runner.run_continuum_trusted_preflight",
                   return_value=(True, "OK")), \
             patch("subprocess.run", side_effect=fake_subprocess_run):
            ok, msg, _ = run_tclean_cont_parallel(
                unit=cont_unit, output_dir=output_dir, row_id=1,
                casa_path=str(casa_path), publish_policy="overwrite",
            )
            assert ok, msg
            full_aux = dict(specs[0]["canonical_paths"]["aux"])
            assert set(full_aux) == {"tt0_image", "tt1_image", "mask",
                                     "residual", "pb", "psf"}
            assert all(Path(p).exists() for p in full_aux.values())

            ok, msg, _ = run_tclean_cont_parallel(
                unit=cont_unit, output_dir=output_dir, row_id=2,
                casa_path=str(casa_path), publish_policy="overwrite",
                aux_products=("mask", "residual", "pb"),
            )
            assert ok, msg
        for kind in ("mask", "residual", "pb"):
            assert Path(full_aux[kind]).read_text() == f"GEN2-{kind}"
        for kind in ("tt0_image", "tt1_image", "psf"):
            assert not Path(full_aux[kind]).exists()
        assert Path(specs[1]["canonical_paths"]["alpha"]).read_text() == "GEN2-alpha"


class TestRunContinuumTrustedPreflight:
    def test_assigns_each_eb_its_own_selection(self, cont_unit):
        cont_unit.recovered_params["spw_by_eb"] = dict(_EB_MAP)
        with patch("panta_rei.imaging.runner._ensure_casatools",
                   return_value=_cont_casatools()):
            ok, msg = run_continuum_trusted_preflight(cont_unit)
        assert ok, msg
        # parallel to vis_tm + vis_sm, each EB's OWN selection (no remap)
        assert cont_unit.spw_selection == [
            "25:90.60~90.70GHz", "25:90.61~90.71GHz", "24:90.59~90.69GHz",
        ]
        assert cont_unit.field_selection == ["SRC1", "SRC1", "SRC1"]
        assert cont_unit.datacolumn == "corrected"

    def test_datacolumn_falls_back_to_data(self, cont_unit):
        cont_unit.recovered_params["spw_by_eb"] = dict(_EB_MAP)
        with patch("panta_rei.imaging.runner._ensure_casatools",
                   return_value=_cont_casatools(has_corrected=False)):
            ok, msg = run_continuum_trusted_preflight(cont_unit)
        assert ok, msg
        assert cont_unit.datacolumn == "data"

    def test_drops_eb_without_selection(self, cont_unit):
        # tm2 absent from the map (e.g. fully flagged) → dropped, tm1 remains.
        m = dict(_EB_MAP)
        del m["tm2.ms"]
        cont_unit.recovered_params["spw_by_eb"] = m
        with patch("panta_rei.imaging.runner._ensure_casatools",
                   return_value=_cont_casatools()):
            ok, msg = run_continuum_trusted_preflight(cont_unit)
        assert ok, msg
        assert cont_unit.vis_tm == ["/data/tm1.ms"]
        assert len(cont_unit.spw_selection) == 2  # tm1 + sm1

    def test_fails_when_no_sm_selection(self, cont_unit):
        cont_unit.recovered_params["spw_by_eb"] = {
            "tm1.ms": "25:90.60~90.70GHz", "tm2.ms": "25:90.61~90.71GHz",
        }
        with patch("panta_rei.imaging.runner._ensure_casatools",
                   return_value=_cont_casatools()):
            ok, msg = run_continuum_trusted_preflight(cont_unit)
        assert not ok
        assert "SM" in msg

    def test_fails_when_no_map(self, cont_unit):
        cont_unit.recovered_params.pop("spw_by_eb", None)
        ok, msg = run_continuum_trusted_preflight(cont_unit)
        assert not ok
