"""Tests for panta_rei.workflows.imaging — RecoverParamsStep and JointImagingStep."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

from panta_rei.config import PipelineConfig
from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import (
    ImagingParamsQueries,
    ImagingParamsStatus,
    ImagingRunsQueries,
    ImagingRunStatus,
)
from panta_rei.workflows.base import WorkflowContext
from panta_rei.workflows.imaging import (
    ImagingOptions,
    JointImagingStep,
    RecoverParamsStep,
    run_imaging,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_CASA_COMMANDS = dedent("""\
    # hif_makeimlist(specmode='cube')
    tclean(vis=['test.ms'], field='"SRC1"', spw=['23'], specmode='cube', nchan=100, start='86.0GHz', width='244.14kHz', imsize=[100, 100], cell=['0.5arcsec'], restoration=True, pbcor=True, niter=30000, threshold='0.1Jy', imagename='test.spw23.cube.I.iter1')
    tclean(vis=['test.ms'], field='"SRC1"', spw=['25'], specmode='cube', nchan=100, start='87.0GHz', width='244.14kHz', imsize=[100, 100], cell=['0.5arcsec'], restoration=True, pbcor=True, niter=30000, threshold='0.1Jy', imagename='test.spw25.cube.I.iter1')
""")


@pytest.fixture
def imaging_tree(tmp_path):
    """Create a realistic TM/SM/TP directory layout with mock weblogs and CSV."""
    base_dir = tmp_path / "2025.1.00383.L"
    data_dir = base_dir / "2025.1.00383.L"

    mous_id_tm = "X3833_X64bc"
    mous_id_sm = "X3833_X64bd"
    mous_id_tp = "X3833_X64be"
    gous_id = "X3833_X64b9"
    sg_id = "X3833_X64b8"

    # TM member with calibrated MS
    tm_member = data_dir / f"science_goal.uid___A001_{sg_id}" / f"group.uid___A001_{gous_id}" / f"member.uid___A001_{mous_id_tm}"
    tm_working = tm_member / "calibrated" / "working"
    tm_working.mkdir(parents=True)
    (tm_working / "uid___A001_X3833_X64bc_targets_line.ms").mkdir()

    # SM member with calibrated MS
    sm_member = data_dir / f"science_goal.uid___A001_{sg_id}" / f"group.uid___A001_{gous_id}" / f"member.uid___A001_{mous_id_sm}"
    sm_working = sm_member / "calibrated" / "working"
    sm_working.mkdir(parents=True)
    (sm_working / "uid___A001_X3833_X64bd_targets_line.ms").mkdir()

    # TP member with product cube
    tp_member = data_dir / f"science_goal.uid___A001_{sg_id}" / f"group.uid___A001_{gous_id}" / f"member.uid___A001_{mous_id_tp}"
    tp_product = tp_member / "product"
    tp_product.mkdir(parents=True)

    # Create a minimal TP FITS cube
    try:
        from astropy.io import fits
        import numpy as np

        tp_data = np.zeros((1, 100, 10, 10), dtype=np.float32)
        hdr = fits.Header()
        hdr["NAXIS"] = 4
        hdr["NAXIS1"] = 10
        hdr["NAXIS2"] = 10
        hdr["NAXIS3"] = 100
        hdr["NAXIS4"] = 1
        hdr["CTYPE1"] = "RA---SIN"
        hdr["CTYPE2"] = "DEC--SIN"
        hdr["CTYPE3"] = "FREQ"
        hdr["CTYPE4"] = "STOKES"
        hdr["CRVAL3"] = 86.0e9
        hdr["CDELT3"] = 244140.0
        hdr["CRPIX3"] = 1
        hdr["BMAJ"] = 0.01
        hdr["BMIN"] = 0.008
        hdr["BPA"] = 0.0

        tp_path = tp_product / "uid___A001_X3833_X64be.SRC1.spw23.cube.I.sd.fits"
        fits.PrimaryHDU(data=tp_data, header=hdr).writeto(str(tp_path))
    except ImportError:
        # If astropy not available, create a placeholder
        (tp_product / "uid___A001_X3833_X64be.SRC1.spw23.cube.I.sd.fits").write_bytes(b"")

    # Weblog with casa_commands.log
    weblog_dir = tmp_path / "weblogs"
    log_dir = weblog_dir / f"uid___A001_{mous_id_tm}" / "pipeline-20250101" / "html"
    log_dir.mkdir(parents=True)
    (log_dir / "casa_commands.log").write_text(MOCK_CASA_COMMANDS)

    # Targets CSV
    csv_path = base_dir / "targets_by_array.csv"
    csv_path.write_text(
        "source_name,array,sb_name,sgous_id,gous_id,mous_ids,Line group\n"
        f"SRC1,TM,sb_tm,{sg_id},{gous_id},{mous_id_tm},N2H+\n"
        f"SRC1,SM,sb_sm,{sg_id},{gous_id},{mous_id_sm},N2H+\n"
        f"SRC1,TP,sb_tp,{sg_id},{gous_id},{mous_id_tp},N2H+\n"
    )

    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "weblog_dir": weblog_dir,
        "csv_path": csv_path,
        "gous_id": gous_id,
        "mous_id_tm": mous_id_tm,
    }


@pytest.fixture
def imaging_ctx(imaging_tree):
    """WorkflowContext wired to the imaging_tree fixture."""
    config = PipelineConfig(
        panta_rei_base=imaging_tree["base_dir"].parent,
        weblog_dir=imaging_tree["weblog_dir"],
    )
    db = DatabaseManager(":memory:")
    return WorkflowContext(
        config=config,
        db_manager=db,
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# RecoverParamsStep
# ---------------------------------------------------------------------------

class TestRecoverParamsStep:
    def test_recovers_params(self, imaging_ctx, imaging_tree):
        opts = ImagingOptions(
            weblog_dir=imaging_tree["weblog_dir"],
            obs_csv=imaging_tree["csv_path"],
            step="recover",
        )
        step = RecoverParamsStep(opts)
        result = step.run(imaging_ctx)

        assert result.success
        assert result.items_processed >= 1

        # Verify DB was populated
        with imaging_ctx.db_manager.connect() as con:
            rows = ImagingParamsQueries.get_all_recovered(con)
            assert len(rows) >= 1
            assert rows[0]["source_name"] == "SRC1"

    def test_skips_when_step_mismatch(self, imaging_ctx, imaging_tree):
        opts = ImagingOptions(
            obs_csv=imaging_tree["csv_path"],
            step="image",
        )
        step = RecoverParamsStep(opts)
        reason = step.should_skip(imaging_ctx)
        assert reason is not None
        assert "recover" in reason

    def test_dry_run(self, imaging_ctx, imaging_tree):
        imaging_ctx.dry_run = True
        opts = ImagingOptions(
            weblog_dir=imaging_tree["weblog_dir"],
            obs_csv=imaging_tree["csv_path"],
            step="recover",
        )
        step = RecoverParamsStep(opts)
        result = step.run(imaging_ctx)
        assert result.success

        # DB should be empty in dry run
        with imaging_ctx.db_manager.connect() as con:
            rows = ImagingParamsQueries.get_all_recovered(con)
            assert len(rows) == 0

    def test_match_filter(self, imaging_ctx, imaging_tree):
        opts = ImagingOptions(
            weblog_dir=imaging_tree["weblog_dir"],
            obs_csv=imaging_tree["csv_path"],
            step="recover",
            match="NONEXISTENT_SOURCE",
        )
        step = RecoverParamsStep(opts)
        result = step.run(imaging_ctx)
        assert result.success
        assert result.items_processed == 0


# ---------------------------------------------------------------------------
# JointImagingStep
# ---------------------------------------------------------------------------

class TestJointImagingStep:
    def test_preflight_mode(self, imaging_ctx, imaging_tree):
        # First recover params
        recover_opts = ImagingOptions(
            weblog_dir=imaging_tree["weblog_dir"],
            obs_csv=imaging_tree["csv_path"],
            step="recover",
        )
        RecoverParamsStep(recover_opts).run(imaging_ctx)

        # Then run preflight
        opts = ImagingOptions(
            obs_csv=imaging_tree["csv_path"],
            step="preflight",
        )
        step = JointImagingStep(opts)
        result = step.run(imaging_ctx)
        assert result.success

    def test_no_params_recovered(self, imaging_ctx, imaging_tree):
        opts = ImagingOptions(
            obs_csv=imaging_tree["csv_path"],
            step="preflight",
        )
        step = JointImagingStep(opts)
        result = step.run(imaging_ctx)
        assert result.success
        assert result.items_processed == 0

    def test_skips_when_step_mismatch(self, imaging_ctx, imaging_tree):
        opts = ImagingOptions(
            obs_csv=imaging_tree["csv_path"],
            step="recover",
        )
        step = JointImagingStep(opts)
        reason = step.should_skip(imaging_ctx)
        assert reason is not None


# ---------------------------------------------------------------------------
# run_imaging orchestration
# ---------------------------------------------------------------------------

class TestRunImaging:
    def test_recover_only(self, imaging_ctx, imaging_tree):
        opts = ImagingOptions(
            weblog_dir=imaging_tree["weblog_dir"],
            obs_csv=imaging_tree["csv_path"],
            step="recover",
        )
        results = run_imaging(imaging_ctx, opts)
        assert "recover" in results
        assert results["recover"].success

    def test_all_steps(self, imaging_ctx, imaging_tree):
        opts = ImagingOptions(
            weblog_dir=imaging_tree["weblog_dir"],
            obs_csv=imaging_tree["csv_path"],
            step="all",
        )
        results = run_imaging(imaging_ctx, opts)
        assert "recover" in results
        assert "image" in results


# ---------------------------------------------------------------------------
# DB lifecycle through execution (mocked CASA)
# ---------------------------------------------------------------------------

class TestDBLifecycle:
    def test_image_step_writes_db_rows(self, imaging_ctx, imaging_tree):
        """Image step should create imaging_runs rows with QUEUED→SUCCESS lifecycle."""
        # First recover
        recover_opts = ImagingOptions(
            weblog_dir=imaging_tree["weblog_dir"],
            obs_csv=imaging_tree["csv_path"],
            step="recover",
        )
        RecoverParamsStep(recover_opts).run(imaging_ctx)

        # Mock run_sdintimaging to succeed
        def mock_run(unit, output_dir, **kwargs):
            return True, "success", "/out/cube.fits"

        output_dir = imaging_tree["base_dir"] / "output"

        opts = ImagingOptions(
            obs_csv=imaging_tree["csv_path"],
            step="image",
            output_dir=output_dir,
            method="sdintimaging",
        )

        import sys
        mock_casatasks = MagicMock()
        with patch("panta_rei.imaging.runner.run_sdintimaging", side_effect=mock_run), \
             patch("panta_rei.imaging.runner.get_casa_version", return_value="6.6.6"), \
             patch.dict(sys.modules, {"casatasks": mock_casatasks}):
            step = JointImagingStep(opts)
            result = step.run(imaging_ctx)

        assert result.success
        assert result.items_processed >= 1

        # Verify DB has imaging_runs entries
        with imaging_ctx.db_manager.connect() as con:
            summary = ImagingRunsQueries.summary(con)
            assert "success" in summary

    def test_image_step_records_failure(self, imaging_ctx, imaging_tree):
        """If sdintimaging fails, DB should have status=failed."""
        # First recover
        recover_opts = ImagingOptions(
            weblog_dir=imaging_tree["weblog_dir"],
            obs_csv=imaging_tree["csv_path"],
            step="recover",
        )
        RecoverParamsStep(recover_opts).run(imaging_ctx)

        def mock_run_fail(unit, output_dir, **kwargs):
            raise RuntimeError("CASA crashed")

        output_dir = imaging_tree["base_dir"] / "output"
        opts = ImagingOptions(
            obs_csv=imaging_tree["csv_path"],
            step="image",
            output_dir=output_dir,
            method="sdintimaging",
        )

        import sys
        mock_casatasks = MagicMock()
        with patch("panta_rei.imaging.runner.run_sdintimaging", side_effect=mock_run_fail), \
             patch("panta_rei.imaging.runner.get_casa_version", return_value="6.6.6"), \
             patch.dict(sys.modules, {"casatasks": mock_casatasks}):
            step = JointImagingStep(opts)
            result = step.run(imaging_ctx)

        assert not result.success
        assert len(result.errors) >= 1

        with imaging_ctx.db_manager.connect() as con:
            summary = ImagingRunsQueries.summary(con)
            assert "failed" in summary

    def test_dry_run_skips_db_writes(self, imaging_ctx, imaging_tree):
        """Dry run on image step should not create imaging_runs rows."""
        imaging_ctx.dry_run = True

        recover_opts = ImagingOptions(
            weblog_dir=imaging_tree["weblog_dir"],
            obs_csv=imaging_tree["csv_path"],
            step="recover",
        )
        # Recovery itself is dry-run, so seed DB manually
        imaging_ctx.dry_run = False
        RecoverParamsStep(recover_opts).run(imaging_ctx)
        imaging_ctx.dry_run = True

        output_dir = imaging_tree["base_dir"] / "output"
        opts = ImagingOptions(
            obs_csv=imaging_tree["csv_path"],
            step="image",
            output_dir=output_dir,
            method="sdintimaging",
        )

        import sys
        mock_casatasks = MagicMock()
        with patch.dict(sys.modules, {"casatasks": mock_casatasks}):
            step = JointImagingStep(opts)
            result = step.run(imaging_ctx)

        assert result.success

        # No imaging_runs rows should exist
        with imaging_ctx.db_manager.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM imaging_runs").fetchone()[0]
            assert count == 0

    def test_idempotence_skips_successful(self, imaging_ctx, imaging_tree):
        """Second run should skip already-successful units."""
        # Recover
        recover_opts = ImagingOptions(
            weblog_dir=imaging_tree["weblog_dir"],
            obs_csv=imaging_tree["csv_path"],
            step="recover",
        )
        RecoverParamsStep(recover_opts).run(imaging_ctx)

        def mock_run(unit, output_dir, **kwargs):
            return True, "success", "/out/cube.fits"

        output_dir = imaging_tree["base_dir"] / "output"
        opts = ImagingOptions(
            obs_csv=imaging_tree["csv_path"],
            step="image",
            output_dir=output_dir,
            method="sdintimaging",
        )

        import sys
        mock_casatasks = MagicMock()

        # First run
        with patch("panta_rei.imaging.runner.run_sdintimaging", side_effect=mock_run) as mock, \
             patch("panta_rei.imaging.runner.get_casa_version", return_value="6.6.6"), \
             patch.dict(sys.modules, {"casatasks": mock_casatasks}):
            step = JointImagingStep(opts)
            result1 = step.run(imaging_ctx)

        first_call_count = mock.call_count
        assert result1.success
        assert result1.items_processed >= 1

        # Second run — should skip all
        with patch("panta_rei.imaging.runner.run_sdintimaging", side_effect=mock_run) as mock2, \
             patch("panta_rei.imaging.runner.get_casa_version", return_value="6.6.6"), \
             patch.dict(sys.modules, {"casatasks": mock_casatasks}):
            step2 = JointImagingStep(opts)
            result2 = step2.run(imaging_ctx)

        assert result2.success
        assert mock2.call_count == 0  # Should not have been called
