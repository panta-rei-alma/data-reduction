"""Integration tests for the calibration workflow.

Tests use the ``alma_tree`` fixture from conftest.py and mock subprocess
so no actual CASA installation is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from panta_rei.config import PipelineConfig
from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import PIRunsQueries, PIRunStatus
from panta_rei.workflows.base import StepResult, WorkflowContext
from panta_rei.workflows.calibration import (
    CalibrationOptions,
    CalibrateStep,
    already_completed,
    discover_scriptforpi,
    find_calibrated_directories,
    parse_hierarchy_from_path,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(
    tmp_path: Path,
    db: DatabaseManager,
    data_dir: Path | None = None,
    dry_run: bool = False,
) -> WorkflowContext:
    """Build a WorkflowContext.  If *data_dir* points into an alma_tree, the
    config must be set so ``config.data_dir`` matches it."""
    config = PipelineConfig(panta_rei_base=tmp_path)
    # The alma_tree fixture puts data under tmp_path/2025.1.00383.L/2025.1.00383.L
    # which matches config.data_dir by default
    return WorkflowContext(config=config, db_manager=db, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscoverScriptForPI:

    def test_discovers_scripts_in_alma_tree(self, alma_tree):
        """discover_scriptforpi finds both ScriptForPI.py files in the fixture."""
        found = list(discover_scriptforpi(alma_tree))
        assert len(found) == 2

        uids = {uid for uid, _script, _mous_dir, _hier in found}
        # Both UIDs should be lowercase canonical form
        assert "uid___a001_x3833_x64bc" in uids
        assert "uid___a001_x3833_x64bd" in uids

    def test_script_path_is_under_script_dir(self, alma_tree):
        """Each discovered script lives in a member/script/ directory."""
        for _uid, script_path, _mous_dir, _hier in discover_scriptforpi(alma_tree):
            assert script_path.parent.name == "script"
            assert script_path.name.lower().startswith("scriptforpi")

    def test_mous_dir_is_member_directory(self, alma_tree):
        """The returned mous_dir is the member.uid___ parent of script/."""
        for _uid, _script, mous_dir, _hier in discover_scriptforpi(alma_tree):
            assert mous_dir.name.startswith("member.")

    def test_hierarchy_extraction(self, alma_tree):
        """parse_hierarchy_from_path extracts SG, GOUS, and MOUS UIDs."""
        for _uid, _script, mous_dir, hier in discover_scriptforpi(alma_tree):
            assert hier["sg_uid"] is not None
            assert hier["gous_uid"] is not None
            assert hier["mous_uid"] is not None
            assert hier["sg_uid"].startswith("uid___")

    def test_no_scripts_in_empty_dir(self, tmp_path):
        """discover_scriptforpi yields nothing for an empty directory."""
        empty = tmp_path / "empty"
        empty.mkdir()
        assert list(discover_scriptforpi(empty)) == []


# ---------------------------------------------------------------------------
# Hierarchy parsing
# ---------------------------------------------------------------------------

class TestParseHierarchy:

    def test_full_path(self):
        p = Path(
            "/data/science_goal.uid___A001_X3833_X64b8/"
            "group.uid___A001_X3833_X64b9/"
            "member.uid___A001_X3833_X64bc"
        )
        h = parse_hierarchy_from_path(p)
        assert h["sg_uid"] == "uid___a001_x3833_x64b8"
        assert h["gous_uid"] == "uid___a001_x3833_x64b9"
        assert h["mous_uid"] == "uid___a001_x3833_x64bc"

    def test_partial_path(self):
        """Only member component present."""
        p = Path("/data/member.uid___A001_X3833_X64bc")
        h = parse_hierarchy_from_path(p)
        assert h["mous_uid"] == "uid___a001_x3833_x64bc"
        assert h["sg_uid"] is None
        assert h["gous_uid"] is None


# ---------------------------------------------------------------------------
# Idempotence: already_completed
# ---------------------------------------------------------------------------

class TestAlreadyCompleted:

    def test_not_completed_without_calibrated_dir(self, alma_tree):
        """No calibrated/ directory means not completed."""
        for _uid, _script, mous_dir, _hier in discover_scriptforpi(alma_tree):
            assert not already_completed(mous_dir)

    def test_completed_with_ms_directory(self, alma_tree):
        """A calibrated/*.ms directory indicates completion."""
        for _uid, _script, mous_dir, _hier in discover_scriptforpi(alma_tree):
            cal_dir = mous_dir / "calibrated"
            cal_dir.mkdir()
            (cal_dir / "uid___A001_X3833_X64bc.ms").mkdir()
            assert already_completed(mous_dir)
            break  # only need one

    def test_completed_with_split_cal_directory(self, alma_tree):
        """A calibrated/*.ms.split.cal directory also indicates completion."""
        for _uid, _script, mous_dir, _hier in discover_scriptforpi(alma_tree):
            cal_dir = mous_dir / "calibrated"
            cal_dir.mkdir()
            (cal_dir / "uid___A001_X3833_X64bc.ms.split.cal").mkdir()
            assert already_completed(mous_dir)
            break

    def test_find_calibrated_directories(self, alma_tree):
        """find_calibrated_directories returns only .ms and .ms.split.cal dirs."""
        for _uid, _script, mous_dir, _hier in discover_scriptforpi(alma_tree):
            cal_dir = mous_dir / "calibrated"
            cal_dir.mkdir()
            (cal_dir / "valid.ms").mkdir()
            (cal_dir / "also_valid.ms.split.cal").mkdir()
            (cal_dir / "not_a_ms.txt").write_text("nope")
            (cal_dir / "not_a_dir.ms").write_text("file, not dir")

            found = find_calibrated_directories(mous_dir)
            names = {p.name for p in found}
            assert "valid.ms" in names
            assert "also_valid.ms.split.cal" in names
            assert "not_a_ms.txt" not in names
            assert len(found) == 2
            break


# ---------------------------------------------------------------------------
# CalibrateStep properties
# ---------------------------------------------------------------------------

class TestCalibrateStepProperties:

    def test_step_name_and_description(self):
        step = CalibrateStep()
        assert step.name == "calibrate"
        assert "CASA" in step.description or "calibrat" in step.description.lower()

    def test_should_skip_in_skip_set(self, tmp_path, db):
        config = PipelineConfig(panta_rei_base=tmp_path)
        ctx = WorkflowContext(
            config=config, db_manager=db,
            skip_steps={"calibrate"},
        )
        reason = CalibrateStep().should_skip(ctx)
        assert reason is not None

    def test_should_skip_when_data_dir_missing(self, tmp_path, db):
        """Step skips when data_dir does not exist."""
        config = PipelineConfig(panta_rei_base=tmp_path)
        ctx = WorkflowContext(config=config, db_manager=db)
        # data_dir = tmp_path / 2025.1.00383.L / 2025.1.00383.L -- doesn't exist
        reason = CalibrateStep().should_skip(ctx)
        assert reason is not None
        assert "data directory" in reason.lower()


# ---------------------------------------------------------------------------
# CalibrateStep.run() with mocked subprocess
# ---------------------------------------------------------------------------

class TestCalibrateStepRun:

    @patch("panta_rei.workflows.calibration.subprocess.run")
    def test_runs_casa_for_each_mous(self, mock_subproc, alma_tree, db):
        """With mocked subprocess, CalibrateStep runs CASA for each discovered script."""
        mock_subproc.return_value = MagicMock(returncode=0)

        ctx = _make_ctx(alma_tree.parent.parent, db)
        step = CalibrateStep()
        result = step.run(ctx)

        assert result.success
        assert result.items_processed == 2  # two member directories in alma_tree
        assert mock_subproc.call_count == 2

    @patch("panta_rei.workflows.calibration.subprocess.run")
    def test_casa_cwd_is_script_parent_dir(self, mock_subproc, alma_tree, db):
        """CASA must be invoked with cwd=script's parent directory (the script/ dir)."""
        mock_subproc.return_value = MagicMock(returncode=0)

        ctx = _make_ctx(alma_tree.parent.parent, db)
        CalibrateStep().run(ctx)

        for call_args in mock_subproc.call_args_list:
            cwd = call_args.kwargs.get("cwd") or call_args[1].get("cwd", "")
            # cwd should end with /script
            assert Path(cwd).name == "script", (
                f"CASA must run from the script/ directory, got cwd={cwd}"
            )

    @patch("panta_rei.workflows.calibration.subprocess.run")
    def test_idempotence_skips_already_calibrated(self, mock_subproc, alma_tree, db):
        """If calibrated/*.ms already exists, the step skips that MOUS."""
        # Create calibrated output for both members
        for _uid, _script, mous_dir, _hier in discover_scriptforpi(alma_tree):
            cal_dir = mous_dir / "calibrated"
            cal_dir.mkdir()
            (cal_dir / "output.ms").mkdir()

        ctx = _make_ctx(alma_tree.parent.parent, db)
        step = CalibrateStep()
        result = step.run(ctx)

        assert result.success
        assert result.items_processed == 0
        assert result.items_skipped == 2
        mock_subproc.assert_not_called()

    @patch("panta_rei.workflows.calibration.subprocess.run")
    def test_dry_run_does_not_execute_casa(self, mock_subproc, alma_tree, db):
        """In dry_run mode, CASA is not executed."""
        ctx = _make_ctx(alma_tree.parent.parent, db, dry_run=True)
        step = CalibrateStep()
        result = step.run(ctx)

        assert result.success
        assert result.items_processed == 2
        mock_subproc.assert_not_called()

    @patch("panta_rei.workflows.calibration.subprocess.run")
    def test_failed_casa_reported_in_result(self, mock_subproc, alma_tree, db):
        """Non-zero CASA exit code is reported as an error in StepResult."""
        mock_subproc.return_value = MagicMock(returncode=1)

        ctx = _make_ctx(alma_tree.parent.parent, db)
        step = CalibrateStep()
        result = step.run(ctx)

        assert not result.success
        assert result.items_processed == 2
        assert len(result.errors) == 2

    @patch("panta_rei.workflows.calibration.subprocess.run")
    def test_db_records_created_for_runs(self, mock_subproc, alma_tree, db):
        """Each CASA run should create a row in the pi_runs table."""
        mock_subproc.return_value = MagicMock(returncode=0)

        ctx = _make_ctx(alma_tree.parent.parent, db)
        CalibrateStep().run(ctx)

        con = db.connect()
        rows = con.execute("SELECT uid, status FROM pi_runs").fetchall()
        assert len(rows) == 2
        for _uid, status in rows:
            assert status == PIRunStatus.SUCCESS


# ---------------------------------------------------------------------------
# CalibrationOptions filtering
# ---------------------------------------------------------------------------

class TestCalibrationOptionsFiltering:

    @patch("panta_rei.workflows.calibration.subprocess.run")
    def test_match_filter(self, mock_subproc, alma_tree, db):
        """The match option filters UIDs by regex."""
        mock_subproc.return_value = MagicMock(returncode=0)

        ctx = _make_ctx(alma_tree.parent.parent, db)
        opts = CalibrationOptions(match="x64bc")
        step = CalibrateStep(options=opts)
        result = step.run(ctx)

        assert result.items_processed == 1
        assert result.items_skipped == 1

    @patch("panta_rei.workflows.calibration.subprocess.run")
    def test_limit_option(self, mock_subproc, alma_tree, db):
        """The limit option stops after N runs."""
        mock_subproc.return_value = MagicMock(returncode=0)

        ctx = _make_ctx(alma_tree.parent.parent, db)
        opts = CalibrationOptions(limit=1)
        step = CalibrateStep(options=opts)
        result = step.run(ctx)

        assert result.items_processed == 1
        assert mock_subproc.call_count == 1

    def test_no_scripts_found_returns_success(self, tmp_path, db):
        """When no ScriptForPI.py files exist, step returns success with 0 processed."""
        # Create an empty data_dir
        config = PipelineConfig(panta_rei_base=tmp_path)
        config.data_dir.mkdir(parents=True)
        ctx = WorkflowContext(config=config, db_manager=db)

        step = CalibrateStep()
        result = step.run(ctx)

        assert result.success
        assert result.items_processed == 0
        assert "No ScriptForPI" in result.summary


# ---------------------------------------------------------------------------
# CASA command template
# ---------------------------------------------------------------------------

class TestCASACommandTemplate:

    @patch("panta_rei.workflows.calibration.subprocess.run")
    def test_default_casa_cmd_template(self, mock_subproc, alma_tree, db):
        """Default CASA command should include --nologger --nogui --pipeline."""
        mock_subproc.return_value = MagicMock(returncode=0)

        ctx = _make_ctx(alma_tree.parent.parent, db)
        CalibrateStep().run(ctx)

        # Check the command passed to subprocess.run
        call_args = mock_subproc.call_args_list[0]
        cmd = call_args[0][0]  # first positional arg (the command list)
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        assert "casa" in cmd_str.lower()
        assert "--nologger" in cmd_str
        assert "--nogui" in cmd_str
        assert "--pipeline" in cmd_str
        assert "scriptForPI.py" in cmd_str or "scriptforpi" in cmd_str.lower()

    @patch("panta_rei.workflows.calibration.subprocess.run")
    def test_config_casa_path_overrides_default(self, mock_subproc, alma_tree, tmp_path, db):
        """When config has casa_path, the step uses it for the CASA command."""
        mock_subproc.return_value = MagicMock(returncode=0)

        casa_path = tmp_path / "casa-6.6.6"
        casa_path.mkdir()
        (casa_path / "bin").mkdir()
        (casa_path / "bin" / "casa").write_text("#!/bin/bash\n")

        config = PipelineConfig(
            panta_rei_base=alma_tree.parent.parent,
            casa_path=casa_path,
        )
        ctx = WorkflowContext(config=config, db_manager=db)
        CalibrateStep().run(ctx)

        call_args = mock_subproc.call_args_list[0]
        cmd = call_args[0][0]
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        assert str(casa_path / "bin" / "casa") in cmd_str
