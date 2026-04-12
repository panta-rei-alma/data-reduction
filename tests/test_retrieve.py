"""Integration tests for the RetrieveStep and retrieve_and_extract.

Tests use mocked astroquery so no external ALMA servers are contacted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from panta_rei.config import PipelineConfig
from panta_rei.core.errors import ALMAError
from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import ObsQueries, ObsStatus
from panta_rei.workflows.base import StepResult, WorkflowContext
from panta_rei.workflows.retrieval import RetrieveStep

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(tmp_path: Path, db: DatabaseManager, username: str = "testuser") -> WorkflowContext:
    """Build a WorkflowContext rooted at *tmp_path*."""
    config = PipelineConfig(panta_rei_base=tmp_path)
    return WorkflowContext(config=config, db_manager=db, username=username)


def _make_tar(tar_path: Path, member_name: str = "data.txt", content: bytes = b"hello") -> None:
    """Create a tiny tar file at *tar_path* containing one text member."""
    import tarfile
    import io

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))


def _make_fake_query_results(uids: list[str], release_dates: list[str] | None = None):
    """Build a mock Astropy Table-like object returned by alma.query()."""
    from astropy.table import Table

    if release_dates is None:
        release_dates = ["2025-06-01"] * len(uids)
    return Table({
        "member_ous_uid": uids,
        "obs_release_date": release_dates,
    })


# ---------------------------------------------------------------------------
# RetrieveStep.run()
# ---------------------------------------------------------------------------

class TestRetrieveStep:

    def test_step_properties(self):
        step = RetrieveStep()
        assert step.name == "retrieve"
        assert "Download" in step.description

    def test_should_skip_no_username(self, tmp_path, db):
        """Step should skip when no ALMA username is provided."""
        config = PipelineConfig(panta_rei_base=tmp_path)
        ctx = WorkflowContext(config=config, db_manager=db, username=None)
        reason = RetrieveStep().should_skip(ctx)
        assert reason is not None
        assert "username" in reason.lower()

    def test_should_skip_in_skip_set(self, tmp_path, db):
        """Step should skip when its name is in skip_steps."""
        config = PipelineConfig(panta_rei_base=tmp_path)
        ctx = WorkflowContext(
            config=config, db_manager=db, username="user",
            skip_steps={"retrieve"},
        )
        reason = RetrieveStep().should_skip(ctx)
        assert reason is not None

    def test_should_not_skip_when_username_present(self, tmp_path, db):
        """Step should proceed when username is given."""
        ctx = _make_ctx(tmp_path, db)
        assert RetrieveStep().should_skip(ctx) is None

    @patch("panta_rei.alma.download.retrieve_and_extract")
    def test_run_success(self, mock_retrieve, tmp_path, db):
        """RetrieveStep.run() returns success when retrieve_and_extract completes."""
        mock_retrieve.return_value = True
        ctx = _make_ctx(tmp_path, db)

        result = RetrieveStep().run(ctx)

        assert result.success
        assert result.items_processed == 1
        mock_retrieve.assert_called_once_with(
            username="testuser",
            project_code=ctx.config.project_code,
            base_dir=ctx.data_dir,
            db_manager=db,
        )

    @patch("panta_rei.alma.download.retrieve_and_extract")
    def test_run_returns_failure_on_alma_error(self, mock_retrieve, tmp_path, db):
        """RetrieveStep.run() returns StepResult(success=False) when ALMAError is raised."""
        mock_retrieve.side_effect = ALMAError("Failed on all ALMA mirrors")
        ctx = _make_ctx(tmp_path, db)

        result = RetrieveStep().run(ctx)

        assert not result.success
        assert "Failed on all ALMA mirrors" in result.summary
        assert len(result.errors) == 1

    @patch("panta_rei.alma.download.retrieve_and_extract")
    def test_run_returns_failure_on_generic_exception(self, mock_retrieve, tmp_path, db):
        """RetrieveStep.run() catches generic exceptions and returns failure."""
        mock_retrieve.side_effect = RuntimeError("network down")
        ctx = _make_ctx(tmp_path, db)

        result = RetrieveStep().run(ctx)

        assert not result.success
        assert "network down" in result.summary


# ---------------------------------------------------------------------------
# DB state transitions via retrieve_and_extract
# ---------------------------------------------------------------------------

class TestRetrieveAndExtractDBTransitions:
    """Test the full retrieve_and_extract function with mocked ALMA client."""

    @patch("panta_rei.alma.download.login_alma")
    @patch("panta_rei.alma.download.setup_alma_client")
    @patch("panta_rei.alma.download.query_project")
    @patch("panta_rei.alma.download.retrieve_uids")
    def test_pending_to_downloaded_to_extracted(
        self, mock_retrieve_uids, mock_query, mock_setup, mock_login,
        tmp_path, db,
    ):
        """Verify the full lifecycle: pending -> downloaded -> extracted."""
        from panta_rei.alma.download import retrieve_and_extract

        uid = "uid://A001/X3833/X64bc"
        uid_canonical = "uid___a001_x3833_x64bc"
        base_dir = tmp_path / "data"
        base_dir.mkdir(parents=True)

        # Set up mocks
        mock_setup.return_value = MagicMock()
        mock_query.return_value = _make_fake_query_results([uid])

        # Create a real tar file that retrieve_uids will "return"
        tar_name = f"uid___A001_X3833_X64bc.tar"
        tar_path = base_dir / "tars" / tar_name
        _make_tar(tar_path, member_name="science_goal.uid___A001/test.txt")
        mock_retrieve_uids.return_value = [tar_path]

        # Run
        result = retrieve_and_extract(
            username="testuser",
            project_code="2025.1.00383.L",
            base_dir=base_dir,
            db_manager=db,
        )

        assert result is True

        # Verify DB state reached extracted
        con = db.connect()
        status = ObsQueries.get_status(con, uid)
        assert status == ObsStatus.EXTRACTED

    @patch("panta_rei.alma.download.login_alma")
    @patch("panta_rei.alma.download.setup_alma_client")
    @patch("panta_rei.alma.download.query_project")
    @patch("panta_rei.alma.download.retrieve_uids")
    def test_idempotence_skips_already_extracted(
        self, mock_retrieve_uids, mock_query, mock_setup, mock_login,
        tmp_path, db,
    ):
        """Re-running retrieve_and_extract skips UIDs already in 'extracted' state."""
        from panta_rei.alma.download import retrieve_and_extract

        uid = "uid://A001/X3833/X64bc"
        uid_canonical = "uid___a001_x3833_x64bc"
        base_dir = tmp_path / "data"
        base_dir.mkdir(parents=True)

        # Pre-seed DB with extracted status
        con = db.connect()
        ObsQueries.upsert_seen(con, uid, "2025-06-01")
        ObsQueries.mark_extracted(con, uid, base_dir, 10, 0, True)
        con.commit()

        # Confirm it is extracted
        assert ObsQueries.get_status(con, uid) == ObsStatus.EXTRACTED

        # Set up mocks — query returns the same UID
        mock_setup.return_value = MagicMock()
        mock_query.return_value = _make_fake_query_results([uid])

        result = retrieve_and_extract(
            username="testuser",
            project_code="2025.1.00383.L",
            base_dir=base_dir,
            db_manager=db,
        )

        assert result is True
        # retrieve_uids should NOT have been called (nothing to download)
        mock_retrieve_uids.assert_not_called()
        # Status should still be extracted
        assert ObsQueries.get_status(con, uid) == ObsStatus.EXTRACTED

    @patch("panta_rei.alma.download.login_alma")
    @patch("panta_rei.alma.download.setup_alma_client")
    @patch("panta_rei.alma.download.query_project")
    def test_no_results_returns_true(
        self, mock_query, mock_setup, mock_login, tmp_path, db,
    ):
        """When ALMA returns no results, retrieve_and_extract returns True (nothing to do)."""
        from panta_rei.alma.download import retrieve_and_extract

        base_dir = tmp_path / "data"
        base_dir.mkdir(parents=True)

        mock_setup.return_value = MagicMock()
        # Return an empty table
        from astropy.table import Table
        mock_query.return_value = Table(
            {"member_ous_uid": [], "obs_release_date": []}
        )

        result = retrieve_and_extract(
            username="testuser",
            project_code="2025.1.00383.L",
            base_dir=base_dir,
            db_manager=db,
        )
        assert result is True


# ---------------------------------------------------------------------------
# ALMAError on all mirrors failing
# ---------------------------------------------------------------------------

class TestALMAErrorOnAllMirrorsFailing:

    def test_alma_error_raised_when_all_mirrors_fail(self, tmp_path, db):
        """retrieve_and_extract raises ALMAError when every mirror fails."""
        from panta_rei.alma.download import retrieve_and_extract

        base_dir = tmp_path / "data"
        base_dir.mkdir(parents=True)

        # Patch setup_alma_client to always raise, simulating all mirrors failing
        with patch(
            "panta_rei.alma.download.setup_alma_client",
            side_effect=Exception("connection refused"),
        ):
            with pytest.raises(ALMAError, match="Failed on all ALMA mirrors"):
                retrieve_and_extract(
                    username="testuser",
                    project_code="2025.1.00383.L",
                    base_dir=base_dir,
                    db_manager=db,
                )

    def test_step_returns_failure_when_all_mirrors_fail(self, tmp_path, db):
        """RetrieveStep.run() returns StepResult(success=False) when ALMAError is raised."""
        ctx = _make_ctx(tmp_path, db)

        with patch(
            "panta_rei.alma.download.setup_alma_client",
            side_effect=Exception("connection refused"),
        ):
            result = RetrieveStep().run(ctx)

        assert not result.success
        assert "Failed on all ALMA mirrors" in result.summary
        assert len(result.errors) >= 1
