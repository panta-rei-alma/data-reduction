"""Integration tests for StageWeblogsStep.

Tests use realistic directory fixtures and small tar archives but do NOT
require the legacy stage_weblogs module to be importable (that module lives
outside the panta_rei package).  Instead, tests exercise the step's error
handling paths (ImportError, exception handling) and verify StepResult
semantics.
"""

from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from panta_rei.config import PipelineConfig
from panta_rei.db.connection import DatabaseManager
from panta_rei.workflows.base import StepResult, WorkflowContext
from panta_rei.workflows.retrieval import StageWeblogsStep

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(
    tmp_path: Path,
    db: DatabaseManager,
    weblog_dir: Path | None = None,
    dry_run: bool = False,
) -> WorkflowContext:
    """Build a WorkflowContext rooted at *tmp_path*."""
    wdir = weblog_dir or (tmp_path / "weblogs")
    wdir.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(panta_rei_base=tmp_path, weblog_dir=wdir)
    return WorkflowContext(config=config, db_manager=db, dry_run=dry_run)


def _build_weblog_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create a realistic ALMA directory with a .weblog.tgz inside qa/.

    Returns (data_dir, weblog_tgz_path).
    """
    data_dir = tmp_path / "2025.1.00383.L" / "2025.1.00383.L"
    sg = data_dir / "science_goal.uid___A001_X3833_X64b8"
    group = sg / "group.uid___A001_X3833_X64b9"
    member = group / "member.uid___A001_X3833_X64bc"
    qa_dir = member / "qa"
    qa_dir.mkdir(parents=True)

    # Create a small but valid .weblog.tgz containing an index.html
    tgz_path = qa_dir / "pipeline-20250601T120000.weblog.tgz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        html = b"<html><body>Weblog</body></html>"
        info = tarfile.TarInfo(name="pipeline-20250601T120000/html/index.html")
        info.size = len(html)
        tf.addfile(info, io.BytesIO(html))
    tgz_path.write_bytes(buf.getvalue())

    return data_dir, tgz_path


# ---------------------------------------------------------------------------
# StageWeblogsStep properties
# ---------------------------------------------------------------------------

class TestStageWeblogsStepProperties:

    def test_step_name_and_description(self):
        step = StageWeblogsStep()
        assert step.name == "stage_weblogs"
        assert "weblog" in step.description.lower()

    def test_should_skip_in_skip_set(self, tmp_path, db):
        """Step skips when its name is listed in skip_steps."""
        config = PipelineConfig(panta_rei_base=tmp_path)
        ctx = WorkflowContext(
            config=config, db_manager=db,
            skip_steps={"stage_weblogs"},
        )
        reason = StageWeblogsStep().should_skip(ctx)
        assert reason is not None


# ---------------------------------------------------------------------------
# ImportError path
# ---------------------------------------------------------------------------

class TestStageWeblogsImport:
    """The staging module is now part of the package and always importable."""

    def test_staging_module_importable(self):
        """panta_rei.alma.staging should always be importable."""
        from panta_rei.alma.staging import WeblogStager, WeblogStateDB
        assert WeblogStager is not None
        assert WeblogStateDB is not None


# ---------------------------------------------------------------------------
# Successful staging (mocked stager)
# ---------------------------------------------------------------------------

class TestStageWeblogsWithMockedStager:
    """Test the step with a mocked WeblogStager injected via import patching."""

    def test_run_success_with_mock_stager(self, tmp_path, db):
        """Step returns success and correct counts when the stager works."""
        ctx = _make_ctx(tmp_path, db)

        # Build directory structure so config paths exist
        _build_weblog_tree(tmp_path)

        # Create a fake module to be "imported" by the step
        mock_stager_instance = MagicMock()
        mock_stager_instance.stage_all.return_value = [
            "/staged/weblog1",
            "/staged/weblog2",
        ]
        mock_stager_instance.corrupted_count = 0

        mock_stager_cls = MagicMock(return_value=mock_stager_instance)
        mock_db_cls = MagicMock()

        with patch("panta_rei.alma.staging.WeblogStager", mock_stager_cls), \
             patch("panta_rei.alma.staging.WeblogStateDB", mock_db_cls):
            result = StageWeblogsStep().run(ctx)

        assert result.success
        assert result.items_processed == 2
        assert result.items_skipped == 0
        assert "Staged 2 weblog(s)" in result.summary

    def test_run_with_corrupted_weblogs(self, tmp_path, db):
        """Step reports corrupted count in summary."""
        ctx = _make_ctx(tmp_path, db)
        _build_weblog_tree(tmp_path)

        mock_stager_instance = MagicMock()
        mock_stager_instance.stage_all.return_value = ["/staged/weblog1"]
        mock_stager_instance.corrupted_count = 3

        with patch("panta_rei.alma.staging.WeblogStager", MagicMock(return_value=mock_stager_instance)), \
             patch("panta_rei.alma.staging.WeblogStateDB", MagicMock()):
            result = StageWeblogsStep().run(ctx)

        assert result.success
        assert result.items_processed == 1
        assert result.items_skipped == 3
        assert "corrupted" in result.summary.lower()

    def test_run_exception_returns_failure(self, tmp_path, db):
        """Step catches exceptions from stager and returns failure."""
        ctx = _make_ctx(tmp_path, db)
        _build_weblog_tree(tmp_path)

        mock_stager_instance = MagicMock()
        mock_stager_instance.stage_all.side_effect = RuntimeError("disk full")

        with patch("panta_rei.alma.staging.WeblogStager", MagicMock(return_value=mock_stager_instance)), \
             patch("panta_rei.alma.staging.WeblogStateDB", MagicMock()):
            result = StageWeblogsStep().run(ctx)

        assert not result.success
        assert "disk full" in result.summary
        assert len(result.errors) >= 1


# ---------------------------------------------------------------------------
# Realistic directory fixture validation
# ---------------------------------------------------------------------------

class TestWeblogDirectoryFixture:
    """Verify the helper produces the expected directory structure."""

    def test_build_weblog_tree_creates_tgz(self, tmp_path):
        """The fixture builder creates a valid .weblog.tgz with HTML inside."""
        data_dir, tgz_path = _build_weblog_tree(tmp_path)

        assert tgz_path.exists()
        assert tgz_path.name.endswith(".weblog.tgz")
        assert tgz_path.stat().st_size > 0

        # Verify tgz is a valid gzipped tar
        with tarfile.open(tgz_path, "r:gz") as tf:
            names = tf.getnames()
            assert any("index.html" in n for n in names)

    def test_directory_hierarchy(self, tmp_path):
        """The fixture creates the science_goal/group/member/qa hierarchy."""
        data_dir, tgz_path = _build_weblog_tree(tmp_path)

        assert data_dir.is_dir()
        sg_dirs = list(data_dir.glob("science_goal.*"))
        assert len(sg_dirs) == 1
        group_dirs = list(sg_dirs[0].glob("group.*"))
        assert len(group_dirs) == 1
        member_dirs = list(group_dirs[0].glob("member.*"))
        assert len(member_dirs) == 1
        qa_dir = member_dirs[0] / "qa"
        assert qa_dir.is_dir()
