"""Tests for panta_rei.workflows — Step interface and workflow orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from panta_rei.config import PipelineConfig
from panta_rei.db.connection import DatabaseManager
from panta_rei.workflows.base import Step, StepResult, WorkflowContext


# ---------------------------------------------------------------------------
# WorkflowContext
# ---------------------------------------------------------------------------

class TestWorkflowContext:

    def test_convenience_properties(self, tmp_path):
        config = PipelineConfig(panta_rei_base=tmp_path)
        db = DatabaseManager(":memory:")
        ctx = WorkflowContext(config=config, db_manager=db)

        assert ctx.base_dir == config.project_dir
        assert ctx.data_dir == config.data_dir
        assert ctx.targets_csv_path == config.targets_csv_path

    def test_skip_steps_default_empty(self, tmp_path):
        config = PipelineConfig(panta_rei_base=tmp_path)
        db = DatabaseManager(":memory:")
        ctx = WorkflowContext(config=config, db_manager=db)
        assert ctx.skip_steps == set()


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class TestStepResult:

    def test_success_result(self):
        r = StepResult(success=True, summary="OK", items_processed=5)
        assert r.success
        assert r.items_processed == 5
        assert r.errors == []

    def test_failure_result(self):
        r = StepResult(
            success=False, summary="Failed", errors=["error1", "error2"]
        )
        assert not r.success
        assert len(r.errors) == 2


# ---------------------------------------------------------------------------
# Step interface
# ---------------------------------------------------------------------------

class TestStepInterface:

    def test_step_is_abstract(self):
        """Cannot instantiate Step directly."""
        with pytest.raises(TypeError):
            Step()

    def test_concrete_step(self, tmp_path):
        """A concrete step subclass works correctly."""

        class TestStep(Step):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "A test step"

            def run(self, ctx: WorkflowContext) -> StepResult:
                return StepResult(success=True, summary="done")

        step = TestStep()
        assert step.name == "test"
        assert step.description == "A test step"
        assert step.should_skip(MagicMock()) is None

        config = PipelineConfig(panta_rei_base=tmp_path)
        db = DatabaseManager(":memory:")
        ctx = WorkflowContext(config=config, db_manager=db)
        result = step.run(ctx)
        assert result.success

    def test_should_skip_override(self, tmp_path):
        """A step can override should_skip to return a reason."""

        class SkippableStep(Step):
            @property
            def name(self) -> str:
                return "skippable"

            @property
            def description(self) -> str:
                return "A step that always skips"

            def should_skip(self, ctx: WorkflowContext) -> str | None:
                return "Always skip this"

            def run(self, ctx: WorkflowContext) -> StepResult:
                return StepResult(success=True, summary="ran")

        step = SkippableStep()
        config = PipelineConfig(panta_rei_base=tmp_path)
        db = DatabaseManager(":memory:")
        ctx = WorkflowContext(config=config, db_manager=db)
        reason = step.should_skip(ctx)
        assert reason == "Always skip this"
