"""Workflow step ABC and supporting types.

Defines the Step interface, WorkflowContext for shared state, and
a run_workflow helper that executes a list of steps with logging.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from panta_rei.config import PipelineConfig
from panta_rei.db.connection import DatabaseManager

log = logging.getLogger(__name__)


@dataclass
class WorkflowContext:
    """Shared context passed to all workflow steps."""

    config: PipelineConfig
    db_manager: DatabaseManager
    username: Optional[str] = None
    skip_steps: set[str] = field(default_factory=set)
    non_interactive: bool = False
    dry_run: bool = False

    @property
    def base_dir(self) -> Path:
        return self.config.project_dir

    @property
    def data_dir(self) -> Path:
        return self.config.data_dir

    @property
    def targets_csv_path(self) -> Path:
        return self.config.targets_csv_path


@dataclass
class StepResult:
    """Outcome of a single workflow step."""

    success: bool
    summary: str
    items_processed: int = 0
    items_skipped: int = 0
    errors: list[str] = field(default_factory=list)


class Step(ABC):
    """Abstract base class for a single workflow step."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    def should_skip(self, ctx: WorkflowContext) -> str | None:
        """Return a reason string to skip, or None to proceed."""
        if self.name in ctx.skip_steps:
            return f"step '{self.name}' listed in skip_steps"
        return None

    @abstractmethod
    def run(self, ctx: WorkflowContext) -> StepResult:
        ...


def run_workflow(
    steps: list[Step],
    ctx: WorkflowContext,
) -> dict[str, StepResult]:
    """Execute *steps* in order, respecting skip checks and logging progress.

    Parameters
    ----------
    steps:
        Ordered list of :class:`Step` instances to execute.
    ctx:
        Shared :class:`WorkflowContext` for every step.

    Returns
    -------
    dict mapping step name to its :class:`StepResult`.
    """
    results: dict[str, StepResult] = {}

    for step in steps:
        skip_reason = step.should_skip(ctx)
        if skip_reason is not None:
            log.info("Skipping %s: %s", step.name, skip_reason)
            results[step.name] = StepResult(
                success=True,
                summary=f"Skipped: {skip_reason}",
            )
            continue

        log.info(
            "--- Running step: %s (%s) ---", step.name, step.description
        )
        try:
            result = step.run(ctx)
        except Exception as exc:
            log.error(
                "Step %s raised an unhandled exception: %s",
                step.name,
                exc,
                exc_info=True,
            )
            result = StepResult(
                success=False,
                summary=f"Unhandled exception: {exc}",
                errors=[str(exc)],
            )

        results[step.name] = result

        status = "OK" if result.success else "FAILED"
        log.info(
            "Step %s finished [%s]: %s "
            "(processed=%d, skipped=%d, errors=%d)",
            step.name,
            status,
            result.summary,
            result.items_processed,
            result.items_skipped,
            len(result.errors),
        )

    return results
