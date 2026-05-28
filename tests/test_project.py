"""Unit tests for GitHubProjectManager status-protection logic.

The daily pipeline auto-syncs each issue's board column to its delivery state
("In progress" / "Delivered"). Columns a human moves an issue into manually must
be protected from that auto-revert; is_status_manually_advanced() is the guard.
A column missing from that guard silently gets reverted on the next run.
"""

from __future__ import annotations

import pytest

from panta_rei.github.project import (
    PROJECT_STATUS_DELIVERED,
    PROJECT_STATUS_DONE,
    PROJECT_STATUS_IN_PROGRESS,
    PROJECT_STATUS_NEEDS_DISCUSSION,
    PROJECT_STATUS_RECALIBRATION,
    PROJECT_STATUS_REIMAGING,
    PROJECT_STATUS_WEBLOG_QA,
    GitHubProjectManager,
)


@pytest.fixture
def manager() -> GitHubProjectManager:
    return GitHubProjectManager(token="x", org="panta-rei-alma", project_number=1)


@pytest.mark.parametrize(
    "status",
    [
        PROJECT_STATUS_WEBLOG_QA,
        PROJECT_STATUS_NEEDS_DISCUSSION,
        PROJECT_STATUS_RECALIBRATION,
        PROJECT_STATUS_REIMAGING,
        PROJECT_STATUS_DONE,
    ],
)
def test_manual_states_are_protected(manager, status):
    assert manager.is_status_manually_advanced(status) is True


@pytest.mark.parametrize(
    "status",
    [PROJECT_STATUS_IN_PROGRESS, PROJECT_STATUS_DELIVERED, None, "Unknown column"],
)
def test_auto_managed_states_are_not_protected(manager, status):
    assert manager.is_status_manually_advanced(status) is False


def test_needs_discussion_constant_matches_board_label():
    # Must match the GitHub column name byte-for-byte or the guard never fires.
    assert PROJECT_STATUS_NEEDS_DISCUSSION == "Needs discussion"
