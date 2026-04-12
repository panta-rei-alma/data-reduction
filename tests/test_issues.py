"""Integration tests for the UpdateIssuesStep.

Tests exercise the GH_DRY_RUN mode (zero API calls), missing token path,
and should_skip when the targets CSV is absent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from panta_rei.config import PipelineConfig
from panta_rei.db.connection import DatabaseManager
from panta_rei.workflows.base import StepResult, WorkflowContext
from panta_rei.workflows.retrieval import UpdateIssuesStep

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_HEADER = "sb_name,array,source_name,gous_id,mous_ids,Line group\n"
_CSV_ROWS = [
    "SB_NGC1234_a_06_TM1,TM,NGC1234,uid___A001_X3833_X64b9,uid___A001_X3833_X64bc,N2H+\n",
    "SB_NGC5678_a_06_SM1,SM,NGC5678,uid___A001_X3833_X64ba,uid___A001_X3833_X64bd,HCO+\n",
    "SB_NGC9012_a_06_TP,TP,NGC9012,uid___A001_X3833_X64bb,uid___A001_X3833_X64be,N2H+\n",
]


def _make_ctx(
    tmp_path: Path,
    db: DatabaseManager,
    write_csv: bool = True,
) -> WorkflowContext:
    """Build a WorkflowContext with an optional targets CSV."""
    config = PipelineConfig(panta_rei_base=tmp_path)
    # Ensure the project dir exists
    config.project_dir.mkdir(parents=True, exist_ok=True)

    if write_csv:
        csv_path = config.targets_csv_path
        csv_path.write_text(_CSV_HEADER + "".join(_CSV_ROWS))

    return WorkflowContext(config=config, db_manager=db)


# ---------------------------------------------------------------------------
# Step properties
# ---------------------------------------------------------------------------

class TestUpdateIssuesStepProperties:

    def test_step_name_and_description(self):
        step = UpdateIssuesStep()
        assert step.name == "update_issues"
        assert "issue" in step.description.lower()

    def test_should_skip_in_skip_set(self, tmp_path, db):
        config = PipelineConfig(panta_rei_base=tmp_path)
        ctx = WorkflowContext(
            config=config, db_manager=db,
            skip_steps={"update_issues"},
        )
        reason = UpdateIssuesStep().should_skip(ctx)
        assert reason is not None
        assert "skip_steps" in reason


# ---------------------------------------------------------------------------
# should_skip when CSV is missing
# ---------------------------------------------------------------------------

class TestShouldSkipMissingCSV:

    def test_should_skip_returns_reason_when_csv_missing(self, tmp_path, db):
        """When targets_by_array.csv does not exist, should_skip returns a reason."""
        ctx = _make_ctx(tmp_path, db, write_csv=False)
        reason = UpdateIssuesStep().should_skip(ctx)
        assert reason is not None
        assert "targets CSV" in reason or "targets_by_array" in reason

    def test_should_not_skip_when_csv_present(self, tmp_path, db):
        """When CSV exists, should_skip returns None (proceed)."""
        ctx = _make_ctx(tmp_path, db, write_csv=True)
        reason = UpdateIssuesStep().should_skip(ctx)
        assert reason is None


# ---------------------------------------------------------------------------
# Missing GitHub token
# ---------------------------------------------------------------------------

class TestMissingGitHubToken:

    def test_returns_failure_without_token(self, tmp_path, db, monkeypatch):
        """Without a GitHub token and not in dry run, step returns failure."""
        ctx = _make_ctx(tmp_path, db, write_csv=True)

        # Ensure no token sources are available
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_DRY_RUN", raising=False)
        monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)

        result = UpdateIssuesStep().run(ctx)

        assert not result.success
        assert "token" in result.summary.lower()
        assert len(result.errors) >= 1


# ---------------------------------------------------------------------------
# GH_DRY_RUN mode (zero API calls)
# ---------------------------------------------------------------------------

class TestGHDryRunMode:

    def test_dry_run_writes_json_payloads(self, tmp_path, db, monkeypatch):
        """GH_DRY_RUN=1 writes JSON payloads with zero GitHub API calls."""
        monkeypatch.setenv("GH_DRY_RUN", "1")
        # Ensure no GitHub token is needed
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)

        ctx = _make_ctx(tmp_path, db, write_csv=True)
        step = UpdateIssuesStep()

        result = step.run(ctx)

        assert result.success
        assert result.items_processed == 3  # 3 CSV rows

        # Verify JSON payload file was written
        json_path = ctx.config.project_dir / "gh_dry_run_payloads.json"
        assert json_path.exists(), f"Expected {json_path} to be written"

        payloads = json.loads(json_path.read_text())
        assert len(payloads) == 3

        # Verify payload structure
        titles = [p["title"] for p in payloads]
        assert "SB: SB_NGC1234_a_06_TM1" in titles
        assert "SB: SB_NGC5678_a_06_SM1" in titles
        assert "SB: SB_NGC9012_a_06_TP" in titles

        # Verify metadata propagated from CSV
        tm_payload = next(p for p in payloads if "TM1" in p["title"])
        assert tm_payload["array"] == "TM"
        assert "NGC1234" in tm_payload["source_names"]
        assert tm_payload["line_group"] == "N2H+"
        # Enriched payloads include body, labels, project_status
        assert "body" in tm_payload
        assert "labels" in tm_payload
        assert "project_status" in tm_payload

    def test_dry_run_makes_zero_api_calls(self, tmp_path, db, monkeypatch):
        """GH_DRY_RUN mode should not call resolve_github_token or GhApi."""
        monkeypatch.setenv("GH_DRY_RUN", "1")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)

        ctx = _make_ctx(tmp_path, db, write_csv=True)

        with patch("panta_rei.auth.resolve_github_token") as mock_token, \
             patch("panta_rei.github.issues.GitHubIssueManager") as mock_manager:

            result = UpdateIssuesStep().run(ctx)

            # resolve_github_token should NOT be called in dry run
            mock_token.assert_not_called()
            # GitHubIssueManager should NOT be constructed
            mock_manager.assert_not_called()

        assert result.success

    def test_dry_run_fails_without_csv(self, tmp_path, db, monkeypatch):
        """GH_DRY_RUN returns failure if the CSV is missing (skipped by should_skip
        in normal flow, but _run_zero_api_dry_run also guards against it)."""
        monkeypatch.setenv("GH_DRY_RUN", "1")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        # Create context WITHOUT the CSV, but bypass should_skip by calling run directly
        config = PipelineConfig(panta_rei_base=tmp_path)
        config.project_dir.mkdir(parents=True, exist_ok=True)
        ctx = WorkflowContext(config=config, db_manager=db)

        # Note: should_skip would normally catch this, but the step's run()
        # also handles it for robustness.
        result = UpdateIssuesStep().run(ctx)

        assert not result.success
        assert "CSV" in result.summary or "csv" in result.summary.lower()

    def test_dry_run_summary_includes_count(self, tmp_path, db, monkeypatch):
        """The summary from dry run includes the payload count."""
        monkeypatch.setenv("GH_DRY_RUN", "1")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        ctx = _make_ctx(tmp_path, db, write_csv=True)
        result = UpdateIssuesStep().run(ctx)

        assert result.success
        assert "3" in result.summary
        assert "payload" in result.summary.lower()


# ---------------------------------------------------------------------------
# CSV with blank sb_name rows
# ---------------------------------------------------------------------------

class TestCSVEdgeCases:

    def test_dry_run_skips_blank_sb_name_rows(self, tmp_path, db, monkeypatch):
        """Rows with empty sb_name are skipped in GH_DRY_RUN payloads."""
        monkeypatch.setenv("GH_DRY_RUN", "1")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        config = PipelineConfig(panta_rei_base=tmp_path)
        config.project_dir.mkdir(parents=True, exist_ok=True)

        csv_content = (
            "sb_name,array,source_name,gous_id,mous_ids,Line group\n"
            "SB_Valid,TM,NGC1234,g1,m1,N2H+\n"
            ",SM,,g2,m2,HCO+\n"  # blank sb_name
            "  ,TP,,g3,m3,N2H+\n"  # whitespace-only sb_name
        )
        config.targets_csv_path.write_text(csv_content)

        ctx = WorkflowContext(config=config, db_manager=db)
        result = UpdateIssuesStep().run(ctx)

        assert result.success
        assert result.items_processed == 1  # only the valid row

        json_path = config.project_dir / "gh_dry_run_payloads.json"
        payloads = json.loads(json_path.read_text())
        assert len(payloads) == 1
        assert payloads[0]["title"] == "SB: SB_Valid"
