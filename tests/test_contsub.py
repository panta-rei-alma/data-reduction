"""Tests for contsub remediation workflow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from panta_rei.core.text import now_iso
from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import ContsubRunsQueries, PIRunStatus
from panta_rei.workflows.contsub import (
    ContsubOptions,
    ContsubStep,
    needs_contsub,
    run_contsub,
    _find_base_ms_files,
)


# =====================================================================
# Detection: needs_contsub
# =====================================================================


class TestNeedsContsub:
    def test_true_when_ms_exists_no_line_ms_and_contdat(self, contsub_tree):
        needed, reason = needs_contsub(contsub_tree["member_needs"])
        assert needed is True
        assert "missing" in reason.lower()

    def test_false_when_all_line_ms_exist(self, contsub_tree):
        needed, reason = needs_contsub(contsub_tree["member_complete"])
        assert needed is False
        assert "present" in reason.lower()

    def test_true_partial_line_ms(self, contsub_tree):
        needed, reason = needs_contsub(contsub_tree["member_partial"])
        assert needed is True
        assert "2 of 3" in reason

    def test_false_no_calibrated_dir(self, tmp_path):
        member = tmp_path / "member.uid___A001_XTEST"
        member.mkdir()
        needed, reason = needs_contsub(member)
        assert needed is False
        assert "no calibrated" in reason.lower()

    def test_false_no_contdat(self, tmp_path):
        member = tmp_path / "member.uid___A001_XTEST"
        working = member / "calibrated" / "working"
        working.mkdir(parents=True)
        (working / "uid___A002_X1_X2.ms").mkdir()
        (member / "calibration").mkdir()
        # No cont.dat
        needed, reason = needs_contsub(member)
        assert needed is False
        assert "cont.dat" in reason.lower()

    def test_false_no_ms(self, tmp_path):
        member = tmp_path / "member.uid___A001_XTEST"
        working = member / "calibrated" / "working"
        working.mkdir(parents=True)
        (member / "calibration").mkdir()
        (member / "calibration" / "cont.dat").write_text("Field: X\n")
        needed, reason = needs_contsub(member)
        assert needed is False
        assert "no calibrated ms" in reason.lower()


class TestFindBaseMs:
    def test_excludes_targets_and_flagversions(self, tmp_path):
        (tmp_path / "uid___A002_X1_X2.ms").mkdir()
        (tmp_path / "uid___A002_X1_X2_targets.ms").mkdir()
        (tmp_path / "uid___A002_X1_X2_targets_line.ms").mkdir()
        result = _find_base_ms_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "uid___A002_X1_X2.ms"


# =====================================================================
# DB queries: ContsubRunsQueries
# =====================================================================


class TestContsubRunsQueries:
    def test_insert_and_mark_done(self, db):
        con = db.connect()
        row_id = ContsubRunsQueries.insert_row(
            con,
            uid="uid___a001_x3833_x64bc",
            member_dir="/path/member",
            working_dir="/path/member/calibrated/working",
            casa_cmd="casa -c script.py",
            log_path="/path/log.txt",
            started_at=now_iso(),
            status=PIRunStatus.QUEUED,
            hostname="testhost",
            eb_count=1,
        )
        assert row_id is not None
        con.commit()

        ContsubRunsQueries.mark_running(con, row_id)
        con.commit()

        ContsubRunsQueries.mark_done(
            con, row_id,
            status=PIRunStatus.SUCCESS,
            retcode=0,
            finished_at=now_iso(),
            duration_sec=42.5,
            targets_line_count=1,
        )
        con.commit()

        row = con.execute(
            "SELECT status, retcode, targets_line_count FROM contsub_runs WHERE id=?",
            (row_id,),
        ).fetchone()
        assert row[0] == PIRunStatus.SUCCESS
        assert row[1] == 0
        assert row[2] == 1

    def test_latest_success_exists(self, db):
        con = db.connect()
        uid = "uid___a001_x3833_x64bc"

        assert ContsubRunsQueries.latest_success_exists(con, uid) is False

        row_id = ContsubRunsQueries.insert_row(
            con,
            uid=uid,
            member_dir="/path",
            working_dir="/path/calibrated/working",
            casa_cmd="casa",
            log_path="/log.txt",
            started_at=now_iso(),
            status=PIRunStatus.QUEUED,
            hostname="test",
        )
        ContsubRunsQueries.mark_done(
            con, row_id,
            status=PIRunStatus.FAILED,
            retcode=1,
            finished_at=now_iso(),
            duration_sec=10.0,
        )
        con.commit()
        assert ContsubRunsQueries.latest_success_exists(con, uid) is False

        row_id2 = ContsubRunsQueries.insert_row(
            con,
            uid=uid,
            member_dir="/path",
            working_dir="/path/calibrated/working",
            casa_cmd="casa",
            log_path="/log2.txt",
            started_at=now_iso(),
            status=PIRunStatus.QUEUED,
            hostname="test",
        )
        ContsubRunsQueries.mark_done(
            con, row_id2,
            status=PIRunStatus.SUCCESS,
            retcode=0,
            finished_at=now_iso(),
            duration_sec=20.0,
        )
        con.commit()
        assert ContsubRunsQueries.latest_success_exists(con, uid) is True

    def test_summary(self, db):
        con = db.connect()
        s = ContsubRunsQueries.summary(con)
        assert "total=0" in s

        ContsubRunsQueries.insert_row(
            con,
            uid="uid1",
            member_dir="/p",
            working_dir="/p/w",
            casa_cmd="casa",
            log_path="/l",
            started_at=now_iso(),
            status=PIRunStatus.SUCCESS,
            hostname="h",
        )
        con.commit()
        s = ContsubRunsQueries.summary(con)
        assert "total=1" in s
        assert "success=1" in s


# =====================================================================
# Workflow step: ContsubStep
# =====================================================================


class TestContsubStep:
    def _make_ctx(self, contsub_tree, db):
        from panta_rei.workflows.base import WorkflowContext
        from panta_rei.config import PipelineConfig

        config = PipelineConfig(
            panta_rei_base=contsub_tree["base_dir"].parent,
        )
        return WorkflowContext(
            config=config,
            db_manager=db,
            dry_run=False,
        )

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_discovers_affected_mous(self, mock_sub, contsub_tree, db):
        mock_sub.return_value = MagicMock(returncode=0)
        ctx = self._make_ctx(contsub_tree, db)
        opts = ContsubOptions(casa_cmd="casa --pipeline")
        result = ContsubStep(opts).run(ctx)
        # Should process at least the two MOUSs that need contsub
        assert result.items_processed >= 2
        assert result.items_skipped >= 1  # member_complete

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_skips_when_needs_contsub_false(self, mock_sub, contsub_tree, db):
        ctx = self._make_ctx(contsub_tree, db)
        opts = ContsubOptions(
            casa_cmd="casa --pipeline",
            match="X64bd",  # the complete MOUS
        )
        result = ContsubStep(opts).run(ctx)
        assert result.items_skipped >= 1
        assert result.items_processed == 0
        mock_sub.assert_not_called()

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_runs_despite_db_success(self, mock_sub, contsub_tree, db):
        """On-disk is authoritative: DB success doesn't suppress rerun."""
        mock_sub.return_value = MagicMock(returncode=0)
        ctx = self._make_ctx(contsub_tree, db)

        # Record a prior success in DB for member_needs
        uid = "uid___a001_x3833_x64bc"
        con = db.connect()
        row_id = ContsubRunsQueries.insert_row(
            con,
            uid=uid,
            member_dir=str(contsub_tree["member_needs"]),
            working_dir=str(contsub_tree["member_needs"] / "calibrated" / "working"),
            casa_cmd="casa",
            log_path="/log.txt",
            started_at=now_iso(),
            status=PIRunStatus.SUCCESS,
            hostname="test",
        )
        con.commit()

        opts = ContsubOptions(casa_cmd="casa --pipeline", match="X64bc")
        result = ContsubStep(opts).run(ctx)
        # Should still run because _targets_line.ms is still missing on disk
        assert result.items_processed == 1

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_dry_run(self, mock_sub, contsub_tree, db):
        ctx = self._make_ctx(contsub_tree, db)
        ctx = WorkflowContext(
            config=ctx.config,
            db_manager=db,
            dry_run=True,
        )
        opts = ContsubOptions(casa_cmd="casa --pipeline", match="X64bc")
        result = ContsubStep(opts).run(ctx)
        assert result.items_processed == 1
        mock_sub.assert_not_called()

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_match_filter(self, mock_sub, contsub_tree, db):
        mock_sub.return_value = MagicMock(returncode=0)
        ctx = self._make_ctx(contsub_tree, db)
        opts = ContsubOptions(
            casa_cmd="casa --pipeline",
            match="X64bc",
        )
        result = ContsubStep(opts).run(ctx)
        assert result.items_processed == 1

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_limit(self, mock_sub, contsub_tree, db):
        mock_sub.return_value = MagicMock(returncode=0)
        ctx = self._make_ctx(contsub_tree, db)
        opts = ContsubOptions(casa_cmd="casa --pipeline", limit=1)
        result = ContsubStep(opts).run(ctx)
        assert result.items_processed == 1

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_re_run_bypasses_disk_check(self, mock_sub, contsub_tree, db):
        mock_sub.return_value = MagicMock(returncode=0)
        ctx = self._make_ctx(contsub_tree, db)
        opts = ContsubOptions(
            casa_cmd="casa --pipeline",
            match="X64bd",  # the complete MOUS
            re_run=True,
        )
        result = ContsubStep(opts).run(ctx)
        assert result.items_processed == 1  # re_run forces processing

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_failed_casa(self, mock_sub, contsub_tree, db):
        mock_sub.return_value = MagicMock(returncode=1)
        ctx = self._make_ctx(contsub_tree, db)
        opts = ContsubOptions(casa_cmd="casa --pipeline", match="X64bc")
        result = ContsubStep(opts).run(ctx)
        assert result.items_processed == 1
        assert len(result.errors) == 1
        assert "failed" in result.errors[0].lower()

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_multi_eb_discovery(self, mock_sub, contsub_tree, db):
        """Partial multi-EB case: 1 of 3 targets_line.ms exists."""
        mock_sub.return_value = MagicMock(returncode=0)
        ctx = self._make_ctx(contsub_tree, db)
        opts = ContsubOptions(casa_cmd="casa --pipeline", match="X64be")
        result = ContsubStep(opts).run(ctx)
        assert result.items_processed == 1

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_db_records_created(self, mock_sub, contsub_tree, db):
        mock_sub.return_value = MagicMock(returncode=0)
        ctx = self._make_ctx(contsub_tree, db)
        opts = ContsubOptions(casa_cmd="casa --pipeline", match="X64bc")
        ContsubStep(opts).run(ctx)

        con = db.connect()
        rows = con.execute("SELECT * FROM contsub_runs").fetchall()
        assert len(rows) >= 1


# =====================================================================
# Import needed for dry_run test WorkflowContext
# =====================================================================

from panta_rei.workflows.base import WorkflowContext
