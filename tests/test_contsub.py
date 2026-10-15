"""Tests for contsub remediation workflow."""

from __future__ import annotations

import os
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
        # Only the fresh MOUS runs; the partial one is refused, the
        # complete one skipped.
        assert result.items_processed == 1
        assert result.items_skipped >= 1  # member_complete
        assert len(result.errors) == 1 and "x64be" in result.errors[0].lower()

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
    def test_re_run_refused_when_outputs_exist(self, mock_sub, contsub_tree, db):
        ctx = self._make_ctx(contsub_tree, db)
        opts = ContsubOptions(
            casa_cmd="casa --pipeline",
            match="X64bd",  # the complete MOUS
            re_run=True,
        )
        result = ContsubStep(opts).run(ctx)
        assert not result.success
        assert result.items_processed == 0
        assert "contsub over existing outputs is unsupported" in result.errors[0]
        mock_sub.assert_not_called()
        # Existing output untouched
        working = contsub_tree["member_complete"] / "calibrated" / "working"
        assert (working / "uid___A002_XABCD_XEFGH_targets_line.ms").is_dir()

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_re_run_forwarded_to_script(self, mock_sub, contsub_tree, db):
        mock_sub.return_value = MagicMock(returncode=0)
        ctx = self._make_ctx(contsub_tree, db)
        ContsubStep(ContsubOptions(
            casa_cmd="casa --pipeline", match="X64bc", re_run=True,
        )).run(ctx)  # no outputs yet → allowed, flag forwarded
        assert mock_sub.call_args[0][0][-1] == "--re-run"

        mock_sub.reset_mock()
        ContsubStep(ContsubOptions(casa_cmd="casa --pipeline", match="X64bc")).run(ctx)
        assert "--re-run" not in mock_sub.call_args[0][0]

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
    def test_partial_multi_eb_refused(self, mock_sub, contsub_tree, db):
        """Partial multi-EB case (1 of 3 targets_line.ms exists): the
        default path refuses too and never launches CASA."""
        ctx = self._make_ctx(contsub_tree, db)
        opts = ContsubOptions(casa_cmd="casa --pipeline", match="X64be")
        result = ContsubStep(opts).run(ctx)
        assert result.items_processed == 0
        assert "contsub over existing outputs is unsupported" in result.errors[0]
        mock_sub.assert_not_called()

    @patch("panta_rei.workflows.contsub.subprocess.run")
    def test_targets_present_line_missing_refused(self, mock_sub, contsub_tree, db):
        working = contsub_tree["member_needs"] / "calibrated" / "working"
        (working / "uid___A002_X12fb842_X4e4e_targets.ms").mkdir()
        ctx = self._make_ctx(contsub_tree, db)
        result = ContsubStep(ContsubOptions(
            casa_cmd="casa --pipeline", match="X64bc",
        )).run(ctx)
        assert result.items_processed == 0 and len(result.errors) == 1
        mock_sub.assert_not_called()

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
# CASA script: contsub_remediation.py (pipeline tasks faked)
# =====================================================================


class TestContsubRemediationScript:
    @pytest.fixture
    def script(self, monkeypatch):
        import panta_rei.casa.contsub_remediation as script
        calls = []

        def fake_mstransform(**_kw):
            calls.append("hif_mstransform")
            for ms in script.find_base_ms_files(os.getcwd()):
                os.mkdir(ms.replace(".ms", "_targets.ms"))

        def fake_uvcontsub(**_kw):
            calls.append("hif_uvcontsub")
            for ms in script.find_base_ms_files(os.getcwd()):
                os.mkdir(ms.replace(".ms", "_targets_line.ms"))

        msmd = MagicMock()
        msmd.fieldnames.return_value = ["SRC1"]
        msmd.spwsforintent.return_value = [22]
        fakes = {
            "h_resume": lambda *_a: calls.append("h_resume"),
            "h_init": lambda: None,
            "hifa_restoredata": lambda **_kw: None,
            "hif_mstransform": fake_mstransform,
            "hifa_flagtargets": lambda **_kw: calls.append("hifa_flagtargets"),
            "hif_uvcontsub": fake_uvcontsub,
            "h_save": lambda: None,
            "msmd": msmd,
        }
        for name, fake in fakes.items():
            monkeypatch.setattr(script, name, fake, raising=False)
        script.calls = calls
        return script

    def _complete_member(self, contsub_tree):
        member = contsub_tree["member_needs"]
        working = member / "calibrated" / "working"
        (working / "uid___A002_X12fb842_X4e4e_targets.ms").mkdir()
        line = working / "uid___A002_X12fb842_X4e4e_targets_line.ms"
        line.mkdir()
        (line / "old_marker").write_text("old")
        return member, working

    def _main(self, script, monkeypatch, *argv):
        monkeypatch.setattr("sys.argv", ["contsub_remediation.py", *argv])
        with pytest.raises(SystemExit) as exc:
            script.main()
        return exc.value.code

    def test_complete_without_re_run_is_noop(self, script, contsub_tree, monkeypatch):
        member, _ = self._complete_member(contsub_tree)
        monkeypatch.chdir(member)
        assert self._main(script, monkeypatch, "--member-dir", str(member)) == 0
        assert script.calls == []

    @pytest.mark.parametrize("extra", [[], ["--dry-run"]])
    def test_re_run_refused_when_outputs_exist(self, script, contsub_tree,
                                               monkeypatch, caplog, extra):
        member, working = self._complete_member(contsub_tree)
        monkeypatch.chdir(member)
        code = self._main(script, monkeypatch, "--member-dir", str(member),
                          "--re-run", *extra)
        assert code == 2
        assert script.calls == []
        assert "contsub over existing outputs is unsupported" in caplog.text
        line = working / "uid___A002_X12fb842_X4e4e_targets_line.ms"
        assert (line / "old_marker").read_text() == "old"
        assert sorted(p.name for p in working.iterdir() if "targets" in p.name) == [
            "uid___A002_X12fb842_X4e4e_targets.ms",
            "uid___A002_X12fb842_X4e4e_targets_line.ms",
        ]

    def test_partial_state_refused_without_re_run(self, script, contsub_tree,
                                                 monkeypatch, caplog):
        """_targets.ms present, _targets_line.ms missing: refuse, exit 2."""
        member = contsub_tree["member_needs"]
        working = member / "calibrated" / "working"
        (working / "uid___A002_X12fb842_X4e4e_targets.ms").mkdir()
        monkeypatch.chdir(member)
        assert self._main(script, monkeypatch, "--member-dir", str(member)) == 2
        assert script.calls == []
        assert "contsub over existing outputs is unsupported" in caplog.text
        assert not (working / "uid___A002_X12fb842_X4e4e_targets_line.ms").exists()

    def test_fresh_member_proceeds(self, script, contsub_tree, monkeypatch):
        member = contsub_tree["member_needs"]
        monkeypatch.chdir(member)
        assert self._main(script, monkeypatch, "--member-dir", str(member)) == 0
        assert "hif_uvcontsub" in script.calls

    def test_re_run_without_outputs_runs_normally(self, script, contsub_tree, monkeypatch):
        member = contsub_tree["member_needs"]
        monkeypatch.chdir(member)
        code = self._main(script, monkeypatch, "--member-dir", str(member), "--re-run")
        assert code == 0
        assert "hif_uvcontsub" in script.calls

    def test_refusal_message_matches_workflow(self, script):
        from panta_rei.workflows.contsub import REFUSED_MSG
        assert script.REFUSED_MSG == REFUSED_MSG

    def test_has_uvcontfit_sees_global_task(self, script, contsub_tree, monkeypatch):
        member = contsub_tree["member_needs"]
        pipescript = next((member / "script").glob("*casa_pipescript.py"))
        pipescript.write_text("hif_uvcontfit()\nhif_uvcontsub()\n")
        assert script.check_has_uvcontfit(str(member)) is False
        monkeypatch.setattr(script, "hif_uvcontfit", lambda **_kw: None, raising=False)
        assert script.check_has_uvcontfit(str(member)) is True


# =====================================================================
# Import needed for dry_run test WorkflowContext
# =====================================================================

from panta_rei.workflows.base import WorkflowContext
