"""Tests for panta_rei.db — CRUD operations on obs and pi_runs tables."""

import sqlite3
from pathlib import Path

import pytest

from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import ObsQueries, ObsStatus, PIRunsQueries, PIRunStatus
from panta_rei.core.text import now_iso


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """In-memory database with full schema."""
    return DatabaseManager(":memory:")


@pytest.fixture
def con(db):
    """Connection from in-memory database."""
    return db.connect()


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

class TestSchemaCreation:

    def test_obs_table_exists(self, con):
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='obs'"
        ).fetchone()
        assert row is not None

    def test_pi_runs_table_exists(self, con):
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pi_runs'"
        ).fetchone()
        assert row is not None

    def test_schema_version_table_exists(self, con):
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()
        assert row is not None

    def test_all_migrations_recorded(self, con):
        count = con.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == 13

    def test_obs_has_all_columns(self, con):
        cols = {row[1] for row in con.execute("PRAGMA table_info(obs)")}
        expected = {
            "uid", "status", "release_date", "tar_path", "tar_deleted",
            "extracted_root", "n_extracted", "n_skipped",
            "created_at", "updated_at", "last_seen_at",
            "weblog_staged", "weblog_path", "weblog_url", "weblog_staged_at",
        }
        assert expected.issubset(cols)

    def test_pi_runs_has_all_columns(self, con):
        cols = {row[1] for row in con.execute("PRAGMA table_info(pi_runs)")}
        expected = {
            "id", "uid", "script_path", "cwd", "casa_cmd", "log_path",
            "started_at", "finished_at", "retcode", "status",
            "hostname", "duration_sec",
            "sg_uid", "gous_uid", "mous_uid", "array",
            "sb_name", "source_name", "line_group",
        }
        assert expected.issubset(cols)

    def test_indexes_exist(self, con):
        indexes = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        expected = {
            "idx_status", "idx_release",
            "idx_pi_runs_uid", "idx_pi_runs_status", "idx_pi_runs_mous",
            "idx_contsub_uid", "idx_contsub_status", "idx_contsub_mous",
        }
        assert expected.issubset(indexes)


# ---------------------------------------------------------------------------
# ObsQueries CRUD
# ---------------------------------------------------------------------------

class TestObsQueries:

    def test_upsert_seen(self, con):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", "2025-06-01")
        con.commit()
        row = con.execute("SELECT uid, status, release_date FROM obs").fetchone()
        assert row[0] == "uid___a001_x123_x456"
        assert row[1] == ObsStatus.PENDING
        assert row[2] == "2025-06-01"

    def test_upsert_seen_updates_existing(self, con):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", "2025-06-01")
        con.commit()
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", "2025-07-01")
        con.commit()
        count = con.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
        assert count == 1
        row = con.execute("SELECT release_date FROM obs").fetchone()
        assert row[0] == "2025-07-01"

    def test_upsert_seen_ignores_invalid_uid(self, con):
        ObsQueries.upsert_seen(con, "", None)
        ObsQueries.upsert_seen(con, None, None)
        con.commit()
        count = con.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
        assert count == 0

    def test_mark_downloaded(self, con, tmp_path):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        con.commit()
        tar = tmp_path / "test.tar"
        tar.touch()
        ObsQueries.mark_downloaded(con, "uid://A001/X123/X456", tar)
        con.commit()
        row = con.execute("SELECT status, tar_path FROM obs").fetchone()
        assert row[0] == ObsStatus.DOWNLOADED
        assert str(tmp_path) in row[1]

    def test_mark_extracted(self, con, tmp_path):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        con.commit()
        ObsQueries.mark_extracted(
            con, "uid://A001/X123/X456", tmp_path, 100, 5, True
        )
        con.commit()
        row = con.execute(
            "SELECT status, n_extracted, n_skipped, tar_deleted FROM obs"
        ).fetchone()
        assert row[0] == ObsStatus.EXTRACTED
        assert row[1] == 100
        assert row[2] == 5
        assert row[3] == 1

    def test_mark_error(self, con):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        con.commit()
        ObsQueries.mark_error(con, "uid://A001/X123/X456")
        con.commit()
        row = con.execute("SELECT status FROM obs").fetchone()
        assert row[0] == ObsStatus.ERROR

    def test_reset_to_pending(self, con, tmp_path):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        ObsQueries.mark_extracted(
            con, "uid://A001/X123/X456", tmp_path, 50, 0, False
        )
        con.commit()
        result = ObsQueries.reset_to_pending(con, "uid://A001/X123/X456")
        con.commit()
        assert result is True
        row = con.execute(
            "SELECT status, tar_path, extracted_root FROM obs"
        ).fetchone()
        assert row[0] == ObsStatus.PENDING
        assert row[1] is None
        assert row[2] is None

    def test_get_status(self, con):
        assert ObsQueries.get_status(con, "uid://A001/X123/X456") is None
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        con.commit()
        assert ObsQueries.get_status(con, "uid://A001/X123/X456") == ObsStatus.PENDING

    def test_uids_to_download(self, con, tmp_path):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        ObsQueries.upsert_seen(con, "uid://A001/X123/X789", None)
        ObsQueries.mark_extracted(
            con, "uid://A001/X123/X456", tmp_path, 10, 0, False
        )
        con.commit()
        to_dl = ObsQueries.uids_to_download(
            con, ["uid://A001/X123/X456", "uid://A001/X123/X789"]
        )
        assert len(to_dl) == 1
        assert "uid://A001/X123/X789" in to_dl

    def test_summary(self, con, tmp_path):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        ObsQueries.upsert_seen(con, "uid://A001/X123/X789", None)
        ObsQueries.mark_extracted(
            con, "uid://A001/X123/X456", tmp_path, 10, 0, False
        )
        con.commit()
        s = ObsQueries.summary(con)
        assert "total=2" in s
        assert "extracted=1" in s
        assert "pending=1" in s

    def test_status_transition_pending_to_error_to_pending(self, con):
        """Test round-trip: pending -> error -> pending (reset)."""
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        con.commit()
        assert ObsQueries.get_status(con, "uid://A001/X123/X456") == ObsStatus.PENDING

        ObsQueries.mark_error(con, "uid://A001/X123/X456")
        con.commit()
        assert ObsQueries.get_status(con, "uid://A001/X123/X456") == ObsStatus.ERROR

        ObsQueries.reset_to_pending(con, "uid://A001/X123/X456")
        con.commit()
        assert ObsQueries.get_status(con, "uid://A001/X123/X456") == ObsStatus.PENDING


# ---------------------------------------------------------------------------
# Weblog operations
# ---------------------------------------------------------------------------

class TestWeblogOperations:

    def test_mark_weblog_staged(self, con, tmp_path):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        ObsQueries.mark_extracted(
            con, "uid://A001/X123/X456", tmp_path, 10, 0, False
        )
        con.commit()

        result = ObsQueries.mark_weblog_staged(
            con,
            "uid://A001/X123/X456",
            tmp_path / "weblog",
            "https://example.com/weblog",
        )
        con.commit()
        assert result is True

        row = con.execute(
            "SELECT weblog_staged, weblog_path, weblog_url FROM obs"
        ).fetchone()
        assert row[0] == 1
        assert "weblog" in row[1]
        assert row[2] == "https://example.com/weblog"

    def test_get_unstaged_extracted(self, con, tmp_path):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        ObsQueries.mark_extracted(
            con, "uid://A001/X123/X456", tmp_path, 10, 0, False
        )
        ObsQueries.upsert_seen(con, "uid://A001/X123/X789", None)
        ObsQueries.mark_extracted(
            con, "uid://A001/X123/X789", tmp_path, 10, 0, False
        )
        ObsQueries.mark_weblog_staged(
            con, "uid://A001/X123/X456", tmp_path / "weblog", None
        )
        con.commit()

        unstaged = ObsQueries.get_unstaged_extracted(con)
        assert len(unstaged) == 1
        assert unstaged[0]["uid"] == "uid___a001_x123_x789"

    def test_get_all_weblog_urls(self, con, tmp_path):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        ObsQueries.mark_extracted(
            con, "uid://A001/X123/X456", tmp_path, 10, 0, False
        )
        ObsQueries.mark_weblog_staged(
            con,
            "uid://A001/X123/X456",
            tmp_path / "weblog",
            "https://example.com/weblog",
        )
        con.commit()

        urls = ObsQueries.get_all_weblog_urls(con)
        assert "uid___a001_x123_x456" in urls
        assert urls["uid___a001_x123_x456"] == "https://example.com/weblog"

    def test_reset_weblog_status(self, con, tmp_path):
        ObsQueries.upsert_seen(con, "uid://A001/X123/X456", None)
        ObsQueries.mark_extracted(
            con, "uid://A001/X123/X456", tmp_path, 10, 0, False
        )
        ObsQueries.mark_weblog_staged(
            con, "uid://A001/X123/X456", tmp_path / "weblog", "https://x.com"
        )
        con.commit()

        result = ObsQueries.reset_weblog_status(con, "uid://A001/X123/X456")
        con.commit()
        assert result is True

        row = con.execute(
            "SELECT weblog_staged, weblog_path, weblog_url FROM obs"
        ).fetchone()
        assert row[0] == 0
        assert row[1] is None
        assert row[2] is None


# ---------------------------------------------------------------------------
# PIRunsQueries CRUD
# ---------------------------------------------------------------------------

class TestPIRunsQueries:

    def test_insert_row(self, con):
        row_id = PIRunsQueries.insert_row(
            con,
            uid="uid___a001_x123_x456",
            script_path="/path/to/script.py",
            cwd="/path/to",
            casa_cmd="casa -c script.py",
            log_path="/path/to/log.txt",
            started_at=now_iso(),
            status=PIRunStatus.QUEUED,
            hostname="testhost",
        )
        con.commit()
        assert row_id is not None
        assert row_id > 0

    def test_mark_running(self, con):
        row_id = PIRunsQueries.insert_row(
            con,
            uid="uid___a001_x123_x456",
            script_path="/path/to/script.py",
            cwd="/path/to",
            casa_cmd="casa -c script.py",
            log_path="/path/to/log.txt",
            started_at=now_iso(),
            status=PIRunStatus.QUEUED,
            hostname="testhost",
        )
        PIRunsQueries.mark_running(con, row_id)
        con.commit()
        row = con.execute(
            "SELECT status FROM pi_runs WHERE id=?", (row_id,)
        ).fetchone()
        assert row[0] == PIRunStatus.RUNNING

    def test_mark_done(self, con):
        row_id = PIRunsQueries.insert_row(
            con,
            uid="uid___a001_x123_x456",
            script_path="/path/to/script.py",
            cwd="/path/to",
            casa_cmd="casa -c script.py",
            log_path="/path/to/log.txt",
            started_at=now_iso(),
            status=PIRunStatus.QUEUED,
            hostname="testhost",
        )
        PIRunsQueries.mark_done(
            con, row_id, PIRunStatus.SUCCESS, 0, now_iso(), 123.4
        )
        con.commit()
        row = con.execute(
            "SELECT status, retcode, duration_sec FROM pi_runs WHERE id=?",
            (row_id,),
        ).fetchone()
        assert row[0] == PIRunStatus.SUCCESS
        assert row[1] == 0
        assert abs(row[2] - 123.4) < 0.01

    def test_latest_success_exists(self, con):
        assert not PIRunsQueries.latest_success_exists(con, "uid___a001_x123_x456")

        row_id = PIRunsQueries.insert_row(
            con,
            uid="uid___a001_x123_x456",
            script_path="/path/to/script.py",
            cwd="/path/to",
            casa_cmd="casa -c script.py",
            log_path="/path/to/log.txt",
            started_at=now_iso(),
            status=PIRunStatus.QUEUED,
            hostname="testhost",
        )
        PIRunsQueries.mark_done(
            con, row_id, PIRunStatus.SUCCESS, 0, now_iso(), 100.0
        )
        con.commit()
        assert PIRunsQueries.latest_success_exists(con, "uid___a001_x123_x456")

    def test_latest_success_false_for_failed(self, con):
        row_id = PIRunsQueries.insert_row(
            con,
            uid="uid___a001_x123_x456",
            script_path="/path/to/script.py",
            cwd="/path/to",
            casa_cmd="casa -c script.py",
            log_path="/path/to/log.txt",
            started_at=now_iso(),
            status=PIRunStatus.QUEUED,
            hostname="testhost",
        )
        PIRunsQueries.mark_done(
            con, row_id, PIRunStatus.FAILED, 1, now_iso(), 50.0
        )
        con.commit()
        assert not PIRunsQueries.latest_success_exists(con, "uid___a001_x123_x456")

    def test_summary(self, con):
        for status in [PIRunStatus.SUCCESS, PIRunStatus.SUCCESS, PIRunStatus.FAILED]:
            row_id = PIRunsQueries.insert_row(
                con,
                uid="uid___a001_x123_x456",
                script_path="/path/to/script.py",
                cwd="/path/to",
                casa_cmd="casa -c script.py",
                log_path="/path/to/log.txt",
                started_at=now_iso(),
                status=status,
                hostname="testhost",
            )
        con.commit()
        s = PIRunsQueries.summary(con)
        assert "total=3" in s
        assert "success=2" in s
        assert "failed=1" in s
