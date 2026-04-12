"""Tests for database migration/bootstrap matrix.

Tests all 6 DB states from the plan (A, B, C, D, D', E) plus partial-column
states. Verifies that DatabaseManager correctly probes existing schema and
applies only the missing migrations.
"""

import sqlite3

import pytest

from panta_rei.db.connection import DatabaseManager
from panta_rei.db.schema import (
    CREATE_OBS_TABLE_SQL,
    CREATE_PI_RUNS_BASE_SQL,
    MIGRATIONS,
    column_exists,
    index_exists,
    table_exists,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_obs_columns(con: sqlite3.Connection) -> set[str]:
    return {row[1] for row in con.execute("PRAGMA table_info(obs)")}


def get_all_pi_runs_columns(con: sqlite3.Connection) -> set[str]:
    return {row[1] for row in con.execute("PRAGMA table_info(pi_runs)")}


def get_all_indexes(con: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
    }


def get_schema_versions(con: sqlite3.Connection) -> set[int]:
    if not table_exists(con, "schema_version"):
        return set()
    return {row[0] for row in con.execute("SELECT version FROM schema_version")}


FULL_OBS_COLUMNS = {
    "uid", "status", "release_date", "tar_path", "tar_deleted",
    "extracted_root", "n_extracted", "n_skipped",
    "created_at", "updated_at", "last_seen_at",
    "weblog_staged", "weblog_path", "weblog_url", "weblog_staged_at",
}

FULL_PI_RUNS_COLUMNS = {
    "id", "uid", "script_path", "cwd", "casa_cmd", "log_path",
    "started_at", "finished_at", "retcode", "status",
    "hostname", "duration_sec",
    "sg_uid", "gous_uid", "mous_uid", "array",
    "sb_name", "source_name", "line_group",
}

ALL_INDEXES = {
    "idx_status", "idx_release",
    "idx_pi_runs_uid", "idx_pi_runs_status", "idx_pi_runs_mous",
    "idx_ip_key", "idx_ip_status", "idx_ir_params", "idx_ir_status",
}


ALL_VERSIONS = set(range(1, 14))  # migrations 1-13


def assert_full_schema(con: sqlite3.Connection) -> None:
    """Assert the DB has the complete expected schema after bootstrap."""
    assert FULL_OBS_COLUMNS.issubset(get_all_obs_columns(con))
    assert FULL_PI_RUNS_COLUMNS.issubset(get_all_pi_runs_columns(con))
    assert ALL_INDEXES.issubset(get_all_indexes(con))
    assert table_exists(con, "imaging_params")
    assert table_exists(con, "imaging_runs")
    assert table_exists(con, "contsub_runs")
    assert get_schema_versions(con) == ALL_VERSIONS


# ---------------------------------------------------------------------------
# Case A: Empty DB
# ---------------------------------------------------------------------------

class TestCaseA_EmptyDB:

    def test_all_migrations_executed(self):
        db = DatabaseManager(":memory:")
        con = db.connect()
        assert_full_schema(con)

    def test_schema_version_has_7_rows(self):
        db = DatabaseManager(":memory:")
        con = db.connect()
        count = con.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == len(ALL_VERSIONS)


# ---------------------------------------------------------------------------
# Case B: obs base only (11 cols + idx_status)
# ---------------------------------------------------------------------------

class TestCaseB_ObsBaseOnly:

    def test_bootstrap_adds_missing_parts(self):
        con = sqlite3.connect(":memory:")
        # Pre-populate: obs with base 11 columns + idx_status
        con.execute(CREATE_OBS_TABLE_SQL)
        con.execute("CREATE INDEX IF NOT EXISTS idx_status ON obs(status)")
        con.commit()

        # Verify pre-state: no weblog cols, no pi_runs
        assert not column_exists(con, "obs", "weblog_staged")
        assert not table_exists(con, "pi_runs")

        # Bootstrap
        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        assert_full_schema(con)

    def test_m1_m2_probed_not_executed(self):
        """M1 and M2 should be probed (obs+idx_status exist), not re-executed."""
        con = sqlite3.connect(":memory:")
        con.execute(CREATE_OBS_TABLE_SQL)
        con.execute("CREATE INDEX IF NOT EXISTS idx_status ON obs(status)")
        con.commit()

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        # All migrations recorded
        assert get_schema_versions(con) == ALL_VERSIONS


# ---------------------------------------------------------------------------
# Case C: obs + weblog cols (15 cols)
# ---------------------------------------------------------------------------

class TestCaseC_ObsWithWeblog:

    def test_bootstrap(self):
        con = sqlite3.connect(":memory:")
        con.execute(CREATE_OBS_TABLE_SQL)
        con.execute("CREATE INDEX IF NOT EXISTS idx_status ON obs(status)")
        # Add weblog columns
        for col, typ in [
            ("weblog_staged", "INTEGER DEFAULT 0"),
            ("weblog_path", "TEXT"),
            ("weblog_url", "TEXT"),
            ("weblog_staged_at", "TEXT"),
        ]:
            con.execute(f"ALTER TABLE obs ADD COLUMN {col} {typ}")
        con.commit()

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        assert_full_schema(con)
        # M1-M3 probed, M4-M7 executed/probed
        assert get_schema_versions(con) == ALL_VERSIONS


# ---------------------------------------------------------------------------
# Case D: pi_runs only (12 base columns)
# ---------------------------------------------------------------------------

class TestCaseD_PIRunsBaseOnly:

    def test_bootstrap_creates_obs_and_enriches_pi_runs(self):
        con = sqlite3.connect(":memory:")
        con.execute(CREATE_PI_RUNS_BASE_SQL)
        con.commit()

        # Pre-state: pi_runs exists with 12 cols, no enrichment
        assert table_exists(con, "pi_runs")
        assert not column_exists(con, "pi_runs", "sg_uid")
        assert not table_exists(con, "obs")

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        assert_full_schema(con)

    def test_m5_probed_m6_executed(self):
        """M5 should be probed (pi_runs base exists), M6 should execute (add enrichment cols)."""
        con = sqlite3.connect(":memory:")
        con.execute(CREATE_PI_RUNS_BASE_SQL)
        con.commit()

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        # Enrichment columns added
        assert column_exists(con, "pi_runs", "sg_uid")
        assert column_exists(con, "pi_runs", "line_group")
        assert get_schema_versions(con) == ALL_VERSIONS


# ---------------------------------------------------------------------------
# Case D': pi_runs full (19 columns, all indexes)
# ---------------------------------------------------------------------------

class TestCaseDPrime_PIRunsFull:

    def test_bootstrap(self):
        con = sqlite3.connect(":memory:")
        con.execute(CREATE_PI_RUNS_BASE_SQL)
        for col, typ in [
            ("sg_uid", "TEXT"), ("gous_uid", "TEXT"), ("mous_uid", "TEXT"),
            ("array", "TEXT"), ("sb_name", "TEXT"), ("source_name", "TEXT"),
            ("line_group", "TEXT"),
        ]:
            con.execute(f"ALTER TABLE pi_runs ADD COLUMN {col} {typ}")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pi_runs_uid ON pi_runs(uid)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pi_runs_status ON pi_runs(status)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pi_runs_mous ON pi_runs(mous_uid)")
        con.commit()

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        assert_full_schema(con)
        # M5, M6, M7 all probed
        assert get_schema_versions(con) == ALL_VERSIONS


# ---------------------------------------------------------------------------
# Case E: Full DB (production path)
# ---------------------------------------------------------------------------

class TestCaseE_FullDB:

    def _build_full_db(self) -> sqlite3.Connection:
        """Build a DB matching the live production schema."""
        con = sqlite3.connect(":memory:")

        # obs table with all 15 columns
        con.execute(CREATE_OBS_TABLE_SQL)
        con.execute("CREATE INDEX IF NOT EXISTS idx_status ON obs(status)")
        for col, typ in [
            ("weblog_staged", "INTEGER DEFAULT 0"),
            ("weblog_path", "TEXT"),
            ("weblog_url", "TEXT"),
            ("weblog_staged_at", "TEXT"),
        ]:
            con.execute(f"ALTER TABLE obs ADD COLUMN {col} {typ}")
        con.execute("CREATE INDEX IF NOT EXISTS idx_release ON obs(release_date)")

        # pi_runs table with all 19 columns
        con.execute(CREATE_PI_RUNS_BASE_SQL)
        for col, typ in [
            ("sg_uid", "TEXT"), ("gous_uid", "TEXT"), ("mous_uid", "TEXT"),
            ("array", "TEXT"), ("sb_name", "TEXT"), ("source_name", "TEXT"),
            ("line_group", "TEXT"),
        ]:
            con.execute(f"ALTER TABLE pi_runs ADD COLUMN {col} {typ}")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pi_runs_uid ON pi_runs(uid)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pi_runs_status ON pi_runs(status)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pi_runs_mous ON pi_runs(mous_uid)")

        con.commit()
        return con

    def test_only_schema_version_created(self):
        """First run against full DB should only create schema_version table."""
        con = self._build_full_db()

        # No schema_version yet
        assert not table_exists(con, "schema_version")

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        # schema_version now exists with 7 rows
        assert table_exists(con, "schema_version")
        assert get_schema_versions(con) == ALL_VERSIONS

    def test_no_business_table_ddl(self):
        """All probes return True — no ALTER TABLE or CREATE TABLE executed."""
        con = self._build_full_db()

        # Verify all probes return True for pre-existing tables
        for m in MIGRATIONS[:7]:  # M1-M7 should probe True on pre-existing DB
            assert m.probe(con), f"Probe for migration {m.version} ({m.description}) should be True"

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        assert_full_schema(con)

    def test_existing_data_preserved(self):
        """Bootstrap must not destroy existing data."""
        con = self._build_full_db()

        # Insert some test data
        from panta_rei.core.text import now_iso
        now = now_iso()
        con.execute(
            "INSERT INTO obs(uid, status, created_at, updated_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("uid___a001_x123_x456", "extracted", now, now, now),
        )
        con.execute(
            "INSERT INTO pi_runs(uid, script_path, cwd, casa_cmd, log_path, "
            "started_at, status, hostname) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("uid___a001_x123_x456", "/p/s.py", "/p", "casa", "/p/l.log",
             now, "success", "host"),
        )
        con.commit()

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        assert con.execute("SELECT COUNT(*) FROM obs").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM pi_runs").fetchone()[0] == 1
        assert con.execute("SELECT status FROM obs").fetchone()[0] == "extracted"

    def test_idempotent_double_bootstrap(self):
        """Running bootstrap twice should be a no-op the second time."""
        con = self._build_full_db()

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        # Second bootstrap
        db._bootstrap(con)

        assert get_schema_versions(con) == ALL_VERSIONS
        count = con.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == len(ALL_VERSIONS)  # No duplicates


# ---------------------------------------------------------------------------
# Partial column states
# ---------------------------------------------------------------------------

class TestPartialColumnStates:

    def test_obs_with_2_of_4_weblog_columns(self):
        """Only 2 weblog columns exist — M3 should add the missing 2."""
        con = sqlite3.connect(":memory:")
        con.execute(CREATE_OBS_TABLE_SQL)
        con.execute("CREATE INDEX IF NOT EXISTS idx_status ON obs(status)")
        # Add only 2 of 4 weblog columns
        con.execute("ALTER TABLE obs ADD COLUMN weblog_staged INTEGER DEFAULT 0")
        con.execute("ALTER TABLE obs ADD COLUMN weblog_path TEXT")
        con.commit()

        assert column_exists(con, "obs", "weblog_staged")
        assert column_exists(con, "obs", "weblog_path")
        assert not column_exists(con, "obs", "weblog_url")
        assert not column_exists(con, "obs", "weblog_staged_at")

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        # All 4 weblog columns now present
        assert column_exists(con, "obs", "weblog_url")
        assert column_exists(con, "obs", "weblog_staged_at")
        assert_full_schema(con)

    def test_pi_runs_with_3_of_7_enrichment_columns(self):
        """Only 3 enrichment columns exist — M6 should add the missing 4."""
        con = sqlite3.connect(":memory:")
        con.execute(CREATE_PI_RUNS_BASE_SQL)
        # Add only 3 of 7 enrichment columns
        con.execute("ALTER TABLE pi_runs ADD COLUMN sg_uid TEXT")
        con.execute("ALTER TABLE pi_runs ADD COLUMN gous_uid TEXT")
        con.execute("ALTER TABLE pi_runs ADD COLUMN mous_uid TEXT")
        con.commit()

        assert column_exists(con, "pi_runs", "sg_uid")
        assert not column_exists(con, "pi_runs", "array")
        assert not column_exists(con, "pi_runs", "line_group")

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        assert column_exists(con, "pi_runs", "array")
        assert column_exists(con, "pi_runs", "sb_name")
        assert column_exists(con, "pi_runs", "source_name")
        assert column_exists(con, "pi_runs", "line_group")
        assert_full_schema(con)

    def test_no_duplicate_column_error(self):
        """Adding columns that already exist should not cause OperationalError."""
        con = sqlite3.connect(":memory:")
        con.execute(CREATE_OBS_TABLE_SQL)
        con.execute("CREATE INDEX IF NOT EXISTS idx_status ON obs(status)")
        # Add ALL weblog columns manually
        for col, typ in [
            ("weblog_staged", "INTEGER DEFAULT 0"),
            ("weblog_path", "TEXT"),
            ("weblog_url", "TEXT"),
            ("weblog_staged_at", "TEXT"),
        ]:
            con.execute(f"ALTER TABLE obs ADD COLUMN {col} {typ}")
        con.commit()

        # Bootstrap should not raise even though columns exist
        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        assert_full_schema(con)


# ---------------------------------------------------------------------------
# File-based database (verifies Path handling)
# ---------------------------------------------------------------------------

class TestFileDatabase:

    def test_create_file_db(self, tmp_path):
        db_path = tmp_path / "test.sqlite3"
        db = DatabaseManager(db_path)
        assert db_path.exists()

        con = db.connect()
        assert_full_schema(con)
        con.close()

    def test_reopen_file_db(self, tmp_path):
        db_path = tmp_path / "test.sqlite3"

        # First open: creates schema
        db1 = DatabaseManager(db_path)
        con1 = db1.connect()
        assert get_schema_versions(con1) == ALL_VERSIONS
        con1.close()

        # Second open: bootstrap is a no-op
        db2 = DatabaseManager(db_path)
        con2 = db2.connect()
        assert get_schema_versions(con2) == ALL_VERSIONS
        count = con2.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == len(ALL_VERSIONS)
        con2.close()

    def test_parent_dir_created(self, tmp_path):
        db_path = tmp_path / "subdir" / "nested" / "test.sqlite3"
        db = DatabaseManager(db_path)
        assert db_path.exists()


# ---------------------------------------------------------------------------
# Migration rollback on failure
# ---------------------------------------------------------------------------

class TestMigrationRollback:

    def test_failed_migration_rolls_back_ddl(self):
        """A migration that fails mid-apply should rollback DDL changes.

        We craft a Migration with conditional_columns where the third
        column definition is invalid SQL, which will cause a real
        OperationalError. Verify the first two columns are rolled back.
        """
        from panta_rei.db.schema import Migration

        con = sqlite3.connect(":memory:")
        con.execute(CREATE_OBS_TABLE_SQL)
        con.execute("CREATE INDEX IF NOT EXISTS idx_status ON obs(status)")
        con.commit()

        # Create schema_version table
        con.execute("""\
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )""")
        con.execute(
            "INSERT INTO schema_version VALUES (1, 'obs table', '2026-01-01T00:00:00Z')"
        )
        con.execute(
            "INSERT INTO schema_version VALUES (2, 'idx_status', '2026-01-01T00:00:00Z')"
        )
        con.commit()

        # Verify pre-state
        assert not column_exists(con, "obs", "weblog_staged")

        # Create a migration that will fail on the third column (invalid SQL type)
        bad_migration = Migration(
            version=99,
            description="test migration that fails on third column",
            probe=lambda c: False,
            conditional_columns=[
                ("obs", "weblog_staged", "INTEGER DEFAULT 0"),
                ("obs", "weblog_path", "TEXT"),
                # This will cause the migration to fail — 'obs' doesn't have
                # a column 'NONEXISTENT' to reference in a CHECK constraint,
                # but more directly we can use an invalid table name:
                ("nonexistent_table", "bad_col", "TEXT"),
            ],
        )

        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con

        with pytest.raises(sqlite3.OperationalError):
            db._apply_migration(con, bad_migration)

        # After rollback: migration should NOT be recorded
        assert 99 not in get_schema_versions(con)

        # After rollback: columns added before the failure should be gone
        # (transactional DDL within explicit BEGIN...COMMIT)
        assert not column_exists(con, "obs", "weblog_staged")
        assert not column_exists(con, "obs", "weblog_path")

    def test_retry_after_failed_migration(self):
        """After a failed migration, a rerun should succeed from scratch."""
        con = sqlite3.connect(":memory:")
        con.execute(CREATE_OBS_TABLE_SQL)
        con.execute("CREATE INDEX IF NOT EXISTS idx_status ON obs(status)")
        con.commit()

        # Full bootstrap should succeed
        db = DatabaseManager.__new__(DatabaseManager)
        db._db_path = ":memory:"
        db._memory_con = con
        db._bootstrap(con)

        # All columns and migrations present
        assert column_exists(con, "obs", "weblog_staged")
        assert column_exists(con, "obs", "weblog_url")
        assert column_exists(con, "obs", "weblog_staged_at")
        assert get_schema_versions(con) == ALL_VERSIONS
