"""Database connection management and migration runner.

DatabaseManager is the single entry point for obtaining a database connection.
On initialization, it creates the schema_version table, probes existing schema
state, and applies any missing migrations.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from panta_rei.core.text import now_iso
from panta_rei.db.schema import (
    CREATE_SCHEMA_VERSION_SQL,
    MIGRATIONS,
    Migration,
    column_exists,
    table_exists,
)

log = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite connections and schema migrations.

    Parameters
    ----------
    db_path : Path or str or \":memory:\"
        Path to the SQLite database file. Use \":memory:\" for in-memory DBs.
    """

    def __init__(self, db_path: Path | str) -> None:
        if str(db_path) == ":memory:":
            self._db_path = ":memory:"
            # Keep a persistent connection for in-memory DBs
            self._memory_con = sqlite3.connect(":memory:")
            self._bootstrap(self._memory_con)
        else:
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory_con = None
            with self.connect() as con:
                self._bootstrap(con)

    @property
    def db_path(self) -> Path | str:
        return self._db_path

    def connect(self) -> sqlite3.Connection:
        """Return a new connection to the database.

        For in-memory databases, returns the single persistent connection.
        For file databases, returns a new connection each call.
        """
        if self._memory_con is not None:
            return self._memory_con
        return sqlite3.connect(self._db_path)

    def _bootstrap(self, con: sqlite3.Connection) -> None:
        """Run migration bootstrap: create schema_version, probe, apply."""
        # Step 1: Create schema_version table
        con.execute(CREATE_SCHEMA_VERSION_SQL)
        con.commit()

        # Step 2: Determine which migrations are already recorded
        recorded = set()
        if table_exists(con, "schema_version"):
            rows = con.execute("SELECT version FROM schema_version").fetchall()
            recorded = {row[0] for row in rows}

        # Step 3: Process each migration
        for migration in MIGRATIONS:
            if migration.version in recorded:
                continue

            probe_result = migration.probe(con)

            if probe_result:
                # Migration was already applied (schema matches) but not recorded
                log.debug(
                    "Migration %d (%s) already applied — recording",
                    migration.version,
                    migration.description,
                )
                con.execute(
                    "INSERT INTO schema_version (version, description, applied_at) "
                    "VALUES (?, ?, ?)",
                    (migration.version, migration.description, now_iso()),
                )
                con.commit()
            else:
                # Migration needs to be executed
                log.info(
                    "Applying migration %d: %s",
                    migration.version,
                    migration.description,
                )
                self._apply_migration(con, migration)

    def _apply_migration(
        self, con: sqlite3.Connection, migration: Migration
    ) -> None:
        """Apply a single migration within an explicit transaction.

        The migration's DDL and its schema_version record are committed
        atomically. If any statement fails, the entire migration is rolled
        back and NOT recorded — so a rerun will safely retry.

        SQLite supports transactional DDL (CREATE TABLE, ALTER TABLE) within
        explicit BEGIN...COMMIT blocks. We must issue BEGIN ourselves because
        the default autocommit mode would implicitly commit DDL, making
        rollback() ineffective for partial failures.
        """
        # Save and temporarily set isolation_level=None (autocommit) so we
        # have full control over the transaction.  Python's sqlite3 module
        # in its default mode issues implicit BEGIN which can conflict with
        # our explicit transaction management.
        saved_isolation = con.isolation_level
        con.isolation_level = None
        try:
            # Start an explicit transaction so DDL is rollback-safe
            con.execute("BEGIN")

            # Execute DDL statements
            if migration.sql:
                for stmt in migration.sql:
                    con.execute(stmt)

            # Execute conditional column additions
            if migration.conditional_columns:
                for table, col_name, col_type in migration.conditional_columns:
                    if not column_exists(con, table, col_name):
                        con.execute(
                            f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                        )

            # Record the migration
            con.execute(
                "INSERT INTO schema_version (version, description, applied_at) "
                "VALUES (?, ?, ?)",
                (migration.version, migration.description, now_iso()),
            )
            con.execute("COMMIT")

        except Exception:
            con.execute("ROLLBACK")
            log.error(
                "Migration %d (%s) failed — rolled back",
                migration.version,
                migration.description,
                exc_info=True,
            )
            raise
        finally:
            con.isolation_level = saved_isolation
