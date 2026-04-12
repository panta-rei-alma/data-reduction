"""Database schema definitions and versioned migrations.

All DDL lives here. Never ALTER TABLE outside this file.

The migration system uses probe functions to detect what already exists,
so it safely handles databases in any state from empty to fully populated.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Callable

from panta_rei.core.text import now_iso


# ---------------------------------------------------------------------------
# Schema introspection helpers
# ---------------------------------------------------------------------------

def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def column_exists(con: sqlite3.Connection, table: str, column: str) -> bool:
    if not table_exists(con, table):
        return False
    cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    return column in cols


def index_exists(con: sqlite3.Connection, index: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (index,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

CREATE_OBS_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS obs (
    uid TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    release_date TEXT,
    tar_path TEXT,
    tar_deleted INTEGER DEFAULT 0,
    extracted_root TEXT,
    n_extracted INTEGER,
    n_skipped INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
)"""

CREATE_PI_RUNS_BASE_SQL = """\
CREATE TABLE IF NOT EXISTS pi_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL,
    script_path TEXT NOT NULL,
    cwd TEXT NOT NULL,
    casa_cmd TEXT NOT NULL,
    log_path TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    retcode INTEGER,
    status TEXT,
    hostname TEXT,
    duration_sec REAL
)"""

CREATE_SCHEMA_VERSION_SQL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TEXT NOT NULL
)"""

CREATE_IMAGING_PARAMS_SQL = """\
CREATE TABLE IF NOT EXISTS imaging_params (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    gous_uid        TEXT NOT NULL,
    source_name     TEXT NOT NULL,
    line_group      TEXT,
    spw_id          TEXT NOT NULL,
    mous_uids_tm    TEXT NOT NULL,
    params_json_path TEXT,
    params_source   TEXT NOT NULL,
    weblog_path     TEXT,
    status          TEXT NOT NULL,
    recovered_at    TEXT NOT NULL,
    error_message   TEXT,
    imsize          TEXT,
    cell            TEXT,
    nchan           INTEGER,
    phasecenter     TEXT,
    robust          REAL,
    UNIQUE(gous_uid, source_name, line_group, spw_id)
)"""

CREATE_CONTSUB_RUNS_SQL = """\
CREATE TABLE IF NOT EXISTS contsub_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL,
    sg_uid TEXT,
    gous_uid TEXT,
    mous_uid TEXT,
    array TEXT,
    sb_name TEXT,
    source_name TEXT,
    line_group TEXT,
    member_dir TEXT NOT NULL,
    working_dir TEXT NOT NULL,
    casa_cmd TEXT NOT NULL,
    log_path TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    retcode INTEGER,
    status TEXT NOT NULL,
    hostname TEXT,
    duration_sec REAL,
    eb_count INTEGER,
    targets_line_count INTEGER
)"""

CREATE_IMAGING_RUNS_SQL = """\
CREATE TABLE IF NOT EXISTS imaging_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    params_id       INTEGER NOT NULL,
    gous_uid        TEXT NOT NULL,
    source_name     TEXT NOT NULL,
    line_group      TEXT,
    spw_id          TEXT NOT NULL,
    vis_tm          TEXT,
    vis_sm          TEXT,
    spw_selection   TEXT,
    field_selection TEXT,
    sdimage         TEXT,
    mous_uids_tm    TEXT,
    mous_uids_sm    TEXT,
    mous_uids_tp    TEXT,
    output_dir      TEXT,
    imagename       TEXT,
    output_fits     TEXT,
    job_json_path   TEXT,
    sdgain          REAL DEFAULT 1.0,
    deconvolver     TEXT DEFAULT 'multiscale',
    scales          TEXT DEFAULT '[0,5,10,15,20]',
    casa_version    TEXT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    retcode         INTEGER,
    status          TEXT NOT NULL,
    hostname        TEXT,
    duration_sec    REAL
)"""


# ---------------------------------------------------------------------------
# Migration dataclass
# ---------------------------------------------------------------------------

@dataclass
class Migration:
    """A single schema migration step.

    Either ``sql`` or ``conditional_columns`` (or both) must be provided.

    - ``sql``: list of SQL statements executed unconditionally.
    - ``conditional_columns``: list of (table, column, type) tuples;
      each column is added only if it does not already exist.
    - ``probe``: callable that returns True if this migration has already
      been applied (checked against the live schema, not schema_version).
    """
    version: int
    description: str
    probe: Callable[[sqlite3.Connection], bool]
    sql: list[str] = field(default_factory=list)
    conditional_columns: list[tuple[str, str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Migration definitions
# ---------------------------------------------------------------------------

MIGRATIONS: list[Migration] = [
    # --- obs table ---
    Migration(
        version=1,
        description="create obs table with base columns",
        probe=lambda con: (
            table_exists(con, "obs")
            and all(
                column_exists(con, "obs", c)
                for c in [
                    "uid", "status", "release_date", "tar_path", "tar_deleted",
                    "extracted_root", "n_extracted", "n_skipped",
                    "created_at", "updated_at", "last_seen_at",
                ]
            )
        ),
        sql=[CREATE_OBS_TABLE_SQL],
    ),
    Migration(
        version=2,
        description="add idx_status index on obs",
        probe=lambda con: index_exists(con, "idx_status"),
        sql=["CREATE INDEX IF NOT EXISTS idx_status ON obs(status)"],
    ),
    Migration(
        version=3,
        description="add weblog columns to obs",
        probe=lambda con: all(
            column_exists(con, "obs", c)
            for c in ["weblog_staged", "weblog_path", "weblog_url", "weblog_staged_at"]
        ),
        conditional_columns=[
            ("obs", "weblog_staged", "INTEGER DEFAULT 0"),
            ("obs", "weblog_path", "TEXT"),
            ("obs", "weblog_url", "TEXT"),
            ("obs", "weblog_staged_at", "TEXT"),
        ],
    ),
    Migration(
        version=4,
        description="add idx_release index on obs",
        probe=lambda con: index_exists(con, "idx_release"),
        sql=["CREATE INDEX IF NOT EXISTS idx_release ON obs(release_date)"],
    ),

    # --- pi_runs table ---
    Migration(
        version=5,
        description="create pi_runs table with base columns",
        probe=lambda con: (
            table_exists(con, "pi_runs")
            and all(
                column_exists(con, "pi_runs", c)
                for c in [
                    "id", "uid", "script_path", "cwd", "casa_cmd", "log_path",
                    "started_at", "finished_at", "retcode", "status",
                    "hostname", "duration_sec",
                ]
            )
        ),
        sql=[CREATE_PI_RUNS_BASE_SQL],
    ),
    Migration(
        version=6,
        description="add enrichment columns to pi_runs",
        probe=lambda con: all(
            column_exists(con, "pi_runs", c)
            for c in [
                "sg_uid", "gous_uid", "mous_uid", "array",
                "sb_name", "source_name", "line_group",
            ]
        ),
        conditional_columns=[
            ("pi_runs", "sg_uid", "TEXT"),
            ("pi_runs", "gous_uid", "TEXT"),
            ("pi_runs", "mous_uid", "TEXT"),
            ("pi_runs", "array", "TEXT"),
            ("pi_runs", "sb_name", "TEXT"),
            ("pi_runs", "source_name", "TEXT"),
            ("pi_runs", "line_group", "TEXT"),
        ],
    ),
    Migration(
        version=7,
        description="add indexes on pi_runs",
        probe=lambda con: all(
            index_exists(con, idx)
            for idx in ["idx_pi_runs_uid", "idx_pi_runs_status", "idx_pi_runs_mous"]
        ),
        sql=[
            "CREATE INDEX IF NOT EXISTS idx_pi_runs_uid ON pi_runs(uid)",
            "CREATE INDEX IF NOT EXISTS idx_pi_runs_status ON pi_runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_pi_runs_mous ON pi_runs(mous_uid)",
        ],
    ),

    # --- imaging tables ---
    Migration(
        version=8,
        description="create imaging_params table",
        probe=lambda con: (
            table_exists(con, "imaging_params")
            and all(
                column_exists(con, "imaging_params", c)
                for c in [
                    "id", "gous_uid", "source_name", "line_group", "spw_id",
                    "mous_uids_tm", "params_json_path", "params_source",
                    "weblog_path", "status", "recovered_at", "error_message",
                    "imsize", "cell", "nchan", "phasecenter", "robust",
                ]
            )
        ),
        sql=[CREATE_IMAGING_PARAMS_SQL],
    ),
    Migration(
        version=9,
        description="create imaging_runs table",
        probe=lambda con: (
            table_exists(con, "imaging_runs")
            and all(
                column_exists(con, "imaging_runs", c)
                for c in [
                    "id", "params_id", "gous_uid", "source_name", "line_group",
                    "spw_id", "vis_tm", "vis_sm", "spw_selection",
                    "field_selection", "sdimage", "mous_uids_tm", "mous_uids_sm",
                    "mous_uids_tp", "output_dir", "imagename", "output_fits",
                    "job_json_path", "sdgain", "deconvolver", "scales",
                    "casa_version", "started_at", "finished_at", "retcode",
                    "status", "hostname", "duration_sec",
                ]
            )
        ),
        sql=[CREATE_IMAGING_RUNS_SQL],
    ),
    Migration(
        version=10,
        description="add indexes on imaging tables",
        probe=lambda con: all(
            index_exists(con, idx)
            for idx in [
                "idx_ip_key", "idx_ip_status",
                "idx_ir_params", "idx_ir_status",
            ]
        ),
        sql=[
            "CREATE INDEX IF NOT EXISTS idx_ip_key ON imaging_params(gous_uid, source_name, line_group, spw_id)",
            "CREATE INDEX IF NOT EXISTS idx_ip_status ON imaging_params(status)",
            "CREATE INDEX IF NOT EXISTS idx_ir_params ON imaging_runs(params_id)",
            "CREATE INDEX IF NOT EXISTS idx_ir_status ON imaging_runs(status)",
        ],
    ),

    # --- contsub_runs table ---
    Migration(
        version=11,
        description="create contsub_runs table",
        probe=lambda con: (
            table_exists(con, "contsub_runs")
            and all(
                column_exists(con, "contsub_runs", c)
                for c in [
                    "id", "uid", "sg_uid", "gous_uid", "mous_uid",
                    "array", "sb_name", "source_name", "line_group",
                    "member_dir", "working_dir", "casa_cmd", "log_path",
                    "started_at", "finished_at", "retcode", "status",
                    "hostname", "duration_sec", "eb_count",
                    "targets_line_count",
                ]
            )
        ),
        sql=[CREATE_CONTSUB_RUNS_SQL],
    ),
    Migration(
        version=12,
        description="add indexes on contsub_runs",
        probe=lambda con: all(
            index_exists(con, idx)
            for idx in [
                "idx_contsub_uid", "idx_contsub_status",
                "idx_contsub_mous",
            ]
        ),
        sql=[
            "CREATE INDEX IF NOT EXISTS idx_contsub_uid ON contsub_runs(uid)",
            "CREATE INDEX IF NOT EXISTS idx_contsub_status ON contsub_runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_contsub_mous ON contsub_runs(mous_uid)",
        ],
    ),
    Migration(
        version=13,
        description="add method and parallel columns to imaging_runs",
        probe=lambda con: (
            column_exists(con, "imaging_runs", "method")
            and column_exists(con, "imaging_runs", "parallel")
        ),
        conditional_columns=[
            ("imaging_runs", "method", "TEXT DEFAULT 'sdintimaging'"),
            ("imaging_runs", "parallel", "INTEGER DEFAULT 0"),
        ],
    ),
]
