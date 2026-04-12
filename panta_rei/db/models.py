"""Query helpers and status constants for the database tables.

All database queries are raw SQL, organized as static methods on classes.
Each method takes a sqlite3.Connection as its first argument.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional


from panta_rei.core.text import now_iso
from panta_rei.core.uid import canonical_uid


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

class ObsStatus:
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    EXTRACTED = "extracted"
    ERROR = "error"


class PIRunStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Obs table queries
# ---------------------------------------------------------------------------

class ObsQueries:
    """Query helpers for the obs table."""

    @staticmethod
    def upsert_seen(
        con: sqlite3.Connection,
        uid: str,
        release_date: Optional[str] = None,
    ) -> None:
        """Insert or update an observation as seen in the archive."""
        uidc = canonical_uid(uid)
        if not uidc:
            return
        now = now_iso()
        con.execute(
            """
            INSERT INTO obs(uid, status, release_date, created_at, updated_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(uid) DO UPDATE SET
                release_date=COALESCE(excluded.release_date, release_date),
                last_seen_at=excluded.last_seen_at,
                updated_at=excluded.updated_at
            """,
            (uidc, ObsStatus.PENDING, release_date, now, now, now),
        )

    @staticmethod
    def mark_downloaded(
        con: sqlite3.Connection, uid: str, tar_path: Path
    ) -> None:
        uidc = canonical_uid(uid)
        if not uidc:
            return
        con.execute(
            "UPDATE obs SET status=?, tar_path=?, updated_at=? WHERE uid=?",
            (ObsStatus.DOWNLOADED, str(Path(tar_path).resolve()), now_iso(), uidc),
        )

    @staticmethod
    def mark_many_downloaded(
        con: sqlite3.Connection,
        uid_path_pairs: list[tuple[str, Path]],
    ) -> None:
        now = now_iso()
        updates = []
        for uid, tar_path in uid_path_pairs:
            uidc = canonical_uid(uid)
            if uidc:
                updates.append(
                    (ObsStatus.DOWNLOADED, str(Path(tar_path).resolve()), now, uidc)
                )
        if updates:
            con.executemany(
                "UPDATE obs SET status=?, tar_path=?, updated_at=? WHERE uid=?",
                updates,
            )

    @staticmethod
    def mark_extracted(
        con: sqlite3.Connection,
        uid: str,
        extracted_root: Path,
        n_extracted: int,
        n_skipped: int,
        tar_deleted: bool,
    ) -> None:
        uidc = canonical_uid(uid)
        if not uidc:
            return
        con.execute(
            """
            UPDATE obs SET status=?, extracted_root=?, n_extracted=?,
                   n_skipped=?, tar_deleted=?, updated_at=?
            WHERE uid=?
            """,
            (
                ObsStatus.EXTRACTED,
                str(Path(extracted_root).resolve()),
                n_extracted,
                n_skipped,
                int(tar_deleted),
                now_iso(),
                uidc,
            ),
        )

    @staticmethod
    def mark_error(con: sqlite3.Connection, uid: str) -> None:
        uidc = canonical_uid(uid)
        if not uidc:
            return
        con.execute(
            "UPDATE obs SET status=?, updated_at=? WHERE uid=?",
            (ObsStatus.ERROR, now_iso(), uidc),
        )

    @staticmethod
    def reset_to_pending(con: sqlite3.Connection, uid: str) -> bool:
        """Reset a UID to pending status for re-download.

        Returns True if a row was updated.
        """
        uidc = canonical_uid(uid)
        if not uidc:
            return False
        con.execute(
            """
            UPDATE obs SET
                status = ?,
                tar_path = NULL,
                tar_deleted = 0,
                extracted_root = NULL,
                n_extracted = NULL,
                n_skipped = NULL,
                updated_at = ?
            WHERE uid = ?
            """,
            (ObsStatus.PENDING, now_iso(), uidc),
        )
        return con.total_changes > 0

    @staticmethod
    def get_status(con: sqlite3.Connection, uid: str) -> Optional[str]:
        uidc = canonical_uid(uid)
        if not uidc:
            return None
        row = con.execute(
            "SELECT status FROM obs WHERE uid=?", (uidc,)
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def uids_to_download(
        con: sqlite3.Connection, uids: list[str]
    ) -> list[str]:
        """Return UIDs that are not yet extracted."""
        to_get: list[str] = []
        for uid in uids:
            st = ObsQueries.get_status(con, uid)
            if st != ObsStatus.EXTRACTED:
                to_get.append(uid)
        return to_get

    @staticmethod
    def summary(con: sqlite3.Connection) -> str:
        totals = dict(
            con.execute("SELECT status, COUNT(*) FROM obs GROUP BY status")
        )
        total = sum(totals.values()) if totals else 0
        parts = [f"total={total}"] + [
            f"{k}={v}" for k, v in sorted(totals.items())
        ]
        return " | ".join(parts)

    # --- Weblog operations ---

    @staticmethod
    def mark_weblog_staged(
        con: sqlite3.Connection,
        uid: str,
        weblog_path: Path,
        weblog_url: Optional[str] = None,
    ) -> bool:
        uidc = canonical_uid(uid)
        if not uidc:
            return False
        now = now_iso()
        con.execute(
            """
            UPDATE obs SET
                weblog_staged = 1,
                weblog_path = ?,
                weblog_url = ?,
                weblog_staged_at = ?,
                updated_at = ?
            WHERE uid = ?
            """,
            (str(weblog_path), weblog_url, now, now, uidc),
        )
        return con.total_changes > 0

    @staticmethod
    def get_unstaged_extracted(con: sqlite3.Connection) -> list[dict]:
        """Get all extracted obs that don't have weblogs staged yet."""
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT uid, status, extracted_root, tar_path
            FROM obs
            WHERE status = 'extracted'
              AND (weblog_staged IS NULL OR weblog_staged = 0)
            """
        ).fetchall()
        result = [dict(row) for row in rows]
        con.row_factory = None
        return result

    @staticmethod
    def get_all_weblog_urls(con: sqlite3.Connection) -> dict[str, str]:
        rows = con.execute(
            "SELECT uid, weblog_url FROM obs "
            "WHERE weblog_staged = 1 AND weblog_url IS NOT NULL"
        ).fetchall()
        return {row[0]: row[1] for row in rows if row[1]}

    @staticmethod
    def reset_weblog_status(con: sqlite3.Connection, uid: str) -> bool:
        uidc = canonical_uid(uid)
        if not uidc:
            return False
        now = now_iso()
        con.execute(
            """
            UPDATE obs SET
                weblog_staged = 0,
                weblog_path = NULL,
                weblog_url = NULL,
                weblog_staged_at = NULL,
                updated_at = ?
            WHERE uid = ?
            """,
            (now, uidc),
        )
        return con.total_changes > 0


# ---------------------------------------------------------------------------
# PI runs table queries
# ---------------------------------------------------------------------------

class PIRunsQueries:
    """Query helpers for the pi_runs table."""

    @staticmethod
    def insert_row(con: sqlite3.Connection, **kw) -> int:
        """Insert a pi_runs row. Returns the new row ID."""
        cols = ", ".join(kw.keys())
        qmarks = ", ".join(["?"] * len(kw))
        cur = con.execute(
            f"INSERT INTO pi_runs ({cols}) VALUES ({qmarks})",
            tuple(kw.values()),
        )
        return cur.lastrowid

    @staticmethod
    def mark_running(con: sqlite3.Connection, row_id: int) -> None:
        con.execute(
            "UPDATE pi_runs SET status=? WHERE id=?",
            (PIRunStatus.RUNNING, row_id),
        )

    @staticmethod
    def mark_done(
        con: sqlite3.Connection,
        row_id: int,
        status: str,
        retcode: int,
        finished_at: str,
        duration_sec: float,
    ) -> None:
        con.execute(
            """
            UPDATE pi_runs
               SET status=?, retcode=?, finished_at=?, duration_sec=?
             WHERE id=?
            """,
            (status, int(retcode), finished_at, float(duration_sec), row_id),
        )

    @staticmethod
    def latest_success_exists(con: sqlite3.Connection, uid: str) -> bool:
        row = con.execute(
            "SELECT 1 FROM pi_runs WHERE uid=? AND status=? "
            "ORDER BY finished_at DESC LIMIT 1",
            (uid, PIRunStatus.SUCCESS),
        ).fetchone()
        return bool(row)

    @staticmethod
    def summary(con: sqlite3.Connection) -> str:
        totals = dict(
            con.execute(
                "SELECT status, COUNT(*) FROM pi_runs GROUP BY status"
            )
        )
        total = sum(totals.values()) if totals else 0
        parts = [f"total={total}"] + [
            f"{k}={v}" for k, v in sorted(totals.items())
        ]
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Imaging parameter status constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Contsub runs table queries
# ---------------------------------------------------------------------------

class ContsubRunsQueries:
    """Query helpers for the contsub_runs table."""

    @staticmethod
    def insert_row(con: sqlite3.Connection, **kw) -> int:
        """Insert a contsub_runs row. Returns the new row ID."""
        cols = ", ".join(kw.keys())
        qmarks = ", ".join(["?"] * len(kw))
        cur = con.execute(
            f"INSERT INTO contsub_runs ({cols}) VALUES ({qmarks})",
            tuple(kw.values()),
        )
        return cur.lastrowid

    @staticmethod
    def mark_running(con: sqlite3.Connection, row_id: int) -> None:
        con.execute(
            "UPDATE contsub_runs SET status=? WHERE id=?",
            (PIRunStatus.RUNNING, row_id),
        )

    @staticmethod
    def mark_done(
        con: sqlite3.Connection,
        row_id: int,
        status: str,
        retcode: int,
        finished_at: str,
        duration_sec: float,
        targets_line_count: Optional[int] = None,
    ) -> None:
        con.execute(
            """
            UPDATE contsub_runs
               SET status=?, retcode=?, finished_at=?, duration_sec=?,
                   targets_line_count=COALESCE(?, targets_line_count)
             WHERE id=?
            """,
            (status, int(retcode), finished_at, float(duration_sec),
             targets_line_count, row_id),
        )

    @staticmethod
    def latest_success_exists(con: sqlite3.Connection, uid: str) -> bool:
        row = con.execute(
            "SELECT 1 FROM contsub_runs WHERE uid=? AND status=? "
            "ORDER BY finished_at DESC LIMIT 1",
            (uid, PIRunStatus.SUCCESS),
        ).fetchone()
        return bool(row)

    @staticmethod
    def summary(con: sqlite3.Connection) -> str:
        totals = dict(
            con.execute(
                "SELECT status, COUNT(*) FROM contsub_runs GROUP BY status"
            )
        )
        total = sum(totals.values()) if totals else 0
        parts = [f"total={total}"] + [
            f"{k}={v}" for k, v in sorted(totals.items())
        ]
        return " | ".join(parts)


class ImagingParamsStatus:
    RECOVERED = "recovered"
    FAILED = "failed"
    MANUAL = "manual"


class ImagingRunStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Imaging params table queries
# ---------------------------------------------------------------------------

class ImagingParamsQueries:
    """Query helpers for the imaging_params table."""

    @staticmethod
    def upsert(
        con: sqlite3.Connection,
        *,
        gous_uid: str,
        source_name: str,
        line_group: Optional[str],
        spw_id: str,
        mous_uids_tm: list[str],
        params_json_path: Optional[str] = None,
        params_source: str = "weblog",
        weblog_path: Optional[str] = None,
        status: str = ImagingParamsStatus.RECOVERED,
        error_message: Optional[str] = None,
        imsize: Optional[str] = None,
        cell: Optional[str] = None,
        nchan: Optional[int] = None,
        phasecenter: Optional[str] = None,
        robust: Optional[float] = None,
    ) -> int:
        """Insert or update an imaging_params row. Returns the row ID."""
        now = now_iso()
        mous_json = json.dumps(mous_uids_tm)
        cur = con.execute(
            """
            INSERT INTO imaging_params (
                gous_uid, source_name, line_group, spw_id,
                mous_uids_tm, params_json_path, params_source,
                weblog_path, status, recovered_at, error_message,
                imsize, cell, nchan, phasecenter, robust
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(gous_uid, source_name, line_group, spw_id) DO UPDATE SET
                mous_uids_tm = excluded.mous_uids_tm,
                params_json_path = excluded.params_json_path,
                params_source = excluded.params_source,
                weblog_path = excluded.weblog_path,
                status = excluded.status,
                recovered_at = excluded.recovered_at,
                error_message = excluded.error_message,
                imsize = excluded.imsize,
                cell = excluded.cell,
                nchan = excluded.nchan,
                phasecenter = excluded.phasecenter,
                robust = excluded.robust
            """,
            (
                gous_uid, source_name, line_group, spw_id,
                mous_json, params_json_path, params_source,
                weblog_path, status, now, error_message,
                imsize, cell, nchan, phasecenter, robust,
            ),
        )
        return cur.lastrowid

    @staticmethod
    def get_by_key(
        con: sqlite3.Connection,
        gous_uid: str,
        source_name: str,
        line_group: Optional[str],
        spw_id: str,
    ) -> Optional[dict]:
        """Look up a single imaging_params row by its unique key."""
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT * FROM imaging_params
            WHERE gous_uid = ? AND source_name = ?
              AND line_group IS ? AND spw_id = ?
            """,
            (gous_uid, source_name, line_group, spw_id),
        ).fetchone()
        result = dict(row) if row else None
        con.row_factory = None
        return result

    @staticmethod
    def get_by_id(con: sqlite3.Connection, params_id: int) -> Optional[dict]:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM imaging_params WHERE id = ?", (params_id,)
        ).fetchone()
        result = dict(row) if row else None
        con.row_factory = None
        return result

    @staticmethod
    def get_all_recovered(con: sqlite3.Connection) -> list[dict]:
        """Return all rows with status='recovered'."""
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM imaging_params WHERE status = ?",
            (ImagingParamsStatus.RECOVERED,),
        ).fetchall()
        result = [dict(r) for r in rows]
        con.row_factory = None
        return result

    @staticmethod
    def summary(con: sqlite3.Connection) -> str:
        totals = dict(
            con.execute(
                "SELECT status, COUNT(*) FROM imaging_params GROUP BY status"
            )
        )
        total = sum(totals.values()) if totals else 0
        parts = [f"total={total}"] + [
            f"{k}={v}" for k, v in sorted(totals.items())
        ]
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Imaging runs table queries
# ---------------------------------------------------------------------------

class ImagingRunsQueries:
    """Query helpers for the imaging_runs table."""

    @staticmethod
    def insert_row(con: sqlite3.Connection, **kw) -> int:
        """Insert an imaging_runs row. Returns the new row ID."""
        cols = ", ".join(kw.keys())
        qmarks = ", ".join(["?"] * len(kw))
        cur = con.execute(
            f"INSERT INTO imaging_runs ({cols}) VALUES ({qmarks})",
            tuple(kw.values()),
        )
        return cur.lastrowid

    @staticmethod
    def mark_running(con: sqlite3.Connection, row_id: int) -> None:
        con.execute(
            "UPDATE imaging_runs SET status=? WHERE id=?",
            (ImagingRunStatus.RUNNING, row_id),
        )

    @staticmethod
    def update_resolved(
        con: sqlite3.Connection,
        row_id: int,
        spw_selection: Optional[str] = None,
        field_selection: Optional[str] = None,
    ) -> None:
        """Store resolved SPW/field selections after trusted preflight."""
        con.execute(
            """
            UPDATE imaging_runs
               SET spw_selection = COALESCE(?, spw_selection),
                   field_selection = COALESCE(?, field_selection)
             WHERE id = ?
            """,
            (spw_selection, field_selection, row_id),
        )

    @staticmethod
    def mark_done(
        con: sqlite3.Connection,
        row_id: int,
        status: str,
        retcode: int,
        finished_at: str,
        duration_sec: float,
        output_fits: Optional[str] = None,
    ) -> None:
        con.execute(
            """
            UPDATE imaging_runs
               SET status=?, retcode=?, finished_at=?, duration_sec=?,
                   output_fits=COALESCE(?, output_fits)
             WHERE id=?
            """,
            (status, int(retcode), finished_at, float(duration_sec),
             output_fits, row_id),
        )

    @staticmethod
    def success_exists(
        con: sqlite3.Connection,
        params_id: int,
        method: str = "tclean_feather",
        sdgain: Optional[float] = None,
        deconvolver: str = "multiscale",
        scales: str = "[0,5,10,15,20]",
    ) -> bool:
        """Check if a successful run exists with matching identity.

        Identity is method-specific:
        - ``tclean_feather``: params_id + deconvolver + scales + method
          (sdgain is irrelevant)
        - ``sdintimaging``: params_id + sdgain + deconvolver + scales + method
        """
        if method == "tclean_feather":
            row = con.execute(
                """
                SELECT 1 FROM imaging_runs
                WHERE params_id = ? AND status = ?
                  AND COALESCE(method, 'sdintimaging') = ?
                  AND deconvolver = ? AND scales = ?
                ORDER BY finished_at DESC LIMIT 1
                """,
                (params_id, ImagingRunStatus.SUCCESS, method,
                 deconvolver, scales),
            ).fetchone()
        else:
            # sdintimaging: identity includes sdgain
            if sdgain is None:
                sdgain = 1.0
            row = con.execute(
                """
                SELECT 1 FROM imaging_runs
                WHERE params_id = ? AND status = ?
                  AND COALESCE(method, 'sdintimaging') = ?
                  AND sdgain = ? AND deconvolver = ? AND scales = ?
                ORDER BY finished_at DESC LIMIT 1
                """,
                (params_id, ImagingRunStatus.SUCCESS, method,
                 sdgain, deconvolver, scales),
            ).fetchone()
        return bool(row)

    @staticmethod
    def summary(con: sqlite3.Connection) -> str:
        totals = dict(
            con.execute(
                "SELECT status, COUNT(*) FROM imaging_runs GROUP BY status"
            )
        )
        total = sum(totals.values()) if totals else 0
        parts = [f"total={total}"] + [
            f"{k}={v}" for k, v in sorted(totals.items())
        ]
        return " | ".join(parts)
