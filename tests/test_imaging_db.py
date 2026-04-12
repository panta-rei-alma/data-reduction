"""Tests for imaging DB tables: migrations, query helpers, idempotence."""

from __future__ import annotations

import json

import pytest

from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import (
    ImagingParamsQueries,
    ImagingParamsStatus,
    ImagingRunsQueries,
    ImagingRunStatus,
)
from panta_rei.db.schema import table_exists, index_exists
from panta_rei.core.text import now_iso


# ---------------------------------------------------------------------------
# Migration bootstrap
# ---------------------------------------------------------------------------

class TestImagingMigrations:
    def test_fresh_db_has_imaging_tables(self):
        db = DatabaseManager(":memory:")
        con = db.connect()
        assert table_exists(con, "imaging_params")
        assert table_exists(con, "imaging_runs")

    def test_indexes_created(self):
        db = DatabaseManager(":memory:")
        con = db.connect()
        assert index_exists(con, "idx_ip_key")
        assert index_exists(con, "idx_ip_status")
        assert index_exists(con, "idx_ir_params")
        assert index_exists(con, "idx_ir_status")

    def test_re_bootstrap_is_idempotent(self):
        """Running DatabaseManager twice on same DB should not fail."""
        db = DatabaseManager(":memory:")
        con = db.connect()
        # Simulate a second bootstrap (as if restarting the app)
        db2 = DatabaseManager(":memory:")
        con2 = db2.connect()
        assert table_exists(con2, "imaging_params")


# ---------------------------------------------------------------------------
# ImagingParamsQueries
# ---------------------------------------------------------------------------

class TestImagingParamsQueries:
    @pytest.fixture
    def con(self):
        db = DatabaseManager(":memory:")
        return db.connect()

    def test_upsert_and_get(self, con):
        row_id = ImagingParamsQueries.upsert(
            con,
            gous_uid="g1",
            source_name="SRC1",
            line_group="N2H+",
            spw_id="23",
            mous_uids_tm=["m1", "m2"],
            params_json_path="/tmp/params.json",
            params_source="weblog",
            status=ImagingParamsStatus.RECOVERED,
            imsize="[480, 450]",
            cell='["0.42arcsec"]',
            nchan=956,
            phasecenter="J2000 07:27:07.5 -18:55:30",
            robust=0.5,
        )
        con.commit()
        assert row_id is not None

        row = ImagingParamsQueries.get_by_key(con, "g1", "SRC1", "N2H+", "23")
        assert row is not None
        assert row["gous_uid"] == "g1"
        assert json.loads(row["mous_uids_tm"]) == ["m1", "m2"]
        assert row["nchan"] == 956
        assert row["robust"] == 0.5

    def test_upsert_updates_on_conflict(self, con):
        ImagingParamsQueries.upsert(
            con,
            gous_uid="g1", source_name="SRC1", line_group="N2H+",
            spw_id="23", mous_uids_tm=["m1"],
            status=ImagingParamsStatus.RECOVERED,
            nchan=100,
        )
        con.commit()

        # Upsert again with different nchan
        ImagingParamsQueries.upsert(
            con,
            gous_uid="g1", source_name="SRC1", line_group="N2H+",
            spw_id="23", mous_uids_tm=["m1"],
            status=ImagingParamsStatus.RECOVERED,
            nchan=200,
        )
        con.commit()

        row = ImagingParamsQueries.get_by_key(con, "g1", "SRC1", "N2H+", "23")
        assert row["nchan"] == 200

    def test_get_by_id(self, con):
        row_id = ImagingParamsQueries.upsert(
            con,
            gous_uid="g1", source_name="SRC1", line_group="N2H+",
            spw_id="23", mous_uids_tm=["m1"],
            status=ImagingParamsStatus.RECOVERED,
        )
        con.commit()

        row = ImagingParamsQueries.get_by_id(con, row_id)
        assert row is not None
        assert row["source_name"] == "SRC1"

    def test_get_all_recovered(self, con):
        ImagingParamsQueries.upsert(
            con,
            gous_uid="g1", source_name="SRC1", line_group="N2H+",
            spw_id="23", mous_uids_tm=["m1"],
            status=ImagingParamsStatus.RECOVERED,
        )
        ImagingParamsQueries.upsert(
            con,
            gous_uid="g1", source_name="SRC1", line_group="N2H+",
            spw_id="25", mous_uids_tm=["m1"],
            status=ImagingParamsStatus.FAILED,
            error_message="divergence",
        )
        con.commit()

        recovered = ImagingParamsQueries.get_all_recovered(con)
        assert len(recovered) == 1
        assert recovered[0]["spw_id"] == "23"

    def test_summary(self, con):
        ImagingParamsQueries.upsert(
            con,
            gous_uid="g1", source_name="SRC1", line_group="N2H+",
            spw_id="23", mous_uids_tm=["m1"],
            status=ImagingParamsStatus.RECOVERED,
        )
        con.commit()
        s = ImagingParamsQueries.summary(con)
        assert "total=1" in s
        assert "recovered=1" in s


# ---------------------------------------------------------------------------
# ImagingRunsQueries
# ---------------------------------------------------------------------------

class TestImagingRunsQueries:
    @pytest.fixture
    def con(self):
        db = DatabaseManager(":memory:")
        return db.connect()

    def test_lifecycle(self, con):
        """QUEUED -> RUNNING -> SUCCESS lifecycle."""
        now = now_iso()
        row_id = ImagingRunsQueries.insert_row(
            con,
            params_id=1,
            gous_uid="g1",
            source_name="SRC1",
            line_group="N2H+",
            spw_id="23",
            started_at=now,
            status=ImagingRunStatus.QUEUED,
        )
        con.commit()
        assert row_id is not None

        ImagingRunsQueries.mark_running(con, row_id)
        con.commit()

        ImagingRunsQueries.mark_done(
            con, row_id,
            status=ImagingRunStatus.SUCCESS,
            retcode=0,
            finished_at=now_iso(),
            duration_sec=120.5,
            output_fits="/out/cube.fits",
        )
        con.commit()

        # Verify
        row = con.execute(
            "SELECT status, retcode, output_fits, duration_sec FROM imaging_runs WHERE id=?",
            (row_id,),
        ).fetchone()
        assert row[0] == "success"
        assert row[1] == 0
        assert row[2] == "/out/cube.fits"
        assert row[3] == 120.5

    def test_success_exists(self, con):
        now = now_iso()
        row_id = ImagingRunsQueries.insert_row(
            con,
            params_id=1, gous_uid="g1", source_name="SRC1",
            line_group="N2H+", spw_id="23",
            sdgain=1.0, deconvolver="multiscale",
            scales="[0,5,10,15,20]",
            started_at=now, status=ImagingRunStatus.QUEUED,
        )
        ImagingRunsQueries.mark_done(
            con, row_id,
            status=ImagingRunStatus.SUCCESS,
            retcode=0, finished_at=now_iso(), duration_sec=10.0,
        )
        con.commit()

        assert ImagingRunsQueries.success_exists(
            con, 1, method="sdintimaging", sdgain=1.0,
        )

    def test_success_exists_different_overrides(self, con):
        """Different sdgain = different run identity for sdintimaging."""
        now = now_iso()
        row_id = ImagingRunsQueries.insert_row(
            con,
            params_id=1, gous_uid="g1", source_name="SRC1",
            line_group="N2H+", spw_id="23",
            sdgain=1.0, deconvolver="multiscale",
            scales="[0,5,10,15,20]",
            started_at=now, status=ImagingRunStatus.QUEUED,
        )
        ImagingRunsQueries.mark_done(
            con, row_id,
            status=ImagingRunStatus.SUCCESS,
            retcode=0, finished_at=now_iso(), duration_sec=10.0,
        )
        con.commit()

        # Same overrides: exists
        assert ImagingRunsQueries.success_exists(
            con, 1, method="sdintimaging", sdgain=1.0,
        )
        # Different sdgain: does not exist
        assert not ImagingRunsQueries.success_exists(
            con, 1, method="sdintimaging", sdgain=0.5,
        )
        # tclean_feather: does not exist (different method)
        assert not ImagingRunsQueries.success_exists(
            con, 1, method="tclean_feather",
        )

    def test_summary(self, con):
        now = now_iso()
        ImagingRunsQueries.insert_row(
            con,
            params_id=1, gous_uid="g1", source_name="SRC1",
            spw_id="23", started_at=now, status=ImagingRunStatus.QUEUED,
        )
        con.commit()
        s = ImagingRunsQueries.summary(con)
        assert "total=1" in s
