"""Tests for panta_rei.imaging.unit_selection."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import (
    DispatchesQueries,
    DispatchState,
    ImagingParamsQueries,
    ImagingParamsStatus,
    ImagingRunsQueries,
    ImagingRunStatus,
)
from panta_rei.imaging.matching import ImagingUnit
from panta_rei.imaging.unit_selection import (
    SelectionFilters,
    select_units_to_image,
)


def _ready_unit(gous="G", source="SRC1", spw="23", params_id=1):
    return ImagingUnit(
        gous_uid=gous, source_name=source, line_group="N2H+",
        spw_id=spw, params_id=params_id, ready=True,
    )


def _seed_recovered_param(db, **kwargs):
    defaults = {
        "gous_uid": "X3833_X64b9",
        "source_name": "SRC1",
        "line_group": "N2H+",
        "spw_id": "23",
        "mous_uids_tm": ["X3833_X64bc"],
        "params_json_path": "/some/path.json",
        "params_source": "weblog",
        "weblog_path": "/p/weblog",
        "status": ImagingParamsStatus.RECOVERED,
        "imsize": "[100,100]",
        "cell": "[0.5arcsec]",
        "nchan": 100,
        "phasecenter": "",
        "robust": 0.5,
    }
    defaults.update(kwargs)
    with db.connect() as con:
        return ImagingParamsQueries.upsert(con, **defaults)


def test_selection_returns_empty_when_no_recovered(tmp_path):
    db = DatabaseManager(":memory:")
    csv = tmp_path / "targets.csv"
    csv.write_text("source_name,array,sb_name,sgous_id,gous_id,mous_ids,Line group\n")
    res = select_units_to_image(
        db, csv, tmp_path, SelectionFilters(scales=[0, 5, 10, 15, 20]),
    )
    assert res.ready == []
    assert res.skipped_already_done == 0


def test_exclude_active_drops_in_flight_under_running_dispatch(tmp_path):
    """Selection must NOT enqueue a new row for a unit whose previous run
    is still active under a non-terminal dispatch."""
    db = DatabaseManager(":memory:")
    pid = _seed_recovered_param(db)

    # Seed a running dispatch + running run for this params_id
    with db.connect() as con:
        DispatchesQueries.insert(
            con, dispatch_id="d_old",
            coordinator_host="h", coordinator_pid=42,
            machines_json="{}", cli_args="",
        )
        ImagingRunsQueries.insert_row(
            con,
            params_id=pid, gous_uid="X3833_X64b9",
            source_name="SRC1", line_group="N2H+", spw_id="23",
            started_at="2026-01-01T00:00:00",
            status=ImagingRunStatus.RUNNING,
            dispatch_id="d_old",
            method="tclean_feather",
            deconvolver="multiscale",
            scales=json.dumps([0, 5, 10, 15, 20]),
        )
        con.commit()

    csv = tmp_path / "targets.csv"
    csv.write_text(
        "source_name,array,sb_name,sgous_id,gous_id,mous_ids,Line group\n"
        "SRC1,TM,sb,X3833_X64b8,X3833_X64b9,X3833_X64bc,N2H+\n"
    )

    with mock.patch(
        "panta_rei.imaging.unit_selection.build_imaging_units_advisory",
        return_value=[_ready_unit(params_id=pid)],
    ):
        res = select_units_to_image(
            db, csv, tmp_path,
            SelectionFilters(scales=[0, 5, 10, 15, 20], exclude_active=True),
        )
    assert res.skipped_active == 1
    assert res.ready == []


def test_exclude_active_allows_when_prior_dispatch_terminal(tmp_path):
    """Once the prior dispatch is marked DONE, the selection can enqueue
    a fresh row even if a stale row from that dispatch lingers."""
    db = DatabaseManager(":memory:")
    pid = _seed_recovered_param(db)
    with db.connect() as con:
        DispatchesQueries.insert(
            con, dispatch_id="d_old",
            coordinator_host="h", coordinator_pid=42,
            machines_json="{}", cli_args="",
        )
        ImagingRunsQueries.insert_row(
            con,
            params_id=pid, gous_uid="X3833_X64b9",
            source_name="SRC1", line_group="N2H+", spw_id="23",
            started_at="2026-01-01T00:00:00",
            status=ImagingRunStatus.FAILED,   # terminal
            dispatch_id="d_old",
            method="tclean_feather",
            deconvolver="multiscale",
            scales=json.dumps([0, 5, 10, 15, 20]),
        )
        DispatchesQueries.mark_terminal(con, "d_old", DispatchState.DONE)
        con.commit()

    csv = tmp_path / "targets.csv"
    csv.write_text(
        "source_name,array,sb_name,sgous_id,gous_id,mous_ids,Line group\n"
        "SRC1,TM,sb,X3833_X64b8,X3833_X64b9,X3833_X64bc,N2H+\n"
    )
    with mock.patch(
        "panta_rei.imaging.unit_selection.build_imaging_units_advisory",
        return_value=[_ready_unit(params_id=pid)],
    ):
        res = select_units_to_image(
            db, csv, tmp_path,
            SelectionFilters(scales=[0, 5, 10, 15, 20], exclude_active=True),
        )
    # Terminal status row → not active.  Unit should pass through.
    assert res.skipped_active == 0
    assert len(res.ready) == 1
