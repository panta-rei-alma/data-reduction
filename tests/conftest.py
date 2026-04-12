"""Shared test fixtures for Panta Rei tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import ObsQueries, ObsStatus, PIRunsQueries, PIRunStatus
from panta_rei.core.text import now_iso


@pytest.fixture
def db():
    """In-memory DatabaseManager with full schema."""
    return DatabaseManager(":memory:")


@pytest.fixture
def con(db):
    """Connection from in-memory database."""
    return db.connect()


@pytest.fixture
def populated_db(db):
    """In-memory DB with representative obs and pi_runs rows."""
    con = db.connect()
    now = now_iso()

    # 3 obs rows in different states
    for uid, status in [
        ("uid___a001_x123_x001", ObsStatus.EXTRACTED),
        ("uid___a001_x123_x002", ObsStatus.EXTRACTED),
        ("uid___a001_x123_x003", ObsStatus.PENDING),
    ]:
        ObsQueries.upsert_seen(con, uid, "2025-06-01")
        if status == ObsStatus.EXTRACTED:
            ObsQueries.mark_extracted(con, uid, Path("/tmp/data"), 50, 0, True)
    con.commit()

    # 2 pi_runs rows
    PIRunsQueries.insert_row(
        con,
        uid="uid___a001_x123_x001",
        script_path="/p/script.py",
        cwd="/p",
        casa_cmd="casa -c script.py",
        log_path="/p/log.txt",
        started_at=now,
        status=PIRunStatus.SUCCESS,
        hostname="testhost",
    )
    PIRunsQueries.insert_row(
        con,
        uid="uid___a001_x123_x002",
        script_path="/p/script.py",
        cwd="/p",
        casa_cmd="casa -c script.py",
        log_path="/p/log.txt",
        started_at=now,
        status=PIRunStatus.FAILED,
        hostname="testhost",
    )
    con.commit()

    return db


@pytest.fixture
def alma_tree(tmp_path):
    """Create a realistic science_goal/group/member directory layout."""
    data_dir = tmp_path / "2025.1.00383.L" / "2025.1.00383.L"
    sg = data_dir / "science_goal.uid___A001_X3833_X64b8"
    group = sg / "group.uid___A001_X3833_X64b9"
    member = group / "member.uid___A001_X3833_X64bc"
    script_dir = member / "script"
    script_dir.mkdir(parents=True)
    (script_dir / "scriptForPI.py").write_text("# ScriptForPI\n")

    # Create a second member
    member2 = group / "member.uid___A001_X3833_X64bd"
    script_dir2 = member2 / "script"
    script_dir2.mkdir(parents=True)
    (script_dir2 / "scriptForPI.py").write_text("# ScriptForPI\n")

    return data_dir


@pytest.fixture
def imaging_tree(tmp_path):
    """Create a realistic TM/SM/TP directory layout for imaging tests.

    Includes mock weblogs, calibrated MS dirs, TP FITS cubes, and
    a targets_by_array.csv.
    """
    base_dir = tmp_path / "2025.1.00383.L"
    data_dir = base_dir / "2025.1.00383.L"

    mous_id_tm = "X3833_X64bc"
    mous_id_sm = "X3833_X64bd"
    mous_id_tp = "X3833_X64be"
    gous_id = "X3833_X64b9"
    sg_id = "X3833_X64b8"

    # TM member with calibrated MS
    tm_member = (
        data_dir / f"science_goal.uid___A001_{sg_id}"
        / f"group.uid___A001_{gous_id}"
        / f"member.uid___A001_{mous_id_tm}"
    )
    tm_working = tm_member / "calibrated" / "working"
    tm_working.mkdir(parents=True)
    (tm_working / "uid___A001_X3833_X64bc_targets_line.ms").mkdir()

    # SM member with calibrated MS
    sm_member = (
        data_dir / f"science_goal.uid___A001_{sg_id}"
        / f"group.uid___A001_{gous_id}"
        / f"member.uid___A001_{mous_id_sm}"
    )
    sm_working = sm_member / "calibrated" / "working"
    sm_working.mkdir(parents=True)
    (sm_working / "uid___A001_X3833_X64bd_targets_line.ms").mkdir()

    # TP member with product cube
    tp_member = (
        data_dir / f"science_goal.uid___A001_{sg_id}"
        / f"group.uid___A001_{gous_id}"
        / f"member.uid___A001_{mous_id_tp}"
    )
    tp_product = tp_member / "product"
    tp_product.mkdir(parents=True)

    # Create a minimal TP FITS cube
    try:
        from astropy.io import fits
        import numpy as np

        tp_data = np.zeros((1, 100, 10, 10), dtype=np.float32)
        hdr = fits.Header()
        hdr["NAXIS"] = 4
        hdr["NAXIS1"] = 10
        hdr["NAXIS2"] = 10
        hdr["NAXIS3"] = 100
        hdr["NAXIS4"] = 1
        hdr["CTYPE1"] = "RA---SIN"
        hdr["CTYPE2"] = "DEC--SIN"
        hdr["CTYPE3"] = "FREQ"
        hdr["CTYPE4"] = "STOKES"
        hdr["CRVAL3"] = 86.0e9
        hdr["CDELT3"] = 244140.0
        hdr["CRPIX3"] = 1
        hdr["BMAJ"] = 0.01
        hdr["BMIN"] = 0.008
        hdr["BPA"] = 0.0

        tp_path = tp_product / "uid___A001_X3833_X64be.SRC1.spw23.cube.I.sd.fits"
        fits.PrimaryHDU(data=tp_data, header=hdr).writeto(str(tp_path))
    except ImportError:
        # astropy not available — create placeholder
        (tp_product / "uid___A001_X3833_X64be.SRC1.spw23.cube.I.sd.fits").write_bytes(b"")

    # Weblog with casa_commands.log
    weblog_dir = tmp_path / "weblogs"
    log_dir = (
        weblog_dir / f"uid___A001_{mous_id_tm}"
        / "pipeline-20250101" / "html"
    )
    log_dir.mkdir(parents=True)
    (log_dir / "casa_commands.log").write_text(dedent("""\
        # hif_makeimlist(specmode='cube')
        tclean(vis=['test.ms'], field='"SRC1"', spw=['23'], specmode='cube', nchan=100, start='86.0GHz', width='244.14kHz', imsize=[100, 100], cell=['0.5arcsec'], restoration=True, pbcor=True, niter=30000, threshold='0.1Jy', imagename='test.spw23.cube.I.iter1')
        tclean(vis=['test.ms'], field='"SRC1"', spw=['25'], specmode='cube', nchan=100, start='87.0GHz', width='244.14kHz', imsize=[100, 100], cell=['0.5arcsec'], restoration=True, pbcor=True, niter=30000, threshold='0.1Jy', imagename='test.spw25.cube.I.iter1')
    """))

    # Targets CSV
    csv_path = base_dir / "targets_by_array.csv"
    csv_path.write_text(
        "source_name,array,sb_name,sgous_id,gous_id,mous_ids,Line group\n"
        f"SRC1,TM,sb_tm,{sg_id},{gous_id},{mous_id_tm},N2H+\n"
        f"SRC1,SM,sb_sm,{sg_id},{gous_id},{mous_id_sm},N2H+\n"
        f"SRC1,TP,sb_tp,{sg_id},{gous_id},{mous_id_tp},N2H+\n"
    )

    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "weblog_dir": weblog_dir,
        "csv_path": csv_path,
        "gous_id": gous_id,
        "sg_id": sg_id,
        "mous_id_tm": mous_id_tm,
        "mous_id_sm": mous_id_sm,
        "mous_id_tp": mous_id_tp,
    }


@pytest.fixture
def contsub_tree(tmp_path):
    """Create a MOUS directory in the 'needs contsub' state.

    Has a calibrated MS, cont.dat, piperestorescript, and scriptForPI,
    but NO _targets_line.ms.  Also includes a second MOUS that already
    has _targets_line.ms (for skip testing).
    """
    data_dir = tmp_path / "2025.1.00383.L" / "2025.1.00383.L"

    # --- MOUS needing contsub (1 EB) ---
    sg = data_dir / "science_goal.uid___A001_X3833_X64b8"
    group = sg / "group.uid___A001_X3833_X64b9"
    member = group / "member.uid___A001_X3833_X64bc"

    # script/
    script_dir = member / "script"
    script_dir.mkdir(parents=True)
    (script_dir / "member.uid___A001_X3833_X64bc.scriptForPI.py").write_text(
        "# ScriptForPI\n"
    )
    (script_dir / "member.uid___A001_X3833_X64bc.hifa_calimage.casa_piperestorescript.py").write_text(
        "h_init()\n"
        "try:\n"
        "    hifa_restoredata (vis=['uid___A002_X12fb842_X4e4e'], "
        "session=['session_1'], ocorr_mode='ca')\n"
        "finally:\n"
        "    h_save()\n"
    )
    (script_dir / "member.uid___A001_X3833_X64bc.hifa_calimage.casa_pipescript.py").write_text(
        "context = h_init()\nhif_mstransform()\nhif_findcont()\nhif_uvcontsub()\nh_save()\n"
    )

    # calibrated/working/ with MS but no _targets_line.ms
    working = member / "calibrated" / "working"
    working.mkdir(parents=True)
    (working / "uid___A002_X12fb842_X4e4e.ms").mkdir()
    (working / "pipeline-20251016T221022.context").write_text("")

    # calibrated/rawdata/ and products/
    rawdata = member / "calibrated" / "rawdata"
    rawdata.mkdir()
    (member / "calibrated" / "products").symlink_to(
        Path("..") / "calibration"
    )

    # calibration/
    cal_dir = member / "calibration"
    cal_dir.mkdir()
    (cal_dir / "cont.dat").write_text(
        "Field: SRC1\n\n"
        "SpectralWindow: 22\n"
        "90.62GHz~90.68GHz LSRK\n"
    )
    (cal_dir / "pipeline-20251014T163250.selfcal.json").write_text(
        '{"scal_targets": []}'
    )

    # --- MOUS already complete (has _targets_line.ms) ---
    member2 = group / "member.uid___A001_X3833_X64bd"
    script_dir2 = member2 / "script"
    script_dir2.mkdir(parents=True)
    (script_dir2 / "member.uid___A001_X3833_X64bd.scriptForPI.py").write_text(
        "# ScriptForPI\n"
    )
    working2 = member2 / "calibrated" / "working"
    working2.mkdir(parents=True)
    (working2 / "uid___A002_XABCD_XEFGH.ms").mkdir()
    (working2 / "uid___A002_XABCD_XEFGH_targets_line.ms").mkdir()
    cal_dir2 = member2 / "calibration"
    cal_dir2.mkdir()
    (cal_dir2 / "cont.dat").write_text("Field: SRC1\n")

    # --- MOUS with 3 EBs, only 1 has _targets_line.ms (partial) ---
    member3 = group / "member.uid___A001_X3833_X64be"
    script_dir3 = member3 / "script"
    script_dir3.mkdir(parents=True)
    (script_dir3 / "member.uid___A001_X3833_X64be.scriptForPI.py").write_text(
        "# ScriptForPI\n"
    )
    working3 = member3 / "calibrated" / "working"
    working3.mkdir(parents=True)
    (working3 / "uid___A002_X111_X001.ms").mkdir()
    (working3 / "uid___A002_X111_X002.ms").mkdir()
    (working3 / "uid___A002_X111_X003.ms").mkdir()
    (working3 / "uid___A002_X111_X001_targets_line.ms").mkdir()  # only 1 of 3
    cal_dir3 = member3 / "calibration"
    cal_dir3.mkdir()
    (cal_dir3 / "cont.dat").write_text("Field: SRC2\n")

    return {
        "data_dir": data_dir,
        "base_dir": tmp_path / "2025.1.00383.L",
        "member_needs": member,
        "member_complete": member2,
        "member_partial": member3,
    }
