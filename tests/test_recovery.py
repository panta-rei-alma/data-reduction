"""Tests for panta_rei.imaging.recovery — tclean parameter recovery."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from panta_rei.imaging.recovery import (
    extract_by_field_spw,
    filter_cube_iter1_calls,
    find_casa_commands_log,
    has_staged_weblog,
    parse_tclean_calls,
    recover_params_for_mous,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SINGLE_LINE_TCLEAN = dedent("""\
    # hif_makeimlist(specmode='cube')
    tclean(vis=['test.ms'], field='"AG231.7986-1.9684"', spw=['23'], specmode='cube', nchan=956, start='86.05GHz', width='244.140625kHz', imsize=[480, 450], cell=['0.42arcsec'], phasecenter='J2000 07:27:07.5 -18:55:30', gridder='mosaic', deconvolver='hogbom', weighting='briggsbwtaper', robust=0.5, niter=30000, threshold='0.276Jy', restoration=True, pbcor=True, imagename='oussid.s39_0.AG231.7986m1.9684_sci.spw23.cube.I.iter1')
""")

MULTI_LINE_TCLEAN = dedent("""\
    # hif_makeimlist(specmode='cube')
    tclean(vis=['test.ms'],
           field='"AG231.7986-1.9684"',
           spw=['23'],
           specmode='cube',
           nchan=956,
           start='86.05GHz',
           width='244.140625kHz',
           imsize=[480, 450],
           cell=['0.42arcsec'],
           phasecenter='J2000 07:27:07.5 -18:55:30',
           gridder='mosaic',
           deconvolver='hogbom',
           weighting='briggsbwtaper',
           robust=0.5,
           niter=30000,
           threshold='0.276Jy',
           restoration=True,
           pbcor=True,
           imagename='oussid.s39_0.AG231.7986m1.9684_sci.spw23.cube.I.iter1')
""")

ITER0_AND_ITER1 = dedent("""\
    # hif_makeimlist(specmode='cube')
    tclean(vis=['test.ms'], field='"SRC1"', spw=['23'], specmode='cube', nchan=100, start='86.0GHz', width='244.14kHz', imsize=[100, 100], cell=['0.5arcsec'], restoration=False, niter=0, pbcor=False, imagename='test.spw23.cube.I.iter0')
    tclean(vis=['test.ms'], field='"SRC1"', spw=['23'], specmode='cube', nchan=100, start='86.0GHz', width='244.14kHz', imsize=[100, 100], cell=['0.5arcsec'], restoration=True, pbcor=True, niter=30000, threshold='0.1Jy', imagename='test.spw23.cube.I.iter1')
    tclean(vis=['test.ms'], field='"SRC1"', spw=['25'], specmode='cube', nchan=100, start='87.0GHz', width='244.14kHz', imsize=[100, 100], cell=['0.5arcsec'], restoration=True, pbcor=True, niter=30000, threshold='0.1Jy', imagename='test.spw25.cube.I.iter1')
""")


# ---------------------------------------------------------------------------
# parse_tclean_calls
# ---------------------------------------------------------------------------

class TestParseTcleanCalls:
    def test_single_line(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(SINGLE_LINE_TCLEAN)
        calls = parse_tclean_calls(p)
        assert len(calls) == 1
        assert calls[0]["specmode"] == "cube"
        assert calls[0]["nchan"] == 956

    def test_multi_line(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(MULTI_LINE_TCLEAN)
        calls = parse_tclean_calls(p)
        assert len(calls) == 1
        assert calls[0]["specmode"] == "cube"
        assert calls[0]["robust"] == 0.5

    def test_multiple_calls(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(ITER0_AND_ITER1)
        calls = parse_tclean_calls(p)
        assert len(calls) == 3

    def test_empty_file(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text("")
        calls = parse_tclean_calls(p)
        assert calls == []


# ---------------------------------------------------------------------------
# filter_cube_iter1_calls
# ---------------------------------------------------------------------------

class TestFilterCubeIter1:
    def test_filters_iter0(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(ITER0_AND_ITER1)
        calls = parse_tclean_calls(p)
        iter1 = filter_cube_iter1_calls(calls, p)
        assert len(iter1) == 2
        for c in iter1:
            assert c["restoration"] is True
            assert c["pbcor"] is True

    def test_no_cube_calls(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text("tclean(specmode='mfs', restoration=True, pbcor=True, imagename='test.cont.I.iter1')\n")
        calls = parse_tclean_calls(p)
        iter1 = filter_cube_iter1_calls(calls, p)
        assert iter1 == []


# ---------------------------------------------------------------------------
# extract_by_field_spw
# ---------------------------------------------------------------------------

class TestExtractByFieldSpw:
    def test_basic(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(ITER0_AND_ITER1)
        calls = parse_tclean_calls(p)
        iter1 = filter_cube_iter1_calls(calls, p)
        result = extract_by_field_spw(iter1)
        assert len(result) == 2
        assert ("SRC1", "23") in result
        assert ("SRC1", "25") in result

    def test_cleans_embedded_quotes(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(SINGLE_LINE_TCLEAN)
        calls = parse_tclean_calls(p)
        iter1 = filter_cube_iter1_calls(calls, p)
        result = extract_by_field_spw(iter1)
        assert ("AG231.7986-1.9684", "23") in result


# ---------------------------------------------------------------------------
# find_casa_commands_log
# ---------------------------------------------------------------------------

class TestFindCasaCommandsLog:
    def test_staged_weblog(self, tmp_path):
        weblog_base = tmp_path / "weblogs"
        # sanitize_uid normalizes to uid___A001_X3833_X64bc
        log_dir = weblog_base / "uid___A001_X3833_X64bc" / "pipeline-20250101" / "html"
        log_dir.mkdir(parents=True)
        (log_dir / "casa_commands.log").write_text("# test\n")

        result = find_casa_commands_log(weblog_base, "uid___a001_x3833_x64bc")
        assert result is not None
        assert result.name == "casa_commands.log"

    def test_missing_weblog(self, tmp_path):
        result = find_casa_commands_log(tmp_path, "uid___a001_x0000_x0000")
        assert result is None

    def test_raw_log_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        # sanitize_uid gives uid___A001_X3833_X64bc — the glob uses **/ so case must match on disk
        member = data_dir / "science_goal.uid___A001_X3833_X64b8" / "group.uid___A001_X3833_X64b9" / "member.uid___A001_X3833_X64bc"
        log_dir = member / "log"
        log_dir.mkdir(parents=True)
        (log_dir / "member.uid___A001_X3833_X64bc.hifa_calimage.casa_commands.log").write_text("# test\n")

        result = find_casa_commands_log(tmp_path / "no_weblogs", "uid___a001_x3833_x64bc", data_dir)
        assert result is not None


# ---------------------------------------------------------------------------
# recover_params_for_mous (end-to-end)
# ---------------------------------------------------------------------------

class TestRecoverParamsForMous:
    def test_full_recovery(self, tmp_path):
        weblog_base = tmp_path / "weblogs"
        # Directory must match sanitize_uid output
        log_dir = weblog_base / "uid___A001_X3833_X64bc" / "pipeline-20250101" / "html"
        log_dir.mkdir(parents=True)
        (log_dir / "casa_commands.log").write_text(ITER0_AND_ITER1)

        result = recover_params_for_mous("uid___a001_x3833_x64bc", weblog_base)
        assert result is not None
        assert len(result) == 2

    def test_no_weblog(self, tmp_path):
        result = recover_params_for_mous("uid___a001_x0000_x0000", tmp_path)
        assert result is None

    def test_empty_weblog(self, tmp_path):
        weblog_base = tmp_path / "weblogs"
        log_dir = weblog_base / "uid___A001_X3833_X64bc" / "pipeline-20250101" / "html"
        log_dir.mkdir(parents=True)
        (log_dir / "casa_commands.log").write_text("# empty\n")

        result = recover_params_for_mous("uid___a001_x3833_x64bc", weblog_base)
        assert result is None


# ---------------------------------------------------------------------------
# Staged weblog availability
# ---------------------------------------------------------------------------

class TestHasStagedWeblog:

    def test_true_when_weblog_exists(self, tmp_path):
        weblog_base = tmp_path / "weblogs"
        log_dir = weblog_base / "uid___A001_X3833_X64bc" / "pipeline-20250101" / "html"
        log_dir.mkdir(parents=True)
        (log_dir / "casa_commands.log").write_text("# present\n")

        assert has_staged_weblog(weblog_base, "uid___a001_x3833_x64bc") is True

    def test_false_when_no_weblog(self, tmp_path):
        weblog_base = tmp_path / "weblogs"
        weblog_base.mkdir()
        assert has_staged_weblog(weblog_base, "uid___a001_x9999_x9999") is False

    def test_false_when_no_base_dir(self, tmp_path):
        assert has_staged_weblog(tmp_path / "nonexistent", "uid___a001_x3833_x64bc") is False


class TestStagedWeblogWithoutMemberDir:

    def test_finds_weblog_when_member_dir_absent(self, tmp_path):
        """Staged weblog should be found even if member dir doesn't exist."""
        weblog_base = tmp_path / "weblogs"
        log_dir = weblog_base / "uid___A001_X3833_X64bc" / "pipeline-20250101" / "html"
        log_dir.mkdir(parents=True)
        (log_dir / "casa_commands.log").write_text(
            "tclean(vis=['test.ms'], field='\"SRC1\"', spw=['23'], "
            "specmode='cube', nchan=100, start='86.0GHz', width='244.14kHz', "
            "imsize=[100, 100], cell=['0.5arcsec'], restoration=True, "
            "pbcor=True, niter=30000, threshold='0.1Jy', "
            "imagename='test.spw23.cube.I.iter1')\n"
        )

        # data_dir has no member directory for this MOUS
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = find_casa_commands_log(weblog_base, "uid___a001_x3833_x64bc", data_dir)
        assert result is not None
        assert "casa_commands.log" in str(result)
