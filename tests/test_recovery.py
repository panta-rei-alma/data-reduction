"""Tests for panta_rei.imaging.recovery — tclean parameter recovery."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from panta_rei.imaging.recovery import (
    extract_by_field_spw,
    extract_continuum_by_field,
    filter_continuum_iter1_calls,
    filter_cube_iter1_calls,
    find_casa_commands_log,
    has_staged_weblog,
    parse_tclean_calls,
    recover_continuum_params_for_mous,
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

# A real-format QA2 cube call (casa_commands.log line-wrapping preserved),
# including values the old regex fallback dropped: nested lists, restfreq,
# uvtaper, scales, a value split across a line break, negative numbers.
REAL_FORMAT_TCLEAN = dedent("""\
    # hif_makeimlist(specmode='cube')
    tclean(vis=['uid___A002_X135387b_Xa317_targets_line.ms'],
           field='"AG221.9599-1.9932"', spw=['27'],
           antenna=['0,1,2,3,4,5,6,7&'],
           scan=['7'], intent='OBSERVE_TARGET#ON_SOURCE', datacolumn='data',
           imagename='uid___A001_X3833_X64ba.s45_0._AG221.9599-1.9932__sci.spw27.cube.regcal.I.iter1',
           imsize=[480, 450], cell=['0.36arcsec'], phasecenter='ICRS 07:00:50.9517
           -008.56.30.169', stokes='I', specmode='cube', nchan=1916,
           start='102.4527117095GHz', width='0.0610408MHz', outframe='LSRK',
           restfreq=['102.5479844GHz'], uvtaper=[], scales=[0, 5, 15],
           perchanweightdensity=True, gridder='mosaic', mosweight=True,
           usepointing=False, pblimit=0.2, deconvolver='hogbom', restoration=True,
           restoringbeam='common', pbcor=True, weighting='briggsbwtaper',
           robust=-0.5, npixels=0, niter=4360, threshold='0.0145Jy', nsigma=0.0,
           cyclefactor=3.0, interactive=False, fullsummary=False, usemask='auto-multithresh',
           sidelobethreshold=2.0, noisethreshold=4.25, lownoisethreshold=1.5,
           negativethreshold=15.0, minbeamfrac=0.3, growiterations=50,
           dogrowprune=True, minpercentchange=1.0, fastnoise=False,
           savemodel='none', parallel=True)
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

    def test_real_format_keeps_every_literal(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(REAL_FORMAT_TCLEAN)
        (c,) = parse_tclean_calls(p)
        assert c["field"] == '"AG221.9599-1.9932"'
        assert c["antenna"] == ["0,1,2,3,4,5,6,7&"]
        assert c["imsize"] == [480, 450]
        assert c["restfreq"] == ["102.5479844GHz"]
        assert c["uvtaper"] == []
        assert c["scales"] == [0, 5, 15]
        assert c["robust"] == -0.5
        assert c["cyclefactor"] == 3.0
        assert c["fastnoise"] is False and c["dogrowprune"] is True
        assert c["minpercentchange"] == 1.0
        assert c["niter"] == 4360 and isinstance(c["niter"], int)
        # Value wrapped across a line break: whitespace collapsed later by
        # _normalize_string_values, but the value itself is intact.
        assert " ".join(c["phasecenter"].split()) == "ICRS 07:00:50.9517 -008.56.30.169"

    def test_nested_lists_and_negative_numbers(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(
            "tclean(vis=['a.ms', 'b.ms'], spw=['24:86.1~86.2GHz;86.3~86.4GHz', '18'], "
            "mask=[[1, 2], [3, 4]], negativethreshold=-1.5e-3, nterms=2, "
            "reffreq='86.5GHz', calcres=False)\n"
        )
        (c,) = parse_tclean_calls(p)
        assert c["vis"] == ["a.ms", "b.ms"]
        assert c["spw"] == ["24:86.1~86.2GHz;86.3~86.4GHz", "18"]
        assert c["mask"] == [[1, 2], [3, 4]]
        assert c["negativethreshold"] == -1.5e-3
        assert c["nterms"] == 2
        assert c["reffreq"] == "86.5GHz"
        assert c["calcres"] is False

    def test_non_literal_value_skipped_rest_kept(self, tmp_path):
        """Self-cal solve calls carry ``np.float64(...)`` reprs; that key is
        skipped, never evaluated, and the other keywords survive."""
        p = tmp_path / "casa_commands.log"
        p.write_text(
            "tclean(vis=['a.ms'], noisethreshold=np.float64(5.0), "
            "niter=100, imagename=__import__('os').getcwd())\n"
        )
        (c,) = parse_tclean_calls(p)
        assert c == {"vis": ["a.ms"], "niter": 100}

    def test_unparseable_call_falls_back_to_regex(self, tmp_path):
        """A truncated/garbled call still yields its recognisable params."""
        p = tmp_path / "casa_commands.log"
        p.write_text("tclean(field='SRC1', niter=50, restoration=True, imsize=[10, 10] ,,)\n")
        (c,) = parse_tclean_calls(p)
        assert c["field"] == "SRC1"
        assert c["niter"] == 50
        assert c["restoration"] is True
        assert c["imsize"] == [10, 10]


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

    def test_repairs_dash_linebreak_in_usemask(self, tmp_path):
        """When the multi-line tclean assembler line-wraps a hyphenated
        value (e.g. 'auto-multithresh'), the resulting ' ' join produces
        'auto- multithresh', which tclean rejects.  The post-parse
        normalizer must rejoin it.

        Reproduction: a real ALMA pipeline weblog from this project's
        recovered/ tree shows ``usemask='auto-`` at the end of one line
        and ``multithresh'`` at the start of the next.  This test pins
        the repair behaviour."""
        log_text = (
            "# 2026-01-23 ... cube imaging\n"
            "# hif_makeimlist(specmode='cube')\n"
            "tclean(field='SRC1', spw=['23', '23'], specmode='cube', "
            "usemask='auto-\n"
            "multithresh', restoration=True, pbcor=True, "
            "imagename='/x/SRC1.cube.iter1', niter=100)\n"
        )
        p = tmp_path / "casa_commands.log"
        p.write_text(log_text)
        calls = parse_tclean_calls(p)
        iter1 = filter_cube_iter1_calls(calls, p)
        result = extract_by_field_spw(iter1)

        params = result.get(("SRC1", "23"))
        assert params is not None, result
        assert params["usemask"] == "auto-multithresh", repr(params["usemask"])

    def test_normalize_preserves_unrelated_strings(self):
        """The dash-space repair must only affect ``letter-WS-letter``;
        legitimate strings containing hyphens or spaces are untouched."""
        from panta_rei.imaging.recovery import _normalize_string_values
        out = _normalize_string_values({
            "specmode": "cube",
            "outframe": "LSRK",
            "weighting": "briggsbwtaper",
            "usemask": "auto- multithresh",  # the bug we fix
            "phasecenter": "ICRS 12:00:00.000 -30.00.00.000",  # has dash, no fix
            "field": "AG231.7986-1.9684",  # source name with dash
        })
        assert out["usemask"] == "auto-multithresh"
        # Numeric/space-after-dash patterns inside coords/names must not change:
        assert out["phasecenter"] == "ICRS 12:00:00.000 -30.00.00.000"
        assert out["field"] == "AG231.7986-1.9684"
        # Unrelated values pass through with internal whitespace collapsed:
        assert out["specmode"] == "cube"
        assert out["weighting"] == "briggsbwtaper"


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
# Continuum recovery
# ---------------------------------------------------------------------------

# A realistic continuum section: a per-spw findcont decoy (mfs, but imagename
# has '.findcont' not '.cont.'), the aggregate continuum iter1 for two fields
# (mfs, '.cont.', frequency-range spw selection), and a cube iter1 decoy
# (specmode='cube', '.cube.').  Only the two aggregate continuum calls must be
# recovered.
CONTINUUM_LOG = dedent("""\
    # hif_makeimlist(specmode='cont')
    tclean(vis=['eb.ms'], field='"AG344.8921+0.4602"', spw=['24'], specmode='mfs', restoration=True, pbcor=True, gridder='mosaic', imagename='uid.s32_0._AG344.8921+0.4602__sci.spw24.mfs.regcal.I.findcont')
    tclean(vis=['eb.ms'], field='"AG344.8921+0.4602"', spw=['24:86.3229~86.3305GHz;86.3324~86.3801GHz,26:86.8295~86.8602GHz'], datacolumn='data', specmode='mfs', deconvolver='hogbom', gridder='mosaic', mosweight=True, pblimit=0.2, perchanweightdensity=False, weighting='briggs', robust=0.5, imsize=[160, 160], cell=['1.9arcsec'], phasecenter='ICRS 17:01:55.4 -041.09.30', threshold='0.00311Jy', usemask='auto-multithresh', restoration=True, pbcor=True, imagename='uid.s36_0._AG344.8921+0.4602__sci.spw24_26.cont.regcal.I.iter1')
    tclean(vis=['eb.ms'], field='"AG345.0884+0.0818"', spw=['24:86.3~86.4GHz,26:86.8~86.9GHz'], datacolumn='corrected', specmode='mfs', deconvolver='hogbom', gridder='mosaic', mosweight=True, imsize=[160, 160], cell=['1.9arcsec'], restoration=True, pbcor=True, imagename='uid.s36_0._AG345.0884+0.0818__sci.spw24_26.cont.regcal.I.iter1')
    tclean(vis=['eb.ms'], field='"AG344.8921+0.4602"', spw=['24'], specmode='cube', nchan=100, restoration=True, pbcor=True, imagename='uid.s45_0._AG344.8921+0.4602__sci.spw24.cube.regcal.I.iter1')
""")


class TestFilterContinuumIter1Calls:
    def test_excludes_findcont_cube_keeps_aggregate(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(CONTINUUM_LOG)
        cont = filter_continuum_iter1_calls(parse_tclean_calls(p))
        assert len(cont) == 2
        names = {c["imagename"] for c in cont}
        assert all(".cont." in n for n in names)
        assert not any(".findcont" in n or ".cube." in n for n in names)

    def test_preserves_full_freq_range_spw_and_params(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(CONTINUUM_LOG)
        cont = filter_continuum_iter1_calls(parse_tclean_calls(p))
        c = next(c for c in cont if "AG344.8921+0.4602" in c["field"])
        # full line-free GHz-range selection preserved verbatim (as a 1-elem list)
        assert c["spw"] == ["24:86.3229~86.3305GHz;86.3324~86.3801GHz,26:86.8295~86.8602GHz"]
        # continuum-specific fidelity preserved
        assert c["mosweight"] is True
        assert c["pblimit"] == 0.2
        assert c["perchanweightdensity"] is False
        assert c["datacolumn"] == "data"


class TestExtractContinuumByField:
    def test_keys_by_field_one_per_field(self, tmp_path):
        p = tmp_path / "casa_commands.log"
        p.write_text(CONTINUUM_LOG)
        by_field = extract_continuum_by_field(
            filter_continuum_iter1_calls(parse_tclean_calls(p))
        )
        assert set(by_field) == {"AG344.8921+0.4602", "AG345.0884+0.0818"}
        # field quotes cleaned; spw NOT part of the key
        assert by_field["AG345.0884+0.0818"]["datacolumn"] == "corrected"


class TestParseCache:
    def test_second_parse_served_from_cache(self, tmp_path):
        from panta_rei.imaging.recovery import clear_parse_cache
        clear_parse_cache()
        p = tmp_path / "casa_commands.log"
        p.write_text(ITER0_AND_ITER1)
        first = parse_tclean_calls(p)
        assert len(first) == 3
        # Unchanged file → second call returns the SAME cached object (a
        # re-parse would build a new, merely-equal list).
        assert parse_tclean_calls(p) is first
        clear_parse_cache()

    def test_cache_invalidates_on_rewrite(self, tmp_path):
        from panta_rei.imaging.recovery import clear_parse_cache
        clear_parse_cache()
        p = tmp_path / "casa_commands.log"
        p.write_text(SINGLE_LINE_TCLEAN)
        assert len(parse_tclean_calls(p)) == 1
        p.write_text(ITER0_AND_ITER1)  # size + mtime change → re-parse
        assert len(parse_tclean_calls(p)) == 3
        clear_parse_cache()


class TestEbSpwMapFromCall:
    def test_parallel_vis_spw_per_eb(self):
        from panta_rei.imaging.recovery import eb_spw_map_from_call
        call = {
            "vis": ["uid___A002_X1_Xa_targets.ms", "uid___A002_X2_Xb_targets.ms"],
            "spw": ["24:86.0~86.1GHz", "24:86.01~86.11GHz"],
        }
        assert eb_spw_map_from_call(call) == {
            "uid___A002_X1_Xa": "24:86.0~86.1GHz",
            "uid___A002_X2_Xb": "24:86.01~86.11GHz",
        }

    def test_single_selection_applied_to_all_ebs(self):
        from panta_rei.imaging.recovery import eb_spw_map_from_call
        call = {
            "vis": ["uid___A002_X1_Xa_targets.ms", "uid___A002_X2_Xb_targets.ms"],
            "spw": ["24:86.0~86.1GHz"],
        }
        m = eb_spw_map_from_call(call)
        assert set(m) == {"uid___A002_X1_Xa", "uid___A002_X2_Xb"}
        assert all(v == "24:86.0~86.1GHz" for v in m.values())

    def test_length_mismatch_raises(self):
        import pytest
        from panta_rei.imaging.recovery import eb_spw_map_from_call
        three_vis = {
            "vis": [f"uid___A002_X{i}_Xa_targets.ms" for i in range(3)],
            "spw": ["24:86.0~86.1GHz", "24:86.01~86.11GHz"],
        }
        with pytest.raises(ValueError, match="length mismatch"):
            eb_spw_map_from_call(three_vis)
        more_spw = {
            "vis": ["uid___A002_X1_Xa_targets.ms", "uid___A002_X2_Xb_targets.ms"],
            "spw": ["24:a", "24:b", "24:c"],
        }
        with pytest.raises(ValueError, match="length mismatch"):
            eb_spw_map_from_call(more_spw)


class TestRecoverContinuumParamsForMous:
    def test_full_recovery(self, tmp_path):
        weblog_base = tmp_path / "weblogs"
        log_dir = weblog_base / "uid___A001_X3833_X64bc" / "pipeline-20250101" / "html"
        log_dir.mkdir(parents=True)
        (log_dir / "casa_commands.log").write_text(CONTINUUM_LOG)
        result = recover_continuum_params_for_mous(
            "uid___a001_x3833_x64bc", weblog_base
        )
        assert result is not None
        assert set(result) == {"AG344.8921+0.4602", "AG345.0884+0.0818"}

    def test_none_when_only_cube_present(self, tmp_path):
        weblog_base = tmp_path / "weblogs"
        log_dir = weblog_base / "uid___A001_X3833_X64bc" / "pipeline-20250101" / "html"
        log_dir.mkdir(parents=True)
        (log_dir / "casa_commands.log").write_text(ITER0_AND_ITER1)  # cube only
        result = recover_continuum_params_for_mous(
            "uid___a001_x3833_x64bc", weblog_base
        )
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


class TestGlobCachesScopedToOperation:
    def test_select_units_sees_new_member_after_prior_miss(self, tmp_path):
        """A member-dir miss cached by one selection pass must not hide a
        member extracted before the next pass."""
        from panta_rei.imaging.matching import find_member_dir
        from panta_rei.imaging.recovery import clear_parse_cache

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        assert find_member_dir(data_dir, "X1_X2") is None  # cached miss
        member = data_dir / "science_goal.a" / "group.b" / "member.uid___A001_X1_X2"
        member.mkdir(parents=True)
        # Within one operation the miss is memoized...
        assert find_member_dir(data_dir, "X1_X2") is None
        # ...and the operation-boundary reset makes it visible again.
        clear_parse_cache()
        assert find_member_dir(data_dir, "X1_X2") == member
