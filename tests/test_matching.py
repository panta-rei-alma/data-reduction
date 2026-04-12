"""Tests for panta_rei.imaging.matching — input mapping and preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from panta_rei.imaging.matching import (
    ImagingUnit,
    TargetGroup,
    _compute_tm_freq_range,
    _parse_freq_quantity,
    build_output_path,
    get_freq_range_string,
    load_targets_csv,
    match_cubes_by_frequency,
    sanitize_source_name,
    sanitize_source_name_for_glob,
    targets_by_array,
    validate_tp_beams,
    validate_tp_spectral_axis,
)


# ---------------------------------------------------------------------------
# Source name helpers
# ---------------------------------------------------------------------------

class TestSanitizeSourceName:
    def test_plus_and_minus(self):
        assert sanitize_source_name("AG342.0584+0.4213") == "AG342.0584p0.4213"
        assert sanitize_source_name("AG221.9599-1.9932") == "AG221.9599m1.9932"

    def test_for_glob(self):
        assert sanitize_source_name_for_glob("AG342.0584+0.4213") == "AG342.0584p0.4213"
        # Minus is kept for glob (only + is changed)
        assert sanitize_source_name_for_glob("AG221.9599-1.9932") == "AG221.9599-1.9932"


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------

class TestParseFreqQuantity:
    def test_ghz(self):
        assert _parse_freq_quantity("86.05GHz") == pytest.approx(86.05e9)

    def test_khz(self):
        assert _parse_freq_quantity("244.140625kHz") == pytest.approx(244140.625)

    def test_mhz(self):
        assert _parse_freq_quantity("100MHz") == pytest.approx(1e8)

    def test_bare_float(self):
        assert _parse_freq_quantity("1e9") == pytest.approx(1e9)

    def test_empty(self):
        assert _parse_freq_quantity("") is None

    def test_invalid(self):
        assert _parse_freq_quantity("not_a_freq") is None


class TestComputeTmFreqRange:
    def test_basic(self):
        result = _compute_tm_freq_range("86.05GHz", "244.140625kHz", 956)
        assert result is not None
        lo, hi = result
        assert lo == pytest.approx(86.05e9, rel=1e-6)
        # hi ≈ 86.05e9 + 956 * 244140.625
        assert hi > lo

    def test_missing_start(self):
        assert _compute_tm_freq_range("", "244.14kHz", 100) is None

    def test_missing_nchan(self):
        assert _compute_tm_freq_range("86GHz", "244kHz", None) is None


class TestMatchCubesByFrequency:
    def test_simple_match(self):
        a = [{"freq_min": 86e9, "freq_max": 86.2e9}]
        b = [{"freq_min": 86.01e9, "freq_max": 86.21e9}]
        matched = match_cubes_by_frequency(a, b)
        assert len(matched) == 1

    def test_no_match(self):
        a = [{"freq_min": 86e9, "freq_max": 86.2e9}]
        b = [{"freq_min": 100e9, "freq_max": 100.2e9}]
        matched = match_cubes_by_frequency(a, b)
        assert len(matched) == 0

    def test_multiple_match(self):
        a = [
            {"freq_min": 86e9, "freq_max": 86.2e9},
            {"freq_min": 87e9, "freq_max": 87.2e9},
        ]
        b = [
            {"freq_min": 87.01e9, "freq_max": 87.21e9},
            {"freq_min": 86.01e9, "freq_max": 86.21e9},
        ]
        matched = match_cubes_by_frequency(a, b)
        assert len(matched) == 2
        # Should be sorted by frequency
        assert matched[0][0]["freq_min"] < matched[1][0]["freq_min"]


class TestGetFreqRangeString:
    def test_basic(self):
        assert get_freq_range_string(86e9, 87e9) == "86.0-87.0"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoadTargetsCsv:
    def test_round_trip(self, tmp_path):
        csv_path = tmp_path / "targets_by_array.csv"
        csv_path.write_text(
            "source_name,array,sb_name,sgous_id,gous_id,mous_ids,Line group\n"
            "SRC1,TM,sb_tm,sg1,g1,x3833_x64bc,N2H+\n"
            "SRC1,SM,sb_sm,sg1,g1,x3833_x64bd,N2H+\n"
            "SRC1,TP,sb_tp,sg1,g1,x3833_x64be,N2H+\n"
        )
        result = load_targets_csv(csv_path)
        assert "g1" in result
        assert len(result["g1"]) == 3

    def test_multi_mous(self, tmp_path):
        csv_path = tmp_path / "targets_by_array.csv"
        csv_path.write_text(
            "source_name,array,sb_name,sgous_id,gous_id,mous_ids,Line group\n"
            "SRC1,TM,sb_tm,sg1,g1,x3833_x64bc;x3833_x64bd,N2H+\n"
        )
        result = load_targets_csv(csv_path)
        tg = result["g1"][0]
        assert len(tg.mous_ids) == 2

    def test_empty_file(self, tmp_path):
        csv_path = tmp_path / "targets_by_array.csv"
        csv_path.write_text(
            "source_name,array,sb_name,sgous_id,gous_id,mous_ids,Line group\n"
        )
        result = load_targets_csv(csv_path)
        assert result == {}


class TestTargetsByArray:
    def test_partition(self):
        groups = [
            TargetGroup("SRC1", "TM", "sb_tm", "sg1", "g1", ["m1"], "N2H+"),
            TargetGroup("SRC1", "SM", "sb_sm", "sg1", "g1", ["m2"], "N2H+"),
            TargetGroup("SRC1", "TP", "sb_tp", "sg1", "g1", ["m3"], "N2H+"),
        ]
        by_arr = targets_by_array(groups)
        assert set(by_arr.keys()) == {"TM", "SM", "TP"}
        assert len(by_arr["TM"]) == 1


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

class TestBuildOutputPath:
    def test_basic(self):
        p = build_output_path(
            Path("/out"),
            "X3833_X64b9",
            "AG231.7986-1.9684",
            86e9,
            87e9,
        )
        assert p.name == (
            "group.uid___A001_X3833_X64b9.lp_nperetto."
            "AG231.7986m1.9684.12m7mTP.86.0-87.0GHz.cube.pbcor.fits"
        )
        assert "group.uid___A001_X3833_X64b9.lp_nperetto" in str(p.parent)


# ---------------------------------------------------------------------------
# TP validation (requires astropy)
# ---------------------------------------------------------------------------

class TestValidateTP:
    @pytest.fixture
    def tp_fits(self, tmp_path):
        """Create a minimal TP FITS file with frequency axis and beams."""
        try:
            from astropy.io import fits
            import numpy as np
        except ImportError:
            pytest.skip("astropy not available")

        data = np.zeros((1, 100, 10, 10), dtype=np.float32)
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
        hdr["CRVAL3"] = 86.1e9
        hdr["CDELT3"] = 244140.625
        hdr["CRPIX3"] = 1
        hdr["BMAJ"] = 0.01
        hdr["BMIN"] = 0.008
        hdr["BPA"] = 0.0

        path = tmp_path / "tp_cube.fits"
        fits.PrimaryHDU(data=data, header=hdr).writeto(str(path))
        return path

    def test_spectral_axis_ok(self, tp_fits):
        ok, msg, info = validate_tp_spectral_axis(tp_fits, recovered_nchan=100)
        assert ok
        assert info["nchan"] == 100

    def test_spectral_nchan_diff_is_warning_not_fail(self, tp_fits):
        """nchan mismatch logs a warning but does not hard-fail.

        TP and TM may have different channelization; sdintimaging regrids
        to the TM grid.
        """
        ok, msg, info = validate_tp_spectral_axis(tp_fits, recovered_nchan=200)
        assert ok  # Should pass — nchan diff is only a warning now
        assert info["nchan"] == 100  # TP has 100 channels

    def test_beams_ok(self, tp_fits):
        ok, msg = validate_tp_beams(tp_fits)
        assert ok

    def test_beams_missing_bpa(self, tmp_path):
        """BMAJ/BMIN without BPA should fail."""
        try:
            from astropy.io import fits
            import numpy as np
        except ImportError:
            pytest.skip("astropy not available")

        data = np.zeros((1, 100, 10, 10), dtype=np.float32)
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
        hdr["CRVAL3"] = 86.1e9
        hdr["CDELT3"] = 244140.625
        hdr["CRPIX3"] = 1
        hdr["BMAJ"] = 0.01
        hdr["BMIN"] = 0.008
        # No BPA

        path = tmp_path / "tp_no_bpa.fits"
        fits.PrimaryHDU(data=data, header=hdr).writeto(str(path))

        ok, msg = validate_tp_beams(path)
        assert not ok
        assert "BPA missing" in msg

    def test_beams_missing(self, tmp_path):
        try:
            from astropy.io import fits
            import numpy as np
        except ImportError:
            pytest.skip("astropy not available")

        data = np.zeros((1, 100, 10, 10), dtype=np.float32)
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
        hdr["CRVAL3"] = 86.1e9
        hdr["CDELT3"] = 244140.625
        hdr["CRPIX3"] = 1
        # No BMAJ/BMIN

        path = tmp_path / "tp_no_beam.fits"
        fits.PrimaryHDU(data=data, header=hdr).writeto(str(path))

        ok, msg = validate_tp_beams(path)
        assert not ok


# ---------------------------------------------------------------------------
# ImagingUnit dataclass
# ---------------------------------------------------------------------------

class TestImagingUnit:
    def test_defaults(self):
        u = ImagingUnit(
            gous_uid="g1",
            source_name="SRC1",
            line_group="N2H+",
            spw_id="23",
            params_id=1,
        )
        assert not u.ready
        assert u.vis_tm == []
        assert u.sdimage is None


# ---------------------------------------------------------------------------
# find_member_dir
# ---------------------------------------------------------------------------

class TestFindMemberDir:

    def test_compact_uid(self, imaging_tree):
        from panta_rei.imaging.matching import find_member_dir
        result = find_member_dir(imaging_tree["data_dir"], "X3833_X64bc")
        assert result is not None
        assert "member.uid___A001_X3833_X64bc" in str(result)

    def test_full_uid(self, imaging_tree):
        from panta_rei.imaging.matching import find_member_dir
        result = find_member_dir(imaging_tree["data_dir"], "uid___A001_X3833_X64bc")
        assert result is not None
        assert "member.uid___A001_X3833_X64bc" in str(result)

    def test_returns_none_when_missing(self, imaging_tree):
        from panta_rei.imaging.matching import find_member_dir
        result = find_member_dir(imaging_tree["data_dir"], "X9999_X9999")
        assert result is None


class TestFindMsFilesDiagnostic:

    def test_member_not_available_wording(self, imaging_tree):
        from panta_rei.imaging.matching import find_ms_files
        result = find_ms_files(imaging_tree["data_dir"], "X9999_X9999")
        assert not result
        assert "not available on disk" in result.missing_reason
