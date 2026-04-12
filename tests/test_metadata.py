"""Tests for panta_rei.alma.metadata — array classification, SB parsing, and frequency support."""

from __future__ import annotations

import pytest

from panta_rei.alma.metadata import (
    _ranges_from_freqsupport,
    classify_array,
    sb_family,
    to_compact_ous,
)


# ---------------------------------------------------------------------------
# classify_array
# ---------------------------------------------------------------------------

class TestClassifyArray:
    """classify_array should identify TP, SM (7m), and TM (12m) arrays."""

    # --- TP detection from SB name ---

    def test_tp_suffix_uppercase(self):
        assert classify_array("AG231.79_a_03_TP") == "TP"

    def test_tp_suffix_lowercase(self):
        assert classify_array("AG231.79_a_03_tp") == "TP"

    def test_tp_suffix_mixed_case(self):
        assert classify_array("AG231.79_a_03_Tp") == "TP"

    def test_tp_standalone_word(self):
        """TP as a standalone word matches via word boundary."""
        assert classify_array("tp observations") == "TP"

    # --- SM (7m) detection from SB name ---

    def test_7m_suffix(self):
        assert classify_array("AG231.79_a_03_7M") == "SM"

    def test_7m_suffix_lowercase(self):
        assert classify_array("AG231.79_a_03_7m") == "SM"

    def test_aca_standalone_word(self):
        """'aca' as standalone word (with word boundaries) matches SM."""
        assert classify_array("aca array") == "SM"

    # --- TM (12m) detection from SB name ---

    def test_tm1_suffix(self):
        assert classify_array("AG231.79_a_03_TM1") == "TM"

    def test_tm2_suffix(self):
        assert classify_array("AG231.79_a_03_TM2") == "TM"

    def test_tm_suffix_no_number(self):
        assert classify_array("AG231.79_a_03_TM") == "TM"

    def test_tm_suffix_lowercase(self):
        assert classify_array("AG231.79_a_03_tm1") == "TM"

    def test_tm_suffix_mixed_case(self):
        assert classify_array("AG231.79_a_03_Tm2") == "TM"

    # --- Classification from antenna_arrays fallback ---

    def test_antenna_arrays_totalpower(self):
        assert classify_array("unknown_sb", antenna_arrays="TotalPower") == "TP"

    def test_antenna_arrays_total_power_with_space(self):
        assert classify_array("unknown_sb", antenna_arrays="Total Power") == "TP"

    def test_antenna_arrays_tp(self):
        assert classify_array("unknown_sb", antenna_arrays="tp") == "TP"

    def test_antenna_arrays_12m(self):
        assert classify_array("unknown_sb", antenna_arrays="12m") == "TM"

    def test_antenna_arrays_7m(self):
        assert classify_array("unknown_sb", antenna_arrays="7m") == "SM"

    def test_antenna_arrays_aca(self):
        assert classify_array("unknown_sb", antenna_arrays="aca") == "SM"

    # --- SB name takes precedence over antenna_arrays ---

    def test_sb_name_wins_over_antenna_arrays(self):
        assert classify_array("AG231.79_a_03_TP", antenna_arrays="12m") == "TP"

    # --- Unclassifiable ---

    def test_no_match_returns_none(self):
        assert classify_array("AG231.79_a_03") is None

    def test_empty_sb_name_no_antenna(self):
        assert classify_array("") is None

    def test_unrelated_name_no_antenna(self):
        assert classify_array("random_name_here") is None


# ---------------------------------------------------------------------------
# sb_family
# ---------------------------------------------------------------------------

class TestSbFamily:
    """sb_family should strip the array suffix from SB names."""

    def test_strip_7m_suffix(self):
        assert sb_family("AG231.79_a_03_7M") == "AG231.79_a_03"

    def test_strip_7m_lowercase(self):
        assert sb_family("AG231.79_a_03_7m") == "AG231.79_a_03"

    def test_strip_tm1_suffix(self):
        assert sb_family("AG231.79_a_03_TM1") == "AG231.79_a_03"

    def test_strip_tm2_suffix(self):
        assert sb_family("AG231.79_a_03_TM2") == "AG231.79_a_03"

    def test_strip_tm_suffix_no_number(self):
        assert sb_family("AG231.79_a_03_TM") == "AG231.79_a_03"

    def test_strip_tp_suffix(self):
        assert sb_family("AG231.79_a_03_TP") == "AG231.79_a_03"

    def test_strip_tp_lowercase(self):
        assert sb_family("AG231.79_a_03_tp") == "AG231.79_a_03"

    def test_no_suffix_unchanged(self):
        assert sb_family("AG231.79_a_03") == "AG231.79_a_03"

    def test_multiple_underscores_only_strip_last(self):
        assert sb_family("AG231.79_a_03_extra_TM1") == "AG231.79_a_03_extra"

    def test_real_sb_name_7m(self):
        assert sb_family("AG231.79_b_06_7M") == "AG231.79_b_06"

    def test_aca_suffix_not_stripped(self):
        """The regex only strips _tm, _7m, _tp — not _aca."""
        assert sb_family("AG231.79_a_03_ACA") == "AG231.79_a_03_ACA"


# ---------------------------------------------------------------------------
# _ranges_from_freqsupport
# ---------------------------------------------------------------------------

class TestRangesFromFreqsupport:
    """_ranges_from_freqsupport should parse ALMA frequency_support strings.

    The regex matches 'lo-hi' or 'lo?hi' format with optional unit suffix.
    The '..' separator used in some ALMA formats does NOT match.
    """

    def test_dash_separator(self):
        """Standard dash-separated range with GHz unit."""
        result = _ranges_from_freqsupport("86.63-88.50GHz")
        assert len(result) == 1
        lo, hi = result[0]
        assert pytest.approx(lo, abs=0.01) == 86.63
        assert pytest.approx(hi, abs=0.01) == 88.50

    def test_two_ranges(self):
        s = "86.63-88.50GHz 88.37-90.24GHz"
        result = _ranges_from_freqsupport(s)
        assert len(result) == 2

    def test_returns_tuples_of_floats(self):
        result = _ranges_from_freqsupport("86.63-88.50GHz")
        lo, hi = result[0]
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_lo_less_than_hi(self):
        result = _ranges_from_freqsupport("86.63-88.50GHz")
        lo, hi = result[0]
        assert lo < hi

    def test_reversed_range_swapped(self):
        """If hi < lo in input, they should be swapped."""
        result = _ranges_from_freqsupport("90.0-86.0GHz")
        lo, hi = result[0]
        assert lo < hi

    def test_mhz_unit(self):
        result = _ranges_from_freqsupport("86630-88500MHz")
        assert len(result) == 1
        lo, hi = result[0]
        assert pytest.approx(lo, abs=0.1) == 86.63
        assert pytest.approx(hi, abs=0.1) == 88.50

    def test_empty_string(self):
        assert _ranges_from_freqsupport("") == []

    def test_no_match(self):
        assert _ranges_from_freqsupport("no frequency info here") == []

    def test_dotdot_format_does_not_match(self):
        """The '..' separator in ALMA bracket format is not matched by the regex."""
        result = _ranges_from_freqsupport("[86.63..88.50GHz,976.6kHz, XX YY]")
        assert result == []


# ---------------------------------------------------------------------------
# to_compact_ous
# ---------------------------------------------------------------------------

class TestToCompactOus:
    """to_compact_ous should convert UIDs to compact X####_X#### format."""

    def test_url_format(self):
        assert to_compact_ous("uid://A001/X3833/X64bc") == "X3833_X64bc"

    def test_underscore_format(self):
        assert to_compact_ous("uid___A001_X3833_X64bc") == "X3833_X64bc"

    def test_lowercase_preserves_case(self):
        """to_compact_ous preserves the case of the input."""
        result = to_compact_ous("uid://a001/x3833/x64bc")
        assert result == "x3833_x64bc"

    def test_uppercase_preserves_case(self):
        result = to_compact_ous("uid://A001/X3833/X64BC")
        assert result == "X3833_X64BC"

    def test_different_uid_values(self):
        assert to_compact_ous("uid://A001/X15a0/Xbb") == "X15a0_Xbb"

    def test_short_hex(self):
        assert to_compact_ous("uid://A001/X1/X2") == "X1_X2"

    def test_long_hex(self):
        assert to_compact_ous("uid://A001/Xabcdef/X123456") == "Xabcdef_X123456"

    def test_preserves_x_prefix(self):
        result = to_compact_ous("uid://A001/XABCD/XEF01")
        assert result.startswith("X")
        assert "_X" in result


# ---------------------------------------------------------------------------
# Line group inference
# ---------------------------------------------------------------------------

class TestLineGroupInference:
    """Line group is inferred from frequency in write_csv/build_index.

    The logic: format freq as string, check if it starts with "97." → N2H+,
    "87." → HCO+, else empty string.
    """

    def _infer_line_group(self, freq: float) -> str:
        """Replicate the line group inference logic from write_csv."""
        s = f"{freq:.3f}"
        if s.startswith("97."):
            return "N2H+"
        if s.startswith("87."):
            return "HCO+"
        return ""

    def test_n2hp_exact(self):
        assert self._infer_line_group(97.0) == "N2H+"

    def test_n2hp_with_decimals(self):
        assert self._infer_line_group(97.345) == "N2H+"

    def test_hcop_exact(self):
        assert self._infer_line_group(87.0) == "HCO+"

    def test_hcop_with_decimals(self):
        assert self._infer_line_group(87.921) == "HCO+"

    def test_other_frequency(self):
        assert self._infer_line_group(110.0) == ""

    def test_close_but_not_97(self):
        assert self._infer_line_group(96.999) == ""

    def test_close_but_not_87(self):
        assert self._infer_line_group(86.999) == ""

    def test_between_lines(self):
        assert self._infer_line_group(92.0) == ""
