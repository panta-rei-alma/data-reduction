"""Tests for panta_rei.core.uid — UID parsing and normalization."""

import pytest

from panta_rei.core.uid import UID_CORE_RE, canonical_uid, extract_uid_from_path, sanitize_uid


# ---------------------------------------------------------------------------
# canonical_uid
# ---------------------------------------------------------------------------

class TestCanonicalUid:
    """canonical_uid should return fully lowercase uid___... format."""

    def test_url_format(self):
        assert canonical_uid("uid://A001/X123/X456") == "uid___a001_x123_x456"

    def test_underscore_format(self):
        assert canonical_uid("uid___A001_X123_X456") == "uid___a001_x123_x456"

    def test_already_lowercase(self):
        assert canonical_uid("uid___a001_x123_x456") == "uid___a001_x123_x456"

    def test_mixed_case(self):
        assert canonical_uid("uid__A001_X123_x456") == "uid__a001_x123_x456"

    def test_uppercase_uid_prefix(self):
        assert canonical_uid("UID___A001_X123_X456") == "uid___a001_x123_x456"

    def test_real_uid_from_live_data(self):
        # A real UID from the Panta Rei project
        assert canonical_uid("uid://A001/X3833/X64bc") == "uid___a001_x3833_x64bc"

    def test_real_uid_sanitized_format(self):
        assert canonical_uid("uid___A001_X3833_X64bc") == "uid___a001_x3833_x64bc"

    def test_none_input(self):
        assert canonical_uid(None) is None

    def test_empty_string(self):
        assert canonical_uid("") is None

    def test_garbage_input(self):
        assert canonical_uid("not_a_uid_at_all") is None

    def test_partial_uid(self):
        # The fallback path returns lowercase if it starts with uid___
        # even without a full UID structure — this is existing behavior
        assert canonical_uid("uid___") == "uid___"

    def test_bytes_input(self):
        assert canonical_uid(b"uid://A001/X123/X456") == "uid___a001_x123_x456"

    def test_numpy_bytes(self):
        import numpy as np
        assert canonical_uid(np.bytes_(b"uid://A001/X123/X456")) == "uid___a001_x123_x456"

    def test_longer_hex_values(self):
        assert canonical_uid("uid://A001/X3833/X64b8") == "uid___a001_x3833_x64b8"


# ---------------------------------------------------------------------------
# sanitize_uid
# ---------------------------------------------------------------------------

class TestSanitizeUid:
    """sanitize_uid should produce filesystem-safe format with normalized casing."""

    def test_url_format(self):
        assert sanitize_uid("uid://A001/X123/X456") == "uid___A001_X123_X456"

    def test_lowercase_input(self):
        result = sanitize_uid("uid___a001_x123_x456")
        assert result == "uid___A001_X123_X456"

    def test_mixed_case(self):
        result = sanitize_uid("uid___a001_X123_x456")
        assert result == "uid___A001_X123_X456"

    def test_already_correct(self):
        assert sanitize_uid("uid___A001_X123_X456") == "uid___A001_X123_X456"

    def test_hex_lowercase(self):
        """Hex digits after X should be lowercase."""
        result = sanitize_uid("uid___A001_XABCD_XEFGH")
        # XEFGH doesn't match the regex (G, H not hex), so falls through
        # This tests the non-matching path
        assert sanitize_uid("uid___A001_XABCD_XEF01") == "uid___A001_Xabcd_Xef01"

    def test_empty_string(self):
        assert sanitize_uid("") == ""

    def test_none_input(self):
        assert sanitize_uid(None) == ""

    def test_whitespace_stripped(self):
        assert sanitize_uid("  uid://A001/X123/X456  ") == "uid___A001_X123_X456"

    def test_real_uid(self):
        assert sanitize_uid("uid://A001/X3833/X64bc") == "uid___A001_X3833_X64bc"

    def test_double_underscore_variant(self):
        """uid__A001_X123_X456 (single underscore variant) should still work."""
        result = sanitize_uid("uid__A001_X123_X456")
        assert result == "uid__A001_X123_X456"


# ---------------------------------------------------------------------------
# extract_uid_from_path
# ---------------------------------------------------------------------------

class TestExtractUidFromPath:
    """extract_uid_from_path should find and sanitize UIDs embedded in paths."""

    def test_member_path(self):
        path = "science_goal.uid___A001_X3833_X64b8/group.uid___A001_X3833_X64b9/member.uid___A001_X3833_X64bc"
        result = extract_uid_from_path(path)
        # Should extract the first UID found
        assert result == "uid___A001_X3833_X64b8"

    def test_tar_filename(self):
        path = "/data/tars/uid___A001_X3833_X64bc.tar"
        result = extract_uid_from_path(path)
        assert result == "uid___A001_X3833_X64bc"

    def test_pathlib_path(self):
        from pathlib import Path
        path = Path("/data/member.uid___A001_X123_X456/calibrated")
        result = extract_uid_from_path(path)
        assert result == "uid___A001_X123_X456"

    def test_no_uid_in_path(self):
        assert extract_uid_from_path("/data/some/random/path") is None

    def test_empty_path(self):
        assert extract_uid_from_path("") is None


# ---------------------------------------------------------------------------
# UID_CORE_RE
# ---------------------------------------------------------------------------

class TestUidRegex:
    """Verify the regex matches expected patterns."""

    def test_triple_underscore(self):
        assert UID_CORE_RE.search("uid___A001_X123_X456") is not None

    def test_double_underscore(self):
        assert UID_CORE_RE.search("uid__A001_X123_X456") is not None

    def test_no_match_random(self):
        assert UID_CORE_RE.search("random_string") is None

    def test_case_insensitive(self):
        assert UID_CORE_RE.search("UID___A001_X123_X456") is not None
        assert UID_CORE_RE.search("uid___a001_x123_x456") is not None
