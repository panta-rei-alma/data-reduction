"""Tests for panta_rei.core.text — text conversion and timestamps."""

import re

import numpy as np
import pytest

from panta_rei.core.text import as_text, now_iso


# ---------------------------------------------------------------------------
# as_text
# ---------------------------------------------------------------------------

class TestAsText:
    """as_text should safely convert various types to strings."""

    def test_regular_string(self):
        assert as_text("hello") == "hello"

    def test_none(self):
        assert as_text(None) == ""

    def test_bytes_utf8(self):
        assert as_text(b"hello") == "hello"

    def test_bytes_latin1_fallback(self):
        # Latin-1 encoded bytes that are not valid UTF-8
        latin1_bytes = b"\xe9\xe8\xea"  # e-acute, e-grave, e-circumflex
        result = as_text(latin1_bytes)
        assert isinstance(result, str)
        assert len(result) == 3

    def test_numpy_bytes(self):
        assert as_text(np.bytes_(b"hello")) == "hello"

    def test_numpy_int(self):
        assert as_text(np.int64(42)) == "42"

    def test_numpy_float(self):
        assert as_text(np.float64(3.14)) == "3.14"

    def test_numpy_str(self):
        assert as_text(np.str_("hello")) == "hello"

    def test_integer(self):
        assert as_text(42) == "42"

    def test_float(self):
        assert as_text(3.14) == "3.14"

    def test_empty_string(self):
        assert as_text("") == ""

    def test_empty_bytes(self):
        assert as_text(b"") == ""


# ---------------------------------------------------------------------------
# now_iso
# ---------------------------------------------------------------------------

class TestNowIso:
    """now_iso should return a UTC ISO 8601 timestamp."""

    def test_format(self):
        result = now_iso()
        # Should match YYYY-MM-DDTHH:MM:SSZ
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", result)

    def test_ends_with_z(self):
        assert now_iso().endswith("Z")

    def test_no_microseconds(self):
        result = now_iso()
        # Should not contain a decimal point (no microseconds)
        assert "." not in result

    def test_returns_string(self):
        assert isinstance(now_iso(), str)
