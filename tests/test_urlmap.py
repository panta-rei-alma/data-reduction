"""Tests for the shared path<->URL mapping helpers."""

from __future__ import annotations

from pathlib import Path

from panta_rei.core.urlmap import path_to_url, url_to_path

MAPPINGS = {"/scratch/almanas": "https://www.alma.ac.uk/nas"}


class TestPathToUrl:
    def test_basic_mapping(self, tmp_path):
        url = path_to_url(
            Path("/scratch/almanas/dwalker2/panta-rei/weblogs/uid___A001_X1_X2"),
            MAPPINGS,
        )
        assert url == (
            "https://www.alma.ac.uk/nas/dwalker2/panta-rei/weblogs/uid___A001_X1_X2"
        )

    def test_component_boundary_not_matched(self):
        # /scratch/almanas-other must NOT match the /scratch/almanas prefix
        assert path_to_url(Path("/scratch/almanas-other/foo"), MAPPINGS) is None

    def test_unmapped_path_returns_none(self):
        assert path_to_url(Path("/data/elsewhere/foo"), MAPPINGS) is None

    def test_prefix_itself_maps_to_base(self):
        assert path_to_url(Path("/scratch/almanas"), MAPPINGS) == (
            "https://www.alma.ac.uk/nas"
        )


class TestUrlToPath:
    def test_basic_mapping(self):
        p = url_to_path(
            "https://www.alma.ac.uk/nas/dwalker2/panta-rei/weblogs/x/html/index.html",
            MAPPINGS,
        )
        assert p == Path(
            "/scratch/almanas/dwalker2/panta-rei/weblogs/x/html/index.html"
        )

    def test_scheme_insensitive(self):
        # Issue bodies use http:// while the mapping is declared https://
        p = url_to_path("http://www.alma.ac.uk/nas/dwalker2/foo", MAPPINGS)
        assert p == Path("/scratch/almanas/dwalker2/foo")

    def test_component_boundary_not_matched(self):
        # /nastier must not match the /nas URL prefix
        assert url_to_path("https://www.alma.ac.uk/nastier/foo", MAPPINGS) is None

    def test_other_host_returns_none(self):
        assert url_to_path("https://example.org/nas/foo", MAPPINGS) is None

    def test_non_http_scheme_returns_none(self):
        assert url_to_path("ftp://www.alma.ac.uk/nas/foo", MAPPINGS) is None
        assert url_to_path("", MAPPINGS) is None
        assert url_to_path(None, MAPPINGS) is None

    def test_roundtrip(self):
        original = Path("/scratch/almanas/dwalker2/panta-rei/weblogs/u/html/index.html")
        assert url_to_path(path_to_url(original, MAPPINGS), MAPPINGS) == original
