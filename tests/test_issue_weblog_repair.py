"""Tests for weblog link handling in GitHub issue bodies.

Covers the removal of path-derived URL fabrication (the issue #21 root
cause), exact-UID weblog lookup, and the guarded self-healing of broken
weblog links in ``update_issue``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from panta_rei.github.issues import (
    GitHubIssueManager,
    SchedulingBlock,
    build_sb_issue_body,
)

URL_PREFIX = "https://test.example/nas"


def _make_manager(tmp_path: Path, weblog_dir: Path) -> GitHubIssueManager:
    with patch("ghapi.all.GhApi"):
        return GitHubIssueManager(
            project_code="2025.1.00383.L",
            base_dir=tmp_path,
            gh_owner="owner",
            gh_repo="repo",
            gh_token="dummy-token",
            weblog_dir=weblog_dir,
            dry_run=True,
            url_mappings={str(tmp_path): URL_PREFIX},
        )


def _sb(**kwargs) -> SchedulingBlock:
    defaults = dict(
        sb_name="AG000.00_a_00_TP",
        array="TP",
        gous_id="X1_X2",
        mous_ids=["X3833_X64d8"],
        targets={"AG000.00+0.00"},
    )
    defaults.update(kwargs)
    return SchedulingBlock(**defaults)


def _body_with_weblog_line(line: str) -> str:
    return (
        "## Scheduling Block: AG000.00_a_00_TP\n\n"
        "### Data Status\n\n"
        "* [x] Delivered\n"
        "* [x] Downloaded\n"
        "* [x] Extracted\n"
        f"{line}\n\n"
        "### Quality Assessment\n\n"
        "* [ ] Weblog reviewed\n"
    )


@pytest.fixture
def staged(tmp_path):
    """A staged weblog tree with one valid and one absent pipeline dir."""
    weblog_dir = tmp_path / "weblogs"
    good = weblog_dir / "uid___A001_X3833_X64d8" / "pipeline-NEW" / "html"
    good.mkdir(parents=True)
    (good / "index.html").write_text("<html></html>")
    return {
        "weblog_dir": weblog_dir,
        "good_url": f"{URL_PREFIX}/weblogs/uid___A001_X3833_X64d8/pipeline-NEW/html/index.html",
        "bad_url": f"{URL_PREFIX}/weblogs/uid___A001_X3833_X64d8/pipeline-OLD/html/index.html",
    }


# ---------------------------------------------------------------------------
# build_sb_issue_body: no URL fabrication from paths
# ---------------------------------------------------------------------------

class TestNoUrlFabrication:
    def test_weblog_url_used_verbatim(self):
        sb = _sb(weblog_url="http://x/y/html/index.html")
        assert "* [x] [Weblog](http://x/y/html/index.html) available" in (
            build_sb_issue_body(sb)
        )

    def test_path_without_url_renders_plain_path(self):
        sb = _sb(weblog_path=Path("/local/scratch/pipeline-123"))
        body = build_sb_issue_body(sb)
        assert "[Weblog](" not in body
        assert "* [x] Weblog available at: `/local/scratch/pipeline-123`" in body

    def test_no_weblog_renders_unchecked(self):
        assert "* [ ] Weblog available" in build_sb_issue_body(_sb())


# ---------------------------------------------------------------------------
# find_weblog: exact canonical UID matching
# ---------------------------------------------------------------------------

class TestFindWeblogExactMatch:
    def _weblog_info(self):
        return {
            "uid___a001_x3833_x64d80": ("/w/uid___A001_X3833_X64d80/p", "http://u/80"),
            "uid___a001_x3833_x64d8": ("/w/uid___A001_X3833_X64d8/p", "http://u/8"),
        }

    def test_compact_id_matches_exactly(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(mous_ids=["X3833_X64d8"])
        path, url = mgr.find_weblog(sb, self._weblog_info())
        # Must match x64d8 exactly, never the x64d80 entry via substring
        assert path == Path("/w/uid___A001_X3833_X64d8/p")
        assert url == "http://u/8/html/index.html"

    def test_full_uid_form_matches(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(mous_ids=["uid___A001_X3833_X64d8"])
        path, _ = mgr.find_weblog(sb, self._weblog_info())
        assert path == Path("/w/uid___A001_X3833_X64d8/p")

    def test_no_match_returns_none(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(mous_ids=["X3833_Xffff"])
        assert mgr.find_weblog(sb, self._weblog_info()) == (None, None)

    def test_empty_info_returns_none(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        assert mgr.find_weblog(_sb(), {}) == (None, None)


# ---------------------------------------------------------------------------
# _repair_weblog_link: guarded self-healing
# ---------------------------------------------------------------------------

class TestRepairWeblogLink:
    def test_repairs_broken_link(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(weblog_url=staged["good_url"])
        body = _body_with_weblog_line(
            f"* [x] [Weblog]({staged['bad_url']}) available"
        )
        repaired = mgr._repair_weblog_link(sb, body, 21)
        assert repaired is not None
        assert staged["good_url"] in repaired
        assert staged["bad_url"] not in repaired
        # Only the weblog line changed
        assert repaired.replace(staged["good_url"], staged["bad_url"]) == body

    def test_working_link_never_touched(self, tmp_path, staged):
        """A link whose target exists on disk is left alone, even if it
        differs from the staged URL (protects deliberate manual edits)."""
        other = staged["weblog_dir"] / "uid___A001_X9_X9" / "html"
        other.mkdir(parents=True)
        (other / "index.html").write_text("x")
        other_url = f"{URL_PREFIX}/weblogs/uid___A001_X9_X9/html/index.html"
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(weblog_url=staged["good_url"])
        body = _body_with_weblog_line(f"* [x] [Weblog]({other_url}) available")
        assert mgr._repair_weblog_link(sb, body, 21) is None

    def test_external_url_never_touched(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(weblog_url=staged["good_url"])
        body = _body_with_weblog_line(
            "* [x] [Weblog](https://elsewhere.example/weblog.html) available"
        )
        assert mgr._repair_weblog_link(sb, body, 21) is None

    def test_url_outside_weblog_dir_never_touched(self, tmp_path, staged):
        # Maps under the fs prefix but not under weblog_dir
        url = f"{URL_PREFIX}/other-tree/foo/html/index.html"
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(weblog_url=staged["good_url"])
        body = _body_with_weblog_line(f"* [x] [Weblog]({url}) available")
        assert mgr._repair_weblog_link(sb, body, 21) is None

    def test_unverifiable_replacement_blocks_repair(self, tmp_path, staged):
        missing = (
            f"{URL_PREFIX}/weblogs/uid___A001_X3833_X64d8/"
            "pipeline-MISSING/html/index.html"
        )
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(weblog_url=missing)
        body = _body_with_weblog_line(
            f"* [x] [Weblog]({staged['bad_url']}) available"
        )
        assert mgr._repair_weblog_link(sb, body, 21) is None

    def test_missing_weblog_dir_blocks_repair(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, tmp_path / "not-mounted")
        sb = _sb(weblog_url=staged["good_url"])
        body = _body_with_weblog_line(
            f"* [x] [Weblog]({staged['bad_url']}) available"
        )
        assert mgr._repair_weblog_link(sb, body, 21) is None

    def test_empty_weblog_dir_blocks_repair(self, tmp_path):
        """An empty weblog dir looks unmounted; repair must not fire."""
        empty = tmp_path / "weblogs-empty"
        empty.mkdir()
        mgr = _make_manager(tmp_path, empty)
        sb = _sb(weblog_url=f"{URL_PREFIX}/weblogs-empty/u/html/index.html")
        body = _body_with_weblog_line(
            f"* [x] [Weblog]({URL_PREFIX}/weblogs-empty/old/html/index.html) available"
        )
        assert mgr._repair_weblog_link(sb, body, 21) is None

    def test_duplicate_weblog_lines_block_repair(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(weblog_url=staged["good_url"])
        line = f"* [x] [Weblog]({staged['bad_url']}) available"
        body = _body_with_weblog_line(line) + f"\n{line}\n"
        assert mgr._repair_weblog_link(sb, body, 21) is None

    def test_no_staged_url_blocks_repair(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb()  # no weblog_url
        body = _body_with_weblog_line(
            f"* [x] [Weblog]({staged['bad_url']}) available"
        )
        assert mgr._repair_weblog_link(sb, body, 21) is None

    def test_identical_url_no_repair(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(weblog_url=staged["good_url"])
        body = _body_with_weblog_line(
            f"* [x] [Weblog]({staged['good_url']}) available"
        )
        assert mgr._repair_weblog_link(sb, body, 21) is None

    def test_no_weblog_line_no_repair(self, tmp_path, staged):
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(weblog_url=staged["good_url"])
        body = _body_with_weblog_line("* [ ] Weblog available")
        assert mgr._repair_weblog_link(sb, body, 21) is None

    def test_http_https_scheme_mismatch_still_repairs(self, tmp_path, staged):
        """Bodies use http:// while mappings declare https:// (or vice
        versa); scheme differences must not block the repair."""
        bad_http = staged["bad_url"].replace("https://", "http://")
        mgr = _make_manager(tmp_path, staged["weblog_dir"])
        sb = _sb(weblog_url=staged["good_url"])
        body = _body_with_weblog_line(f"* [x] [Weblog]({bad_http}) available")
        repaired = mgr._repair_weblog_link(sb, body, 21)
        assert repaired is not None
        assert staged["good_url"] in repaired
