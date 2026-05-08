"""Tests for panta_rei.core.tar — tarball extraction helpers."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from panta_rei.core.tar import safe_extract_tar


def _make_tar(tmp_path: Path, members: dict[str, bytes]) -> Path:
    """Build a tarball at *tmp_path/archive.tar* with the given member
    name → bytes mapping (or empty bytes for directory entries)."""
    tar_path = tmp_path / "archive.tar"
    with tarfile.open(tar_path, "w") as tf:
        for name, content in members.items():
            if content is None:
                # Directory entry
                ti = tarfile.TarInfo(name=name)
                ti.type = tarfile.DIRTYPE
                ti.mode = 0o755
                tf.addfile(ti)
            else:
                ti = tarfile.TarInfo(name=name)
                ti.size = len(content)
                ti.mode = 0o644
                tf.addfile(ti, io.BytesIO(content))
    return tar_path


class TestSafeExtractTar:
    def test_basic_extract(self, tmp_path):
        """Without strip_top_level, members extract verbatim."""
        tar = _make_tar(tmp_path, {
            "file1.txt": b"hello",
            "subdir/file2.txt": b"world",
        })
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(tar) as tf:
            extracted, skipped = safe_extract_tar(tf, dest)
        assert extracted == 2
        assert (dest / "file1.txt").read_bytes() == b"hello"
        assert (dest / "subdir/file2.txt").read_bytes() == b"world"

    def test_strip_top_level_removes_uniform_prefix(self, tmp_path):
        """When all members share a top-level dir matching strip_top_level,
        the prefix is removed during extraction.  This is the ALMA
        archive case: tarball wraps under <project_code>/."""
        tar = _make_tar(tmp_path, {
            "2025.1.00383.L/": None,
            "2025.1.00383.L/science_goal.uid___A001_X3833_X65c4/": None,
            "2025.1.00383.L/science_goal.uid___A001_X3833_X65c4/group.uid___A001_X3833_X65c5/": None,
            "2025.1.00383.L/science_goal.uid___A001_X3833_X65c4/group.uid___A001_X3833_X65c5/data.txt": b"science",
        })
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(tar) as tf:
            extracted, skipped = safe_extract_tar(
                tf, dest, strip_top_level="2025.1.00383.L",
            )
        # The placeholder root dir entry produces 0 extracts; the
        # nested dirs are mkdir'd and don't count as extracted.  The
        # leaf file extracts.
        assert extracted == 1
        # Files now live at dest/<sg>/<g>/, not dest/<project>/<sg>/<g>/.
        assert (dest / "science_goal.uid___A001_X3833_X65c4" /
                "group.uid___A001_X3833_X65c5" / "data.txt").read_bytes() == b"science"
        # The project_code dir level should NOT exist at dest.
        assert not (dest / "2025.1.00383.L").exists()

    def test_strip_no_op_when_prefix_absent(self, tmp_path):
        """If members don't share the prefix, strip is a no-op (members
        extract verbatim)."""
        tar = _make_tar(tmp_path, {
            "science_goal.uid___A001_X3833_X65c4/data.txt": b"hi",
            "other.txt": b"unrelated",
        })
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(tar) as tf:
            extracted, skipped = safe_extract_tar(
                tf, dest, strip_top_level="2025.1.00383.L",
            )
        assert extracted == 2
        assert (dest / "science_goal.uid___A001_X3833_X65c4" / "data.txt").exists()
        assert (dest / "other.txt").exists()

    def test_strip_no_op_on_mixed_content(self, tmp_path):
        """If SOME members have the prefix but others don't, do not
        strip — extracts everything verbatim (defensive)."""
        tar = _make_tar(tmp_path, {
            "2025.1.00383.L/inside.txt": b"a",
            "outside.txt": b"b",  # no prefix
        })
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(tar) as tf:
            extracted, skipped = safe_extract_tar(
                tf, dest, strip_top_level="2025.1.00383.L",
            )
        # Both extracted at their declared paths.
        assert extracted == 2
        assert (dest / "2025.1.00383.L" / "inside.txt").exists()
        assert (dest / "outside.txt").exists()

    def test_strip_preserves_path_traversal_guard(self, tmp_path):
        """The strip step must not let path-traversal members slip
        through — the guard still applies after stripping."""
        tar = _make_tar(tmp_path, {
            "2025.1.00383.L/../escapee.txt": b"oops",
            "2025.1.00383.L/legit.txt": b"ok",
        })
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(tar) as tf:
            extracted, skipped = safe_extract_tar(
                tf, dest, strip_top_level="2025.1.00383.L",
            )
        # Traversal member is rejected by the guard; legit one extracts.
        assert (dest / "legit.txt").exists()
        assert not (tmp_path / "escapee.txt").exists()

    def test_strip_handles_trailing_slash_arg(self, tmp_path):
        """strip_top_level='X/' should work the same as 'X'."""
        tar = _make_tar(tmp_path, {
            "X/a.txt": b"a",
        })
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(tar) as tf:
            extracted, skipped = safe_extract_tar(tf, dest, strip_top_level="X/")
        assert (dest / "a.txt").read_bytes() == b"a"
        assert not (dest / "X").exists()

    def test_strip_skips_existing_files(self, tmp_path):
        """Existing files in dest are skipped (counted), not overwritten."""
        tar = _make_tar(tmp_path, {
            "X/a.txt": b"new",
        })
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "a.txt").write_bytes(b"old")
        with tarfile.open(tar) as tf:
            extracted, skipped = safe_extract_tar(tf, dest, strip_top_level="X")
        assert extracted == 0
        assert skipped == 1
        assert (dest / "a.txt").read_bytes() == b"old"
