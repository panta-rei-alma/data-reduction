"""Safe tar extraction utilities."""

from __future__ import annotations

import logging
import os
import tarfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _is_within_directory(base: Path, target: Path) -> bool:
    """Check that target path is within base directory (path traversal guard)."""
    try:
        base = base.resolve()
        target = target.resolve()
    except FileNotFoundError:
        base = base.absolute()
        target = target.absolute()
    return str(target) == str(base) or str(target).startswith(str(base) + os.sep)


def safe_extract_tar(
    tf: tarfile.TarFile,
    dest_dir: Path,
    *,
    strip_top_level: Optional[str] = None,
) -> tuple[int, int]:
    """Safely extract a tarball, skipping path traversal attempts and existing files.

    If *strip_top_level* is provided and **every** non-empty member's
    name starts with ``<strip_top_level>/`` (or equals the bare top
    level), that prefix is removed from each member before extraction.
    This handles ALMA archive tarballs whose internal layout is
    ``<project_code>/<sg>/<g>/<member>/…`` — without stripping, those
    extract into ``<dest_dir>/<project_code>/<sg>/…`` and produce the
    triple-nested ``…/<project>/<project>/<project>/<sg>/…`` layout
    when *dest_dir* is itself the project's data_dir
    (``<base>/<project>``).

    The strip is conditional on *all-or-nothing* matching: if any member
    falls outside the prefix, no stripping happens (defence against
    mixed-content tarballs that we don't yet know about).

    Returns ``(extracted_count, skipped_count)``.
    """
    members = tf.getmembers()

    if strip_top_level:
        bare = strip_top_level.rstrip("/")
        prefix = bare + "/"
        all_prefixed = all(
            (not m.name) or m.name == bare or m.name.startswith(prefix)
            for m in members
        )
        if all_prefixed:
            log.info(
                "tarball wraps under %r; stripping top-level prefix on extract",
                bare,
            )
            for m in members:
                if m.name == bare:
                    m.name = ""  # placeholder, will be skipped below
                elif m.name.startswith(prefix):
                    m.name = m.name[len(prefix):]
        else:
            log.debug(
                "strip_top_level=%r requested but tarball is not uniformly "
                "wrapped — extracting members verbatim",
                bare,
            )

    extracted = 0
    skipped = 0
    for member in members:
        if not member.name:
            continue  # placeholder from a fully-stripped top-level dir
        member_path = dest_dir / member.name
        if not _is_within_directory(dest_dir, member_path):
            log.warning("Skipping unsafe member path: %s", member.name)
            continue
        if member.isdir():
            (dest_dir / member.name).mkdir(parents=True, exist_ok=True)
            continue
        if (dest_dir / member.name).exists():
            skipped += 1
            continue
        tf.extract(member, path=dest_dir)
        extracted += 1
        if extracted and extracted % 50 == 0:
            log.info("  Extracted %d files...", extracted)
    return extracted, skipped
