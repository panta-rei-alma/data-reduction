"""Safe tar extraction utilities."""

from __future__ import annotations

import logging
import os
import tarfile
from pathlib import Path

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


def safe_extract_tar(tf: tarfile.TarFile, dest_dir: Path) -> tuple[int, int]:
    """Safely extract a tarball, skipping path traversal attempts and existing files.

    Returns (extracted_count, skipped_count).
    """
    extracted = 0
    skipped = 0
    for member in tf.getmembers():
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
