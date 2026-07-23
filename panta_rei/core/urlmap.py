"""Bidirectional filesystem-path <-> public-URL mapping.

Centralizes the conversion between NAS filesystem paths (e.g.
``/scratch/almanas/...``) and their public web URLs (e.g.
``https://www.alma.ac.uk/nas/...``) so that all callers share one
validated implementation.

Matching is component-boundary safe: ``/scratch/almanas-other`` does
NOT match the ``/scratch/almanas`` prefix. URL matching is
scheme-insensitive (``http://`` and ``https://`` are equivalent).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Dict, Optional
from urllib.parse import unquote, urlsplit

# Default mapping used across the pipeline (see PipelineConfig.url_mappings).
DEFAULT_URL_MAPPINGS: Dict[str, str] = {
    "/scratch/almanas": "https://www.alma.ac.uk/nas",
}


def path_to_url(path: Path, url_mappings: Dict[str, str]) -> Optional[str]:
    """Convert a filesystem path to its public URL via configured mappings.

    Returns None if the path is not under any mapped prefix. Prefix
    matching respects path-component boundaries.
    """
    resolved = Path(path).resolve()
    for fs_prefix, url_prefix in url_mappings.items():
        try:
            rel = resolved.relative_to(fs_prefix)
        except ValueError:
            continue
        rel_str = rel.as_posix()
        base = url_prefix.rstrip("/")
        return base if rel_str == "." else f"{base}/{rel_str}"
    return None


def url_to_path(url: str, url_mappings: Dict[str, str]) -> Optional[Path]:
    """Convert a public URL back to its filesystem path.

    Scheme-insensitive (http/https treated as equivalent) and
    component-boundary safe on the URL path. URLs containing dot
    segments (``.`` or ``..``, plain or percent-encoded) are rejected
    outright so a mapped path can never escape its prefix. Returns None
    if the URL does not match any configured mapping.
    """
    if not url:
        return None
    target = urlsplit(url)
    if target.scheme not in ("http", "https") or not target.netloc:
        return None
    target_path = unquote(target.path)
    if any(part in (".", "..") for part in target_path.split("/")):
        return None
    for fs_prefix, url_prefix in url_mappings.items():
        base = urlsplit(url_prefix)
        if target.netloc.lower() != base.netloc.lower():
            continue
        base_path = PurePosixPath(base.path or "/")
        try:
            rel = PurePosixPath(target_path).relative_to(base_path)
        except ValueError:
            continue
        return Path(fs_prefix) / Path(*rel.parts)
    return None
