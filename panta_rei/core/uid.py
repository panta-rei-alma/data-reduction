"""UID parsing and normalization for ALMA Member Observing Unit Set identifiers.

Handles conversion between formats:
    uid://A001/X123/X456   (ALMA canonical URL format)
    uid___A001_X123_X456   (filesystem-safe format)
    uid___a001_x123_x456   (lowercase canonical for DB lookups)
"""

from __future__ import annotations

import re
from typing import Optional

# Regex pattern for ALMA UIDs - handles various formats:
#   uid___A001_X123_X456
#   uid__A001_X123_X456 (single underscore variant)
#   uid://A001/X123/X456
# Case-insensitive for both the 'A' prefix and 'X' hex markers
UID_CORE_RE = re.compile(
    r"(uid___?[aA]\d{3}_[xX][0-9a-fA-F]+_[xX][0-9a-fA-F]+)",
    re.IGNORECASE,
)


def canonical_uid(uid: str) -> Optional[str]:
    """Convert a UID to canonical lowercase form for database lookups.

    Handles various input formats:
        uid://A001/X123/X456 -> uid___a001_x123_x456
        uid___A001_X123_X456 -> uid___a001_x123_x456
        UID___A001_X123_X456 -> uid___a001_x123_x456

    Returns None if input is empty/invalid.
    """
    from panta_rei.core.text import as_text

    s = as_text(uid)
    if not s:
        return None
    if "://" in s:
        s = s.replace("://", "___").replace("/", "_")
    m = UID_CORE_RE.search(s)
    if m:
        return m.group(1).lower()
    if s.lower().startswith("uid___"):
        return s.lower()
    return None


def sanitize_uid(uid: str) -> str:
    """Sanitize a UID for use as a directory or filename.

    Converts various UID formats to a consistent underscore format
    with normalized casing (uppercase A, uppercase X with lowercase hex):
        uid://A001/X123/X456 -> uid___A001_X123_X456
        uid___a001_x123_x456 -> uid___A001_X123_X456

    Returns empty string if input is empty/invalid.
    """
    if not uid:
        return ""

    s = uid.strip()
    s = s.replace("://", "___").replace("/", "_")

    m = UID_CORE_RE.search(s)
    if m:
        core = m.group(1)
        parts = core.split("_")
        normalized_parts = []
        for p in parts:
            if p.lower().startswith("a") and len(p) == 4:
                # Archive identifier (e.g., A001) - uppercase
                normalized_parts.append(p.upper())
            elif p.lower().startswith("x"):
                # Hex component (e.g., X123) - X uppercase, hex lowercase
                normalized_parts.append("X" + p[1:].lower())
            else:
                normalized_parts.append(p.lower())
        return "_".join(normalized_parts)

    return s


def extract_uid_from_path(path) -> Optional[str]:
    """Extract and sanitize a UID from a file or directory path.

    Returns sanitized UID, or None if no UID found.
    """
    s = str(path)
    m = UID_CORE_RE.search(s)
    if m:
        return sanitize_uid(m.group(1))
    return None
