"""Text conversion and timestamp utilities."""

from __future__ import annotations

import datetime as dt

import numpy as np


def as_text(x) -> str:
    """Convert a value to a string, handling bytes and numpy types.

    Safely handles:
        None -> ""
        bytes/np.bytes_ -> decoded string (UTF-8 with Latin-1 fallback)
        np.generic -> str(x.item())
        Everything else -> str(x)
    """
    if x is None:
        return ""
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            return x.decode("latin-1", errors="ignore")
    if isinstance(x, np.generic):
        return str(x.item())
    return str(x)


def now_iso() -> str:
    """Get current UTC time as ISO 8601 string.

    Returns timestamp in format: 2025-01-20T12:34:56Z
    """
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
