"""Custom exception hierarchy for the Panta Rei pipeline."""

from __future__ import annotations


class PantaReiError(Exception):
    """Base exception for all Panta Rei pipeline errors."""


class ConfigError(PantaReiError):
    """Missing or invalid configuration."""


class AuthError(PantaReiError):
    """ALMA or GitHub authentication failure."""


class ALMAError(PantaReiError):
    """ALMA archive communication error."""


class ALMATimeoutError(ALMAError):
    """ALMA server timeout."""


class CorruptDataError(PantaReiError):
    """Corrupt data detected (zlib errors, bad tarballs)."""


class DatabaseError(PantaReiError):
    """SQLite operation failure."""
