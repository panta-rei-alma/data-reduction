"""Resolve the TP cube corresponding to a 12m7mTP feathered output.

Lookup chain:

1. Parse ``gous_id``, source name (sanitized), and freq range from the
   feathered FITS filename.
2. Read ``targets_by_array.csv`` to map ``(gous_id, source, array='TP')``
   → TP ``mous_id``.
3. List candidate TP cubes under the matching member directory.
4. Select the cube whose frequency range best overlaps the feathered cube.

Returns ``None`` (with a warning log) at any step where the lookup fails;
callers fall back to plotting the feathered spectrum on its own.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

from panta_rei.imaging.matching import find_tp_cube

logger = logging.getLogger(__name__)


_GROUP_DIR_RE = re.compile(r"^group\.uid___A001_(?P<gous>[^.]+)\.")
_FEATHERED_NAME_RE = re.compile(
    r"\.lp_nperetto\.(?P<src>.+?)\.12m7mTP\."
    r"(?P<lo>[\d.]+)-(?P<hi>[\d.]+)GHz\.cube\.pbcor\.fits$"
)


def parse_feathered_filename(feathered_fits: Path) -> dict:
    """Return ``{gous_id, source_sanitized, freq_lo_ghz, freq_hi_ghz}``.

    Raises ``ValueError`` if the filename doesn't follow the convention.
    """
    g_match = _GROUP_DIR_RE.match(feathered_fits.parent.name)
    if g_match is None:
        raise ValueError(
            f"cannot parse GOUS id from group dir: {feathered_fits.parent.name}"
        )
    n_match = _FEATHERED_NAME_RE.search(feathered_fits.name)
    if n_match is None:
        raise ValueError(
            f"cannot parse source/freq from feathered name: {feathered_fits.name}"
        )
    return {
        "gous_id": g_match.group("gous"),
        "source_sanitized": n_match.group("src"),
        "freq_lo_ghz": float(n_match.group("lo")),
        "freq_hi_ghz": float(n_match.group("hi")),
    }


def desanitize_source_name(sanitized: str) -> str:
    """Invert ``imaging.matching.sanitize_source_name``.

    Source names look like ``AG221.9599-1.9932`` or ``G034.997+0.330``;
    the sanitized form replaces ``-`` / ``+`` with ``m`` / ``p``. Only
    numeric-context ``m``/``p`` get swapped back, so alphabetic tokens
    like ``AG`` are untouched.
    """
    return re.sub(
        r"(?<=\d)([pm])(?=\d)",
        lambda m: {"p": "+", "m": "-"}[m.group(1)],
        sanitized,
    )


def lookup_tp_mous_id(
    targets_csv: Path, gous_id: str, source_name: str,
) -> Optional[str]:
    """Return the TP MOUS id for ``(gous_id, source)`` from the targets CSV, or None."""
    with open(targets_csv, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("gous_id") == gous_id
                and row.get("source_name") == source_name
                and row.get("array") == "TP"
            ):
                return row.get("mous_ids")
    return None


def find_tp_cube_for_feathered(
    feathered_fits: Path,
    project_dir: Path,
    targets_csv: Path,
) -> Optional[Path]:
    """Resolve the TP cube path for a 12m7mTP feathered output.

    ``project_dir`` is the project root (the one containing
    ``targets_by_array.csv`` and the nested ``<project_code>/`` ALMA
    data tree).
    """
    try:
        parsed = parse_feathered_filename(feathered_fits)
    except ValueError as exc:
        logger.warning("TP lookup: %s", exc)
        return None

    source = desanitize_source_name(parsed["source_sanitized"])
    gous = parsed["gous_id"]

    if not targets_csv.is_file():
        logger.warning("TP lookup: targets CSV missing at %s", targets_csv)
        return None

    tp_mous = lookup_tp_mous_id(targets_csv, gous, source)
    if tp_mous is None:
        logger.warning(
            "TP lookup: no TP row in CSV for gous=%s source=%s", gous, source,
        )
        return None

    data_dir = project_dir / project_dir.name
    tp_cubes = find_tp_cube(data_dir, tp_mous, source)
    if not tp_cubes:
        logger.warning(
            "TP lookup: TP MOUS %s has no cubes under %s", tp_mous, data_dir,
        )
        return None

    lo_hz = parsed["freq_lo_ghz"] * 1e9
    hi_hz = parsed["freq_hi_ghz"] * 1e9
    overlapping = [
        c for c in tp_cubes
        if c["freq_min"] < hi_hz and c["freq_max"] > lo_hz
    ]
    if not overlapping:
        logger.warning(
            "TP lookup: no TP cube overlaps %.3f-%.3f GHz for gous=%s source=%s",
            parsed["freq_lo_ghz"], parsed["freq_hi_ghz"], gous, source,
        )
        return None

    target_center = 0.5 * (lo_hz + hi_hz)
    best = min(
        overlapping,
        key=lambda c: abs(0.5 * (c["freq_min"] + c["freq_max"]) - target_center),
    )
    return Path(best["file"])
