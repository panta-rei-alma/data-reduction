"""ALMA metadata: array classification, frequency parsing, SB family, table building.

Extracted from make_mous_table.py — queries the ALMA archive for
scheduling-block metadata, classifies arrays (TM / SM / TP), and writes
the ``targets_by_array.csv`` index file.
"""

from __future__ import annotations

import csv
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from panta_rei.auth import login_alma
from panta_rei.core.text import as_text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SB_SUFFIX_RE = re.compile(r"(_tm\d*|_7m|_tp)$", re.IGNORECASE)
_FREQ_PAIR_RE = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?)\s*[-?]\s*(?P<hi>\d+(?:\.\d+)?)\s*(?P<Unit>GHz|MHz|kHz)?",
    re.IGNORECASE,
)

# ALMA mirror URLs — fallback order
ALMA_SERVERS = [
    "https://almascience.nrao.edu",
    "https://almascience.eso.org",
    "https://almascience.nao.ac.jp",
]


# ---------------------------------------------------------------------------
# SB family / array helpers
# ---------------------------------------------------------------------------

def sb_family(sb: str) -> str:
    """Strip the array suffix (``_tm``, ``_7m``, ``_tp``) from a scheduling-block name."""
    return _SB_SUFFIX_RE.sub("", as_text(sb).strip())


def classify_array(sb_name: str, antenna_arrays: str = "") -> Optional[str]:
    """Return ``"TM"``, ``"SM"``, ``"TP"``, or ``None`` based on SB name / antenna arrays."""
    sb = as_text(sb_name).lower()
    if re.search(r"(?:^|_|\b)tp(?:\b|$)", sb):
        return "TP"
    if re.search(r"(?:^|_|\b)7m(?:\b|$)|\baca\b", sb):
        return "SM"
    if re.search(r"(?:^|_|\b)tm\d*(?:\b|$)", sb):
        return "TM"
    aa = as_text(antenna_arrays).lower()
    if any(k in aa for k in ["tp", "total power", "totalpower"]):
        return "TP"
    if any(k in aa for k in ["7m", "aca"]):
        return "SM"
    if any(k in aa for k in ["12m", "tm"]):
        return "TM"
    return None


# ---------------------------------------------------------------------------
# UID helpers
# ---------------------------------------------------------------------------

def to_compact_ous(uid: str) -> str:
    """Convert a UID to compact ``X_X`` format (e.g. ``X3833_X64b8``)."""
    s = as_text(uid).strip()
    s = s.replace("UID", "uid")
    if "uid___" in s:
        s = s.split("uid___", 1)[-1].replace("_", "/")
        s = "uid://" + s
    if not s.startswith("uid://"):
        s = s.replace("uid:///", "uid://").replace("uid:/", "uid://")
        if not s.startswith("uid://"):
            s = "uid://" + s.lstrip("/")
    parts = s.split("/")
    xs = [p for p in parts if p.lower().startswith("x")]
    if len(xs) >= 2:
        return f"{xs[-2]}_{xs[-1]}"
    return s


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------

def _ranges_from_freqsupport(s: str) -> List[Tuple[float, float]]:
    """Parse ``[lo, hi]`` GHz frequency ranges from an ALMA ``frequency_support`` string."""
    out: List[Tuple[float, float]] = []
    for m in _FREQ_PAIR_RE.finditer(as_text(s)):
        lo = float(m.group("lo"))
        hi = float(m.group("hi"))
        unit = (m.group("Unit") or "GHz").lower()
        scale = 1.0 if unit == "ghz" else (1e-3 if unit == "mhz" else 1e-6)
        lo *= scale
        hi *= scale
        if hi < lo:
            lo, hi = hi, lo
        out.append((lo, hi))
    return out


def _range_from_em_minmax(
    em_min_m: float, em_max_m: float
) -> Optional[Tuple[float, float]]:
    """Convert wavelength min/max (metres) to a ``(lo_GHz, hi_GHz)`` frequency range."""
    import astropy.constants as const

    _C = const.c.to("m/s").value
    try:
        vmin = float(getattr(em_min_m, "value", em_min_m))
        vmax = float(getattr(em_max_m, "value", em_max_m))
    except (ValueError, TypeError):
        return None
    if vmin <= 0 or vmax <= 0:
        return None
    nu_lo = _C / max(vmin, vmax) / 1e9
    nu_hi = _C / min(vmin, vmax) / 1e9
    if nu_hi < nu_lo:
        nu_lo, nu_hi = nu_hi, nu_lo
    return (nu_lo, nu_hi)


def _midpoints_avg(ranges_ghz: List[Tuple[float, float]]) -> Optional[float]:
    """Return the average of the midpoints of the given GHz ranges."""
    if not ranges_ghz:
        return None
    mids = [(a + b) / 2.0 for a, b in ranges_ghz]
    return sum(mids) / len(mids)


# ---------------------------------------------------------------------------
# Directory-structure helpers
# ---------------------------------------------------------------------------

def _extract_compact_uid(dirname: str, prefix: str) -> Optional[str]:
    """Extract compact UID (e.g. ``X3833_X64b8``) from a directory name.

    Example: ``science_goal.uid___A001_X3833_X64b8`` with *prefix*
    ``"science_goal."`` yields ``"X3833_X64b8"``.
    """
    if not dirname.startswith(prefix):
        return None
    uid_part = dirname[len(prefix):]
    match = re.search(r"uid___A\d+_(X[0-9a-fA-F]+_X[0-9a-fA-F]+)$", uid_part)
    if match:
        return match.group(1)
    return None


def build_gous_to_sgous_map(data_dir: Optional[Path]) -> Dict[str, str]:
    """Scan the extracted data directory to build a GOUS -> SGOUS compact-UID mapping.

    Directory structure expected::

        data_dir/
          science_goal.uid___A001_X3833_X64b8/
            group.uid___A001_X3833_X64b9/
              member.uid___...
    """
    gous_to_sgous: Dict[str, str] = {}

    if data_dir is None or not data_dir.exists():
        return gous_to_sgous

    for sg_dir in data_dir.iterdir():
        if not sg_dir.is_dir():
            continue
        if not sg_dir.name.startswith("science_goal."):
            continue

        sgous_id = _extract_compact_uid(sg_dir.name, "science_goal.")
        if not sgous_id:
            log.warning(f"Could not parse SGOUS ID from: {sg_dir.name}")
            continue

        for gous_dir in sg_dir.iterdir():
            if not gous_dir.is_dir():
                continue
            if not gous_dir.name.startswith("group."):
                continue

            gous_id = _extract_compact_uid(gous_dir.name, "group.")
            if not gous_id:
                log.warning(f"Could not parse GOUS ID from: {gous_dir.name}")
                continue

            gous_to_sgous[gous_id] = sgous_id

    log.info(f"Built GOUS\u2192SGOUS mapping with {len(gous_to_sgous)} entries from {data_dir}")
    return gous_to_sgous


# ---------------------------------------------------------------------------
# Query and aggregation
# ---------------------------------------------------------------------------

def _client(url: str) -> Alma:
    """Return an :class:`Alma` client pointed at *url*."""
    from astroquery.alma import Alma

    a = Alma()
    a.TIMEOUT = 300
    a.archive_url = url
    a.dataarchive_url = url
    return a


def query_rows(project_code: str, username: Optional[str]) -> List[dict]:
    """Query all ALMA mirrors for *project_code* and return a list of row dicts.

    Each dict has keys: ``target_name``, ``schedblock_name``, ``group_ous_uid``,
    ``member_ous_uid``, ``antenna_arrays``, ``frequency``.
    """
    last_err = None
    for server in ALMA_SERVERS:
        try:
            alma = _client(server)
            login_alma(alma, username)
            tab = alma.query(
                payload=dict(project_code=project_code),
                public=None,
                science=True,
                legacy_columns=False,
            )
            if tab is None or len(tab) == 0:
                log.warning(f"No rows from {server}")
                continue
            cols = tab.colnames

            def get(row, key):
                return row[key] if key in cols else ""

            rows: List[dict] = []
            for row in tab:
                target = as_text(get(row, "target_name"))
                if "off" in target.lower():  # drop OFF positions
                    continue
                freq_support = as_text(get(row, "frequency_support"))
                em_min = get(row, "em_min")
                em_max = get(row, "em_max")
                ranges = _ranges_from_freqsupport(freq_support) or (
                    [_range_from_em_minmax(em_min, em_max)]
                    if _range_from_em_minmax(em_min, em_max)
                    else []
                )
                nu_ref = _midpoints_avg(ranges) if ranges else None
                rows.append(
                    dict(
                        target_name=target,
                        schedblock_name=as_text(get(row, "schedblock_name")),
                        group_ous_uid=to_compact_ous(get(row, "group_ous_uid")),
                        member_ous_uid=to_compact_ous(get(row, "member_ous_uid")),
                        antenna_arrays=as_text(get(row, "antenna_arrays")),
                        frequency=nu_ref,
                    )
                )
            log.info(f"Got {len(rows)} usable rows from {server}")
            return rows
        except Exception as e:
            last_err = e
            log.warning(f"Query failed on {server}: {e}")
            continue
    from panta_rei.core.errors import ALMAError

    raise ALMAError(f"Failed to query any ALMA mirrors: {last_err}")


def build_index(
    rows: Iterable[dict],
) -> Dict[Tuple[str, str, str, str], dict]:
    """Aggregate *rows* by ``(source, array, sb_family, gous)``."""
    agg: Dict[Tuple[str, str, str, str], dict] = defaultdict(
        lambda: {"mous": set(), "freqs": set()}
    )
    for r in rows:
        src = r["target_name"]
        sbfam = r["schedblock_name"]
        gous = r["group_ous_uid"]
        mous = r["member_ous_uid"]
        arr = classify_array(r["schedblock_name"], r.get("antenna_arrays", ""))
        if arr is None:
            continue
        key = (src, arr, sbfam, gous)
        if mous:
            agg[key]["mous"].add(mous)
        f = r.get("frequency")
        if isinstance(f, (float, int)) and math.isfinite(f):
            agg[key]["freqs"].add(float(f))
    return agg


def write_csv(
    agg: Dict[Tuple[str, str, str, str], dict],
    out_path: Path,
    gous_to_sgous: Dict[str, str],
) -> None:
    """Write ``targets_by_array.csv`` from an aggregated index.

    Line group inference: ``"N2H+"`` if mean freq starts with ``97.``,
    ``"HCO+"`` if it starts with ``87.``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["source_name", "array", "sb_name", "sgous_id", "gous_id", "mous_ids", "Line group"]
        )
        for (src, arr, sbfam, gous), payload in sorted(agg.items()):
            mous = ";".join(sorted(payload["mous"]))
            sgous = gous_to_sgous.get(gous, "")
            line_group = ""
            if payload["freqs"]:
                vals = sorted(payload["freqs"])
                mean_val = sum(vals) / len(vals)
                s = f"{mean_val:.3f}"
                line_group = (
                    "N2H+"
                    if s.startswith("97.")
                    else ("HCO+" if s.startswith("87.") else "")
                )
            w.writerow([src, arr, sbfam, sgous, gous, mous, line_group])
    log.info(f"Wrote: {out_path}")
