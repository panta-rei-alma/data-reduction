"""Input mapping, readiness checks, and TP preflight for joint imaging.

Determines which TM/SM/TP inputs are available for each imaging unit,
validates TP spectral axes and beams, and builds ImagingUnit objects
ready for imaging execution (tclean+feather or sdintimaging).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FREQ_TOLERANCE_GHZ = 0.05


# ---------------------------------------------------------------------------
# Source name helpers (ported from feather_sm_tp.py)
# ---------------------------------------------------------------------------

def sanitize_source_name(source_name: str) -> str:
    """Sanitize source name for output filenames (``+`` -> ``p``, ``-`` -> ``m``)."""
    return source_name.replace("+", "p").replace("-", "m")


def sanitize_source_name_for_glob(source_name: str) -> str:
    """Sanitize source name for glob matching (``+`` -> ``p``)."""
    return source_name.replace("+", "p")


# ---------------------------------------------------------------------------
# Frequency helpers (ported from feather_sm_tp.py)
# ---------------------------------------------------------------------------

def get_freq_bounds_from_fits(fits_path: Path) -> tuple[float, float, Optional[int]]:
    """Extract frequency bounds from a FITS header.

    Returns ``(freq_min_hz, freq_max_hz, spw_id)`` where *spw_id* is
    parsed from the filename (or *None*).
    """
    from astropy.io import fits

    with fits.open(str(fits_path)) as hdul:
        header = hdul[0].header

        freq_axis = None
        for i in range(1, 5):
            if "FREQ" in header.get(f"CTYPE{i}", "").upper():
                freq_axis = i
                break

        if freq_axis is None:
            raise ValueError(f"No frequency axis in {fits_path}")

        crval = header[f"CRVAL{freq_axis}"]
        cdelt = header[f"CDELT{freq_axis}"]
        crpix = header[f"CRPIX{freq_axis}"]
        naxis = header[f"NAXIS{freq_axis}"]

        freq_first = crval + (1 - crpix) * cdelt
        freq_last = crval + (naxis - crpix) * cdelt

    spw_match = re.search(r"spw(\d+)", fits_path.name)
    spw_id = int(spw_match.group(1)) if spw_match else None

    return (min(freq_first, freq_last), max(freq_first, freq_last), spw_id)


def get_freq_range_string(freq_min_hz: float, freq_max_hz: float) -> str:
    """Format frequency bounds as ``XX.X-YY.Y`` GHz string for filenames."""
    return f"{freq_min_hz / 1e9:.1f}-{freq_max_hz / 1e9:.1f}"


def match_cubes_by_frequency(
    cubes_a: list[dict],
    cubes_b: list[dict],
    tolerance_ghz: float = DEFAULT_FREQ_TOLERANCE_GHZ,
) -> list[tuple[dict, dict]]:
    """Match two lists of cube dicts by center frequency.

    Each dict must have ``freq_min`` and ``freq_max`` keys (in Hz).
    Returns list of ``(a, b)`` pairs sorted by frequency.
    """
    tolerance_hz = tolerance_ghz * 1e9
    matched = []
    used_b = set()

    for a in cubes_a:
        a_center = (a["freq_min"] + a["freq_max"]) / 2
        best_b = None
        best_diff = float("inf")
        best_idx = None

        for i, b in enumerate(cubes_b):
            if i in used_b:
                continue
            b_center = (b["freq_min"] + b["freq_max"]) / 2
            diff = abs(a_center - b_center)
            if diff < best_diff and diff < tolerance_hz:
                best_diff = diff
                best_b = b
                best_idx = i

        if best_b is not None:
            matched.append((a, best_b))
            used_b.add(best_idx)

    matched.sort(key=lambda x: x[0]["freq_min"])
    return matched


# ---------------------------------------------------------------------------
# ImagingUnit dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImagingUnit:
    """Everything needed to run one imaging call (tclean+feather or sdintimaging).

    Built during preflight from recovered params + input mapping.
    """

    gous_uid: str
    source_name: str
    line_group: Optional[str]
    spw_id: str
    params_id: int

    # Recovered TM tclean params (full dict)
    recovered_params: dict = field(default_factory=dict)

    # Input paths
    vis_tm: list[str] = field(default_factory=list)
    vis_sm: list[str] = field(default_factory=list)
    sdimage: Optional[str] = None

    # Resolved per-MS selections (built in trusted preflight)
    spw_selection: list[str] = field(default_factory=list)
    field_selection: list[str] = field(default_factory=list)
    datacolumn: str = "corrected"

    # MOUS UIDs for provenance
    mous_uids_tm: list[str] = field(default_factory=list)
    mous_uids_sm: list[str] = field(default_factory=list)
    mous_uids_tp: list[str] = field(default_factory=list)

    # TP metadata
    tp_freq_min: Optional[float] = None
    tp_freq_max: Optional[float] = None
    tp_nchan: Optional[int] = None

    # Preflight status
    ready: bool = False
    skip_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# CSV loading for target groups
# ---------------------------------------------------------------------------

@dataclass
class TargetGroup:
    """A row from targets_by_array.csv, keyed by (source, array, line_group, gous)."""

    source_name: str
    array: str
    sb_name: str
    sgous_id: str
    gous_id: str
    mous_ids: list[str]
    line_group: str


def load_targets_csv(csv_path: Path) -> dict[str, list[TargetGroup]]:
    """Load targets_by_array.csv grouped by GOUS ID.

    Returns ``{gous_id: [TargetGroup, ...]}``.
    """
    groups: dict[str, list[TargetGroup]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gous_id = row.get("gous_id", "").strip()
            if not gous_id:
                continue
            mous_raw = row.get("mous_ids", "").strip()
            mous_list = [m.strip() for m in mous_raw.split(";") if m.strip()]
            tg = TargetGroup(
                source_name=row.get("source_name", "").strip(),
                array=row.get("array", "").strip(),
                sb_name=row.get("sb_name", "").strip(),
                sgous_id=row.get("sgous_id", "").strip(),
                gous_id=gous_id,
                mous_ids=mous_list,
                line_group=row.get("Line group", "").strip(),
            )
            groups[gous_id].append(tg)
    return dict(groups)


def targets_by_array(
    groups: list[TargetGroup],
) -> dict[str, list[TargetGroup]]:
    """Partition a GOUS's TargetGroups by array type."""
    by_array: dict[str, list[TargetGroup]] = defaultdict(list)
    for tg in groups:
        by_array[tg.array].append(tg)
    return dict(by_array)


# ---------------------------------------------------------------------------
# MS file discovery
# ---------------------------------------------------------------------------

def find_member_dir(data_dir: Path, mous_id: str) -> Optional[Path]:
    """Locate the member directory for a MOUS using the known hierarchy.

    Accepts both compact (``X3833_X64bc``) and full
    (``uid___A001_X3833_X64bc``) MOUS IDs — the ``uid___A001_`` prefix
    is stripped automatically if present.

    Uses the structured path ``science_goal.*/group.*/member.uid___A001_{mous_id}``
    instead of a ``**`` recursive glob (which is prohibitively slow on large
    data trees because it descends into MS directories).
    """
    compact = mous_id
    if compact.lower().startswith("uid___a001_"):
        compact = compact[len("uid___A001_"):]
    pattern = f"science_goal.*/group.*/member.uid___A001_{compact}"
    candidates = sorted(data_dir.glob(pattern))
    return candidates[0] if candidates else None


class MSSearchResult:
    """Result of searching for calibrated MS files."""

    def __init__(
        self,
        ms_files: list[Path],
        missing_reason: Optional[str] = None,
    ):
        self.ms_files = ms_files
        self.missing_reason = missing_reason

    def __bool__(self) -> bool:
        return len(self.ms_files) > 0


def find_ms_files(data_dir: Path, mous_id: str) -> MSSearchResult:
    """Find continuum-subtracted calibrated MS files for a MOUS.

    Looks for ``*_targets_line.ms`` under ``calibrated/working/``.  These
    contain the continuum-subtracted DATA column required for line imaging.

    Returns an :class:`MSSearchResult` whose ``missing_reason`` explains
    why no suitable MS was found (member missing, not calibrated, or
    calibrated but ScriptForPI did not produce the contsub split).
    """
    member_dir = find_member_dir(data_dir, mous_id)
    if member_dir is None:
        return MSSearchResult([], f"MOUS {mous_id}: member not available on disk")

    working = member_dir / "calibrated" / "working"
    if not working.is_dir():
        return MSSearchResult([], f"no calibrated/working/ directory for MOUS {mous_id}")

    ms_dirs = sorted(working.glob("*_targets_line.ms"))
    if ms_dirs:
        return MSSearchResult(ms_dirs)

    # targets_line.ms is missing — diagnose why
    any_ms = sorted(working.glob("*.ms"))
    if any_ms:
        # Calibrated MS exists but no contsub split.  This happens when
        # ScriptForPI detects selfcal failed for all targets and skips
        # the hif_uvcontsub() branch, even though cont.dat exists.
        return MSSearchResult(
            [],
            f"MOUS {mous_id}: calibrated MS exists but _targets_line.ms "
            f"missing (contsub not performed — likely ScriptForPI skipped "
            f"hif_uvcontsub because selfcal was unsuccessful). "
            f"Run: panta-rei-contsub --base-dir <DIR> --match {mous_id}",
        )

    return MSSearchResult([], f"no MS files in calibrated/working/ for MOUS {mous_id}")


def find_tp_cube(
    data_dir: Path,
    mous_id: str,
    source_name: str,
) -> list[dict]:
    """Find TP cube FITS files for a MOUS and source.

    Returns list of dicts with ``file``, ``freq_min``, ``freq_max``, ``spw`` keys.
    """
    member_dir = find_member_dir(data_dir, mous_id)
    if member_dir is None:
        log.debug("TP MOUS %s: member not available on disk", mous_id)
        return []

    product_dir = member_dir / "product"
    if not product_dir.is_dir():
        return []

    source_glob = sanitize_source_name_for_glob(source_name)
    pattern = f"*{source_glob}*spw*.cube.I.sd.fits"
    cubes = []
    for fits_path in product_dir.glob(pattern):
        try:
            freq_min, freq_max, spw_id = get_freq_bounds_from_fits(fits_path)
            cubes.append({
                "file": str(fits_path),
                "freq_min": freq_min,
                "freq_max": freq_max,
                "spw": spw_id,
            })
        except Exception as e:
            log.warning("Could not read frequency from %s: %s", fits_path.name, e)
    cubes.sort(key=lambda x: x["freq_min"])
    return cubes


# ---------------------------------------------------------------------------
# TP preflight validation (advisory mode — no casatasks)
# ---------------------------------------------------------------------------

def validate_tp_spectral_axis(
    tp_fits_path: Path,
    recovered_nchan: Optional[int] = None,
) -> tuple[bool, str, dict]:
    """Validate that a TP cube has a readable frequency axis.

    The TP spectral grid is **not** expected to match the TM grid exactly —
    doppler shifts, different observation epochs, and correlator configs
    all produce legitimate differences. ``sdintimaging`` regrids all inputs
    to the common grid defined by the recovered ``start``/``width``/``nchan``
    parameters, so strict start/width comparison would be wrong here.

    We check:
    - The FITS has a frequency axis at all (hard fail if missing).
    - nchan is logged but not enforced (TP may have different channelization).

    Returns ``(ok, message, info_dict)`` where *info_dict* has
    ``nchan``, ``crval``, ``cdelt``, ``crpix`` for the frequency axis.
    """
    from astropy.io import fits

    info: dict = {}
    with fits.open(str(tp_fits_path)) as hdul:
        header = hdul[0].header

        freq_axis = None
        for i in range(1, 5):
            if "FREQ" in header.get(f"CTYPE{i}", "").upper():
                freq_axis = i
                break

        if freq_axis is None:
            return False, f"No frequency axis in {tp_fits_path.name}", info

        info["nchan"] = header[f"NAXIS{freq_axis}"]
        info["crval"] = header[f"CRVAL{freq_axis}"]
        info["cdelt"] = header[f"CDELT{freq_axis}"]
        info["crpix"] = header[f"CRPIX{freq_axis}"]

    # Log nchan difference but do not hard-fail — TP channelization may
    # legitimately differ from TM (sdintimaging regrids to the TM grid).
    if recovered_nchan is not None and info["nchan"] != recovered_nchan:
        log.warning(
            "TP nchan=%d differs from TM nchan=%d for %s "
            "(sdintimaging will regrid)",
            info["nchan"], recovered_nchan, tp_fits_path.name,
        )

    return True, "OK", info


def validate_tp_beams(tp_fits_path: Path) -> tuple[bool, str]:
    """Check that TP cube has per-plane restoring beams.

    Requires a BEAMS extension table, or BMAJ + BMIN + BPA in the primary
    header. BMAJ/BMIN without BPA is rejected (beam position angle is
    undefined).
    """
    from astropy.io import fits

    with fits.open(str(tp_fits_path)) as hdul:
        # Check for BEAMS extension
        for ext in hdul:
            if hasattr(ext, "name") and ext.name == "BEAMS":
                return True, "BEAMS extension found"

        # Fall back to primary header beam keywords — require all three
        header = hdul[0].header
        if "BMAJ" in header and "BMIN" in header and "BPA" in header:
            return True, "BMAJ/BMIN/BPA in primary header"

        if "BMAJ" in header and "BMIN" in header:
            return False, f"BMAJ/BMIN present but BPA missing in {tp_fits_path.name}"

    return False, f"No restoring beams in {tp_fits_path.name}"


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

def build_output_path(
    output_dir: Path,
    gous_id: str,
    source_name: str,
    freq_min_hz: float,
    freq_max_hz: float,
) -> Path:
    """Build the output FITS path for the feathered 12m+7m+TP product.

    Pattern: ``{output_dir}/group.uid___A001_{gous_id}.lp_nperetto/
    group.uid___A001_{gous_id}.lp_nperetto.{source}.12m7mTP.{freq}GHz.cube.pbcor.fits``
    """
    sanitized = sanitize_source_name(source_name)
    freq_range = get_freq_range_string(freq_min_hz, freq_max_hz)
    subdir = f"group.uid___A001_{gous_id}.lp_nperetto"
    filename = (
        f"group.uid___A001_{gous_id}.lp_nperetto."
        f"{sanitized}.12m7mTP.{freq_range}GHz.cube.pbcor.fits"
    )
    return output_dir / subdir / filename


def build_tclean_only_output_path(
    output_dir: Path,
    gous_id: str,
    source_name: str,
    freq_min_hz: float,
    freq_max_hz: float,
) -> Path:
    """Build the output FITS path for the tclean-only 12m+7m product.

    Pattern: ``{output_dir}/group.uid___A001_{gous_id}.lp_nperetto/
    group.uid___A001_{gous_id}.lp_nperetto.{source}.12m7m.{freq}GHz.cube.pbcor.fits``
    """
    sanitized = sanitize_source_name(source_name)
    freq_range = get_freq_range_string(freq_min_hz, freq_max_hz)
    subdir = f"group.uid___A001_{gous_id}.lp_nperetto"
    filename = (
        f"group.uid___A001_{gous_id}.lp_nperetto."
        f"{sanitized}.12m7m.{freq_range}GHz.cube.pbcor.fits"
    )
    return output_dir / subdir / filename


# ---------------------------------------------------------------------------
# Preflight assembly (advisory mode)
# ---------------------------------------------------------------------------

def build_imaging_units_advisory(
    recovered_params: list[dict],
    targets: dict[str, list[TargetGroup]],
    data_dir: Path,
    freq_tolerance_ghz: float = DEFAULT_FREQ_TOLERANCE_GHZ,
) -> list[ImagingUnit]:
    """Build ImagingUnit objects with advisory-mode preflight.

    For each recovered param set, finds matching SM/TP inputs, validates
    TP spectral axis and beams. Does NOT resolve SPW/field/datacolumn
    (that requires casatasks in trusted mode).

    Parameters
    ----------
    recovered_params:
        List of dicts from imaging_params DB rows (must have
        ``id``, ``gous_uid``, ``source_name``, ``line_group``, ``spw_id``,
        ``mous_uids_tm``, and optionally ``params_json_path``).
    targets:
        Output of :func:`load_targets_csv`.
    data_dir:
        Project data directory.

    Returns list of ImagingUnit objects (check ``.ready`` and ``.skip_reason``).
    """
    units: list[ImagingUnit] = []

    for row in recovered_params:
        gous_uid = row["gous_uid"]
        source = row["source_name"]
        lg = row.get("line_group")
        spw_id = row["spw_id"]
        params_id = row["id"]
        mous_uids_tm = json.loads(row["mous_uids_tm"])

        # Load full recovered params from JSON if available
        recovered = {}
        pjp = row.get("params_json_path")
        if pjp and Path(pjp).exists():
            recovered = json.loads(Path(pjp).read_text())

        unit = ImagingUnit(
            gous_uid=gous_uid,
            source_name=source,
            line_group=lg,
            spw_id=spw_id,
            params_id=params_id,
            recovered_params=recovered,
            mous_uids_tm=mous_uids_tm,
        )

        # Find GOUS members from CSV
        gous_targets = targets.get(gous_uid)
        if not gous_targets:
            unit.skip_reason = f"GOUS {gous_uid} not found in targets CSV"
            units.append(unit)
            continue

        by_array = targets_by_array(gous_targets)

        # Filter to matching source + line_group
        def _match(tg: TargetGroup) -> bool:
            return tg.source_name == source and (
                not lg or tg.line_group == lg
            )

        # TM MS files — require artifacts from ALL TM members (GOUS completeness)
        tm_groups = [tg for tg in by_array.get("TM", []) if _match(tg)]
        tm_missing: list[str] = []
        for tg in tm_groups:
            for mous_id in tg.mous_ids:
                result = find_ms_files(data_dir, mous_id)
                if result:
                    unit.vis_tm.extend(str(p) for p in result.ms_files)
                else:
                    tm_missing.append(result.missing_reason or mous_id)

        if not unit.vis_tm:
            unit.skip_reason = (
                f"No TM MS files found: {'; '.join(tm_missing)}"
                if tm_missing else "No TM MS files found"
            )
            units.append(unit)
            continue

        if tm_missing:
            unit.skip_reason = (
                f"GOUS incomplete — TM: {'; '.join(tm_missing)}"
            )
            units.append(unit)
            continue

        # SM MS files — require artifacts from ALL SM members
        sm_groups = [tg for tg in by_array.get("SM", []) if _match(tg)]
        sm_missing: list[str] = []
        for tg in sm_groups:
            unit.mous_uids_sm.extend(tg.mous_ids)
            for mous_id in tg.mous_ids:
                result = find_ms_files(data_dir, mous_id)
                if result:
                    unit.vis_sm.extend(str(p) for p in result.ms_files)
                else:
                    sm_missing.append(result.missing_reason or mous_id)

        if not unit.vis_sm:
            unit.skip_reason = (
                f"No SM MS files found: {'; '.join(sm_missing)}"
                if sm_missing else "No SM MS files found"
            )
            units.append(unit)
            continue

        if sm_missing:
            unit.skip_reason = (
                f"GOUS incomplete — SM: {'; '.join(sm_missing)}"
            )
            units.append(unit)
            continue

        # TP cubes — require cubes from ALL TP members (GOUS completeness)
        tp_groups = [tg for tg in by_array.get("TP", []) if _match(tg)]
        all_tp_cubes: list[dict] = []
        tp_missing: list[str] = []
        for tg in tp_groups:
            unit.mous_uids_tp.extend(tg.mous_ids)
            for mous_id in tg.mous_ids:
                cubes = find_tp_cube(data_dir, mous_id, source)
                if cubes:
                    all_tp_cubes.extend(cubes)
                else:
                    tp_missing.append(mous_id)

        if not all_tp_cubes:
            unit.skip_reason = "No TP cubes found"
            units.append(unit)
            continue

        if tp_missing:
            unit.skip_reason = (
                f"GOUS incomplete: TP MOUS(es) missing cubes: "
                f"{', '.join(tp_missing)}"
            )
            units.append(unit)
            continue

        # Match TP cube to this SPW by frequency from recovered params
        nchan = recovered.get("nchan") or row.get("nchan")
        start = recovered.get("start", "")
        width = recovered.get("width", "")

        tp_match = _match_tp_cube_for_spw(
            all_tp_cubes, start, width, nchan, freq_tolerance_ghz
        )

        if tp_match is None:
            unit.skip_reason = (
                f"Cannot match TP cube to SPW {spw_id}: "
                f"{len(all_tp_cubes)} TP cubes, no frequency match"
            )
            units.append(unit)
            continue

        # Check for ambiguous match (multiple TP cubes within tolerance)
        tp_matches = _all_matching_tp_cubes(
            all_tp_cubes, start, width, nchan, freq_tolerance_ghz
        )
        if len(tp_matches) > 1:
            unit.skip_reason = (
                f"Ambiguous TP match for SPW {spw_id}: "
                f"{len(tp_matches)} TP cubes within frequency tolerance"
            )
            units.append(unit)
            continue

        unit.sdimage = tp_match["file"]
        unit.tp_freq_min = tp_match["freq_min"]
        unit.tp_freq_max = tp_match["freq_max"]

        # Validate TP spectral axis (frequency axis exists, log nchan diff)
        ok, msg, tp_info = validate_tp_spectral_axis(
            Path(tp_match["file"]),
            recovered_nchan=int(nchan) if nchan else None,
        )
        if not ok:
            unit.skip_reason = f"TP spectral validation failed: {msg}"
            units.append(unit)
            continue

        unit.tp_nchan = tp_info.get("nchan")

        # Validate TP beams (requires BMAJ+BMIN+BPA or BEAMS table)
        ok, msg = validate_tp_beams(Path(tp_match["file"]))
        if not ok:
            unit.skip_reason = f"TP beam validation failed: {msg}"
            units.append(unit)
            continue

        # Advisory mode: SPW/field/datacolumn NOT resolved (needs casatasks)
        unit.ready = True
        units.append(unit)

    return units


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _all_matching_tp_cubes(
    tp_cubes: list[dict],
    start: str,
    width: str,
    nchan,
    tolerance_ghz: float,
) -> list[dict]:
    """Return ALL TP cubes that match the TM SPW frequency within tolerance."""
    tm_freq = _compute_tm_freq_range(start, width, nchan)
    if tm_freq is None:
        return []

    tm_center = (tm_freq[0] + tm_freq[1]) / 2
    tolerance_hz = tolerance_ghz * 1e9

    matches = []
    for cube in tp_cubes:
        tp_center = (cube["freq_min"] + cube["freq_max"]) / 2
        if abs(tm_center - tp_center) < tolerance_hz:
            matches.append(cube)
    return matches


def _match_tp_cube_for_spw(
    tp_cubes: list[dict],
    start: str,
    width: str,
    nchan,
    tolerance_ghz: float,
) -> Optional[dict]:
    """Match a TP cube to a TM SPW by computing the TM frequency range
    from recovered start/width/nchan and comparing center frequencies."""
    tm_freq = _compute_tm_freq_range(start, width, nchan)
    if tm_freq is None:
        return None

    tm_center = (tm_freq[0] + tm_freq[1]) / 2
    tolerance_hz = tolerance_ghz * 1e9

    best = None
    best_diff = float("inf")
    for cube in tp_cubes:
        tp_center = (cube["freq_min"] + cube["freq_max"]) / 2
        diff = abs(tm_center - tp_center)
        if diff < best_diff and diff < tolerance_hz:
            best_diff = diff
            best = cube

    return best


def _compute_tm_freq_range(
    start: str, width: str, nchan
) -> Optional[tuple[float, float]]:
    """Compute TM frequency range from recovered start/width/nchan.

    The start and width are CASA quantity strings like ``'86.05GHz'`` or
    ``'244.140625kHz'``.
    """
    start_hz = _parse_freq_quantity(start)
    width_hz = _parse_freq_quantity(width)
    if start_hz is None or width_hz is None or nchan is None:
        return None
    try:
        n = int(nchan)
    except (ValueError, TypeError):
        return None

    freq_end = start_hz + n * width_hz
    return (min(start_hz, freq_end), max(start_hz, freq_end))


def _parse_freq_quantity(s: str) -> Optional[float]:
    """Parse a CASA frequency quantity string to Hz."""
    if not s:
        return None
    s = s.strip()

    # Try direct float (already in Hz)
    try:
        return float(s)
    except ValueError:
        pass

    m = re.match(r"^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(Hz|kHz|MHz|GHz|THz)$", s, re.IGNORECASE)
    if not m:
        return None

    val = float(m.group(1))
    unit = m.group(2).lower()
    multipliers = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}
    return val * multipliers.get(unit, 1.0)
