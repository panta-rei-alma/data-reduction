"""Recover tclean imaging parameters from ALMA pipeline weblogs.

Parses ``casa_commands.log`` from TM (12m) pipeline weblogs to extract
the tclean parameters used for cube imaging.  Only the restored,
PB-corrected calls (``restoration=True, pbcor=True``) are returned — the
final cleaned ``iter1`` images, or the published dirty ``iter0`` image when
QA2 found nothing to clean (see :func:`extract_by_field_spw`).

The recovered parameters serve as the baseline for ``sdintimaging``.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Section marker that precedes cube imaging tclean calls
_CUBE_SECTION_RE = re.compile(
    r"#\s*hif_makeimlist\(.*specmode\s*=\s*['\"]cube['\"]",
)

# Regex to detect the start of a tclean call
_TCLEAN_START_RE = re.compile(r"^tclean\(")

# Repair the dash-space artefact introduced by the multi-line tclean
# assembler: when ``casa_commands.log`` line-wraps a hyphenated value
# like ``auto-multithresh`` between the ``-`` and ``multithresh``, the
# ``" ".join(stripped_lines)`` step in ``parse_tclean_calls`` produces
# ``"auto- multithresh"``.  Collapse whitespace that sits between an
# ASCII hyphen+letter boundary and another ASCII letter — narrow enough
# that legitimate values containing hyphens followed by space-then-word
# would never be touched (no such CASA tclean param value exists; all
# affected enums are single hyphenated tokens like ``auto-multithresh``,
# ``pb-based`` etc.).
_HYPHEN_LINEBREAK_RE = re.compile(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])")

# Parameters that identify an iter1 (final) tclean call
_ITER1_MARKERS = {"restoration": True, "pbcor": True}

# Parameters that identify an iter0 (dirty) tclean call
_ITER0_MARKERS = {"restoration": False, "niter": 0}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Per-process caches for the NFS directory globs below.  Across a full
# recovery these helpers are called thousands of times for the same MOUSs, and
# on NFS each glob costs 0.25-1.2s (metadata round-trips) — far more than the
# parsing.  The filesystem layout is immutable during a recovery, so memoizing
# the lookups is safe.  Cleared via clear_parse_cache() (tests / long processes).
_LOGPATH_CACHE: dict[tuple, "Optional[Path]"] = {}
_STAGED_CACHE: dict[tuple, bool] = {}


def find_casa_commands_log(
    weblog_base: Path,
    mous_uid: str,
    data_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Locate ``casa_commands.log`` for a MOUS, searching multiple locations.

    Search order:
      1. Staged weblog: ``{weblog_base}/{uid}/pipeline-*/html/casa_commands.log``
      2. Raw log dir: ``{data_dir}/.../member.uid___*/log/*.casa_commands.log``
      3. Raw working dir: ``{data_dir}/.../member.uid___*/calibrated/working/pipeline-*/html/casa_commands.log``

    Returns the first existing path, or *None*.  Memoized per-process.
    """
    from panta_rei.core.uid import sanitize_uid

    uid_dir = sanitize_uid(mous_uid) if mous_uid else None
    if not uid_dir:
        return None

    cache_key = (str(weblog_base), uid_dir, str(data_dir) if data_dir else None)
    if cache_key in _LOGPATH_CACHE:
        return _LOGPATH_CACHE[cache_key]

    result: Optional[Path] = None
    # 1. Staged weblog
    if weblog_base and weblog_base.is_dir():
        candidates = sorted(weblog_base.glob(
            f"{uid_dir}/pipeline-*/html/casa_commands.log"
        ))
        if candidates:
            result = candidates[-1]  # latest pipeline run

    # 2-3. Raw extracted data (structured lookup, not recursive **)
    if result is None and data_dir and data_dir.is_dir():
        from panta_rei.imaging.matching import find_member_dir

        member_dir = find_member_dir(data_dir, uid_dir)
        if member_dir is not None:
            for p in sorted(member_dir.glob("log/*.casa_commands.log")):
                result = p
                break
            if result is None:
                candidates = sorted(member_dir.glob(
                    "calibrated/working/pipeline-*/html/casa_commands.log"
                ))
                if candidates:
                    result = candidates[-1]

    _LOGPATH_CACHE[cache_key] = result
    return result


def has_staged_weblog(weblog_base: Path, mous_uid: str) -> bool:
    """Check if a staged weblog exists for a MOUS (fast glob, no parsing).

    Memoized per-process (the glob is a slow NFS metadata round-trip).
    """
    from panta_rei.core.uid import sanitize_uid

    uid_dir = sanitize_uid(mous_uid) if mous_uid else None
    if not uid_dir or not weblog_base or not weblog_base.is_dir():
        return False
    cache_key = (str(weblog_base), uid_dir)
    if cache_key in _STAGED_CACHE:
        return _STAGED_CACHE[cache_key]
    result = (
        next(
            weblog_base.glob(f"{uid_dir}/pipeline-*/html/casa_commands.log"),
            None,
        )
        is not None
    )
    _STAGED_CACHE[cache_key] = result
    return result


# Per-process cache of parsed weblogs, keyed by (path, size, mtime_ns).  A full
# continuum recovery re-parses the same large casa_commands.log many times (once
# per source, plus the SM weblog per unit); parsing dominates the runtime, so
# memoizing it makes a full --re-run ~10-20x faster.  Weblogs are immutable
# during a recovery; the size+mtime key still invalidates if one is rewritten.
# Callers never mutate the returned dicts (extract_* build fresh dicts), so
# returning the cached list by reference is safe.
_PARSE_CACHE: dict[tuple, list[dict]] = {}


def clear_parse_cache() -> None:
    """Drop the recovery per-process caches (parsed weblogs + glob lookups).

    For tests and any long-lived process that must re-see filesystem changes.
    """
    _PARSE_CACHE.clear()
    _LOGPATH_CACHE.clear()
    _STAGED_CACHE.clear()
    from panta_rei.imaging.matching import clear_member_dir_cache
    clear_member_dir_cache()


def parse_tclean_calls(log_path: Path) -> list[dict]:
    """Parse all ``tclean(...)`` invocations from a ``casa_commands.log``.

    Each call may span multiple lines.  The parser accumulates lines from
    ``tclean(`` until a line ending with ``)`` is found, joins them, and
    converts the keyword arguments to a dict (see :func:`_parse_one_call`).

    Results are memoized per-process (see ``_PARSE_CACHE``) since recovery
    re-parses the same weblogs repeatedly.

    Returns a list of parameter dicts, one per tclean call.
    """
    try:
        st = log_path.stat()
        cache_key = (str(log_path), st.st_size, st.st_mtime_ns)
    except OSError:
        cache_key = None
    if cache_key is not None and cache_key in _PARSE_CACHE:
        return _PARSE_CACHE[cache_key]

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    calls: list[dict] = []
    accumulator: list[str] = []
    in_call = False

    for line in lines:
        stripped = line.strip()

        if not in_call:
            if _TCLEAN_START_RE.match(stripped):
                in_call = True
                accumulator = [stripped]
                if stripped.endswith(")"):
                    calls.append(_parse_one_call("".join(accumulator)))
                    in_call = False
                    accumulator = []
        else:
            accumulator.append(stripped)
            if stripped.endswith(")"):
                calls.append(_parse_one_call(" ".join(accumulator)))
                in_call = False
                accumulator = []

    if cache_key is not None:
        _PARSE_CACHE[cache_key] = calls
    return calls


def filter_cube_iter1_calls(
    calls: list[dict],
    log_path: Optional[Path] = None,
) -> list[dict]:
    """Filter to only cube imaging iter1 tclean calls.

    Iter1 calls have ``restoration=True`` and ``pbcor=True``.
    Only calls after the ``hif_makeimlist(specmode='cube')`` section
    marker are considered.

    If *log_path* is provided, the section marker is located in the raw
    file first and only calls whose ``imagename`` contains ``.cube.``
    are included.
    """
    # If we have the log path, use section-aware filtering
    cube_section_line = _find_cube_section_line(log_path) if log_path else 0

    iter1: list[dict] = []
    for call in calls:
        # Must be specmode='cube'
        if call.get("specmode") != "cube":
            continue
        # Must be iter1
        if call.get("restoration") is not True or call.get("pbcor") is not True:
            continue
        # imagename should contain '.cube.' (not continuum)
        imagename = str(call.get("imagename", ""))
        if ".cube." not in imagename:
            continue
        iter1.append(call)

    return iter1


# Specmodes used by ALMA continuum (aggregate) imaging.  The aggregate
# continuum product is never 'cube'.
_CONT_SPECMODES = {"mfs", "cont"}


def filter_continuum_iter1_calls(
    calls: list[dict],
    log_path: Optional[Path] = None,
) -> list[dict]:
    """Filter to only aggregate-continuum iter1 tclean calls.

    A continuum iter1 call has ``restoration=True``, ``pbcor=True``,
    ``specmode in {'mfs', 'cont'}`` and an ``imagename`` carrying the
    dot-bounded ``.cont.`` token (e.g.
    ``...spw24_26_..cont.regcal.I.iter1``).

    This excludes cube calls (``.cube.`` / ``specmode='cube'``) and the
    per-spw ``findcont`` calls (``...mfs.regcal.I.findcont`` — no ``.cont.``,
    not iter1).  ``log_path`` is accepted for signature symmetry with
    :func:`filter_cube_iter1_calls` but is not needed: the ``.cont.`` token
    is an unambiguous discriminator.
    """
    out: list[dict] = []
    for call in calls:
        if call.get("restoration") is not True or call.get("pbcor") is not True:
            continue
        if call.get("specmode") not in _CONT_SPECMODES:
            continue
        if ".cont." not in str(call.get("imagename", "")):
            continue
        out.append(call)
    return out


def extract_continuum_by_field(calls: list[dict]) -> dict[str, dict]:
    """Organize continuum iter1 calls by ``field`` name.

    Continuum is aggregated across all spws, so there is exactly one
    aggregate product per field.  Unlike the line path, the spw is **not**
    part of the identity — the full line-free frequency-range selection
    lives inside the params dict (``spw``).  Returns ``{field: params}``.
    """
    result: dict[str, dict] = {}
    for call in calls:
        call = _normalize_string_values(call)
        field = _clean_field(call.get("field", ""))
        if field in result:
            log.debug("Duplicate continuum field %s; keeping last", field)
        result[field] = call
    return result


def extract_by_field_spw(calls: list[dict]) -> dict[tuple[str, str], dict]:
    """Organize iter1 calls by ``(field, spw)`` key.

    The ``field`` value is cleaned (embedded quotes removed).
    The ``spw`` value is the first element if it's a list.
    All string values are whitespace-normalized (the multi-line parser
    joins lines with spaces, which can introduce spurious whitespace
    inside string values like ``'auto- multithresh'``).

    The restored/PB-corrected filter also admits the QA2 ``iter0`` dirty
    call (``niter=0``) that the pipeline publishes directly when it sees no
    emission.  A cleaned call (``niter>0``) always wins over a dirty one for
    the same key; a key with only a dirty call keeps it (geometry, spectral
    grid and weighting are still the right seed) but is marked
    ``qa2_dirty_only=True`` so the builder gives it a positive clean budget
    — a QA2 nondetection is not a joint 12m+7m nondetection.

    Returns ``{(field, spw): params_dict}``.
    """
    result: dict[tuple[str, str], dict] = {}
    for call in calls:
        # Normalize whitespace in string values
        call = _normalize_string_values(call)

        field = _clean_field(call.get("field", ""))
        spw_raw = call.get("spw", "")
        if isinstance(spw_raw, list):
            spw = str(spw_raw[0]) if spw_raw else ""
        else:
            spw = str(spw_raw)

        key = (field, spw)
        if key in result:
            if _is_dirty_call(call) and not _is_dirty_call(result[key]):
                log.debug("Duplicate (field, spw) = %s; keeping cleaned call "
                          "over dirty iter0", key)
                continue
            log.debug("Duplicate (field, spw) = %s; keeping last", key)
        result[key] = call

    for key, call in result.items():
        if _is_dirty_call(call):
            log.info("(field, spw) = %s has only a QA2 dirty (niter=0) call",
                     key)
            call["qa2_dirty_only"] = True

    return result


def _is_dirty_call(params: dict) -> bool:
    """True for a tclean call that does no deconvolution (``niter=0``)."""
    try:
        return int(params.get("niter", -1)) == 0
    except (TypeError, ValueError):
        return False


def _normalize_string_values(params: dict) -> dict:
    """Collapse internal whitespace in string parameter values.

    The multi-line tclean parser joins lines with ``" ".join()``, which
    can introduce spaces inside string values when a value is split
    across lines in ``casa_commands.log``.  Two cleanups happen here:

    1. ``re.sub(_HYPHEN_LINEBREAK_RE, "-", v)`` rejoins hyphenated
       identifiers that got split across a line break — e.g.
       ``'auto-\\nmultithresh'`` → joined as ``'auto- multithresh'`` →
       repaired to ``'auto-multithresh'``.  Without this, tclean
       rejects ``usemask='auto- multithresh'`` with
       ``{'usemask': ['unallowed value auto- multithresh']}``.
    2. ``" ".join(v.split())`` collapses any other run of internal
       whitespace into a single space.
    """
    cleaned = {}
    for k, v in params.items():
        if isinstance(v, str):
            v = _HYPHEN_LINEBREAK_RE.sub("-", v)
            cleaned[k] = " ".join(v.split())
        else:
            cleaned[k] = v
    return cleaned


def recover_params_for_mous(
    mous_uid: str,
    weblog_base: Path,
    data_dir: Optional[Path] = None,
) -> Optional[dict[tuple[str, str], dict]]:
    """Recover cube imaging tclean parameters from a TM MOUS weblog.

    Returns a dict keyed by ``(field_name, spw_id)`` mapping to the full
    tclean parameter dict, or *None* if no weblog or no cube calls found.
    """
    log_path = find_casa_commands_log(weblog_base, mous_uid, data_dir)
    if log_path is None:
        log.warning("No casa_commands.log found for %s", mous_uid)
        return None

    log.info("Parsing %s for MOUS %s", log_path, mous_uid)
    all_calls = parse_tclean_calls(log_path)
    if not all_calls:
        log.warning("No tclean calls found in %s", log_path)
        return None

    iter1_calls = filter_cube_iter1_calls(all_calls, log_path)
    if not iter1_calls:
        log.warning("No cube iter1 tclean calls found in %s", log_path)
        return None

    result = extract_by_field_spw(iter1_calls)
    log.info(
        "Recovered %d cube iter1 parameter sets from %s: %s",
        len(result),
        log_path.name,
        list(result.keys()),
    )
    return result


def eb_spw_map_from_call(call: dict) -> dict[str, str]:
    """Build ``{eb_uid: spw_selection}`` from one aggregate-continuum call.

    The ALMA continuum ``tclean`` call's ``vis`` and ``spw`` lists are parallel
    — one line-free selection per EB, each already in that EB's own spw ids and
    topocentric frame (verified across all weblogs).  So each EB carries the
    exact selection the pipeline prescribed for it; no cross-array frequency
    remapping is needed.  Keyed by EB uid (MS basename minus ``_targets*.ms``).
    """
    from panta_rei.imaging.matching import eb_uid_from_ms

    vis = call.get("vis")
    spw = call.get("spw")
    vis = vis if isinstance(vis, list) else [vis]
    spw = spw if isinstance(spw, list) else [spw]
    # Defensive: a single shared selection applied to every EB.
    if len(spw) == 1 and len(vis) > 1:
        spw = spw * len(vis)
    if len(vis) != len(spw):
        # The ALMA contract is parallel vis/spw lists; anything else means a
        # parse anomaly. Fail rather than zip()-truncate: this map is
        # load-bearing and dropping EBs would silently change the joint image.
        raise ValueError(
            f"continuum call vis/spw length mismatch ({len(vis)} vis, "
            f"{len(spw)} spw) for {call.get('imagename', '?')}"
        )
    out: dict[str, str] = {}
    for v, s in zip(vis, spw):
        if not v or not s:
            continue
        out[eb_uid_from_ms(str(v))] = str(s)
    return out


def recover_continuum_params_for_mous(
    mous_uid: str,
    weblog_base: Path,
    data_dir: Optional[Path] = None,
) -> Optional[dict[str, dict]]:
    """Recover aggregate-continuum tclean parameters from a TM MOUS weblog.

    Returns a dict keyed by ``field_name`` mapping to the full tclean
    parameter dict (including the line-free frequency-range ``spw``
    selection, ``mosweight``, masking params, etc.), or *None* if no weblog
    or no continuum calls are found.

    Note: ``deconvolver``/``nterms`` are recovered for provenance but the
    continuum runner overrides them to ``mtmfs``/``2`` (see the runner).
    """
    log_path = find_casa_commands_log(weblog_base, mous_uid, data_dir)
    if log_path is None:
        log.warning("No casa_commands.log found for %s", mous_uid)
        return None

    log.info("Parsing %s for MOUS %s (continuum)", log_path, mous_uid)
    all_calls = parse_tclean_calls(log_path)
    if not all_calls:
        log.warning("No tclean calls found in %s", log_path)
        return None

    cont_calls = filter_continuum_iter1_calls(all_calls, log_path)
    if not cont_calls:
        log.warning("No continuum iter1 tclean calls found in %s", log_path)
        return None

    result = extract_continuum_by_field(cont_calls)
    log.info(
        "Recovered %d continuum iter1 parameter sets from %s: %s",
        len(result),
        log_path.name,
        list(result.keys()),
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_one_call(call_str: str) -> dict:
    """Parse a single ``tclean(key=val, ...)`` string into a dict.

    Primary method: ``ast.parse`` the call expression and
    ``ast.literal_eval`` each keyword value (never ``eval``), so every
    literal argument is kept — nested lists, quoted field names, booleans,
    negative numbers.  A keyword whose value is not a literal is logged and
    skipped; the rest of the call is still recovered.

    Fallback: if the call text itself does not parse (e.g. a truncated
    log), regex-based extraction of the most important parameters.
    """
    try:
        node = ast.parse(call_str.strip(), mode="eval").body
    except SyntaxError:
        node = None

    if isinstance(node, ast.Call):
        result: dict = {}
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs splat — nothing literal to keep
                continue
            try:
                result[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, TypeError, SyntaxError, MemoryError,
                    RecursionError):
                # debug, not warning: the only non-literals seen in real
                # weblogs are np.float64(...) reprs in self-cal solve calls.
                log.debug(
                    "Skipping non-literal tclean argument %s=%.60s",
                    kw.arg, ast.unparse(kw.value),
                )
        return result

    # Fallback: regex extraction
    log.debug("ast.parse failed, falling back to regex for: %.100s...", call_str)
    return _regex_parse(call_str)


def _regex_parse(call_str: str) -> dict:
    """Fallback regex parser for tclean parameter strings.

    Only used when the call text is not valid Python.  Handles the most
    important parameters that sdintimaging needs.
    """
    result: dict = {}

    # String parameters
    for key in (
        "field", "imagename", "specmode", "start", "width", "outframe",
        "veltype", "deconvolver", "weighting", "restoringbeam", "usemask",
        "datacolumn", "stokes", "gridder", "phasecenter", "threshold",
        "intent",
    ):
        m = re.search(rf"{key}\s*=\s*'([^']*)'", call_str)
        if not m:
            m = re.search(rf'{key}\s*=\s*"([^"]*)"', call_str)
        if m:
            result[key] = m.group(1)

    # Numeric parameters (int/float)
    for key in (
        "nchan", "niter", "robust", "npixels", "pblimit", "nsigma",
        "sidelobethreshold", "noisethreshold", "lownoisethreshold",
        "negativethreshold", "minbeamfrac", "growiterations",
        "minpercentchange", "cyclefactor",
    ):
        m = re.search(rf"{key}\s*=\s*([\d.eE+-]+)", call_str)
        if m:
            val = m.group(1)
            result[key] = int(val) if "." not in val else float(val)

    # Boolean parameters
    for key in (
        "restoration", "pbcor", "interactive", "perchanweightdensity",
        "mosweight", "usepointing", "dogrowprune", "fastnoise",
        "restart", "calcres", "calcpsf", "fullsummary",
    ):
        m = re.search(rf"{key}\s*=\s*(True|False)", call_str)
        if m:
            result[key] = m.group(1) == "True"

    # List parameters: imsize, cell, spw, vis
    for key in ("imsize", "cell", "spw", "vis"):
        m = re.search(rf"{key}\s*=\s*(\[[^\]]*\])", call_str)
        if m:
            try:
                result[key] = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                pass

    return result


def _clean_field(field_val) -> str:
    """Remove embedded quotes from field values.

    The pipeline writes ``field='"AG231.7986-1.9684"'`` with nested quotes.
    """
    s = str(field_val)
    # Strip outer and inner quotes
    s = s.strip("'\"")
    return s


def _find_cube_section_line(log_path: Optional[Path]) -> int:
    """Return the line number of the cube imaging section marker, or 0."""
    if log_path is None:
        return 0
    try:
        for i, line in enumerate(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            if _CUBE_SECTION_RE.search(line):
                return i
    except OSError:
        pass
    return 0
