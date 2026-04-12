"""Recover tclean imaging parameters from ALMA pipeline weblogs.

Parses ``casa_commands.log`` from TM (12m) pipeline weblogs to extract
the tclean parameters used for cube imaging.  Only the **iter1** calls
(``restoration=True, pbcor=True``) are returned — these represent the
final cleaned images.

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

# Parameters that identify an iter1 (final) tclean call
_ITER1_MARKERS = {"restoration": True, "pbcor": True}

# Parameters that identify an iter0 (dirty) tclean call
_ITER0_MARKERS = {"restoration": False, "niter": 0}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    Returns the first existing path, or *None*.
    """
    from panta_rei.core.uid import sanitize_uid

    uid_dir = sanitize_uid(mous_uid) if mous_uid else None
    if not uid_dir:
        return None

    # 1. Staged weblog
    if weblog_base and weblog_base.is_dir():
        candidates = sorted(weblog_base.glob(
            f"{uid_dir}/pipeline-*/html/casa_commands.log"
        ))
        if candidates:
            return candidates[-1]  # latest pipeline run

    # 2-3. Raw extracted data (structured lookup, not recursive **)
    if data_dir and data_dir.is_dir():
        from panta_rei.imaging.matching import find_member_dir

        member_dir = find_member_dir(data_dir, uid_dir)
        if member_dir is not None:
            # 2. log/ directory
            for p in sorted(member_dir.glob("log/*.casa_commands.log")):
                return p
            # 3. calibrated/working/pipeline-*/html/
            candidates = sorted(member_dir.glob(
                "calibrated/working/pipeline-*/html/casa_commands.log"
            ))
            if candidates:
                return candidates[-1]

    return None


def has_staged_weblog(weblog_base: Path, mous_uid: str) -> bool:
    """Check if a staged weblog exists for a MOUS (fast glob, no parsing)."""
    from panta_rei.core.uid import sanitize_uid

    uid_dir = sanitize_uid(mous_uid) if mous_uid else None
    if not uid_dir or not weblog_base or not weblog_base.is_dir():
        return False
    return (
        next(
            weblog_base.glob(f"{uid_dir}/pipeline-*/html/casa_commands.log"),
            None,
        )
        is not None
    )


def parse_tclean_calls(log_path: Path) -> list[dict]:
    """Parse all ``tclean(...)`` invocations from a ``casa_commands.log``.

    Each call may span multiple lines.  The parser accumulates lines from
    ``tclean(`` until a line ending with ``)`` is found, joins them, and
    converts the parameter string to a dict via :func:`ast.literal_eval`.

    Returns a list of parameter dicts, one per tclean call.
    """
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


def extract_by_field_spw(calls: list[dict]) -> dict[tuple[str, str], dict]:
    """Organize iter1 calls by ``(field, spw)`` key.

    The ``field`` value is cleaned (embedded quotes removed).
    The ``spw`` value is the first element if it's a list.
    All string values are whitespace-normalized (the multi-line parser
    joins lines with spaces, which can introduce spurious whitespace
    inside string values like ``'auto- multithresh'``).

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
            log.debug("Duplicate (field, spw) = %s; keeping last", key)
        result[key] = call

    return result


def _normalize_string_values(params: dict) -> dict:
    """Collapse internal whitespace in string parameter values.

    The multi-line tclean parser joins lines with ``" ".join()``, which
    can introduce spaces inside string values when a value is split
    across lines in ``casa_commands.log`` (e.g. ``'auto-\\nmultithresh'``
    becomes ``'auto- multithresh'``).
    """
    cleaned = {}
    for k, v in params.items():
        if isinstance(v, str):
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_one_call(call_str: str) -> dict:
    """Parse a single ``tclean(key=val, ...)`` string into a dict.

    Primary method: ``ast.literal_eval`` on ``dict(...)`` form.
    Fallback: regex-based extraction of individual parameters.
    """
    # Strip leading 'tclean' and convert to dict(...)
    inner = call_str.strip()
    if inner.startswith("tclean"):
        inner = inner[len("tclean"):]

    # Try ast.literal_eval with dict() wrapper
    try:
        result = ast.literal_eval(f"dict{inner}")
        return result
    except (ValueError, SyntaxError):
        pass

    # Fallback: regex extraction
    log.debug("ast.literal_eval failed, falling back to regex for: %.100s...", call_str)
    return _regex_parse(call_str)


def _regex_parse(call_str: str) -> dict:
    """Fallback regex parser for tclean parameter strings.

    Handles the most important parameters that sdintimaging needs.
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
