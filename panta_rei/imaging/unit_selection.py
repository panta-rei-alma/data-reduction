"""Shared selection logic for imaging units.

Both :class:`panta_rei.workflows.imaging.JointImagingStep` and
the distributed dispatcher pick units to run from the same pool of
recovered params.  The actual filter / preflight / idempotency logic
lives here so it cannot drift between the two callers.

The dispatcher additionally needs to exclude units that are *already
in-flight* under another (non-terminal) dispatch — otherwise a fresh
run can enqueue a duplicate row for a job an adopted worker is still
executing.  ``select_units_to_image`` exposes that via
``exclude_active=True``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import (
    DispatchState,
    ImagingParamsQueries,
    ImagingRunsQueries,
)
from panta_rei.imaging.matching import (
    ImagingUnit,
    build_imaging_units_advisory,
    load_targets_csv,
)

log = logging.getLogger(__name__)


@dataclass
class SelectionFilters:
    """Selection-time filters shared by step and dispatcher."""

    match: Optional[str] = None
    include_sources: Optional[list[str]] = None
    include_line_groups: Optional[list[str]] = None
    limit: Optional[int] = None
    method: str = "tclean_feather"
    sdgain: Optional[float] = None
    deconvolver: str = "multiscale"
    scales: list[int] = None  # type: ignore[assignment]
    re_run: bool = False
    exclude_active: bool = False


@dataclass
class SelectionResult:
    """Outcome of :func:`select_units_to_image`."""

    ready: list[ImagingUnit]
    not_ready: list[ImagingUnit]
    skipped_already_done: int
    skipped_active: int
    skipped_filtered_out: int


def select_units_to_image(
    db_manager: DatabaseManager,
    targets_csv: Path,
    data_dir: Path,
    filters: SelectionFilters,
) -> SelectionResult:
    """Build, filter, and idempotency-check the list of imaging units.

    Steps:
      1. Load all recovered ``imaging_params`` rows.
      2. Apply ``--match`` / ``--include-sources`` / ``--include-line-groups``.
      3. Build :class:`ImagingUnit` records via advisory preflight.
      4. Drop not-ready units.
      5. Drop units with a matching prior SUCCESS unless ``re_run``.
      6. If ``exclude_active``, drop units currently QUEUED/RUNNING under a
         non-terminal dispatch.
      7. Honour ``limit``.
    """
    if filters.scales is None:
        filters.scales = [0, 5, 10, 15, 20]

    targets = load_targets_csv(targets_csv)

    with db_manager.connect() as con:
        all_params = ImagingParamsQueries.get_all_recovered(con)

    if not all_params:
        return SelectionResult(
            ready=[], not_ready=[],
            skipped_already_done=0, skipped_active=0, skipped_filtered_out=0,
        )

    regex: Optional[re.Pattern] = None
    if filters.match:
        try:
            regex = re.compile(filters.match, re.IGNORECASE)
        except re.error:
            regex = None

    filtered = []
    skipped_filter = 0
    for row in all_params:
        src = row["source_name"]
        gous = row["gous_uid"]
        lg = row.get("line_group")
        if filters.include_sources and src not in filters.include_sources:
            skipped_filter += 1
            continue
        if filters.include_line_groups and lg not in filters.include_line_groups:
            skipped_filter += 1
            continue
        if regex and not regex.search(src) and not regex.search(gous):
            skipped_filter += 1
            continue
        filtered.append(row)

    units = build_imaging_units_advisory(filtered, targets, data_dir)
    ready_units = [u for u in units if u.ready]
    not_ready = [u for u in units if not u.ready]

    scales_str = json.dumps(filters.scales)
    to_run: list[ImagingUnit] = []
    skipped_done = 0
    skipped_active = 0

    for u in ready_units:
        with db_manager.connect() as con:
            if not filters.re_run and ImagingRunsQueries.success_exists(
                con,
                u.params_id,
                method=filters.method,
                sdgain=filters.sdgain if filters.method == "sdintimaging" else None,
                deconvolver=filters.deconvolver,
                scales=scales_str,
            ):
                skipped_done += 1
                continue

            if filters.exclude_active:
                active = ImagingRunsQueries.list_active_for_params(
                    con, u.params_id,
                )
                non_terminal = [
                    r for r in active
                    if r.get("dispatch_state") in (None, DispatchState.RUNNING)
                ]
                if non_terminal:
                    skipped_active += 1
                    continue

        to_run.append(u)

    if filters.limit is not None:
        to_run = to_run[: filters.limit]

    return SelectionResult(
        ready=to_run,
        not_ready=not_ready,
        skipped_already_done=skipped_done,
        skipped_active=skipped_active,
        skipped_filtered_out=skipped_filter,
    )
