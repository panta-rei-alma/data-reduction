"""Joint imaging workflow: recover TM params and run sdintimaging.

Two workflow steps:

- **RecoverParamsStep** — parse TM weblogs for tclean cube iter1 params,
  write JSON sidecars, record in ``imaging_params`` table.
- **JointImagingStep** — preflight validation + sdintimaging execution.
  Preflight is always run internally before imaging (not a separate step).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from panta_rei.core.text import now_iso
from panta_rei.db.connection import DatabaseManager
from panta_rei.db.models import (
    ImagingParamsQueries,
    ImagingParamsStatus,
    ImagingRunsQueries,
    ImagingRunStatus,
)
from panta_rei.imaging.matching import (
    ImagingUnit,
    build_imaging_units_advisory,
    build_output_path,
    find_member_dir,
    load_targets_csv,
)
from panta_rei.imaging.recovery import has_staged_weblog, recover_params_for_mous
from panta_rei.workflows.base import Step, StepResult, WorkflowContext

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ImagingOptions:
    """Imaging-specific parameters that augment WorkflowContext."""

    weblog_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    obs_csv: Optional[Path] = None
    step: str = "all"              # 'recover', 'preflight', 'image', 'all'
    re_run: bool = False
    match: Optional[str] = None
    limit: Optional[int] = None
    include_sources: Optional[list[str]] = None
    include_line_groups: Optional[list[str]] = None
    method: str = "tclean_feather" # 'tclean_feather' or 'sdintimaging'
    sdgain: float = 1.0            # only for sdintimaging
    deconvolver: str = "multiscale"
    scales: list[int] = field(default_factory=lambda: [0, 5, 10, 15, 20])
    keep_intermediates: bool = False
    parallel: bool = False          # use mpicasa for tclean
    nproc: int = 4                  # MPI processes (only with --parallel)


# ---------------------------------------------------------------------------
# RecoverParamsStep
# ---------------------------------------------------------------------------

class RecoverParamsStep(Step):
    """Parse TM weblogs and record recovered tclean params."""

    def __init__(self, options: Optional[ImagingOptions] = None) -> None:
        self._opts = options or ImagingOptions()

    @property
    def name(self) -> str:
        return "recover"

    @property
    def description(self) -> str:
        return "Recover tclean imaging parameters from TM weblogs"

    def should_skip(self, ctx: WorkflowContext) -> str | None:
        base = super().should_skip(ctx)
        if base is not None:
            return base
        if self._opts.step not in ("recover", "all"):
            return f"step={self._opts.step} does not include recover"
        return None

    def run(self, ctx: WorkflowContext) -> StepResult:
        opts = self._opts
        weblog_dir = opts.weblog_dir or ctx.config.weblog_dir
        csv_path = opts.obs_csv or ctx.targets_csv_path

        if not csv_path.exists():
            return StepResult(
                success=False,
                summary=f"Targets CSV not found: {csv_path}",
                errors=[str(csv_path)],
            )

        targets = load_targets_csv(csv_path)

        # Compile optional regex filter
        regex: Optional[re.Pattern] = None
        if opts.match:
            try:
                regex = re.compile(opts.match, re.IGNORECASE)
            except re.error:
                regex = None

        # Recovered params output directory
        recovered_dir = ctx.base_dir / "imaging" / "recovered"
        if not ctx.dry_run:
            recovered_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        skipped = 0
        errors: list[str] = []

        for gous_uid, groups in targets.items():
            # Collect TM members for this GOUS
            tm_groups = [g for g in groups if g.array == "TM"]
            if not tm_groups:
                continue

            for tg in tm_groups:
                source = tg.source_name
                lg = tg.line_group

                # Apply filters
                if opts.include_sources and source not in opts.include_sources:
                    continue
                if opts.include_line_groups and lg not in opts.include_line_groups:
                    continue
                if regex and not regex.search(source) and not regex.search(gous_uid):
                    continue

                # Recover params from ALL TM MOUSs, then compare for
                # divergence before recording (plan section 3.3).
                all_mous_results: dict[str, dict[tuple[str, str], dict]] = {}
                for mous_id in tg.mous_ids:
                    full_mous = _ensure_full_uid(mous_id)

                    # Skip MOUSs not available on disk or in staged weblogs
                    if (
                        not has_staged_weblog(weblog_dir, full_mous)
                        and not find_member_dir(ctx.data_dir, mous_id)
                    ):
                        log.debug(
                            "Skipping MOUS %s: not available on disk",
                            mous_id,
                        )
                        skipped += 1
                        continue

                    log.info(
                        "Recovering params: GOUS=%s source=%s MOUS=%s",
                        gous_uid, source, full_mous,
                    )
                    result = recover_params_for_mous(
                        full_mous,
                        weblog_dir,
                        ctx.data_dir,
                    )
                    if result is None:
                        log.warning(
                            "No params recovered for MOUS %s", mous_id
                        )
                        skipped += 1
                        continue
                    all_mous_results[mous_id] = result

                if not all_mous_results:
                    continue

                # Use first MOUS (sorted UID) as canonical
                canonical_mous = sorted(all_mous_results.keys())[0]
                canonical = all_mous_results[canonical_mous]

                # Check key-set consistency: all MOUSs must have the
                # same set of (field, spw) pairs (plan section 3.3).
                canonical_keys = set(canonical.keys())
                keyset_error = _check_keyset_mismatch(
                    canonical_mous, canonical_keys, all_mous_results
                )

                # Iterate ALL keys across all MOUSs (union), not just canonical
                all_keys = set()
                for results in all_mous_results.values():
                    all_keys.update(results.keys())

                for (field_name, spw_id) in sorted(all_keys):
                    # Use keyset error if this key is missing from some MOUSs
                    divergence = keyset_error
                    if divergence is None:
                        divergence = _check_param_divergence(
                            field_name, spw_id, all_mous_results
                        )

                    # Get params from canonical if available, else first MOUS that has it
                    canon_params = canonical.get((field_name, spw_id))
                    if canon_params is None:
                        for results in all_mous_results.values():
                            if (field_name, spw_id) in results:
                                canon_params = results[(field_name, spw_id)]
                                break
                    if canon_params is None:
                        continue

                    # Check existing in DB
                    with ctx.db_manager.connect() as con:
                        existing = ImagingParamsQueries.get_by_key(
                            con, gous_uid, field_name, lg, spw_id
                        )

                    if existing and not opts.re_run:
                        log.debug(
                            "Already recovered: %s/%s/%s/%s",
                            gous_uid, field_name, lg, spw_id,
                        )
                        skipped += 1
                        continue

                    # Write JSON sidecar
                    safe_source = field_name.replace('"', "").replace("'", "")
                    json_name = (
                        f"{gous_uid}.{safe_source}.{lg or 'none'}"
                        f".spw{spw_id}.json"
                    )
                    json_path = recovered_dir / json_name

                    # Determine status — divergence → failed
                    if divergence:
                        status = ImagingParamsStatus.FAILED
                        error_msg = divergence
                        log.error(
                            "DIVERGENCE: %s/%s/spw%s — %s",
                            gous_uid, field_name, spw_id, divergence,
                        )
                    else:
                        status = ImagingParamsStatus.RECOVERED
                        error_msg = None

                    if not ctx.dry_run:
                        json_path.write_text(
                            json.dumps(canon_params, indent=2, default=str)
                        )

                    # Locate the specific weblog used
                    weblog_path_specific = str(weblog_dir)
                    from panta_rei.imaging.recovery import find_casa_commands_log
                    log_file = find_casa_commands_log(
                        weblog_dir,
                        _ensure_full_uid(canonical_mous),
                        ctx.data_dir,
                    )
                    if log_file:
                        weblog_path_specific = str(log_file)

                    # Extract summary fields
                    imsize = json.dumps(canon_params.get("imsize")) if canon_params.get("imsize") else None
                    cell = json.dumps(canon_params.get("cell")) if canon_params.get("cell") else None
                    nchan = canon_params.get("nchan")
                    phasecenter = canon_params.get("phasecenter")
                    robust = canon_params.get("robust")

                    if ctx.dry_run:
                        log.info(
                            "[DRY] Would record: %s/%s/%s/spw%s (status=%s)",
                            gous_uid, field_name, lg, spw_id, status,
                        )
                    else:
                        with ctx.db_manager.connect() as con:
                            ImagingParamsQueries.upsert(
                                con,
                                gous_uid=gous_uid,
                                source_name=field_name,
                                line_group=lg,
                                spw_id=spw_id,
                                mous_uids_tm=tg.mous_ids,
                                params_json_path=str(json_path),
                                params_source="weblog",
                                weblog_path=weblog_path_specific,
                                status=status,
                                error_message=error_msg,
                                imsize=imsize,
                                cell=cell,
                                nchan=int(nchan) if nchan is not None else None,
                                phasecenter=phasecenter,
                                robust=float(robust) if robust is not None else None,
                            )
                            con.commit()

                    processed += 1
                    if divergence:
                        errors.append(
                            f"{gous_uid}/{field_name}/spw{spw_id}: {divergence}"
                        )
                    log.info(
                        "Recorded: %s/%s/%s/spw%s (status=%s)",
                        gous_uid, field_name, lg, spw_id, status,
                    )

                    if opts.limit and processed >= opts.limit:
                        break
                if opts.limit and processed >= opts.limit:
                    break
            if opts.limit and processed >= opts.limit:
                break

        summary_parts = [f"Recovered {processed} param sets"]
        if skipped:
            summary_parts.append(f"{skipped} skipped")
        if errors:
            summary_parts.append(f"{len(errors)} failed")

        return StepResult(
            success=len(errors) == 0,
            summary="; ".join(summary_parts),
            items_processed=processed,
            items_skipped=skipped,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# JointImagingStep
# ---------------------------------------------------------------------------

class JointImagingStep(Step):
    """Run advisory preflight and (optionally) imaging execution."""

    def __init__(self, options: Optional[ImagingOptions] = None) -> None:
        self._opts = options or ImagingOptions()

    @property
    def name(self) -> str:
        return "image"

    @property
    def description(self) -> str:
        if self._opts.method == "sdintimaging":
            return "Joint 12m+7m+TP imaging via sdintimaging"
        return "Joint 12m+7m imaging (tclean) + TP feathering"

    def should_skip(self, ctx: WorkflowContext) -> str | None:
        base = super().should_skip(ctx)
        if base is not None:
            return base
        if self._opts.step not in ("preflight", "image", "all"):
            return f"step={self._opts.step} does not include image/preflight"
        return None

    def run(self, ctx: WorkflowContext) -> StepResult:
        opts = self._opts
        csv_path = opts.obs_csv or ctx.targets_csv_path

        if not csv_path.exists():
            return StepResult(
                success=False,
                summary=f"Targets CSV not found: {csv_path}",
                errors=[str(csv_path)],
            )

        targets = load_targets_csv(csv_path)

        # Load all recovered params from DB
        with ctx.db_manager.connect() as con:
            all_params = ImagingParamsQueries.get_all_recovered(con)

        if not all_params:
            return StepResult(
                success=True,
                summary="No recovered params to process",
                items_processed=0,
            )

        # Apply filters
        regex: Optional[re.Pattern] = None
        if opts.match:
            try:
                regex = re.compile(opts.match, re.IGNORECASE)
            except re.error:
                regex = None

        filtered = []
        for row in all_params:
            src = row["source_name"]
            gous = row["gous_uid"]
            lg = row.get("line_group")
            if opts.include_sources and src not in opts.include_sources:
                continue
            if opts.include_line_groups and lg not in opts.include_line_groups:
                continue
            if regex and not regex.search(src) and not regex.search(gous):
                continue
            filtered.append(row)

        if not filtered:
            return StepResult(
                success=True,
                summary="No params match filters",
                items_processed=0,
            )

        # Build imaging units with advisory preflight
        units = build_imaging_units_advisory(
            filtered, targets, ctx.data_dir
        )

        ready_units = [u for u in units if u.ready]
        not_ready = [u for u in units if not u.ready]

        for u in not_ready:
            log.info(
                "Not ready: %s/%s/spw%s — %s",
                u.gous_uid, u.source_name, u.spw_id, u.skip_reason,
            )

        log.info(
            "Preflight: %d ready, %d not ready out of %d",
            len(ready_units), len(not_ready), len(units),
        )

        # If preflight-only mode, stop here
        if opts.step == "preflight":
            return StepResult(
                success=True,
                summary=f"Preflight: {len(ready_units)} ready, {len(not_ready)} not ready",
                items_processed=len(units),
                items_skipped=len(not_ready),
            )

        # Image step requires output_dir
        if opts.output_dir is None:
            return StepResult(
                success=False,
                summary="--output-dir required for image step",
                errors=["Missing --output-dir"],
            )

        # Image step requires casatasks (trusted mode) — unless parallel
        # mode, where tclean runs in a subprocess via mpicasa
        if opts.step in ("image", "all"):
            if opts.method == "sdintimaging":
                try:
                    from casatasks import sdintimaging  # noqa: F401
                except ImportError:
                    return StepResult(
                        success=False,
                        summary="casatasks not available — required for sdintimaging",
                        errors=["pip install 'panta_rei[casa]'"],
                    )
            elif opts.method == "tclean_feather" and not opts.parallel:
                try:
                    from casatasks import tclean  # noqa: F401
                except ImportError:
                    return StepResult(
                        success=False,
                        summary="casatasks not available — required for tclean",
                        errors=["pip install 'panta_rei[casa]'"],
                    )
            # parallel mode: skip check — tclean runs via mpicasa subprocess

        # Idempotence: filter out already-successful runs
        scales_str = json.dumps(opts.scales)
        to_run: list[ImagingUnit] = []
        for u in ready_units:
            if not opts.re_run:
                with ctx.db_manager.connect() as con:
                    if ImagingRunsQueries.success_exists(
                        con,
                        u.params_id,
                        method=opts.method,
                        sdgain=opts.sdgain if opts.method == "sdintimaging" else None,
                        deconvolver=opts.deconvolver,
                        scales=scales_str,
                    ):
                        log.info(
                            "Skipping %s/%s/spw%s (already successful with %s)",
                            u.gous_uid, u.source_name, u.spw_id, opts.method,
                        )
                        continue
            to_run.append(u)

        if opts.limit:
            to_run = to_run[: opts.limit]

        if not to_run:
            return StepResult(
                success=True,
                summary="All ready units already have successful runs",
                items_processed=0,
                items_skipped=len(ready_units),
            )

        # Execute imaging for each ready unit
        from panta_rei.imaging.matching import _compute_tm_freq_range
        from panta_rei.imaging.runner import get_casa_version

        processed = 0
        errors: list[str] = []

        for u in to_run:
            log.info(
                "%s [%s]: %s/%s/spw%s — TM:%d SM:%d TP:%s",
                "[DRY]" if ctx.dry_run else "QUEUED",
                opts.method,
                u.gous_uid, u.source_name, u.spw_id,
                len(u.vis_tm), len(u.vis_sm),
                Path(u.sdimage).name if u.sdimage else "?",
            )

            if ctx.dry_run:
                # Dry-run: log what would happen, no DB writes
                processed += 1
                continue

            # Compute TM frequency range for output path
            recovered = u.recovered_params
            tm_freq = _compute_tm_freq_range(
                recovered.get("start", ""),
                recovered.get("width", ""),
                recovered.get("nchan"),
            )
            freq_min = tm_freq[0] if tm_freq else (u.tp_freq_min or 0)
            freq_max = tm_freq[1] if tm_freq else (u.tp_freq_max or 0)

            output_fits_canonical = build_output_path(
                opts.output_dir,
                u.gous_uid, u.source_name,
                freq_min, freq_max,
            )

            # Compute imagename for provenance (final stem without .pbcor.fits)
            imagename_stem = str(output_fits_canonical).replace(".pbcor.fits", "")

            # Insert DB row as QUEUED — row_id is needed before execution
            # for tclean_feather (per-run work dir determines job_json_path)
            now = now_iso()
            casa_version = get_casa_version()
            with ctx.db_manager.connect() as con:
                row_id = ImagingRunsQueries.insert_row(
                    con,
                    params_id=u.params_id,
                    gous_uid=u.gous_uid,
                    source_name=u.source_name,
                    line_group=u.line_group,
                    spw_id=u.spw_id,
                    vis_tm=json.dumps(u.vis_tm),
                    vis_sm=json.dumps(u.vis_sm),
                    sdimage=u.sdimage,
                    mous_uids_tm=json.dumps(u.mous_uids_tm),
                    mous_uids_sm=json.dumps(u.mous_uids_sm),
                    mous_uids_tp=json.dumps(u.mous_uids_tp),
                    output_dir=str(opts.output_dir),
                    imagename=imagename_stem,
                    sdgain=opts.sdgain if opts.method == "sdintimaging" else None,
                    deconvolver=opts.deconvolver,
                    scales=scales_str,
                    casa_version=casa_version,
                    started_at=now,
                    status=ImagingRunStatus.QUEUED,
                    hostname=socket.gethostname(),
                    method=opts.method,
                    parallel=1 if opts.parallel else 0,
                )
                con.commit()

            # For tclean_feather, job_json lives in the per-run dir
            # (only known after row_id). Update the row.
            if opts.method == "tclean_feather":
                job_json_path = str(opts.output_dir / "runs" / str(row_id) / "job.json")
            else:
                job_json_path = str(opts.output_dir / "jobs" / f"{output_fits_canonical.stem}.json")
            with ctx.db_manager.connect() as con:
                con.execute(
                    "UPDATE imaging_runs SET job_json_path = ? WHERE id = ?",
                    (job_json_path, row_id),
                )
                con.commit()

            # Mark RUNNING
            with ctx.db_manager.connect() as con:
                ImagingRunsQueries.mark_running(con, row_id)
                con.commit()

            t0 = dt.datetime.now()
            try:
                if opts.method == "tclean_feather" and opts.parallel:
                    from panta_rei.imaging.runner import run_tclean_feather_parallel
                    success, msg, output_fits = run_tclean_feather_parallel(
                        unit=u,
                        output_dir=opts.output_dir,
                        row_id=row_id,
                        nproc=opts.nproc,
                        casa_path=str(ctx.config.casa_path) if ctx.config.casa_path else None,
                        deconvolver=opts.deconvolver,
                        scales=opts.scales,
                        keep_intermediates=opts.keep_intermediates,
                        dry_run=False,
                    )
                elif opts.method == "tclean_feather":
                    from panta_rei.imaging.runner import run_tclean_feather
                    success, msg, output_fits = run_tclean_feather(
                        unit=u,
                        output_dir=opts.output_dir,
                        row_id=row_id,
                        deconvolver=opts.deconvolver,
                        scales=opts.scales,
                        parallel=False,
                        keep_intermediates=opts.keep_intermediates,
                        dry_run=False,
                    )
                else:
                    from panta_rei.imaging.runner import run_sdintimaging
                    success, msg, output_fits = run_sdintimaging(
                        unit=u,
                        output_dir=opts.output_dir,
                        sdgain=opts.sdgain,
                        deconvolver=opts.deconvolver,
                        scales=opts.scales,
                        keep_intermediates=opts.keep_intermediates,
                        dry_run=False,
                    )
            except Exception as exc:
                success = False
                msg = str(exc)
                output_fits = None
                log.error(
                    "%s failed for %s/%s/spw%s: %s",
                    opts.method,
                    u.gous_uid, u.source_name, u.spw_id, exc,
                    exc_info=True,
                )

            dt_sec = (dt.datetime.now() - t0).total_seconds()
            status = ImagingRunStatus.SUCCESS if success else ImagingRunStatus.FAILED
            retcode = 0 if success else 1

            with ctx.db_manager.connect() as con:
                # Store resolved selections (populated by trusted preflight)
                if u.spw_selection or u.field_selection:
                    ImagingRunsQueries.update_resolved(
                        con, row_id,
                        spw_selection=json.dumps(u.spw_selection) if u.spw_selection else None,
                        field_selection=json.dumps(u.field_selection) if u.field_selection else None,
                    )
                ImagingRunsQueries.mark_done(
                    con,
                    row_id,
                    status=status,
                    retcode=retcode,
                    finished_at=now_iso(),
                    duration_sec=dt_sec,
                    output_fits=output_fits,
                )
                con.commit()

            log.info(
                "Finished %s/%s/spw%s [%s]: %s (%.1fs) %s",
                u.gous_uid, u.source_name, u.spw_id,
                opts.method, status, dt_sec, msg,
            )

            processed += 1
            if not success:
                errors.append(
                    f"{u.gous_uid}/{u.source_name}/spw{u.spw_id}: {msg}"
                )

        summary_parts = [f"Processed {processed} imaging units"]
        if not_ready:
            summary_parts.append(f"{len(not_ready)} not ready")
        if errors:
            summary_parts.append(f"{len(errors)} failed")

        return StepResult(
            success=len(errors) == 0,
            summary="; ".join(summary_parts),
            items_processed=processed,
            items_skipped=len(not_ready),
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Keys compared for multi-TM divergence check (plan section 3.3)
_DIVERGENCE_KEYS = [
    "imsize", "cell", "phasecenter", "nchan", "start", "width",
    "outframe", "veltype", "weighting", "robust", "niter", "threshold",
]

# Relative tolerance for float comparisons
_FLOAT_RTOL = 1e-6


def _check_keyset_mismatch(
    canonical_mous: str,
    canonical_keys: set[tuple[str, str]],
    all_mous_results: dict[str, dict[tuple[str, str], dict]],
) -> Optional[str]:
    """Check that all MOUSs have the same (field, spw) key set.

    Returns an error message if any MOUS has extra or missing keys, or None.
    """
    for mous_id, results in sorted(all_mous_results.items()):
        if mous_id == canonical_mous:
            continue
        other_keys = set(results.keys())
        if other_keys != canonical_keys:
            extra = other_keys - canonical_keys
            missing = canonical_keys - other_keys
            parts = []
            if extra:
                parts.append(f"MOUS {mous_id} has extra keys: {sorted(extra)}")
            if missing:
                parts.append(f"MOUS {mous_id} missing keys: {sorted(missing)}")
            return f"TM key-set mismatch: {'; '.join(parts)}"
    return None


def _check_param_divergence(
    field_name: str,
    spw_id: str,
    all_mous_results: dict[str, dict[tuple[str, str], dict]],
) -> Optional[str]:
    """Compare recovered params for (field, spw) across TM MOUSs.

    Returns an error message describing the divergence, or None if they match.
    """
    key = (field_name, spw_id)
    mous_params: list[tuple[str, dict]] = []
    for mous_id, results in sorted(all_mous_results.items()):
        if key in results:
            mous_params.append((mous_id, results[key]))

    if len(mous_params) <= 1:
        return None  # Only one MOUS — no divergence possible

    _, reference = mous_params[0]
    for mous_id, params in mous_params[1:]:
        for k in _DIVERGENCE_KEYS:
            ref_val = reference.get(k)
            cur_val = params.get(k)
            if not _values_match(ref_val, cur_val):
                return (
                    f"TM param divergence on '{k}': "
                    f"MOUS {mous_params[0][0]} has {ref_val!r}, "
                    f"MOUS {mous_id} has {cur_val!r}"
                )

    return None


def _values_match(a, b) -> bool:
    """Compare two recovered param values with float tolerance."""
    if a == b:
        return True
    if a is None or b is None:
        return False
    # Float comparison
    try:
        fa, fb = float(a), float(b)
        if fa == 0 and fb == 0:
            return True
        return abs(fa - fb) / max(abs(fa), abs(fb)) < _FLOAT_RTOL
    except (ValueError, TypeError):
        pass
    # String comparison (case-insensitive for CASA strings)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return str(a) == str(b)


def _ensure_full_uid(mous_id: str) -> str:
    """Ensure a MOUS ID is in full ``uid___A001_...`` form.

    CSV mous_ids are sometimes just xpairs like ``X3833_X64bc``.
    Prepend ``uid___A001_`` if the prefix is missing.
    """
    s = mous_id.strip()
    if s.lower().startswith("uid___"):
        return s
    # Looks like an xpair (e.g. X3833_X64bc) — prepend standard prefix
    return f"uid___A001_{s}"


# ---------------------------------------------------------------------------
# Convenience orchestration
# ---------------------------------------------------------------------------

def run_imaging(
    ctx: WorkflowContext,
    options: Optional[ImagingOptions] = None,
) -> dict[str, StepResult]:
    """Run the imaging workflow.

    If ``options.step`` is ``'recover'``, only :class:`RecoverParamsStep` runs.
    If ``'preflight'`` or ``'image'``, only :class:`JointImagingStep` runs.
    If ``'all'``, both run in sequence.
    """
    from panta_rei.workflows.base import run_workflow

    opts = options or ImagingOptions()
    steps: list[Step] = []

    if opts.step in ("recover", "all"):
        steps.append(RecoverParamsStep(opts))
    if opts.step in ("preflight", "image", "all"):
        steps.append(JointImagingStep(opts))

    return run_workflow(steps, ctx)
