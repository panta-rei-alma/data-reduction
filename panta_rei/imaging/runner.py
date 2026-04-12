"""Imaging execution engine (tclean+feather and sdintimaging).

Handles lazy casatasks import, trusted-mode preflight (SPW/field/datacolumn
resolution via ``casatools.msmetadata``), parameter assembly, execution,
FITS export, and intermediate cleanup.

Only this module touches ``casatasks`` and ``casatools`` imports.

Two imaging paths:

- **tclean_feather** (default): joint 12m+7m ``tclean`` followed by
  ``feather`` with TP data.  All CASA products live in a per-run work
  directory; final FITS are atomically published to canonical paths.
- **sdintimaging**: joint 12m+7m+TP deconvolution via ``sdintimaging``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from panta_rei.imaging.matching import ImagingUnit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy CASA imports
# ---------------------------------------------------------------------------

def _ensure_casatasks():
    """Import and return ``(sdintimaging, exportfits, importfits)``.

    Raises :class:`ImportError` with a clear message if unavailable.
    """
    try:
        from casatasks import sdintimaging, exportfits, importfits
        return sdintimaging, exportfits, importfits
    except ImportError:
        raise ImportError(
            "casatasks not available. Install with: pip install 'panta_rei[casa]'\n"
            "Supported: CASA 6.6.6, Python 3.10, modular casatasks/casatools."
        )


def _ensure_casatasks_tclean():
    """Import and return ``(tclean, feather, exportfits, importfits)``.

    Raises :class:`ImportError` with a clear message if unavailable.
    """
    try:
        from casatasks import tclean, feather, exportfits, importfits
        return tclean, feather, exportfits, importfits
    except ImportError:
        raise ImportError(
            "casatasks not available. Install with: pip install 'panta_rei[casa]'\n"
            "Supported: CASA 6.6.6, Python 3.10, modular casatasks/casatools."
        )


def _ensure_casatools():
    """Import and return ``casatools`` module.

    Raises :class:`ImportError` with a clear message if unavailable.
    """
    try:
        import casatools
        return casatools
    except ImportError:
        raise ImportError(
            "casatools not available. Install with: pip install 'panta_rei[casa]'\n"
            "Supported: CASA 6.6.6, Python 3.10, modular casatasks/casatools."
        )


def get_casa_version() -> str:
    """Return the CASA version string, or ``'unknown'``."""
    try:
        casatools = _ensure_casatools()
        return casatools.version_string()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Trusted-mode preflight
# ---------------------------------------------------------------------------

def resolve_spw_for_ms(
    ms_path: str,
    target_center_hz: float,
    tolerance_hz: float = 50e6,
) -> Optional[str]:
    """Find the science SPW in *ms_path* whose center frequency matches *target_center_hz*.

    Only searches SPWs associated with the ``OBSERVE_TARGET`` intent to
    avoid matching single-channel WVR or calibration SPWs that happen to
    fall at a similar frequency.

    Uses ``casatools.msmetadata`` to read SPW frequencies.
    Returns the SPW ID as a string, or *None* if no match.
    """
    casatools = _ensure_casatools()
    msmd = casatools.msmetadata()
    try:
        msmd.open(ms_path)
        target_spws = set(int(s) for s in msmd.spwsforintent("OBSERVE_TARGET*"))
        best_spw = None
        best_diff = float("inf")
        for spw_id in target_spws:
            freqs = msmd.chanfreqs(spw_id)
            if len(freqs) < 4:
                continue  # skip single-channel or very narrow SPWs
            center = (freqs[0] + freqs[-1]) / 2.0
            diff = abs(center - target_center_hz)
            if diff < tolerance_hz and diff < best_diff:
                best_diff = diff
                best_spw = spw_id
        return str(best_spw) if best_spw is not None else None
    finally:
        msmd.close()


def resolve_field_for_ms(ms_path: str, source_name: str) -> Optional[str]:
    """Verify *source_name* exists as a field in *ms_path*.

    Uses ``casatools.msmetadata.fieldsforname()``.
    Returns the field name (quoted for CASA) or *None* if not found.
    """
    casatools = _ensure_casatools()
    msmd = casatools.msmetadata()
    try:
        msmd.open(ms_path)
        field_ids = msmd.fieldsforname(source_name)
        if len(field_ids) > 0:
            return source_name
        return None
    finally:
        msmd.close()


def resolve_datacolumn(ms_path: str) -> str:
    """Determine whether *ms_path* has CORRECTED_DATA.

    Uses ``casatools.table.colnames()``.
    Returns ``'corrected'`` or ``'data'``.
    """
    casatools = _ensure_casatools()
    tb = casatools.table()
    try:
        tb.open(ms_path)
        cols = tb.colnames()
        return "corrected" if "CORRECTED_DATA" in cols else "data"
    finally:
        tb.close()


def run_trusted_preflight(unit: ImagingUnit) -> tuple[bool, str]:
    """Run trusted-mode preflight: resolve SPW, field, and datacolumn for each MS.

    Populates ``unit.spw_selection``, ``unit.field_selection``, and
    ``unit.datacolumn`` in-place.

    Returns ``(ok, message)``. On failure the unit should NOT proceed to imaging.
    """
    recovered = unit.recovered_params

    # Compute TM center frequency from recovered start/width/nchan
    from panta_rei.imaging.matching import _compute_tm_freq_range
    start = recovered.get("start", "")
    width = recovered.get("width", "")
    nchan = recovered.get("nchan")
    tm_freq = _compute_tm_freq_range(start, width, nchan)

    if tm_freq is None:
        return False, "Cannot compute TM frequency range from recovered params"

    tm_center = (tm_freq[0] + tm_freq[1]) / 2.0

    # Combine all vis (TM + SM) for unified selection lists
    all_vis = unit.vis_tm + unit.vis_sm
    spw_sel: list[str] = []
    field_sel: list[str] = []
    datacolumns: set[str] = set()

    for ms_path in all_vis:
        # SPW resolution
        spw = resolve_spw_for_ms(ms_path, tm_center)
        if spw is None:
            return False, f"No matching SPW in {Path(ms_path).name} for center freq {tm_center/1e9:.3f} GHz"
        spw_sel.append(spw)

        # Field resolution
        field = resolve_field_for_ms(ms_path, unit.source_name)
        if field is None:
            return False, f"Field '{unit.source_name}' not found in {Path(ms_path).name}"
        field_sel.append(field)

        # Datacolumn — log per-MS for diagnostics
        dc = resolve_datacolumn(ms_path)
        datacolumns.add(dc)

    unit.spw_selection = spw_sel
    unit.field_selection = field_sel

    # Always use 'corrected'. CASA automatically falls back to 'data'
    # for MSs that lack CORRECTED_DATA. This is the correct behavior
    # when TM has selfcal'd CORRECTED_DATA but SM only has DATA from
    # contsub — each MS uses its best available column.
    if len(datacolumns) > 1:
        log.info(
            "Mixed datacolumn across MSs (some corrected, some data-only). "
            "Using datacolumn='corrected' — will fall back to "
            "'data' where CORRECTED_DATA is absent."
        )
    unit.datacolumn = "corrected"

    return True, "OK"


# ---------------------------------------------------------------------------
# Parameter assembly
# ---------------------------------------------------------------------------

# Fixed parameters per plan (section 4)
FIXED_PARAMS = {
    "gridder": "mosaic",
    "pbcor": True,
    "stokes": "I",
    "dishdia": 12.0,
    "sdpsf": "",
    "usedata": "sdint",
}


def build_sdintimaging_params(
    unit: ImagingUnit,
    imagename: str,
    sdgain: float = 1.0,
    deconvolver: str = "multiscale",
    scales: Optional[list[int]] = None,
) -> dict:
    """Assemble the full parameter dict for ``sdintimaging()``.

    Merges recovered TM params with fixed overrides and CLI options.
    """
    if scales is None:
        scales = [0, 5, 10, 15, 20]

    recovered = unit.recovered_params

    params = {
        # Vis inputs
        "vis": unit.vis_tm + unit.vis_sm,
        "sdimage": unit.sdimage,
        # Selections
        "spw": unit.spw_selection,
        "field": unit.field_selection,
        "datacolumn": unit.datacolumn,
        # From recovery
        "imagename": imagename,
        "specmode": "cube",
        "imsize": recovered.get("imsize", [480, 450]),
        "cell": recovered.get("cell", ["0.42arcsec"]),
        "phasecenter": recovered.get("phasecenter", ""),
        "nchan": int(recovered.get("nchan", -1)),
        "start": recovered.get("start", ""),
        "width": recovered.get("width", ""),
        "outframe": recovered.get("outframe", "LSRK"),
        "veltype": recovered.get("veltype", "radio"),
        "weighting": recovered.get("weighting", "briggsbwtaper"),
        "robust": float(recovered.get("robust", 0.5)),
        "niter": int(recovered.get("niter", 30000)),
        "threshold": recovered.get("threshold", "0.0Jy"),
        "pblimit": float(recovered.get("pblimit", 0.2)),
        # Auto-masking from recovery
        "usemask": recovered.get("usemask", "auto-multithresh"),
        "sidelobethreshold": float(recovered.get("sidelobethreshold", 2.0)),
        "noisethreshold": float(recovered.get("noisethreshold", 4.25)),
        "lownoisethreshold": float(recovered.get("lownoisethreshold", 1.5)),
        "negativethreshold": float(recovered.get("negativethreshold", 0.0)),
        "minbeamfrac": float(recovered.get("minbeamfrac", 0.3)),
        "growiterations": int(recovered.get("growiterations", 75)),
        # Overridable
        "deconvolver": deconvolver,
        "scales": scales,
        "sdgain": sdgain,
    }

    # Apply fixed params (always override)
    params.update(FIXED_PARAMS)

    return params


# ---------------------------------------------------------------------------
# tclean parameter assembly
# ---------------------------------------------------------------------------

# Fixed parameters for tclean (no sdintimaging-specific keys)
FIXED_TCLEAN_PARAMS = {
    "gridder": "mosaic",
    "pbcor": True,
    "stokes": "I",
    "restoringbeam": "common",
}


def build_tclean_params(
    unit: ImagingUnit,
    imagename: str,
    deconvolver: str = "multiscale",
    scales: Optional[list[int]] = None,
    parallel: bool = False,
) -> dict:
    """Assemble the full parameter dict for ``tclean()``.

    Like :func:`build_sdintimaging_params` but without sdintimaging-specific
    keys (``sdimage``, ``sdgain``, ``sdpsf``, ``usedata``, ``dishdia``).
    """
    if scales is None:
        scales = [0, 5, 10, 15, 20]

    recovered = unit.recovered_params

    params = {
        # Vis inputs
        "vis": unit.vis_tm + unit.vis_sm,
        # Selections
        "spw": unit.spw_selection,
        "field": unit.field_selection,
        "datacolumn": unit.datacolumn,
        # From recovery
        "imagename": imagename,
        "specmode": "cube",
        "imsize": recovered.get("imsize", [480, 450]),
        "cell": recovered.get("cell", ["0.42arcsec"]),
        "phasecenter": recovered.get("phasecenter", ""),
        "nchan": int(recovered.get("nchan", -1)),
        "start": recovered.get("start", ""),
        "width": recovered.get("width", ""),
        "outframe": recovered.get("outframe", "LSRK"),
        "veltype": recovered.get("veltype", "radio"),
        "weighting": recovered.get("weighting", "briggsbwtaper"),
        "robust": float(recovered.get("robust", 0.5)),
        "niter": int(recovered.get("niter", 30000)),
        "threshold": recovered.get("threshold", "0.0Jy"),
        "pblimit": float(recovered.get("pblimit", 0.2)),
        # Auto-masking from recovery
        "usemask": recovered.get("usemask", "auto-multithresh"),
        "sidelobethreshold": float(recovered.get("sidelobethreshold", 2.0)),
        "noisethreshold": float(recovered.get("noisethreshold", 4.25)),
        "lownoisethreshold": float(recovered.get("lownoisethreshold", 1.5)),
        "negativethreshold": float(recovered.get("negativethreshold", 0.0)),
        "minbeamfrac": float(recovered.get("minbeamfrac", 0.3)),
        "growiterations": int(recovered.get("growiterations", 75)),
        # Overridable
        "deconvolver": deconvolver,
        "scales": scales,
        "parallel": parallel,
    }

    # Apply fixed params (always override)
    params.update(FIXED_TCLEAN_PARAMS)

    return params


# ---------------------------------------------------------------------------
# Import TP FITS → CASA image
# ---------------------------------------------------------------------------

def import_tp_to_casa_image(
    tp_fits_path: str,
    spectral_grid: Optional[dict] = None,
) -> str:
    """Import a TP FITS cube to CASA image format if needed.

    After import:
    1. Checks the axis order and transposes to ``[RA, DEC, Stokes, Freq]``
       if the frequency axis is not in position 3.
    2. If *spectral_grid* is provided (with ``nchan``, ``start``, ``width``,
       ``outframe``), regrids the spectral axis to match the interferometric
       cube grid. ``sdintimaging`` only regrids spatial axes — the spectral
       axis must already match.

    Returns the path to the CASA image directory.
    """
    _, _, importfits = _ensure_casatasks()

    casa_image = tp_fits_path.replace(".fits", ".image")
    if Path(casa_image).exists():
        log.debug("TP CASA image already exists: %s", Path(casa_image).name)
        return casa_image

    log.info("Importing TP FITS: %s → %s", Path(tp_fits_path).name, Path(casa_image).name)
    importfits(fitsimage=tp_fits_path, imagename=casa_image, overwrite=True)

    # Check axis order — sdintimaging needs freq on axis 3
    casa_image = _ensure_axis_order(casa_image)

    # Regrid spectral axis to match the interferometric grid
    if spectral_grid:
        casa_image = _regrid_spectral_axis(casa_image, spectral_grid)

    return casa_image


def _ensure_axis_order(casa_image: str) -> str:
    """Transpose a CASA image to [RA, DEC, Stokes, Freq] if needed.

    Some TP cubes have axis order [RA, DEC, Freq, Stokes]. sdintimaging
    requires frequency on axis 3 (0-indexed). Uses ``imtrans`` to fix.

    Returns the path to the (possibly transposed) image.
    """
    casatools = _ensure_casatools()
    ia = casatools.image()
    try:
        ia.open(casa_image)
        csys = ia.coordsys()
        axis_names = [csys.axiscoordinatetypes()[i] for i in range(csys.naxes())]
        ia.close()
        csys.done()
    except Exception:
        ia.close()
        raise

    # Expected order: Direction, Direction, Stokes, Spectral
    # If Spectral is already at index 3, no transpose needed
    if len(axis_names) >= 4 and axis_names[3] == "Spectral":
        return casa_image

    # Build the reordering: we want [Direction, Direction, Stokes, Spectral]
    # Find current positions and construct the digit-string order
    target_order = ["Direction", "Direction", "Stokes", "Spectral"]
    remaining = list(range(len(axis_names)))
    order_indices = []
    for target_type in target_order:
        for i in remaining:
            if axis_names[i] == target_type:
                order_indices.append(i)
                remaining.remove(i)
                break

    order_str = "".join(str(i) for i in order_indices)
    log.info(
        "TP image axis order is %s — transposing with order='%s'",
        axis_names, order_str,
    )
    from casatasks import imtrans

    transposed = casa_image + ".transposed"
    imtrans(imagename=casa_image, outfile=transposed, order=order_str)

    # Replace original with transposed
    import shutil
    shutil.rmtree(casa_image)
    shutil.move(transposed, casa_image)

    return casa_image


def _regrid_spectral_axis(casa_image: str, spectral_grid: dict) -> str:
    """Regrid a CASA image's spectral axis to match the interferometric grid.

    ``sdintimaging`` only regrids spatial axes — the spectral axis must
    already match.  This function uses ``imregrid`` to resample the TP
    cube to the TM spectral grid defined by *spectral_grid* (keys:
    ``nchan``, ``start``, ``width``, ``outframe``).

    Returns the path to the regridded image (replaces the original).
    """
    from casatasks import imregrid

    nchan = spectral_grid.get("nchan")
    start = spectral_grid.get("start", "")
    width = spectral_grid.get("width", "")
    outframe = spectral_grid.get("outframe", "LSRK")

    if not nchan or not start or not width:
        log.warning(
            "Spectral grid incomplete (nchan=%s, start=%s, width=%s) "
            "— skipping spectral regrid",
            nchan, start, width,
        )
        return casa_image

    log.info(
        "Regridding TP spectral axis: nchan=%s start=%s width=%s outframe=%s",
        nchan, start, width, outframe,
    )

    # Build a template coordinate system with the target spectral grid.
    # Read the existing image's csys, modify only the spectral part.
    casatools = _ensure_casatools()
    ia = casatools.image()
    ia.open(casa_image)
    csys = ia.coordsys()

    # Parse start frequency — remove unit suffix for csys
    import re
    start_match = re.match(r"([0-9.eE+-]+)\s*(GHz|MHz|Hz|kHz)?", str(start))
    if start_match:
        start_val = float(start_match.group(1))
        start_unit = start_match.group(2) or "Hz"
    else:
        ia.close()
        log.warning("Cannot parse start frequency '%s' — skipping spectral regrid", start)
        return casa_image

    width_match = re.match(r"([0-9.eE+-]+)\s*(GHz|MHz|Hz|kHz)?", str(width))
    if width_match:
        width_val = float(width_match.group(1))
        width_unit = width_match.group(2) or "Hz"
    else:
        ia.close()
        log.warning("Cannot parse width '%s' — skipping spectral regrid", width)
        return casa_image

    # Convert to Hz for csys
    unit_mult = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
    start_hz = start_val * unit_mult.get(start_unit, 1)
    width_hz = width_val * unit_mult.get(width_unit, 1)

    csys.setreferencecode(outframe, "spectral")
    csys.setreferencevalue(start_hz, "spectral")
    csys.setincrement(width_hz, "spectral")
    csys.setreferencepixel(0, "spectral")

    # Get current shape and replace spectral axis length
    shape = list(ia.shape())
    # Find spectral axis index
    axis_types = [csys.axiscoordinatetypes()[i] for i in range(csys.naxes())]
    spec_idx = axis_types.index("Spectral")
    shape[spec_idx] = int(nchan)

    ia.close()

    from casatasks import imregrid

    regridded = casa_image + ".regridded"
    imregrid(
        imagename=casa_image,
        output=regridded,
        template={
            "csys": csys.torecord(),
            "shap": shape,
        },
        axes=[spec_idx],
        overwrite=True,
    )
    csys.done()

    # Replace original with regridded
    import shutil
    shutil.rmtree(casa_image)
    shutil.move(regridded, casa_image)

    return casa_image


# ---------------------------------------------------------------------------
# FITS export
# ---------------------------------------------------------------------------

def export_pbcor_to_fits(imagename: str, output_fits: str) -> None:
    """Export the ``.joint.cube.image.pbcor`` product to FITS."""
    _, exportfits, _ = _ensure_casatasks()

    pbcor_image = f"{imagename}.joint.cube.image.pbcor"
    if not Path(pbcor_image).exists():
        raise FileNotFoundError(
            f"Expected pbcor image not found: {pbcor_image}"
        )

    Path(output_fits).parent.mkdir(parents=True, exist_ok=True)
    log.info("Exporting to FITS: %s", Path(output_fits).name)
    exportfits(imagename=pbcor_image, fitsimage=output_fits, overwrite=True)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

# Products to keep regardless of --keep-intermediates
_KEEP_SUFFIXES = {".joint.cube.image.pbcor", ".joint.cube.pb"}

# Products to delete unless --keep-intermediates
_DELETE_SUFFIXES = {
    ".joint.cube.psf",
    ".joint.cube.residual",
    ".joint.cube.model",
    ".joint.cube.sumwt",
    ".joint.cube.weight",
    ".cf",
}


def cleanup_intermediates(imagename: str, keep_all: bool = False) -> int:
    """Remove sdintimaging intermediate products.

    Returns the number of directories removed.
    """
    if keep_all:
        log.info("Keeping all intermediate products (--keep-intermediates)")
        return 0

    removed = 0
    for suffix in _DELETE_SUFFIXES:
        product = Path(f"{imagename}{suffix}")
        if product.exists() and product.is_dir():
            shutil.rmtree(product)
            log.debug("Removed: %s", product.name)
            removed += 1

    if removed:
        log.info("Cleaned up %d intermediate products", removed)
    return removed


# ---------------------------------------------------------------------------
# Main execution entry point
# ---------------------------------------------------------------------------

def run_sdintimaging(
    unit: ImagingUnit,
    output_dir: Path,
    sdgain: float = 1.0,
    deconvolver: str = "multiscale",
    scales: Optional[list[int]] = None,
    keep_intermediates: bool = False,
    dry_run: bool = False,
) -> tuple[bool, str, Optional[str]]:
    """Execute sdintimaging for a single ImagingUnit.

    Returns ``(success, message, output_fits_path)``.
    """
    from panta_rei.imaging.matching import build_output_path

    if scales is None:
        scales = [0, 5, 10, 15, 20]

    # 1. Trusted preflight
    ok, msg = run_trusted_preflight(unit)
    if not ok:
        return False, f"Trusted preflight failed: {msg}", None

    # 2. Determine output paths
    output_fits_path = build_output_path(
        output_dir,
        unit.gous_uid,
        unit.source_name,
        unit.tp_freq_min or 0,
        unit.tp_freq_max or 0,
    )

    # imagename goes in the output subdir (without .fits extension)
    imagename = str(output_fits_path).replace(".pbcor.fits", "")

    if dry_run:
        log.info("[DRY] Would run sdintimaging → %s", output_fits_path.name)
        return True, "dry-run", str(output_fits_path)

    # 3. Import TP FITS to CASA image (with spectral regridding to match TM grid)
    recovered = unit.recovered_params
    spectral_grid = {
        "nchan": recovered.get("nchan"),
        "start": recovered.get("start"),
        "width": recovered.get("width"),
        "outframe": recovered.get("outframe", "LSRK"),
    }
    sdimage_casa = import_tp_to_casa_image(unit.sdimage, spectral_grid=spectral_grid)

    # 4. Build parameter dict
    params = build_sdintimaging_params(
        unit,
        imagename=imagename,
        sdgain=sdgain,
        deconvolver=deconvolver,
        scales=scales,
    )
    # Override sdimage with CASA image path
    params["sdimage"] = sdimage_casa

    # 5. Ensure output directory exists
    output_fits_path.parent.mkdir(parents=True, exist_ok=True)

    # 6. Write job spec JSON
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_json_path = jobs_dir / f"{output_fits_path.stem}.json"
    job_spec = {
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()},
        "unit": {
            "gous_uid": unit.gous_uid,
            "source_name": unit.source_name,
            "line_group": unit.line_group,
            "spw_id": unit.spw_id,
            "params_id": unit.params_id,
            "vis_tm": unit.vis_tm,
            "vis_sm": unit.vis_sm,
            "sdimage": unit.sdimage,
            "mous_uids_tm": unit.mous_uids_tm,
            "mous_uids_sm": unit.mous_uids_sm,
            "mous_uids_tp": unit.mous_uids_tp,
        },
        "overrides": {
            "sdgain": sdgain,
            "deconvolver": deconvolver,
            "scales": scales,
        },
        "casa_version": get_casa_version(),
    }
    job_json_path.write_text(json.dumps(job_spec, indent=2, default=str))
    log.info("Wrote job spec: %s", job_json_path.name)

    # 7. Execute sdintimaging
    sdintimaging, _, _ = _ensure_casatasks()
    log.info(
        "Running sdintimaging: %s/%s/spw%s → %s",
        unit.gous_uid, unit.source_name, unit.spw_id,
        Path(imagename).name,
    )
    sdintimaging(**params)

    # 8. Export pbcor to FITS
    export_pbcor_to_fits(imagename, str(output_fits_path))

    # 9. Cleanup
    cleanup_intermediates(imagename, keep_all=keep_intermediates)

    log.info("SUCCESS: %s", output_fits_path.name)
    return True, "success", str(output_fits_path)


# ---------------------------------------------------------------------------
# tclean + feather execution
# ---------------------------------------------------------------------------

def prepare_tp_for_feather(
    tp_fits_path: str,
    tclean_image: str,
    run_dir: Path,
) -> str:
    """Import TP FITS, regrid, and transpose to match tclean output.

    The TP cube is imported to CASA image format under *run_dir*/tp/
    (not beside the source FITS), regridded with ``imregrid`` using
    *tclean_image* as the template, then transposed to match tclean's
    axis order ``[Direction, Direction, Stokes, Spectral]``.

    The transpose is done AFTER imregrid (not before) to avoid
    corrupting per-channel beam sets.  With ``restoringbeam='common'``
    on the tclean side, the beam set is trivial and imtrans is safe.

    Returns the path to the regridded + transposed TP CASA image.
    """
    from casatasks import imregrid

    tp_work = run_dir / "tp"
    tp_work.mkdir(parents=True, exist_ok=True)

    # Import to CASA image under tp_work
    _, _, importfits = _ensure_casatasks()
    tp_basename = Path(tp_fits_path).stem + ".image"
    casa_image = str(tp_work / tp_basename)

    if not Path(casa_image).exists():
        log.info("Importing TP FITS: %s → %s", Path(tp_fits_path).name, tp_basename)
        importfits(fitsimage=tp_fits_path, imagename=casa_image, overwrite=True)
    else:
        log.debug("TP CASA image already exists: %s", tp_basename)

    # Regrid to match the tclean output spatial/spectral grid
    regridded = casa_image + ".regridded"
    if Path(regridded).exists():
        shutil.rmtree(regridded)
    log.info("Regridding TP to match tclean image: %s", Path(tclean_image).name)
    imregrid(
        imagename=casa_image,
        output=regridded,
        template=tclean_image,
        overwrite=True,
    )

    # Transpose to match tclean axis order [RA, DEC, Stokes, Freq].
    # imregrid preserves the TP's native axis order [RA, DEC, Freq, Stokes],
    # but feather requires both images in the same order.
    regridded = _ensure_axis_order(regridded)

    return regridded


def run_feather_task(
    highres: str,
    lowres: str,
    output_image: str,
) -> None:
    """Run CASA ``feather`` to combine interferometric and TP images.

    Parameters
    ----------
    highres : str
        Path to the tclean ``.image.pbcor`` CASA image.
    lowres : str
        Path to the regridded TP CASA image.
    output_image : str
        Path for the feathered output CASA image.
    """
    _, feather, _, _ = _ensure_casatasks_tclean()

    if Path(output_image).exists():
        shutil.rmtree(output_image)

    log.info(
        "Feathering: highres=%s lowres=%s → %s",
        Path(highres).name, Path(lowres).name, Path(output_image).name,
    )
    feather(imagename=output_image, highres=highres, lowres=lowres)


def export_to_fits(casa_image: str, output_fits: str) -> None:
    """Export a CASA image to FITS. No ``dropdeg``."""
    _, _, exportfits, _ = _ensure_casatasks_tclean()

    Path(output_fits).parent.mkdir(parents=True, exist_ok=True)
    log.info("Exporting to FITS: %s", Path(output_fits).name)
    exportfits(imagename=casa_image, fitsimage=output_fits, overwrite=True)


def _atomic_publish(src: str, dst: str) -> None:
    """Copy *src* to *dst* atomically via temp file + ``os.replace``."""
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        dir=str(dst_path.parent),
        prefix=f".{dst_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        log.info("Published: %s", dst_path.name)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def run_tclean_feather(
    unit: ImagingUnit,
    output_dir: Path,
    row_id: int,
    deconvolver: str = "multiscale",
    scales: Optional[list[int]] = None,
    parallel: bool = False,
    keep_intermediates: bool = False,
    dry_run: bool = False,
) -> tuple[bool, str, Optional[str]]:
    """Execute tclean (12m+7m) followed by feather (with TP).

    All CASA products are written under a per-run work directory
    ``output_dir/runs/{row_id}/``.  Final FITS are atomically published
    to canonical output paths on success.

    Returns ``(success, message, canonical_12m7mTP_fits_path)``.
    """
    from panta_rei.imaging.matching import (
        build_output_path,
        build_tclean_only_output_path,
        _compute_tm_freq_range,
    )

    if scales is None:
        scales = [0, 5, 10, 15, 20]

    # 1. Trusted preflight
    ok, msg = run_trusted_preflight(unit)
    if not ok:
        return False, f"Trusted preflight failed: {msg}", None

    # 2. Compute TM frequency range from recovered params
    recovered = unit.recovered_params
    tm_freq = _compute_tm_freq_range(
        recovered.get("start", ""),
        recovered.get("width", ""),
        recovered.get("nchan"),
    )
    if tm_freq is None:
        return False, "Cannot compute TM frequency range from recovered params", None

    freq_min, freq_max = tm_freq

    # 3. Determine canonical output paths
    canonical_feathered = build_output_path(
        output_dir, unit.gous_uid, unit.source_name, freq_min, freq_max,
    )
    canonical_tclean = build_tclean_only_output_path(
        output_dir, unit.gous_uid, unit.source_name, freq_min, freq_max,
    )

    if dry_run:
        log.info("[DRY] Would run tclean+feather → %s", canonical_feathered.name)
        return True, "dry-run", str(canonical_feathered)

    # 4. Set up per-run work directory
    run_dir = output_dir / "runs" / str(row_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # imagename prefix for tclean products (under run_dir).
    # Uses the 12m7m stem — only the feathered output gets '12m7mTP'.
    imagename = str(run_dir / canonical_tclean.stem.replace(".pbcor", ""))

    # 5. Build tclean params
    params = build_tclean_params(
        unit,
        imagename=imagename,
        deconvolver=deconvolver,
        scales=scales,
        parallel=parallel,
    )

    # 6. Write job spec JSON
    job_json_path = run_dir / "job.json"
    job_spec = {
        "method": "tclean_feather",
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()},
        "unit": {
            "gous_uid": unit.gous_uid,
            "source_name": unit.source_name,
            "line_group": unit.line_group,
            "spw_id": unit.spw_id,
            "params_id": unit.params_id,
            "vis_tm": unit.vis_tm,
            "vis_sm": unit.vis_sm,
            "sdimage": unit.sdimage,
            "mous_uids_tm": unit.mous_uids_tm,
            "mous_uids_sm": unit.mous_uids_sm,
            "mous_uids_tp": unit.mous_uids_tp,
        },
        "overrides": {
            "deconvolver": deconvolver,
            "scales": scales,
            "parallel": parallel,
        },
        "canonical_paths": {
            "feathered": str(canonical_feathered),
            "tclean_only": str(canonical_tclean),
        },
        "run_dir": str(run_dir),
        "casa_version": get_casa_version(),
    }
    job_json_path.write_text(json.dumps(job_spec, indent=2, default=str))
    log.info("Wrote job spec: %s", job_json_path)

    # 7. Execute tclean (or skip if products already exist)
    pbcor_image = f"{imagename}.image.pbcor"

    if Path(pbcor_image).exists():
        log.info(
            "tclean products already exist, skipping to feather: %s",
            Path(pbcor_image).name,
        )
    else:
        tclean, _, _, _ = _ensure_casatasks_tclean()
        log.info(
            "Running tclean: %s/%s/spw%s → %s",
            unit.gous_uid, unit.source_name, unit.spw_id,
            Path(imagename).name,
        )
        tclean(**params)

        if not Path(pbcor_image).exists():
            return False, f"tclean did not produce {Path(pbcor_image).name}", None

    # 8. Export tclean-only product to run_dir
    local_tclean_fits = str(run_dir / canonical_tclean.name)
    export_to_fits(pbcor_image, local_tclean_fits)

    # 9. Prepare TP for feathering
    tp_regridded = prepare_tp_for_feather(
        unit.sdimage, f"{imagename}.image", run_dir,
    )

    # 10. Feather
    feathered_image = f"{imagename}.image.pbcor.feather"
    run_feather_task(pbcor_image, tp_regridded, feathered_image)

    # 11. Export feathered product to run_dir
    local_feathered_fits = str(run_dir / canonical_feathered.name)
    export_to_fits(feathered_image, local_feathered_fits)

    # 12. Atomic publish to canonical paths (only on success)
    _atomic_publish(local_tclean_fits, str(canonical_tclean))
    _atomic_publish(local_feathered_fits, str(canonical_feathered))

    log.info("SUCCESS: %s", canonical_feathered.name)
    return True, "success", str(canonical_feathered)


# ---------------------------------------------------------------------------
# Parallel tclean+feather via mpicasa subprocess
# ---------------------------------------------------------------------------

def run_tclean_feather_parallel(
    unit: ImagingUnit,
    output_dir: Path,
    row_id: int,
    nproc: int = 4,
    casa_path: Optional[str] = None,
    deconvolver: str = "multiscale",
    scales: Optional[list[int]] = None,
    keep_intermediates: bool = False,
    dry_run: bool = False,
) -> tuple[bool, str, Optional[str]]:
    """Execute tclean+feather via mpicasa subprocess for MPI parallelism.

    The pipeline handles preflight and DB tracking.  The actual CASA
    execution is delegated to a standalone script
    (``panta_rei/casa/tclean_feather.py``) launched via mpicasa.

    Returns ``(success, message, canonical_12m7mTP_fits_path)``.
    """
    import subprocess

    from panta_rei.imaging.matching import (
        build_output_path,
        build_tclean_only_output_path,
        _compute_tm_freq_range,
    )

    if scales is None:
        scales = [0, 5, 10, 15, 20]

    # Discover CASA executables
    if not casa_path:
        return False, "CASA_PATH not configured — required for --parallel", None

    casa_bin = Path(casa_path) / "bin"
    mpicasa = casa_bin / "mpicasa"
    casa = casa_bin / "casa"

    if not mpicasa.exists():
        return False, f"mpicasa not found at {mpicasa}", None
    if not casa.exists():
        return False, f"casa not found at {casa}", None

    # 1. Trusted preflight (runs in pipeline Python with modular casatools)
    ok, msg = run_trusted_preflight(unit)
    if not ok:
        return False, f"Trusted preflight failed: {msg}", None

    # 2. Compute TM frequency range
    recovered = unit.recovered_params
    tm_freq = _compute_tm_freq_range(
        recovered.get("start", ""),
        recovered.get("width", ""),
        recovered.get("nchan"),
    )
    if tm_freq is None:
        return False, "Cannot compute TM frequency range from recovered params", None

    freq_min, freq_max = tm_freq

    # 3. Canonical output paths
    canonical_feathered = build_output_path(
        output_dir, unit.gous_uid, unit.source_name, freq_min, freq_max,
    )
    canonical_tclean = build_tclean_only_output_path(
        output_dir, unit.gous_uid, unit.source_name, freq_min, freq_max,
    )

    if dry_run:
        log.info("[DRY] Would run parallel tclean+feather → %s", canonical_feathered.name)
        return True, "dry-run", str(canonical_feathered)

    # 4. Per-run work directory
    run_dir = output_dir / "runs" / str(row_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    imagename = str(run_dir / canonical_tclean.stem.replace(".pbcor", ""))

    # 5. Build tclean params (parallel=True is set by the CASA script)
    params = build_tclean_params(
        unit, imagename=imagename,
        deconvolver=deconvolver, scales=scales, parallel=True,
    )

    # 6. Write job spec JSON
    job_json_path = run_dir / "job.json"
    job_spec = {
        "method": "tclean_feather",
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()},
        "unit": {
            "gous_uid": unit.gous_uid,
            "source_name": unit.source_name,
            "line_group": unit.line_group,
            "spw_id": unit.spw_id,
            "params_id": unit.params_id,
            "vis_tm": unit.vis_tm,
            "vis_sm": unit.vis_sm,
            "sdimage": unit.sdimage,
            "mous_uids_tm": unit.mous_uids_tm,
            "mous_uids_sm": unit.mous_uids_sm,
            "mous_uids_tp": unit.mous_uids_tp,
        },
        "overrides": {
            "deconvolver": deconvolver,
            "scales": scales,
            "parallel": True,
            "nproc": nproc,
        },
        "canonical_paths": {
            "feathered": str(canonical_feathered),
            "tclean_only": str(canonical_tclean),
        },
        "run_dir": str(run_dir),
        "casa_version": "unknown",  # determined inside CASA
    }
    job_json_path.write_text(json.dumps(job_spec, indent=2, default=str))
    log.info("Wrote job spec: %s", job_json_path)

    # 7. Locate the checked-in CASA script
    script_path = Path(__file__).resolve().parent.parent / "casa" / "tclean_feather.py"
    if not script_path.exists():
        return False, f"CASA script not found: {script_path}", None

    # 8. Launch via mpicasa
    # CASA's -c flag consumes all remaining args, so the job.json path
    # goes after the script path with no --args separator.
    cmd = [
        str(mpicasa), "-n", str(nproc),
        str(casa), "--nologger", "--nogui",
        "-c", str(script_path), str(job_json_path),
    ]
    log.info("Launching: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("mpicasa stderr:\n%s", result.stderr[-2000:] if result.stderr else "")
        return False, f"mpicasa exited with code {result.returncode}", None

    # 9. Parse result JSON
    result_json_path = run_dir / "result.json"
    if not result_json_path.exists():
        return False, "CASA script did not write result.json", None

    casa_result = json.loads(result_json_path.read_text())

    if not casa_result.get("success"):
        return False, casa_result.get("error_message", "unknown error"), None

    # 10. Atomic publish
    local_feathered = casa_result.get("feathered_fits")
    local_tclean = casa_result.get("tclean_fits")

    if local_feathered and Path(local_feathered).exists():
        _atomic_publish(local_feathered, str(canonical_feathered))
    else:
        return False, "Feathered FITS not produced by CASA script", None

    if local_tclean and Path(local_tclean).exists():
        _atomic_publish(local_tclean, str(canonical_tclean))

    log.info("SUCCESS (parallel): %s", canonical_feathered.name)
    return True, "success", str(canonical_feathered)
