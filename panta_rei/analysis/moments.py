"""Moment-map and mean-spectrum generation from imaging cubes.

Pure compute kernel — no argparse. The CLI driver lives in
``panta_rei.cli.run_moments``.

For each ``*.cube.pbcor.fits`` we produce up to three FITS products:

- ``<basename>.integrated_intensity.fits`` — 2D moment-0 map
- ``<basename>.peak_intensity.fits``        — 2D peak-intensity map
- ``<basename>.mean_spectrum.fits``         — 1D BinTable spectrum

Outputs are written into ``<analysis_dir>/<group_dir>/`` mirroring the
imaging-output layout, via atomic ``.tmp`` + ``os.replace`` writes.
Re-runs are idempotent: an output is skipped if it exists and is newer
than its input cube. Pass ``force=True`` to override.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy import units as u
from astropy.io import fits

logger = logging.getLogger(__name__)

PRODUCT_KINDS: tuple[str, ...] = (
    "integrated_intensity",
    "peak_intensity",
    "mean_spectrum",
)

CUBE_GLOB = "*.cube.pbcor.fits"

# Array-combination tokens we consider QA-worthy. Only the feathered
# (12m+7m+TP) product is included by default — the bare 12m+7m cube is
# an intermediate and its mosaic is what the joint imaging step ships.
DEFAULT_ARRAY_COMBOS: tuple[str, ...] = ("12m7mTP",)


@dataclass
class CubeResult:
    """Per-cube outcome from :func:`process_cube`.

    ``products`` maps each requested kind to one of:
    ``"written"``, ``"skipped"``, ``"dry-run"``, or ``"failed:<reason>"``.
    """

    cube: Path
    products: dict[str, str]

    @property
    def any_failed(self) -> bool:
        return any(s.startswith("failed:") for s in self.products.values())


def derive_output_paths(
    cube_fits: Path,
    analysis_dir: Path,
    products: Iterable[str] = PRODUCT_KINDS,
) -> dict[str, Path]:
    """Map a cube path to its output products, mirroring the group-dir layout.

    ``cube_fits`` is expected at
    ``<imaging_dir>/<group_dir>/<basename>.fits``.
    Returned paths land under ``<analysis_dir>/<group_dir>/``.
    """
    group_dir = cube_fits.parent.name
    if not cube_fits.name.endswith(".fits"):
        raise ValueError(f"expected .fits suffix, got {cube_fits.name}")
    stem = cube_fits.name[: -len(".fits")]
    target_dir = analysis_dir / group_dir
    return {kind: target_dir / f"{stem}.{kind}.fits" for kind in products}


def needs_regeneration(input_path: Path, output_path: Path, force: bool) -> bool:
    """Return True if ``output_path`` is missing, stale, or ``force`` is set."""
    if force or not output_path.exists():
        return True
    return output_path.stat().st_mtime < input_path.stat().st_mtime


def _resolve_rest_frequency(header: fits.Header) -> float | None:
    """Return RESTFRQ (Hz) from a FITS header, or None if absent/zero."""
    for key in ("RESTFRQ", "RESTFREQ"):
        val = header.get(key)
        if val:
            return float(val)
    return None


def load_cube(cube_fits: Path, spectral_unit: str = "auto"):
    """Open a cube via ``spectral_cube`` and (optionally) switch to km/s.

    ``spectral_unit`` controls the spectral-axis convention:

    - ``"auto"`` (default): velocity (km/s, radio) when ``RESTFRQ`` is set
      in the header; otherwise frequency (Hz) with a warning.
    - ``"velocity"``: force km/s (radio); raises if ``RESTFRQ`` is missing.
    - ``"freq"``: leave as the cube's native frequency axis.
    """
    from spectral_cube import SpectralCube  # local import — heavy dep

    cube = SpectralCube.read(str(cube_fits))

    if spectral_unit == "freq":
        return cube

    restfrq = _resolve_rest_frequency(cube.header)

    if spectral_unit == "velocity":
        if restfrq is None:
            raise ValueError(
                f"--spectral-unit velocity requires RESTFRQ in header: "
                f"{cube_fits.name}"
            )
        return cube.with_spectral_unit(
            u.km / u.s,
            velocity_convention="radio",
            rest_value=restfrq * u.Hz,
        )

    # auto
    if restfrq is None:
        logger.warning(
            "%s has no RESTFRQ; emitting moment-0 in frequency units",
            cube_fits.name,
        )
        return cube
    return cube.with_spectral_unit(
        u.km / u.s,
        velocity_convention="radio",
        rest_value=restfrq * u.Hz,
    )


def compute_moment0(cube):
    """Integrated intensity along the spectral axis."""
    cube.allow_huge_operations = True
    return cube.moment(order=0)


def compute_peak(cube):
    """Peak intensity along the spectral axis (axis=0)."""
    cube.allow_huge_operations = True
    return cube.max(axis=0)


def compute_mean_spectrum(cube) -> tuple[np.ndarray, "u.Quantity", str]:
    """Unweighted nan-mean over the spatial axes, channel by channel.

    Returns ``(values, spectral_axis, flux_unit_str)``. Iterating
    channel-at-a-time keeps RAM at one plane (~few MB) regardless of
    cube size.
    """
    n_chan = cube.shape[0]
    values = np.full(n_chan, np.nan, dtype=np.float64)
    for i in range(n_chan):
        plane = cube.unmasked_data[i, :, :]
        # SpectralCube returns a Quantity; .value strips units cheaply.
        try:
            arr = plane.value
        except AttributeError:
            arr = np.asarray(plane)
        values[i] = np.nanmean(arr)
    flux_unit = str(cube.unit) if cube.unit is not None else ""
    return values, cube.spectral_axis, flux_unit


def _atomic_write_fits(hdul: fits.HDUList, out_path: Path) -> None:
    """Write a FITS HDUList atomically: ``out_path.tmp`` then ``os.replace``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    hdul.writeto(str(tmp_path))
    os.replace(tmp_path, out_path)


def _add_provenance(header: fits.Header, source_fits: Path, kind: str) -> None:
    header["SRCFILE"] = (source_fits.name, "Input cube")
    header["PRODUCT"] = (kind, "QA product kind")
    header.add_history(
        f"panta-rei-moments: {kind} from {source_fits.name}"
    )


def write_moment_fits(
    projection,
    out_path: Path,
    source_fits: Path,
    product_kind: str,
) -> None:
    """Persist a 2D ``Projection`` (moment-0 or peak) to FITS atomically."""
    hdu = projection.hdu  # spectral_cube returns a PrimaryHDU with proper WCS+beam
    _add_provenance(hdu.header, source_fits, product_kind)
    hdul = fits.HDUList([hdu])
    _atomic_write_fits(hdul, out_path)


def write_spectrum_fits(
    values: np.ndarray,
    spectral_axis,
    flux_unit: str,
    out_path: Path,
    source_fits: Path,
) -> None:
    """Persist a 1D mean spectrum as a FITS BinTable.

    Two columns: ``SPECTRAL`` (km/s or Hz, per cube convention) and
    ``FLUX`` (typically Jy/beam). Units are recorded via ``TUNITn``.
    """
    spectral_unit_str = str(spectral_axis.unit)
    spectral_values = np.asarray(spectral_axis.value, dtype=np.float64)

    col_spec = fits.Column(
        name="SPECTRAL",
        format="D",
        unit=spectral_unit_str,
        array=spectral_values,
    )
    col_flux = fits.Column(
        name="FLUX",
        format="D",
        unit=flux_unit,
        array=values.astype(np.float64),
    )
    bintable = fits.BinTableHDU.from_columns([col_spec, col_flux], name="SPECTRUM")
    _add_provenance(bintable.header, source_fits, "mean_spectrum")
    bintable.header["SPECCONV"] = (
        spectral_unit_str,
        "Spectral-axis units of SPECTRAL column",
    )

    primary = fits.PrimaryHDU()
    _add_provenance(primary.header, source_fits, "mean_spectrum")
    hdul = fits.HDUList([primary, bintable])
    _atomic_write_fits(hdul, out_path)


def process_cube(
    cube_fits: Path,
    analysis_dir: Path,
    *,
    products: Iterable[str] = PRODUCT_KINDS,
    spectral_unit: str = "auto",
    force: bool = False,
    dry_run: bool = False,
) -> CubeResult:
    """Generate the requested products for a single cube.

    Catches per-cube exceptions, logs, and returns ``failed:<reason>``
    entries; never raises.
    """
    products = tuple(products)
    out_paths = derive_output_paths(cube_fits, analysis_dir, products)
    statuses: dict[str, str] = {}

    needed = {
        kind: path
        for kind, path in out_paths.items()
        if needs_regeneration(cube_fits, path, force)
    }
    for kind, path in out_paths.items():
        if kind not in needed:
            statuses[kind] = "skipped"
            logger.debug("skip %s (output up-to-date)", path)

    if dry_run:
        for kind in needed:
            statuses[kind] = "dry-run"
            logger.info("[dry-run] would write %s", out_paths[kind])
        return CubeResult(cube=cube_fits, products=statuses)

    if not needed:
        return CubeResult(cube=cube_fits, products=statuses)

    try:
        cube = load_cube(cube_fits, spectral_unit=spectral_unit)
    except Exception as exc:  # pragma: no cover - depends on FITS payload
        reason = f"failed:load:{type(exc).__name__}:{exc}"
        logger.error("%s: %s", cube_fits.name, reason)
        for kind in needed:
            statuses[kind] = reason
        return CubeResult(cube=cube_fits, products=statuses)

    if "integrated_intensity" in needed:
        try:
            m0 = compute_moment0(cube)
            write_moment_fits(
                m0, needed["integrated_intensity"], cube_fits, "integrated_intensity"
            )
            statuses["integrated_intensity"] = "written"
            logger.info("wrote %s", needed["integrated_intensity"])
        except Exception as exc:  # pragma: no cover
            reason = f"failed:moment0:{type(exc).__name__}:{exc}"
            logger.error("%s: %s", cube_fits.name, reason)
            statuses["integrated_intensity"] = reason

    if "peak_intensity" in needed:
        try:
            peak = compute_peak(cube)
            write_moment_fits(
                peak, needed["peak_intensity"], cube_fits, "peak_intensity"
            )
            statuses["peak_intensity"] = "written"
            logger.info("wrote %s", needed["peak_intensity"])
        except Exception as exc:  # pragma: no cover
            reason = f"failed:peak:{type(exc).__name__}:{exc}"
            logger.error("%s: %s", cube_fits.name, reason)
            statuses["peak_intensity"] = reason

    if "mean_spectrum" in needed:
        try:
            spec, axis, flux_unit = compute_mean_spectrum(cube)
            write_spectrum_fits(
                spec, axis, flux_unit, needed["mean_spectrum"], cube_fits
            )
            statuses["mean_spectrum"] = "written"
            logger.info("wrote %s", needed["mean_spectrum"])
        except Exception as exc:  # pragma: no cover
            reason = f"failed:spectrum:{type(exc).__name__}:{exc}"
            logger.error("%s: %s", cube_fits.name, reason)
            statuses["mean_spectrum"] = reason

    return CubeResult(cube=cube_fits, products=statuses)


def discover_cubes(
    imaging_dir: Path,
    group_filters: Iterable[str] | None = None,
    array_combos: Iterable[str] | None = DEFAULT_ARRAY_COMBOS,
) -> list[Path]:
    """List ``*.cube.pbcor.fits`` files under ``imaging_dir``'s group dirs.

    ``group_filters`` is an optional iterable of substrings; a group dir
    is included if its name contains any of them (case-sensitive).

    ``array_combos`` selects which array-combination products to keep,
    matched as the dot-bounded token ``.<combo>.`` in the filename. The
    dot bounding matters: a bare ``12m7m`` substring also appears inside
    ``12m7mTP`` filenames, so substring matching would collapse the two.
    Pass ``None`` (or an empty iterable) to skip this filter.
    """
    cubes = sorted(imaging_dir.glob(f"group.*.lp_nperetto/{CUBE_GLOB}"))
    filters = list(group_filters or [])
    if filters:
        cubes = [
            c for c in cubes
            if any(f in c.parent.name for f in filters)
        ]
    combos = list(array_combos or [])
    if combos:
        tokens = tuple(f".{c}." for c in combos)
        cubes = [c for c in cubes if any(t in c.name for t in tokens)]
    return cubes
