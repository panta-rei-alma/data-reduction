"""Clean-imaging diagnostics for the 12m+7m (no TP) product.

Per pbcor cube under ``imaging/output/aux/<group>/`` we resolve its 3
sibling products (residual, mask, pb) and emit:

- Reconstructed-image moment-0 and peak (image = pbcor * pb)
- Residual moment-0 and peak
- Two 2D clean masks (peak / integrated — identical projection; both
  files written for symmetry with the user-facing summary plots)
- Four mean spectra (cube + residual, within each mask)

All Jy/beam → K conversion uses the **pbcor's beam** as the canonical
source for every product, so image and residual sit on the same Tb
scale and are directly comparable in the summary plots.

The compute kernel is pure-Python / numpy; CLI driver lives in
``panta_rei.cli.run_clean_diagnostics``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.io import fits

logger = logging.getLogger(__name__)


# Per-cube product kinds. Order matters only for log readability.
PRODUCT_KINDS: tuple[str, ...] = (
    "integrated_intensity_image",
    "integrated_intensity_residual",
    "peak_intensity_image",
    "peak_intensity_residual",
    "clean_mask",
    "mean_spectrum_image_in_mask",
    "mean_spectrum_residual_in_mask",
)

# Multi-panel summary plot kinds (PNG-only — no FITS counterpart).
SUMMARY_PLOT_KINDS: tuple[str, ...] = (
    "summary_integrated",
    "summary_peak",
)

# Paired-spectrum PNG kind — image + residual on a single axis. FITS
# above stays split into two BinTables for lossless per-product storage.
PAIRED_SPECTRUM_PLOT_KINDS: tuple[str, ...] = (
    "mean_spectra_in_mask",
)

# Single-panel map PNG kinds (one per FITS intensity / mask product). Mean
# spectra do not get individual PNGs — they're paired (see above).
SINGLE_PANEL_MAP_PNG_KINDS: tuple[str, ...] = (
    "integrated_intensity_image",
    "integrated_intensity_residual",
    "peak_intensity_image",
    "peak_intensity_residual",
    "clean_mask",
)


CUBE_GLOB = "*.cube.pbcor.fits"
DEFAULT_ARRAY_COMBOS: tuple[str, ...] = ("12m7m",)

# Aux products live under <imaging_dir>/aux/group.*.lp_nperetto/. The
# moments CLI uses imaging_dir/group.*.lp_nperetto/ directly — different
# subtree.
AUX_SUBDIR = "aux"

SIBLING_KINDS: tuple[str, ...] = ("pbcor", "mask", "residual", "pb")
# Optional siblings: tclean's auto-masking emits no ``.cube.mask.fits`` when
# nothing rose above the masking threshold (i.e. no detectable emission).
# Treating the mask as required would drop the still-useful intensity maps
# for those cubes; instead we synthesise an all-zero mask in that case.
OPTIONAL_SIBLING_KINDS: tuple[str, ...] = ("mask",)


# ---------------------------- typed exceptions -----------------------------


class CleanDiagError(Exception):
    """Base class for clean-diagnostics compute failures."""

    short_tag = "error"


class MissingSibling(CleanDiagError):
    short_tag = "missing_sibling"


class ShapeMismatch(CleanDiagError):
    short_tag = "shape_mismatch"


class NonSingletonStokes(CleanDiagError):
    short_tag = "non_singleton_stokes"


class PbOutOfRange(CleanDiagError):
    short_tag = "pb_out_of_range"


class MaskNotBinary(CleanDiagError):
    short_tag = "mask_not_binary"


class MissingSpectralAxis(CleanDiagError):
    short_tag = "missing_spectral_axis"


# ---------------------------- result dataclasses ---------------------------


@dataclass
class CleanDiagResult:
    """Per-cube outcome from :func:`process_cube`.

    ``products`` maps each requested kind to one of:
    ``"written"``, ``"skipped"``, ``"dry-run"``, or ``"failed:<reason>"``.
    """

    cube: Path
    products: dict[str, str]

    @property
    def any_failed(self) -> bool:
        return any(s.startswith("failed:") for s in self.products.values())


# ---------------------------- path helpers ---------------------------------


_BASE_RE = re.compile(r"^(?P<base>.+)\.cube\.pbcor\.fits$")


def _split_base(pbcor_path: Path) -> str:
    """Strip ``.cube.pbcor.fits`` from a pbcor filename.

    Used by both sibling resolution and output-path derivation so that
    sibling-derived FITS share the same ``<base>.<kind>.fits`` stem as
    the moments outputs.
    """
    m = _BASE_RE.match(pbcor_path.name)
    if m is None:
        raise ValueError(f"not a pbcor cube filename: {pbcor_path.name}")
    return m.group("base")


def resolve_siblings(pbcor_path: Path) -> dict[str, Path | None]:
    """Return a ``{kind: Path | None}`` map for pbcor / mask / residual / pb.

    Required siblings (``pbcor``, ``residual``, ``pb``) must exist on disk
    or :class:`MissingSibling` is raised. Optional siblings (``mask``)
    are returned as ``None`` when absent so the caller can synthesise a
    sensible default (an all-zero mask, in the mask case).
    """
    base = _split_base(pbcor_path)
    parent = pbcor_path.parent
    candidate: dict[str, Path] = {
        kind: parent / f"{base}.cube.{kind}.fits" for kind in SIBLING_KINDS
    }
    missing_required = [
        kind for kind in SIBLING_KINDS
        if kind not in OPTIONAL_SIBLING_KINDS and not candidate[kind].is_file()
    ]
    if missing_required:
        raise MissingSibling(
            f"missing required sibling(s) {missing_required} for {pbcor_path.name}"
        )
    paths: dict[str, Path | None] = dict(candidate)
    for kind in OPTIONAL_SIBLING_KINDS:
        if not candidate[kind].is_file():
            paths[kind] = None
    return paths


def derive_output_paths(
    pbcor_path: Path,
    analysis_dir: Path,
    products: Iterable[str] = PRODUCT_KINDS,
) -> dict[str, Path]:
    """Map a pbcor path to its FITS output paths.

    Layout mirrors moments: ``<analysis_dir>/<group_dir>/fits/<base>.<kind>.fits``.
    """
    group_dir = pbcor_path.parent.name
    base = _split_base(pbcor_path)
    target_dir = analysis_dir / group_dir / "fits"
    return {kind: target_dir / f"{base}.{kind}.fits" for kind in products}


def derive_plot_paths(
    pbcor_path: Path,
    analysis_dir: Path,
    kinds: Iterable[str],
) -> dict[str, Path]:
    """PNG paths under ``<analysis_dir>/<group_dir>/png/``."""
    group_dir = pbcor_path.parent.name
    base = _split_base(pbcor_path)
    target_dir = analysis_dir / group_dir / "png"
    return {kind: target_dir / f"{base}.{kind}.png" for kind in kinds}


# ---------------------------- mtime logic ----------------------------------


def input_mtime(sibling_paths: dict[str, Path | None]) -> float:
    """Return ``max(mtime)`` across the present sibling cubes.

    Outputs are considered fresh when their mtime is ``>=`` this value
    (uses ``>=`` rather than ``>`` to tolerate same-second writes on
    NFS). Optional siblings that are absent (``None`` in the map) are
    skipped — they have no clock to honour.
    """
    return max(p.stat().st_mtime for p in sibling_paths.values() if p is not None)


def needs_regeneration(input_clock: float, output_path: Path, force: bool) -> bool:
    """True if the output is missing, older than input_clock, or forced."""
    if force or not output_path.exists():
        return True
    return output_path.stat().st_mtime < input_clock


# ---------------------------- spectral-axis helpers ------------------------


def _spectral_axis_index(header: fits.Header) -> int:
    """Return the 1-based FITS axis number whose CTYPE starts with FREQ."""
    for i in range(1, int(header.get("NAXIS", 0)) + 1):
        ctype = str(header.get(f"CTYPE{i}", "")).upper()
        if ctype.startswith("FREQ"):
            return i
    raise MissingSpectralAxis(f"no FREQ-like axis in header: {header.get('NAXIS')}D")


def compute_dv_kms(header: fits.Header) -> float:
    """Channel width in km/s (radio convention) from a FITS header.

    Uses ``|CDELT_freq|`` and RESTFRQ; sign is dropped because moment-0
    magnitude is independent of axis direction.
    """
    axis = _spectral_axis_index(header)
    cdelt_hz = float(header[f"CDELT{axis}"])
    rest_hz = None
    for key in ("RESTFRQ", "RESTFREQ"):
        val = header.get(key)
        if val:
            rest_hz = float(val)
            break
    if rest_hz is None or rest_hz <= 0:
        raise MissingSpectralAxis("RESTFRQ missing — cannot compute dv in km/s")
    c_kms = const.c.to(u.km / u.s).value
    return abs(cdelt_hz) * c_kms / rest_hz


def freq_axis_hz(header: fits.Header, n_chan: int) -> np.ndarray:
    """Return per-channel sky frequencies (Hz) from a FITS header."""
    axis = _spectral_axis_index(header)
    crval = float(header[f"CRVAL{axis}"])
    cdelt = float(header[f"CDELT{axis}"])
    crpix = float(header[f"CRPIX{axis}"])
    return crval + (np.arange(1, n_chan + 1) - crpix) * cdelt


# ---------------------------- FITS open + validate -------------------------


def _squeeze_stokes(data: np.ndarray, path: Path) -> np.ndarray:
    """Return the 3D ``(n_chan, ny, nx)`` view, asserting singleton Stokes."""
    if data.ndim == 3:
        return data
    if data.ndim == 4 and data.shape[0] == 1:
        return data[0]
    raise NonSingletonStokes(
        f"{path.name}: expected 3D or (1, n_chan, ny, nx); got shape={data.shape}"
    )


@dataclass
class CubeBundle:
    """Memory-mapped view of the 4 sibling cubes plus their canonical header."""

    pbcor: np.ndarray  # (n_chan, ny, nx) float
    residual: np.ndarray  # (n_chan, ny, nx) float
    mask: np.ndarray  # (n_chan, ny, nx) numeric (0/1)
    pb: np.ndarray  # (n_chan, ny, nx) float
    header: fits.Header  # pbcor primary header (canonical)
    pbcor_path: Path
    sibling_paths: dict[str, Path]


def _open_cube_arrays(
    sibling_paths: dict[str, Path | None],
    exit_stack: contextlib.ExitStack,
) -> CubeBundle:
    """Open the cubes, squeeze Stokes, validate cross-cube consistency.

    HDULs are kept open via ``exit_stack`` for the lifetime of the
    bundle; arrays are numpy views over memmap'd data. Optional siblings
    that are absent (``None``) are synthesised — for mask this is an
    all-zero ``(n_chan, ny, nx)`` array taken from pbcor's shape.
    """
    arrays: dict[str, np.ndarray] = {}
    pbcor_header: fits.Header | None = None
    for kind in SIBLING_KINDS:
        path = sibling_paths[kind]
        if path is None:
            # Optional sibling absent — handled after pbcor is loaded.
            assert kind in OPTIONAL_SIBLING_KINDS, kind
            continue
        hdul = exit_stack.enter_context(fits.open(str(path), memmap=True))
        primary = hdul[0]
        data = primary.data
        if data is None:
            raise CleanDiagError(f"{path.name}: no primary HDU data")
        arr = _squeeze_stokes(data, path)
        arrays[kind] = arr
        if kind == "pbcor":
            pbcor_header = primary.header

    assert pbcor_header is not None  # SIBLING_KINDS starts with pbcor

    # Synthesise any absent optional siblings now that we know the
    # canonical shape from pbcor.
    n_chan, ny, nx = arrays["pbcor"].shape
    if sibling_paths.get("mask") is None:
        logger.info(
            "%s: no .cube.mask.fits — synthesising empty mask (no clean components)",
            sibling_paths["pbcor"].name,
        )
        arrays["mask"] = np.zeros((n_chan, ny, nx), dtype=np.uint8)

    # Cross-cube shape consistency.
    for kind in ("residual", "mask", "pb"):
        if arrays[kind].shape != (n_chan, ny, nx):
            raise ShapeMismatch(
                f"{kind} shape {arrays[kind].shape} != pbcor {(n_chan, ny, nx)}"
            )

    # pb sanity check: values should be in [0, ~1]. Allow 5% numerical slack
    # and ignore NaNs.
    pb_min = float(np.nanmin(arrays["pb"]))
    pb_max = float(np.nanmax(arrays["pb"]))
    if pb_min < -1e-3 or pb_max > 1.05:
        raise PbOutOfRange(
            f"pb data out of [0, 1.05]: min={pb_min}, max={pb_max}"
        )

    # mask sanity check: integer-valued 0/1. CASA emits float32 0.0/1.0;
    # accept anything that compares equal once cast to int.
    mask_arr = arrays["mask"]
    sample = mask_arr[:: max(1, mask_arr.shape[0] // 8)]  # sparse sample
    sample = sample[np.isfinite(sample)]
    if sample.size > 0:
        non_binary = np.any((sample != 0.0) & (sample != 1.0))
        if non_binary:
            raise MaskNotBinary(
                "mask cube has values outside {0, 1} (sampled along spectral axis)"
            )

    return CubeBundle(
        pbcor=arrays["pbcor"],
        residual=arrays["residual"],
        mask=arrays["mask"],
        pb=arrays["pb"],
        header=pbcor_header,
        pbcor_path=sibling_paths["pbcor"],
        sibling_paths=sibling_paths,
    )


# ---------------------------- compute --------------------------------------


@dataclass
class ComputeResult:
    """Numpy outputs from a single streaming pass over the 4 cubes."""

    integ_image_jy: np.ndarray  # (ny, nx) — already multiplied by dv
    integ_resid_jy: np.ndarray
    peak_image_jy: np.ndarray  # (ny, nx)
    peak_resid_jy: np.ndarray
    mask_2d: np.ndarray  # (ny, nx) uint8 0/1
    spec_image_jy: np.ndarray  # (n_chan,) Jy/beam — mean within mask_2d
    spec_resid_jy: np.ndarray  # (n_chan,) Jy/beam — mean within mask_2d
    dv_kms: float
    freq_hz: np.ndarray  # (n_chan,)


def compute_diagnostics(bundle: CubeBundle) -> ComputeResult:
    """Single streaming pass over the 4 cubes.

    Returns numpy arrays in raw Jy/beam units (integration multiplied by
    ``dv`` already). K conversion happens in the writer / plot layer
    using the pbcor beam metadata.
    """
    n_chan, ny, nx = bundle.pbcor.shape
    dv = compute_dv_kms(bundle.header)
    freqs = freq_axis_hz(bundle.header, n_chan)

    # 2D accumulators — float64 for stability of running sums on long
    # spectral axes. Peak arrays start at -inf so np.fmax-ing finite
    # values always wins; peak_seen tracks whether any finite channel
    # contributed (so we can emit NaN where nothing did).
    integ_image = np.zeros((ny, nx), dtype=np.float64)
    integ_resid = np.zeros((ny, nx), dtype=np.float64)
    valid_count_image = np.zeros((ny, nx), dtype=np.int32)
    valid_count_resid = np.zeros((ny, nx), dtype=np.int32)
    peak_image = np.full((ny, nx), -np.inf, dtype=np.float32)
    peak_resid = np.full((ny, nx), -np.inf, dtype=np.float32)
    peak_seen_image = np.zeros((ny, nx), dtype=bool)
    peak_seen_resid = np.zeros((ny, nx), dtype=bool)
    mask_2d = np.zeros((ny, nx), dtype=bool)

    # Pass 1: stream channel-at-a-time and accumulate.
    for c in range(n_chan):
        pbc = bundle.pbcor[c].astype(np.float32, copy=False)
        rsc = bundle.residual[c].astype(np.float32, copy=False)
        pbm = bundle.pb[c].astype(np.float32, copy=False)
        msk = bundle.mask[c]

        valid_recon = np.isfinite(pbc) & np.isfinite(pbm)
        valid_resid = np.isfinite(rsc)

        un_pbcor = pbc * pbm  # NaN where either is NaN

        # Sums (NaNs already contribute 0 because we mask out invalid)
        integ_image += np.where(valid_recon, un_pbcor, 0.0)
        integ_resid += np.where(valid_resid, rsc, 0.0)
        valid_count_image += valid_recon
        valid_count_resid += valid_resid

        # Peaks — np.fmax propagates the non-NaN operand; peak_seen
        # remembers whether any finite value was ever offered so we can
        # restore NaN at write time.
        peak_image = np.fmax(peak_image, un_pbcor)
        peak_resid = np.fmax(peak_resid, rsc)
        peak_seen_image |= valid_recon
        peak_seen_resid |= valid_resid

        mask_2d |= (msk > 0)

    # Post-pass: scale moment-0 by dv, restore NaN where nothing valid
    # contributed.
    integ_image = np.where(valid_count_image > 0, integ_image * dv, np.nan)
    integ_resid = np.where(valid_count_resid > 0, integ_resid * dv, np.nan)
    peak_image = np.where(peak_seen_image, peak_image, np.nan)
    peak_resid = np.where(peak_seen_resid, peak_resid, np.nan)

    # Pass 2 (light): per-channel masked spatial mean of reconstructed
    # image and residual.
    mask_2d_bool = mask_2d  # alias for clarity
    n_in_mask = int(mask_2d_bool.sum())
    spec_image = np.full(n_chan, np.nan, dtype=np.float64)
    spec_resid = np.full(n_chan, np.nan, dtype=np.float64)
    if n_in_mask > 0:
        for c in range(n_chan):
            pbc = bundle.pbcor[c].astype(np.float32, copy=False)
            rsc = bundle.residual[c].astype(np.float32, copy=False)
            pbm = bundle.pb[c].astype(np.float32, copy=False)
            un_pbcor = pbc * pbm
            spec_image[c] = float(np.nanmean(un_pbcor[mask_2d_bool]))
            spec_resid[c] = float(np.nanmean(rsc[mask_2d_bool]))
    else:
        logger.warning(
            "%s: clean mask is empty (no clean components) — spectra all NaN",
            bundle.pbcor_path.name,
        )

    return ComputeResult(
        integ_image_jy=integ_image,
        integ_resid_jy=integ_resid,
        peak_image_jy=peak_image,
        peak_resid_jy=peak_resid,
        mask_2d=mask_2d_bool.astype(np.uint8),
        spec_image_jy=spec_image,
        spec_resid_jy=spec_resid,
        dv_kms=dv,
        freq_hz=freqs,
    )


# ---------------------------- discovery ------------------------------------


def discover_cubes(
    imaging_dir: Path,
    group_filters: Iterable[str] | None = None,
    array_combos: Iterable[str] | None = DEFAULT_ARRAY_COMBOS,
) -> list[Path]:
    """List 12m7m pbcor cubes under ``<imaging_dir>/aux/group.*.lp_nperetto/``.

    Mirrors :func:`panta_rei.analysis.moments.discover_cubes` but globs the
    ``aux/`` subtree where the deferred-aux-products patch publishes the
    12m7m quad sets.
    """
    aux_dir = imaging_dir / AUX_SUBDIR
    cubes = sorted(aux_dir.glob(f"group.*.lp_nperetto/{CUBE_GLOB}"))
    filters = list(group_filters or [])
    if filters:
        cubes = [c for c in cubes if any(f in c.parent.name for f in filters)]
    combos = list(array_combos or [])
    if combos:
        tokens = tuple(f".{c}." for c in combos)
        cubes = [c for c in cubes if any(t in c.name for t in tokens)]
    return cubes


# ---------------------------- FITS writers ---------------------------------


def _atomic_write_fits(hdul: fits.HDUList, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    hdul.writeto(str(tmp_path))
    os.replace(tmp_path, out_path)


def _build_2d_header(
    src_header: fits.Header,
    *,
    bunit: str,
    product_kind: str,
    bundle: CubeBundle,
    jytok: float | None,
    mask_projection: str | None = None,
) -> fits.Header:
    """Construct a 2D FITS header from a 4D pbcor header.

    Preserves spatial WCS + beam keys; strips spectral / Stokes WCS.
    """
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    # Direct passthroughs for spatial WCS + beam metadata. Each key is
    # optional in the source; missing keys are silently skipped.
    passthrough = (
        "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
        "CDELT1", "CDELT2", "CUNIT1", "CUNIT2",
        "PC1_1", "PC1_2", "PC2_1", "PC2_2",
        "RADESYS", "EQUINOX", "LONPOLE", "LATPOLE",
        "BMAJ", "BMIN", "BPA", "RESTFRQ", "SPECSYS", "OBJECT",
    )
    for key in passthrough:
        if key in src_header:
            hdr[key] = src_header[key]
    hdr["BUNIT"] = bunit
    hdr["SRCFILE"] = (bundle.pbcor_path.name, "pbcor input cube")
    hdr["PRODUCT"] = (product_kind, "clean-diagnostics product kind")
    hdr["SIB_PB"] = (bundle.sibling_paths["pb"].name, "pb sibling")
    hdr["SIB_RESI"] = (bundle.sibling_paths["residual"].name, "residual sibling")
    mask_path = bundle.sibling_paths.get("mask")
    if mask_path is None:
        hdr["SIB_MASK"] = ("(absent - empty mask synthesised)", "mask sibling")
        hdr.add_history(
            "panta-rei-clean-diagnostics: no .cube.mask.fits - "
            "mask synthesised as all-zero (no detectable emission)."
        )
    else:
        hdr["SIB_MASK"] = (mask_path.name, "mask sibling")
    if jytok is not None:
        hdr["JYTOK"] = (float(jytok), "K per (Jy/beam) at RESTFRQ (pbcor beam)")
    if mask_projection is not None:
        hdr["PROJ"] = (mask_projection, "spectral->2D projection rule")
    hdr.add_history(
        "panta-rei-clean-diagnostics: "
        "RECON_IMG = pbcor * pb (valid where both finite); "
        "may differ from CASA .image in PB-tapered or blanked regions."
    )
    return hdr


def _write_map_fits(
    arr_k: np.ndarray,
    out_path: Path,
    *,
    bunit: str,
    product_kind: str,
    bundle: CubeBundle,
    jytok: float,
) -> None:
    hdr = _build_2d_header(
        bundle.header,
        bunit=bunit,
        product_kind=product_kind,
        bundle=bundle,
        jytok=jytok,
    )
    hdu = fits.PrimaryHDU(data=arr_k.astype(np.float32), header=hdr)
    _atomic_write_fits(fits.HDUList([hdu]), out_path)


def _write_mask_fits(
    mask_2d: np.ndarray,
    out_path: Path,
    *,
    product_kind: str,
    bundle: CubeBundle,
) -> None:
    hdr = _build_2d_header(
        bundle.header,
        bunit="",
        product_kind=product_kind,
        bundle=bundle,
        jytok=None,
        mask_projection="max(axis=0)",
    )
    hdu = fits.PrimaryHDU(data=mask_2d.astype(np.uint8), header=hdr)
    _atomic_write_fits(fits.HDUList([hdu]), out_path)


def _write_spectrum_fits(
    values_k: np.ndarray,
    spectral_values: np.ndarray,
    spectral_unit: str,
    out_path: Path,
    *,
    product_kind: str,
    bundle: CubeBundle,
    mask_kind: str,
) -> None:
    col_spec = fits.Column(
        name="SPECTRAL", format="D", unit=spectral_unit,
        array=np.asarray(spectral_values, dtype=np.float64),
    )
    col_flux = fits.Column(
        name="FLUX", format="D", unit="K",
        array=np.asarray(values_k, dtype=np.float64),
    )
    bintable = fits.BinTableHDU.from_columns(
        [col_spec, col_flux], name="SPECTRUM",
    )
    bintable.header["SRCFILE"] = (bundle.pbcor_path.name, "pbcor input cube")
    bintable.header["PRODUCT"] = (product_kind, "clean-diagnostics product kind")
    bintable.header["MASKKIND"] = (mask_kind, "2D mask used for spatial averaging")
    bintable.header["SIB_PB"] = (bundle.sibling_paths["pb"].name, "pb sibling")
    bintable.header["SIB_RESI"] = (bundle.sibling_paths["residual"].name, "residual sibling")
    mask_path = bundle.sibling_paths.get("mask")
    bintable.header["SIB_MASK"] = (
        mask_path.name if mask_path is not None else "(absent - empty mask synthesised)",
        "mask sibling",
    )
    bintable.header["SPECCONV"] = (spectral_unit, "SPECTRAL column units")

    primary = fits.PrimaryHDU()
    primary.header["SRCFILE"] = (bundle.pbcor_path.name, "pbcor input cube")
    primary.header["PRODUCT"] = (product_kind, "clean-diagnostics product kind")
    _atomic_write_fits(fits.HDUList([primary, bintable]), out_path)


# ---------------------------- process_cube ---------------------------------


_FILENAME_RE = re.compile(
    r"\.lp_nperetto\.(?P<src>.+?)\.12m7m\."
    r"(?P<lo>[\d.]+)-(?P<hi>[\d.]+)GHz\.cube\.pbcor\.fits$"
)


def human_source_label(pbcor_path: Path) -> str | None:
    """Short ``<source> | <lo>-<hi> GHz`` label for plot titles."""
    m = _FILENAME_RE.search(pbcor_path.name)
    if m is None:
        return None
    src = re.sub(
        r"(?<=\d)([pm])(?=\d)",
        lambda mm: {"p": "+", "m": "-"}[mm.group(1)],
        m.group("src"),
    )
    return f"{src} | {m.group('lo')}-{m.group('hi')} GHz"


def _all_output_kinds(
    products: tuple[str, ...],
    plot: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (fits_kinds, single_panel_png_kinds, summary_png_kinds).

    All three are filtered by the caller's ``products`` selection plus a
    plot flag. ``products`` carries PRODUCT_KINDS + (optionally)
    SUMMARY_PLOT_KINDS — the latter only meaningful when ``plot=True``.
    """
    fits_kinds = tuple(k for k in PRODUCT_KINDS if k in products)
    if not plot:
        return fits_kinds, (), ()
    single = tuple(k for k in SINGLE_PANEL_MAP_PNG_KINDS if k in products)
    paired = tuple(k for k in PAIRED_SPECTRUM_PLOT_KINDS if k in products)
    summary = tuple(k for k in SUMMARY_PLOT_KINDS if k in products)
    return fits_kinds, single + paired, summary


def process_cube(
    pbcor_path: Path,
    analysis_dir: Path,
    *,
    products: Iterable[str] = PRODUCT_KINDS + SUMMARY_PLOT_KINDS
                              + PAIRED_SPECTRUM_PLOT_KINDS,
    force: bool = False,
    dry_run: bool = False,
    plot: bool = True,
) -> CleanDiagResult:
    """Generate clean-diagnostics products for a single 12m7m pbcor cube.

    Catches per-cube exceptions, logs, and returns ``failed:<reason>``
    statuses; never raises. When ``plot=True`` PNG plots accompany each
    FITS product, plus the two multi-panel summary PNGs.
    """
    requested = tuple(products)
    statuses: dict[str, str] = {}

    try:
        sibling_paths = resolve_siblings(pbcor_path)
    except MissingSibling as exc:
        # Surface as failed:missing_<kind> on every requested product.
        reason = f"failed:{exc.short_tag}:{exc}"
        for kind in requested:
            statuses[kind] = reason
        logger.error("%s: %s", pbcor_path.name, reason)
        return CleanDiagResult(cube=pbcor_path, products=statuses)

    clock = input_mtime(sibling_paths)

    fits_paths = derive_output_paths(pbcor_path, analysis_dir, PRODUCT_KINDS)
    all_png_kinds = SINGLE_PANEL_MAP_PNG_KINDS + PAIRED_SPECTRUM_PLOT_KINDS + SUMMARY_PLOT_KINDS
    png_paths = derive_plot_paths(pbcor_path, analysis_dir, all_png_kinds)

    fits_kinds, single_png_kinds, summary_png_kinds = _all_output_kinds(
        requested, plot,
    )

    needed_fits = {
        kind: fits_paths[kind] for kind in fits_kinds
        if needs_regeneration(clock, fits_paths[kind], force)
    }
    needed_png = {
        kind: png_paths[kind] for kind in single_png_kinds
        if needs_regeneration(clock, png_paths[kind], force)
    }
    needed_summary = {
        kind: png_paths[kind] for kind in summary_png_kinds
        if needs_regeneration(clock, png_paths[kind], force)
    }
    for kind in fits_kinds:
        if kind not in needed_fits:
            statuses[kind] = "skipped"
    for kind in single_png_kinds:
        key = f"plot:{kind}"
        if kind not in needed_png:
            statuses[key] = "skipped"
    for kind in summary_png_kinds:
        key = f"plot:{kind}"
        if kind not in needed_summary:
            statuses[key] = "skipped"

    if dry_run:
        for kind in needed_fits:
            statuses[kind] = "dry-run"
            logger.info("[dry-run] would write %s", needed_fits[kind])
        for kind in needed_png:
            statuses[f"plot:{kind}"] = "dry-run"
            logger.info("[dry-run] would write %s", needed_png[kind])
        for kind in needed_summary:
            statuses[f"plot:{kind}"] = "dry-run"
            logger.info("[dry-run] would write %s", needed_summary[kind])
        return CleanDiagResult(cube=pbcor_path, products=statuses)

    if not needed_fits and not needed_png and not needed_summary:
        return CleanDiagResult(cube=pbcor_path, products=statuses)

    # Heavy work begins here.
    try:
        with contextlib.ExitStack() as stack:
            bundle = _open_cube_arrays(sibling_paths, stack)
            result = compute_diagnostics(bundle)

            # K conversion uses pbcor's beam via SpectralCube (metadata-only).
            from spectral_cube import SpectralCube
            from panta_rei.analysis.plots import (
                jtok_scalar,
                jtok_per_channel,
                cube_frequency_axis_hz,
            )

            pbcor_cube = SpectralCube.read(str(bundle.pbcor_path))
            jytok = jtok_scalar(pbcor_cube)
            if jytok is None:
                logger.warning(
                    "%s: cannot derive jtok (missing RESTFRQ or beam) — "
                    "FITS will be in Jy/beam units",
                    pbcor_path.name,
                )
                map_factor = 1.0
                map_bunit_2d = "Jy/beam"
                map_bunit_mom0 = "Jy/beam km/s"
                spec_factor = np.ones(result.freq_hz.size, dtype=np.float64)
                spec_unit = "Jy/beam"
            else:
                map_factor = jytok
                map_bunit_2d = "K"
                map_bunit_mom0 = "K km/s"
                spec_factor = jtok_per_channel(
                    pbcor_cube, cube_frequency_axis_hz(pbcor_cube),
                )
                spec_unit = "K"

            # Build the data we'll feed to the FITS writers, indexed by kind.
            map_payloads = {
                "integrated_intensity_image":
                    (result.integ_image_jy * map_factor, map_bunit_mom0),
                "integrated_intensity_residual":
                    (result.integ_resid_jy * map_factor, map_bunit_mom0),
                "peak_intensity_image":
                    (result.peak_image_jy * map_factor, map_bunit_2d),
                "peak_intensity_residual":
                    (result.peak_resid_jy * map_factor, map_bunit_2d),
            }
            for kind, (arr, bunit) in map_payloads.items():
                if kind in needed_fits:
                    try:
                        _write_map_fits(
                            arr, needed_fits[kind],
                            bunit=bunit, product_kind=kind,
                            bundle=bundle,
                            jytok=jytok if jytok is not None else 1.0,
                        )
                        statuses[kind] = "written"
                        logger.info("wrote %s", needed_fits[kind])
                    except Exception as exc:  # pragma: no cover
                        reason = f"failed:write_map:{type(exc).__name__}:{exc}"
                        logger.error("%s: %s", pbcor_path.name, reason)
                        statuses[kind] = reason

            if "clean_mask" in needed_fits:
                try:
                    _write_mask_fits(
                        result.mask_2d, needed_fits["clean_mask"],
                        product_kind="clean_mask", bundle=bundle,
                    )
                    statuses["clean_mask"] = "written"
                    logger.info("wrote %s", needed_fits["clean_mask"])
                except Exception as exc:  # pragma: no cover
                    reason = f"failed:write_mask:{type(exc).__name__}:{exc}"
                    logger.error("%s: %s", pbcor_path.name, reason)
                    statuses["clean_mask"] = reason

            # Spectra. spec_factor is per-channel; works for the unit
            # fallback path because it's all-ones in that case.
            spec_payloads = {
                "mean_spectrum_image_in_mask":
                    (result.spec_image_jy * spec_factor, "clean_mask"),
                "mean_spectrum_residual_in_mask":
                    (result.spec_resid_jy * spec_factor, "clean_mask"),
            }
            # Spectral-axis values: use km/s (radio convention) if RESTFRQ
            # is set, else Hz.
            try:
                spec_cube_kms = pbcor_cube.with_spectral_unit(
                    u.km / u.s, velocity_convention="radio",
                    rest_value=pbcor_cube.header.get("RESTFRQ") * u.Hz
                    if pbcor_cube.header.get("RESTFRQ") else None,
                )
                spectral_values = np.asarray(
                    spec_cube_kms.spectral_axis.to(u.km / u.s).value,
                    dtype=np.float64,
                )
                spectral_unit = "km/s"
            except Exception:
                spectral_values = result.freq_hz
                spectral_unit = "Hz"

            for kind, (vals, mask_kind) in spec_payloads.items():
                if kind in needed_fits:
                    try:
                        _write_spectrum_fits(
                            vals, spectral_values, spectral_unit,
                            needed_fits[kind],
                            product_kind=kind, bundle=bundle, mask_kind=mask_kind,
                        )
                        statuses[kind] = "written"
                        logger.info("wrote %s", needed_fits[kind])
                    except Exception as exc:  # pragma: no cover
                        reason = f"failed:write_spectrum:{type(exc).__name__}:{exc}"
                        logger.error("%s: %s", pbcor_path.name, reason)
                        statuses[kind] = reason

            # Plots — defer to the plot module so the heavy compute work
            # above remains import-light.
            if needed_png or needed_summary:
                from panta_rei.analysis.clean_diagnostics_plots import (
                    plot_payload_for_cube,
                )
                # Pre-compute K-space arrays for plotting (independent of
                # whether the matching FITS was selected).
                k_payloads = {
                    "integ_image_k": result.integ_image_jy * map_factor,
                    "integ_resid_k": result.integ_resid_jy * map_factor,
                    "peak_image_k":  result.peak_image_jy  * map_factor,
                    "peak_resid_k":  result.peak_resid_jy  * map_factor,
                    "spec_image_k":  result.spec_image_jy  * spec_factor,
                    "spec_resid_k":  result.spec_resid_jy  * spec_factor,
                    "spectral_values": spectral_values,
                    "spectral_unit":   spectral_unit,
                    "mask_2d":         result.mask_2d,
                    "map_bunit_2d":    map_bunit_2d,
                    "map_bunit_mom0":  map_bunit_mom0,
                    "spec_unit":       spec_unit,
                }
                source_label = human_source_label(pbcor_path)
                plot_payload_for_cube(
                    bundle=bundle,
                    payload=k_payloads,
                    needed_png=needed_png,
                    needed_summary=needed_summary,
                    statuses=statuses,
                    source_label=source_label,
                )

    except CleanDiagError as exc:
        reason = f"failed:{exc.short_tag}:{exc}"
        logger.error("%s: %s", pbcor_path.name, reason)
        for kind in fits_kinds:
            statuses.setdefault(kind, reason)
        for kind in single_png_kinds:
            statuses.setdefault(f"plot:{kind}", reason)
        for kind in summary_png_kinds:
            statuses.setdefault(f"plot:{kind}", reason)
    except Exception as exc:  # pragma: no cover - cube-payload-dependent
        reason = f"failed:compute:{type(exc).__name__}:{exc}"
        logger.error("%s: %s", pbcor_path.name, reason)
        for kind in fits_kinds:
            statuses.setdefault(kind, reason)
        for kind in single_png_kinds:
            statuses.setdefault(f"plot:{kind}", reason)
        for kind in summary_png_kinds:
            statuses.setdefault(f"plot:{kind}", reason)

    return CleanDiagResult(cube=pbcor_path, products=statuses)
