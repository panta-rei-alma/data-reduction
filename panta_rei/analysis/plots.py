"""QA plot generation for moment maps and mean spectra.

Outputs are PNGs alongside the FITS products. The Jy/beam → K conversion
uses ``radio_beam.Beam.jtok``: a single-frequency scalar (evaluated at
``RESTFRQ``) for the 2D moment maps, and per-channel jtok for spectra.

All plots are for the feathered 12m+7m+TP product; the title carries a
``12m7mTP`` tag so the array combination is visible at a glance.

Matplotlib's Agg backend is selected on import so workers can render in
ProcessPoolExecutor children without a display.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless — must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

logger = logging.getLogger(__name__)

# Colour conventions per QA spec.
COLOR_SPECTRUM = "cornflowerblue"
CMAP_MOMENT = "inferno"
ARRAY_TAG = "12m7mTP"

# Continuum quicklook conventions. Continuum is the joint 12m+7m MFS product
# (no TP), so its tag differs from the feathered line products above.
CMAP_CONTINUUM = "inferno"
CMAP_ALPHA = "RdBu_r"  # diverging — spectral index is signed, not a flux
ARRAY_TAG_CONTINUUM = "12m7m"
# Spectral-index colour range. Physical alpha is roughly [-1, 4]; clamp so a
# few noisy survivors don't blow out the scale (we also S/N-mask, below).
ALPHA_VMIN, ALPHA_VMAX = -2.0, 4.0
# Default S/N threshold for the alpha mask, applied on the FLAT (non-pbcor)
# image whose noise is spatially uniform (pbcor amplifies edge noise).
ALPHA_SNR_DEFAULT = 5.0


def _resolve_beam_for_scalar(cube):
    """Return a single ``Beam`` for cube-wide jtok.

    For per-channel-beam cubes (``cube.beams`` is a ``Beams`` object) we
    take the median beam — moment-0 and peak both collapse the spectral
    axis, so a single representative beam is the right granularity.
    """
    beam = getattr(cube, "beam", None)
    if beam is not None:
        return beam
    beams = getattr(cube, "beams", None)
    if beams is not None:
        try:
            return beams.common_beam()
        except Exception:
            # common_beam can fail for pathological cubes; median is safe.
            from radio_beam import Beam

            return Beam(
                major=np.nanmedian(beams.major.to(u.arcsec).value) * u.arcsec,
                minor=np.nanmedian(beams.minor.to(u.arcsec).value) * u.arcsec,
                pa=np.nanmedian(beams.pa.to(u.deg).value) * u.deg,
            )
    raise ValueError("cube has no beam information")


def _resolve_restfrq_hz(cube) -> Optional[float]:
    for key in ("RESTFRQ", "RESTFREQ"):
        val = cube.header.get(key)
        if val:
            return float(val)
    return None


def jtok_scalar(cube) -> Optional[float]:
    """Return K per (Jy/beam) at the cube's RESTFRQ, or None if unavailable."""
    restfrq = _resolve_restfrq_hz(cube)
    if restfrq is None:
        return None
    try:
        beam = _resolve_beam_for_scalar(cube)
    except ValueError:
        return None
    return float(beam.jtok(restfrq * u.Hz).to(u.K).value)


def jtok_per_channel(cube, freq_hz: np.ndarray) -> np.ndarray:
    """Return K per (Jy/beam) at each cube frequency.

    Uses ``cube.beams`` (per-channel beams) when present, else the single
    ``cube.beam`` broadcast to every channel.
    """
    freqs = freq_hz * u.Hz
    beams = getattr(cube, "beams", None)
    if beams is not None:
        # Beams.jtok takes a frequency array of matching length.
        return beams.jtok(freqs).to(u.K).value
    beam = getattr(cube, "beam", None)
    if beam is None:
        raise ValueError("cube has no beam information")
    return beam.jtok(freqs).to(u.K).value


def cube_frequency_axis_hz(cube) -> np.ndarray:
    """Return the cube's spectral axis converted to Hz."""
    restfrq = _resolve_restfrq_hz(cube)
    if restfrq is not None:
        freq_cube = cube.with_spectral_unit(
            u.Hz, velocity_convention="radio", rest_value=restfrq * u.Hz,
        )
    else:
        freq_cube = cube.with_spectral_unit(u.Hz)
    return np.asarray(freq_cube.spectral_axis.to(u.Hz).value, dtype=np.float64)


def _atomic_savefig(fig, out_path: Path, dpi: int = 150) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    # ``format`` is passed explicitly because the .tmp suffix would
    # otherwise mislead matplotlib's extension-based format inference.
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", format="png")
    os.replace(tmp, out_path)


def plot_moment_map(
    projection,
    cube,
    out_path: Path,
    kind: str,
    *,
    source_label: str | None = None,
) -> None:
    """Plot a 2D moment-0 or peak map converted to K (·km/s for moment-0).

    ``kind`` must be ``"integrated_intensity"`` or ``"peak_intensity"``.
    The plot uses inferno with a colourbar; WCS axes when available.
    """
    if kind not in ("integrated_intensity", "peak_intensity"):
        raise ValueError(f"unsupported kind: {kind}")

    factor = jtok_scalar(cube)
    if factor is None:
        raise ValueError("cannot derive jtok scalar (missing RESTFRQ or beam)")

    data = np.asarray(
        projection.value if hasattr(projection, "value") else projection,
        dtype=np.float64,
    )
    data_k = data * factor

    if kind == "integrated_intensity":
        cbar_label = r"Integrated $T_B$ (K km s$^{-1}$)"
        title_kind = "Integrated intensity"
    else:
        cbar_label = r"Peak $T_B$ (K)"
        title_kind = "Peak intensity"

    wcs = None
    try:
        wcs = WCS(projection.header).celestial
    except Exception:
        wcs = None

    fig = plt.figure(figsize=(7.5, 6.0))
    if wcs is not None:
        ax = fig.add_subplot(111, projection=wcs)
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")

    finite = np.isfinite(data_k)
    if finite.any():
        vmin = float(np.nanpercentile(data_k[finite], 1.0))
        vmax = float(np.nanpercentile(data_k[finite], 99.5))
        if vmax <= vmin:
            vmin, vmax = float(np.nanmin(data_k[finite])), float(np.nanmax(data_k[finite]))
    else:
        vmin, vmax = 0.0, 1.0

    try:
        im = ax.imshow(
            data_k, origin="lower", cmap=CMAP_MOMENT,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

        title = f"{ARRAY_TAG} | {title_kind}"
        if source_label:
            title = f"{title} | {source_label}"
        ax.set_title(title, fontsize=10)

        fig.tight_layout()
        _atomic_savefig(fig, out_path)
    finally:
        plt.close(fig)


def plot_mean_spectrum(
    values_jy: np.ndarray,
    cube,
    out_path: Path,
    *,
    source_label: str | None = None,
) -> None:
    """Plot the 12m7mTP mean spectrum in K vs GHz.

    ``values_jy`` is the unweighted spatial mean per channel in Jy/beam
    (the same array written to the BinTable FITS).
    """
    freq_hz = cube_frequency_axis_hz(cube)
    factor = jtok_per_channel(cube, freq_hz)
    spec_k = np.asarray(values_jy, dtype=np.float64) * factor
    freq_ghz = freq_hz / 1e9

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    try:
        ax.plot(freq_ghz, spec_k, color=COLOR_SPECTRUM, linewidth=1.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(r"Brightness Temperature (K)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        title = f"{ARRAY_TAG} | Mean spectrum"
        if source_label:
            title = f"{title} | {source_label}"
        ax.set_title(title, fontsize=10)

        fig.tight_layout()
        _atomic_savefig(fig, out_path)
    finally:
        plt.close(fig)


# --- Continuum quicklooks -------------------------------------------------
# 2D MFS images (no cube). The renderers take plain numpy arrays + the FITS
# header (for WCS), mirroring the moment-map look: WCS axes, a colourbar, an
# inferno/diverging map, and an atomic PNG write.


def _celestial_axes(fig, header):
    """Add a subplot with WCS celestial axes when possible, else pixel axes."""
    wcs = None
    try:
        wcs = WCS(header).celestial
    except Exception:
        wcs = None
    if wcs is not None:
        ax = fig.add_subplot(111, projection=wcs)
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
    return ax


def mask_alpha_by_snr(alpha, flat, snr: float = ALPHA_SNR_DEFAULT):
    """Return ``(masked_alpha, n_significant)``.

    The spectral-index map is meaningless where there is no continuum signal,
    so we blank it outside ``flat > snr * sigma``. ``flat`` is the non-pbcor
    (flat-noise) image; ``sigma`` is its 3-sigma-clipped standard deviation.
    """
    from astropy.stats import sigma_clipped_stats

    alpha = np.asarray(alpha, dtype=np.float64)
    flat = np.asarray(flat, dtype=np.float64)
    flat_finite = np.isfinite(flat)
    if flat_finite.any():
        _, _, sigma = sigma_clipped_stats(flat[flat_finite], sigma=3.0)
        mask = flat_finite & np.isfinite(alpha) & (flat > snr * float(sigma))
    else:
        mask = np.zeros(alpha.shape, dtype=bool)
    masked = np.where(mask, alpha, np.nan)
    return masked, int(mask.sum())


def plot_continuum_image(data_jy, header, out_path: Path, *, source_label: str | None = None) -> None:
    """Plot a 2D continuum image (Jy/beam → mJy/beam) with an asinh stretch.

    asinh keeps the faint extended emission visible alongside bright cores
    (continuum has a very wide dynamic range); the softening width is set at
    the image noise so it is ~linear below the rms and logarithmic above.
    """
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import AsinhStretch, ImageNormalize, ManualInterval

    data = np.asarray(data_jy, dtype=np.float64)
    mjy = data * 1000.0  # Jy/beam → mJy/beam
    finite = np.isfinite(mjy)
    if not finite.any():
        raise ValueError("continuum image is entirely non-finite")

    _, _, sigma = sigma_clipped_stats(mjy[finite], sigma=3.0)
    sigma = float(sigma) if np.isfinite(sigma) and sigma > 0 else float(np.nanstd(mjy[finite]))
    vmin = -2.0 * sigma
    vmax = float(np.nanmax(mjy[finite]))
    if vmax <= vmin:
        vmax = vmin + (abs(vmin) or 1.0)
    a = float(np.clip(sigma / (vmax - vmin), 1e-3, 0.5))
    norm = ImageNormalize(mjy, interval=ManualInterval(vmin, vmax), stretch=AsinhStretch(a=a))

    fig = plt.figure(figsize=(7.5, 6.0))
    ax = _celestial_axes(fig, header)
    try:
        im = ax.imshow(
            mjy, origin="lower", cmap=CMAP_CONTINUUM, norm=norm, interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label="Flux density (mJy/beam)", fraction=0.046, pad=0.04)
        title = f"{ARRAY_TAG_CONTINUUM} | Continuum (tt0, PB-corrected)"
        if source_label:
            title = f"{title} | {source_label}"
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        _atomic_savefig(fig, out_path)
    finally:
        plt.close(fig)


def plot_continuum_alpha(
    masked_alpha,
    header,
    out_path: Path,
    *,
    source_label: str | None = None,
    snr: float = ALPHA_SNR_DEFAULT,
) -> None:
    """Plot a S/N-masked spectral-index map (diverging cmap, fixed [-2, 4] range).

    ``masked_alpha`` should already be blanked outside the significant region
    (see :func:`mask_alpha_by_snr`); this function only renders.
    """
    alpha = np.asarray(masked_alpha, dtype=np.float64)
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = _celestial_axes(fig, header)
    try:
        im = ax.imshow(
            alpha, origin="lower", cmap=CMAP_ALPHA,
            vmin=ALPHA_VMIN, vmax=ALPHA_VMAX, interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label=r"Spectral index $\alpha$", fraction=0.046, pad=0.04)
        title = f"{ARRAY_TAG_CONTINUUM} | Spectral index (S/N>{snr:g})"
        if source_label:
            title = f"{title} | {source_label}"
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        _atomic_savefig(fig, out_path)
    finally:
        plt.close(fig)
