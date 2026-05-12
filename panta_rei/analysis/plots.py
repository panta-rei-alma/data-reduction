"""QA plot generation for moment maps and mean spectra.

Outputs are PNGs alongside the FITS products. The Jy/beam → K conversion
uses ``radio_beam.Beam.jtok``: a single-frequency scalar (evaluated at
``RESTFRQ``) for the 2D moment maps, and per-channel jtok for spectra.

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
COLOR_FEATHERED = "firebrick"
COLOR_TP = "cornflowerblue"
CMAP_MOMENT = "inferno"


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

        title = title_kind
        if source_label:
            title = f"{title} — {source_label}"
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
    tp_values_jy: np.ndarray | None = None,
    tp_cube=None,
    source_label: str | None = None,
) -> None:
    """Plot the 12m7mTP mean spectrum in K vs GHz, with optional TP overlay.

    ``values_jy`` is the unweighted spatial mean per channel in Jy/beam
    (the same array written to the BinTable FITS). The TP overlay is
    drawn only when both ``tp_values_jy`` and ``tp_cube`` are provided.
    """
    freq_hz = cube_frequency_axis_hz(cube)
    factor = jtok_per_channel(cube, freq_hz)
    spec_k = np.asarray(values_jy, dtype=np.float64) * factor
    freq_ghz = freq_hz / 1e9

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    try:
        ax.plot(
            freq_ghz, spec_k,
            color=COLOR_FEATHERED, linewidth=1.5,
            label="12m+7m+TP",
        )

        if tp_values_jy is not None and tp_cube is not None:
            tp_freq_hz = cube_frequency_axis_hz(tp_cube)
            tp_factor = jtok_per_channel(tp_cube, tp_freq_hz)
            tp_spec_k = np.asarray(tp_values_jy, dtype=np.float64) * tp_factor
            ax.plot(
                tp_freq_hz / 1e9, tp_spec_k,
                color=COLOR_TP, linewidth=1.5, linestyle="--", label="TP only",
            )

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(r"Brightness Temperature (K)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="upper right")

        if source_label:
            ax.set_title(source_label, fontsize=10)

        fig.tight_layout()
        _atomic_savefig(fig, out_path)
    finally:
        plt.close(fig)
