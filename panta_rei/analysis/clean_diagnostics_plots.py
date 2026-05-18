"""QA plot generation for the 12m+7m clean-diagnostics products.

Two flavours of plot:

- **Per-product PNGs** — one PNG per intensity map and mask, plus a
  paired-spectrum PNG per mask (cube + residual on one axis).
- **Summary multi-panel PNGs** — two 3-panel summaries (integrated, peak)
  with shared colour scale across the map panels and the mask contour
  overlaid on both.

All map values are in K (or K·km/s for moment-0); the conversion from
Jy/beam → K happens in the compute layer using the pbcor's beam as the
canonical source.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

logger = logging.getLogger(__name__)

ARRAY_TAG = "12m7m"
CMAP_MAP = "inferno"
CMAP_MASK = "Greys"
COLOR_IMAGE = "cornflowerblue"
COLOR_RESIDUAL = "firebrick"
LINESTYLE_IMAGE = "-"
LINESTYLE_RESIDUAL = "--"
MASK_CONTOUR_COLOR = "white"


def _atomic_savefig(fig, out_path: Path, dpi: int = 150) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", format="png")
    os.replace(tmp, out_path)


def _percentile_limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Joint 1st / 99.5th percentile across multiple 2D arrays."""
    stacked = np.concatenate([
        np.asarray(a).ravel() for a in arrays if a is not None
    ])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, 1.0))
    vmax = float(np.nanpercentile(finite, 99.5))
    if vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def _wcs_from_header(header) -> WCS | None:
    try:
        return WCS(header).celestial
    except Exception:
        return None


def _cbar_label_for(map_bunit: str, kind: str) -> str:
    if kind.startswith("integrated"):
        if map_bunit.startswith("K"):
            return r"Integrated $T_B$ (K km s$^{-1}$)"
        return f"Integrated intensity ({map_bunit})"
    if kind.startswith("peak"):
        if map_bunit.startswith("K"):
            return r"Peak $T_B$ (K)"
        return f"Peak intensity ({map_bunit})"
    return map_bunit


def _title_for_kind(kind: str, source_label: str | None) -> str:
    pretty = kind.replace("_", " ")
    title = f"{ARRAY_TAG} | {pretty}"
    if source_label:
        title = f"{title} | {source_label}"
    return title


def _plot_single_map(
    arr: np.ndarray,
    header,
    out_path: Path,
    kind: str,
    bunit: str,
    *,
    source_label: str | None,
) -> None:
    wcs = _wcs_from_header(header)
    fig = plt.figure(figsize=(7.5, 6.0))
    if wcs is not None:
        ax = fig.add_subplot(111, projection=wcs)
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
    try:
        vmin, vmax = _percentile_limits(arr)
        im = ax.imshow(
            arr, origin="lower", cmap=CMAP_MAP,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label=_cbar_label_for(bunit, kind),
                     fraction=0.046, pad=0.04)
        ax.set_title(_title_for_kind(kind, source_label), fontsize=10)
        fig.tight_layout()
        _atomic_savefig(fig, out_path)
    finally:
        plt.close(fig)


def _plot_mask(
    mask_2d: np.ndarray,
    header,
    out_path: Path,
    kind: str,
    *,
    source_label: str | None,
) -> None:
    wcs = _wcs_from_header(header)
    fig = plt.figure(figsize=(7.5, 6.0))
    if wcs is not None:
        ax = fig.add_subplot(111, projection=wcs)
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")
    else:
        ax = fig.add_subplot(111)
    try:
        ax.imshow(mask_2d, origin="lower", cmap=CMAP_MASK,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(_title_for_kind(kind, source_label), fontsize=10)
        fig.tight_layout()
        _atomic_savefig(fig, out_path)
    finally:
        plt.close(fig)


def _plot_paired_spectrum(
    spec_image_k: np.ndarray,
    spec_resid_k: np.ndarray,
    spectral_values: np.ndarray,
    spectral_unit: str,
    spec_unit: str,
    out_path: Path,
    mask_kind: str,
    *,
    source_label: str | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    try:
        x = spectral_values
        if spectral_unit == "Hz":
            x = x / 1e9
            xlabel = "Frequency (GHz)"
        elif spectral_unit == "km/s":
            xlabel = "Velocity (km/s, radio)"
        else:
            xlabel = f"Spectral ({spectral_unit})"
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
        if np.any(np.isfinite(spec_image_k)):
            ax.plot(x, spec_image_k, color=COLOR_IMAGE, linewidth=1.5,
                    linestyle=LINESTYLE_IMAGE, label="image (no pbcor)")
        else:
            ax.text(0.5, 0.6, "image spectrum: all NaN",
                    transform=ax.transAxes, ha="center", color=COLOR_IMAGE)
        if np.any(np.isfinite(spec_resid_k)):
            ax.plot(x, spec_resid_k, color=COLOR_RESIDUAL, linewidth=1.5,
                    linestyle=LINESTYLE_RESIDUAL, label="residual")
        else:
            ax.text(0.5, 0.5, "residual spectrum: all NaN",
                    transform=ax.transAxes, ha="center", color=COLOR_RESIDUAL)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(rf"Mean within mask ({spec_unit})")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        kind_pretty = mask_kind.replace("_", " ")
        title = f"{ARRAY_TAG} | {kind_pretty}"
        if source_label:
            title = f"{title} | {source_label}"
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        _atomic_savefig(fig, out_path)
    finally:
        plt.close(fig)


def _plot_summary(
    map_image_k: np.ndarray,
    map_resid_k: np.ndarray,
    mask_2d: np.ndarray,
    map_bunit: str,
    header,
    out_path: Path,
    flavour: str,  # "integrated" or "peak"
    *,
    source_label: str | None,
) -> None:
    """Two-panel summary: map(image) and map(residual) with shared cbar.

    Layout: 3-column gridspec — two equal map columns + a narrow cbar
    column. The shared colour scale (joint 1st/99.5th percentile) makes
    image vs residual directly comparable. The clean-mask contour is
    overlaid on both panels.

    The mask-averaged spectrum is its own product
    (``mean_spectra_in_mask.png``) and is not duplicated here.
    """
    wcs = _wcs_from_header(header)
    fig = plt.figure(figsize=(11.0, 4.7))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 1.0, 0.05],
        wspace=0.15,
    )
    if wcs is not None:
        ax1 = fig.add_subplot(gs[0, 0], projection=wcs)
        ax2 = fig.add_subplot(gs[0, 1], projection=wcs)
    else:
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    try:
        vmin, vmax = _percentile_limits(map_image_k, map_resid_k)
        im1 = ax1.imshow(map_image_k, origin="lower", cmap=CMAP_MAP,
                         vmin=vmin, vmax=vmax, interpolation="nearest")
        ax2.imshow(map_resid_k, origin="lower", cmap=CMAP_MAP,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
        if mask_2d.any():
            ax1.contour(mask_2d.astype(np.float32), levels=[0.5],
                        colors=MASK_CONTOUR_COLOR, linewidths=0.7, alpha=0.9)
            ax2.contour(mask_2d.astype(np.float32), levels=[0.5],
                        colors=MASK_CONTOUR_COLOR, linewidths=0.7, alpha=0.9)

        fig.colorbar(im1, cax=cax, label=_cbar_label_for(map_bunit, flavour))

        for ax, panel in ((ax1, "image"), (ax2, "residual")):
            ax.set_title(f"{flavour}: {panel}", fontsize=10)
            if wcs is not None:
                ax.set_xlabel("RA (J2000)")
                ax.set_ylabel("Dec (J2000)")
        # ax2 shares ax1's WCS exactly — hide the residual panel's Dec
        # axis label + tick labels so they don't bleed into ax1's frame.
        # On WCSAxes we use coords[1] (Dec); on plain axes the regular
        # ylabel API.
        if wcs is not None:
            ax2.coords[1].set_ticklabel_visible(False)
            ax2.coords[1].set_axislabel("")
        else:
            ax2.set_yticklabels([])
            ax2.set_ylabel("")

        suptitle = f"{ARRAY_TAG} | {flavour} summary"
        if source_label:
            suptitle = f"{suptitle} | {source_label}"
        fig.suptitle(suptitle, fontsize=11)
        _atomic_savefig(fig, out_path)
    finally:
        plt.close(fig)


def plot_payload_for_cube(
    *,
    bundle,
    payload: dict[str, Any],
    needed_png: dict[str, Path],
    needed_summary: dict[str, Path],
    statuses: dict[str, str],
    source_label: str | None,
) -> None:
    """Driver: turn a compute result into the requested PNGs.

    Called from :func:`process_cube` so the heavy compute / FITS-write
    work stays in ``clean_diagnostics.py`` and the matplotlib import
    happens lazily.
    """
    header = bundle.header

    # Per-product map / mask / paired-spectrum PNGs.
    map_dispatch = {
        "integrated_intensity_image":
            (payload["integ_image_k"], payload["map_bunit_mom0"]),
        "integrated_intensity_residual":
            (payload["integ_resid_k"], payload["map_bunit_mom0"]),
        "peak_intensity_image":
            (payload["peak_image_k"], payload["map_bunit_2d"]),
        "peak_intensity_residual":
            (payload["peak_resid_k"], payload["map_bunit_2d"]),
    }
    for kind, (arr, bunit) in map_dispatch.items():
        if kind in needed_png:
            try:
                _plot_single_map(
                    arr, header, needed_png[kind],
                    kind=kind, bunit=bunit, source_label=source_label,
                )
                statuses[f"plot:{kind}"] = "written"
                logger.info("wrote %s", needed_png[kind])
            except Exception as exc:  # pragma: no cover
                reason = f"failed:plot_{kind}:{type(exc).__name__}:{exc}"
                logger.error("%s: %s", bundle.pbcor_path.name, reason)
                statuses[f"plot:{kind}"] = reason

    if "clean_mask" in needed_png:
        try:
            _plot_mask(
                payload["mask_2d"], header, needed_png["clean_mask"],
                kind="clean_mask", source_label=source_label,
            )
            statuses["plot:clean_mask"] = "written"
            logger.info("wrote %s", needed_png["clean_mask"])
        except Exception as exc:  # pragma: no cover
            reason = f"failed:plot_clean_mask:{type(exc).__name__}:{exc}"
            logger.error("%s: %s", bundle.pbcor_path.name, reason)
            statuses["plot:clean_mask"] = reason

    if "mean_spectra_in_mask" in needed_png:
        try:
            _plot_paired_spectrum(
                payload["spec_image_k"], payload["spec_resid_k"],
                payload["spectral_values"], payload["spectral_unit"],
                payload["spec_unit"],
                needed_png["mean_spectra_in_mask"],
                mask_kind="clean_mask",
                source_label=source_label,
            )
            statuses["plot:mean_spectra_in_mask"] = "written"
            logger.info("wrote %s", needed_png["mean_spectra_in_mask"])
        except Exception as exc:  # pragma: no cover
            reason = f"failed:plot_mean_spectra_in_mask:{type(exc).__name__}:{exc}"
            logger.error("%s: %s", bundle.pbcor_path.name, reason)
            statuses["plot:mean_spectra_in_mask"] = reason

    # Summary plots (image | residual, with shared cbar + mask contour).
    summary_inputs = {
        "summary_integrated": (
            payload["integ_image_k"], payload["integ_resid_k"],
            payload["map_bunit_mom0"], "integrated",
        ),
        "summary_peak": (
            payload["peak_image_k"], payload["peak_resid_k"],
            payload["map_bunit_2d"], "peak",
        ),
    }
    for kind, (map_img, map_res, map_bunit, flavour) in summary_inputs.items():
        if kind in needed_summary:
            try:
                _plot_summary(
                    map_img, map_res, payload["mask_2d"],
                    map_bunit, header, needed_summary[kind],
                    flavour=flavour, source_label=source_label,
                )
                statuses[f"plot:{kind}"] = "written"
                logger.info("wrote %s", needed_summary[kind])
            except Exception as exc:  # pragma: no cover
                reason = f"failed:plot_{kind}:{type(exc).__name__}:{exc}"
                logger.error("%s: %s", bundle.pbcor_path.name, reason)
                statuses[f"plot:{kind}"] = reason
