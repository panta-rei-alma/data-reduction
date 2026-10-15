"""Continuum quicklook-PNG generation from the joint 12m+7m MFS products.

Pure compute kernel — no argparse. The CLI driver lives in
``panta_rei.cli.run_continuum_quicklooks``.

For each continuum unit (one ``*.cont.tt0.pbcor.fits`` science image) we
render up to two PNG quicklooks *beside the FITS*, named so the portal
indexer attaches each as that product's preview (stem == FITS stem):

- ``<base>.cont.tt0.pbcor.png`` — asinh flux map (the headline product)
- ``<base>.cont.alpha.png``     — S/N-masked spectral-index map

The alpha mask uses the *flat* (non-pbcor) ``cont.tt0.image`` from the
sibling ``aux/`` dir, whose noise is spatially uniform. Units with no
significant continuum produce no alpha PNG (the product stays preview-less).

Outputs are atomic (``.tmp`` + ``os.replace``, via the plot helpers). Re-runs
are idempotent: a PNG is skipped if it exists and is newer than its FITS
input(s). Pass ``force=True`` to override.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# Science image that anchors a continuum unit. tt1/alpha live beside it; the
# flat (non-pbcor) image lives in the sibling aux/ tree.
CONTINUUM_GLOB = "*.cont.tt0.pbcor.fits"

# Suffixes that turn a unit base into each sibling FITS.
_SCI_TT0 = ".cont.tt0.pbcor.fits"
_SCI_ALPHA = ".cont.alpha.fits"
_AUX_FLAT = ".cont.tt0.image.fits"

# Below this many significant pixels the alpha map is effectively empty, so we
# skip it rather than ship a blank blue thumbnail (the product stays
# preview-less and the portal falls back to its placeholder card).
MIN_ALPHA_PIXELS = 25

PRODUCT_KINDS: tuple[str, ...] = ("tt0_pbcor", "alpha")


@dataclass
class ContinuumResult:
    """Per-unit outcome. ``products`` maps each kind to a status string:
    ``"written"``, ``"skipped"``, ``"dry-run"``, ``"skipped:<reason>"``, or
    ``"failed:<reason>"``."""

    unit: Path
    products: dict[str, str]

    @property
    def any_failed(self) -> bool:
        return any(s.startswith("failed:") for s in self.products.values())


def _unit_base(tt0_fits: Path) -> str:
    """Return the unit basename prefix (everything before ``.cont.tt0.pbcor.fits``)."""
    name = tt0_fits.name
    if not name.endswith(_SCI_TT0):
        raise ValueError(f"not a continuum tt0.pbcor image: {name}")
    return name[: -len(_SCI_TT0)]


def sibling_paths(tt0_fits: Path) -> dict[str, Path]:
    """Map a tt0.pbcor science image to its sibling inputs/outputs.

    Science (tt0.pbcor, alpha) sit in ``imaging/output/<group>/``; the flat
    image sits in ``imaging/output/aux/<group>/``. PNGs are written beside
    their FITS (same dir, ``.fits`` → ``.png``).
    """
    base = _unit_base(tt0_fits)
    sci_dir = tt0_fits.parent
    group = sci_dir.name
    aux_dir = sci_dir.parent / "aux" / group
    return {
        "tt0_fits": tt0_fits,
        "alpha_fits": sci_dir / f"{base}{_SCI_ALPHA}",
        "flat_fits": aux_dir / f"{base}{_AUX_FLAT}",
        "tt0_png": sci_dir / f"{base}.cont.tt0.pbcor.png",
        "alpha_png": sci_dir / f"{base}.cont.alpha.png",
    }


def needs_regeneration(inputs: Iterable[Path], output: Path, force: bool) -> bool:
    """True if ``output`` is missing, older than any input, or ``force`` is set."""
    if force or not output.exists():
        return True
    out_mtime = output.stat().st_mtime
    return any(p.exists() and out_mtime < p.stat().st_mtime for p in inputs)


_NAME_RE = re.compile(
    r"\.lp_nperetto\.(?P<src>.+?)\.12m7m\.(?P<lo>[\d.]+)-(?P<hi>[\d.]+)GHz\.cont\."
)


def human_source_label(fits_path: Path) -> str | None:
    """``<source> | <lo>-<hi> GHz`` for the plot title; ``None`` on parse failure.

    Source name is de-sanitised (numeric-context ``p``/``m`` → ``+``/``-``).
    """
    m = _NAME_RE.search(fits_path.name)
    if m is None:
        return None
    src = re.sub(
        r"(?<=\d)([pm])(?=\d)",
        lambda x: {"p": "+", "m": "-"}[x.group(1)],
        m.group("src"),
    )
    return f"{src} | {m.group('lo')}-{m.group('hi')} GHz"


def _read_2d(fits_path: Path) -> tuple[np.ndarray, fits.Header]:
    """Read a (possibly degenerate-axis) image as a 2D array + its header."""
    with fits.open(fits_path, memmap=False) as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=np.float64))
        header = hdul[0].header
    if data.ndim != 2:
        raise ValueError(f"expected a 2D image after squeeze, got {data.ndim}D: {fits_path.name}")
    return data, header


def process_unit(
    tt0_fits: Path,
    *,
    snr: float = 5.0,
    force: bool = False,
    dry_run: bool = False,
) -> ContinuumResult:
    """Render the tt0.pbcor + alpha quicklooks for one continuum unit.

    Never raises — per-product exceptions are caught and recorded as
    ``failed:<reason>``.
    """
    paths = sibling_paths(tt0_fits)
    label = human_source_label(tt0_fits)
    statuses: dict[str, str] = {}

    # --- tt0.pbcor flux quicklook ---
    if not needs_regeneration([paths["tt0_fits"]], paths["tt0_png"], force):
        statuses["tt0_pbcor"] = "skipped"
    elif dry_run:
        statuses["tt0_pbcor"] = "dry-run"
    else:
        try:
            from panta_rei.analysis.plots import plot_continuum_image

            data, header = _read_2d(paths["tt0_fits"])
            plot_continuum_image(data, header, paths["tt0_png"], source_label=label)
            statuses["tt0_pbcor"] = "written"
            logger.info("wrote %s", paths["tt0_png"])
        except Exception as exc:  # pragma: no cover - depends on FITS payload
            statuses["tt0_pbcor"] = f"failed:tt0:{type(exc).__name__}:{exc}"
            logger.error("%s: %s", tt0_fits.name, statuses["tt0_pbcor"])

    # --- alpha (spectral index) quicklook ---
    if not paths["alpha_fits"].exists():
        statuses["alpha"] = "skipped:no_alpha_fits"
    elif not paths["flat_fits"].exists():
        statuses["alpha"] = "skipped:no_flat_image"
    elif not needs_regeneration(
        [paths["alpha_fits"], paths["flat_fits"]], paths["alpha_png"], force
    ):
        statuses["alpha"] = "skipped"
    elif dry_run:
        statuses["alpha"] = "dry-run"
    else:
        try:
            from panta_rei.analysis.plots import mask_alpha_by_snr, plot_continuum_alpha

            alpha, header = _read_2d(paths["alpha_fits"])
            flat, _ = _read_2d(paths["flat_fits"])
            masked, n_sig = mask_alpha_by_snr(alpha, flat, snr=snr)
            if n_sig < MIN_ALPHA_PIXELS:
                statuses["alpha"] = "skipped:no_significant_alpha"
                logger.info("skip alpha %s (only %d sig px)", paths["alpha_png"].name, n_sig)
            else:
                plot_continuum_alpha(masked, header, paths["alpha_png"], source_label=label, snr=snr)
                statuses["alpha"] = "written"
                logger.info("wrote %s (%d sig px)", paths["alpha_png"], n_sig)
        except Exception as exc:  # pragma: no cover
            statuses["alpha"] = f"failed:alpha:{type(exc).__name__}:{exc}"
            logger.error("%s: %s", tt0_fits.name, statuses["alpha"])

    return ContinuumResult(unit=tt0_fits, products=statuses)


def discover_units(
    imaging_dir: Path,
    group_filters: Iterable[str] | None = None,
) -> list[Path]:
    """List ``*.cont.tt0.pbcor.fits`` science images under ``imaging_dir``.

    ``imaging_dir`` is the imaging-output root (its ``aux/`` subtree holds the
    flat images). ``group_filters`` is an optional iterable of case-sensitive
    substrings matched against the group dirname.
    """
    units = sorted(imaging_dir.glob(f"group.*.lp_nperetto/{CONTINUUM_GLOB}"))
    filters = list(group_filters or [])
    if filters:
        units = [u for u in units if any(f in u.parent.name for f in filters)]
    return units
