"""Tests for continuum quicklook-PNG generation."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from panta_rei.analysis.continuum_quicklook import (
    MIN_ALPHA_PIXELS,
    discover_units,
    human_source_label,
    needs_regeneration,
    process_unit,
    sibling_paths,
)
from panta_rei.analysis.plots import mask_alpha_by_snr

GROUP = "group.uid___A001_X3833_X64cf.lp_nperetto"
BASE = f"{GROUP}.AG301.1365m0.2259.12m7m.90.6-104.5GHz"


def _wcs_header(nx: int = 32, ny: int = 32) -> fits.Header:
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"], h["NAXIS2"] = nx, ny
    h["CTYPE1"], h["CTYPE2"] = "RA---SIN", "DEC--SIN"
    h["CRVAL1"], h["CRVAL2"] = 188.9, -63.0
    h["CRPIX1"], h["CRPIX2"] = nx / 2, ny / 2
    h["CDELT1"], h["CDELT2"] = -1e-4, 1e-4
    h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
    return h


def _write_image(path: Path, data: np.ndarray, bunit: str = "Jy/beam") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = _wcs_header(*data.shape[::-1])
    h["BUNIT"] = bunit
    fits.PrimaryHDU(data=data.astype(np.float32), header=h).writeto(path, overwrite=True)


def _make_unit(imaging_dir: Path, *, bright: bool = True) -> Path:
    """Create a continuum unit (tt0.pbcor + alpha + aux flat) and return the tt0 path."""
    sci = imaging_dir / GROUP
    aux = imaging_dir / "aux" / GROUP
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1e-3, size=(32, 32))  # 1 mJy/beam rms
    flux = noise.copy()
    if bright:
        flux[13:21, 13:21] = 0.5  # 64-pixel bright patch (> MIN_ALPHA_PIXELS)
    _write_image(sci / f"{BASE}.cont.tt0.pbcor.fits", flux)
    _write_image(aux / f"{BASE}.cont.tt0.image.fits", flux)  # flat ≈ pbcor near centre
    alpha = np.full((32, 32), 2.5)  # physical spectral index at the source
    _write_image(sci / f"{BASE}.cont.alpha.fits", alpha, bunit="")
    return sci / f"{BASE}.cont.tt0.pbcor.fits"


# --- path / helper unit tests ------------------------------------------------

def test_sibling_paths(tmp_path: Path):
    tt0 = tmp_path / "imaging" / "output" / GROUP / f"{BASE}.cont.tt0.pbcor.fits"
    p = sibling_paths(tt0)
    assert p["alpha_fits"].name == f"{BASE}.cont.alpha.fits"
    assert p["flat_fits"] == tmp_path / "imaging" / "output" / "aux" / GROUP / f"{BASE}.cont.tt0.image.fits"
    # PNGs sit beside their FITS, stem == FITS stem (so the portal attaches them).
    assert p["tt0_png"].name == f"{BASE}.cont.tt0.pbcor.png"
    assert p["alpha_png"].name == f"{BASE}.cont.alpha.png"
    assert p["tt0_png"].parent == tt0.parent


def test_sibling_paths_rejects_non_tt0(tmp_path: Path):
    with pytest.raises(ValueError):
        sibling_paths(tmp_path / "x.cont.alpha.fits")


def test_human_source_label_desanitises_sign():
    p = Path(f"{BASE}.cont.tt0.pbcor.fits")
    assert human_source_label(p) == "AG301.1365-0.2259 | 90.6-104.5 GHz"


def test_needs_regeneration(tmp_path: Path):
    src = tmp_path / "a.fits"; src.write_text("x")
    out = tmp_path / "a.png"
    assert needs_regeneration([src], out, force=False) is True  # missing
    out.write_text("y")
    os.utime(out, (time.time() + 10, time.time() + 10))  # out newer
    assert needs_regeneration([src], out, force=False) is False
    assert needs_regeneration([src], out, force=True) is True   # force overrides
    os.utime(src, (time.time() + 100, time.time() + 100))       # src newer → stale
    assert needs_regeneration([src], out, force=False) is True


def test_mask_alpha_by_snr_blanks_noise():
    rng = np.random.default_rng(1)
    flat = rng.normal(0, 1e-3, size=(32, 32))
    flat[16, 16] = 0.5  # one bright pixel
    alpha = np.full((32, 32), 3.0)
    masked, n_sig = mask_alpha_by_snr(alpha, flat, snr=5.0)
    assert n_sig >= 1
    assert np.isfinite(masked[16, 16])
    assert np.isnan(masked[0, 0])  # noise blanked


# --- process_unit integration ------------------------------------------------

def test_process_unit_writes_both_pngs(tmp_path: Path):
    imaging = tmp_path / "imaging" / "output"
    tt0 = _make_unit(imaging, bright=True)
    res = process_unit(tt0, snr=5.0)
    assert res.products["tt0_pbcor"] == "written"
    assert res.products["alpha"] == "written"
    assert not res.any_failed
    p = sibling_paths(tt0)
    assert p["tt0_png"].exists() and p["tt0_png"].stat().st_size > 0
    assert p["alpha_png"].exists()


def test_process_unit_idempotent_skip(tmp_path: Path):
    imaging = tmp_path / "imaging" / "output"
    tt0 = _make_unit(imaging, bright=True)
    process_unit(tt0, snr=5.0)
    res2 = process_unit(tt0, snr=5.0)
    assert res2.products["tt0_pbcor"] == "skipped"
    assert res2.products["alpha"] == "skipped"


def test_process_unit_dry_run_writes_nothing(tmp_path: Path):
    imaging = tmp_path / "imaging" / "output"
    tt0 = _make_unit(imaging, bright=True)
    res = process_unit(tt0, snr=5.0, dry_run=True)
    assert res.products["tt0_pbcor"] == "dry-run"
    assert not sibling_paths(tt0)["tt0_png"].exists()


def test_process_unit_skips_empty_alpha(tmp_path: Path):
    # No bright source → no pixel passes S/N → alpha skipped, tt0 still written.
    imaging = tmp_path / "imaging" / "output"
    tt0 = _make_unit(imaging, bright=False)
    res = process_unit(tt0, snr=5.0)
    assert res.products["tt0_pbcor"] == "written"
    assert res.products["alpha"] == "skipped:no_significant_alpha"
    assert not sibling_paths(tt0)["alpha_png"].exists()


def test_process_unit_missing_alpha_fits(tmp_path: Path):
    imaging = tmp_path / "imaging" / "output"
    tt0 = _make_unit(imaging, bright=True)
    sibling_paths(tt0)["alpha_fits"].unlink()
    res = process_unit(tt0, snr=5.0)
    assert res.products["tt0_pbcor"] == "written"
    assert res.products["alpha"] == "skipped:no_alpha_fits"


def test_discover_units_filters_by_group(tmp_path: Path):
    imaging = tmp_path / "imaging" / "output"
    _make_unit(imaging, bright=True)
    assert len(discover_units(imaging)) == 1
    assert len(discover_units(imaging, group_filters=["X64cf"])) == 1
    assert discover_units(imaging, group_filters=["NOPE"]) == []
