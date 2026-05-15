"""Tests for ``panta_rei.analysis.clean_diagnostics``."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


# ---------- path / discovery tests (no FITS payload) ----------


def test_discover_cubes_only_12m7m_under_aux(tmp_path: Path):
    from panta_rei.analysis.clean_diagnostics import discover_cubes

    imaging = tmp_path / "imaging" / "output"
    aux_group = imaging / "aux" / "group.uid___A001_X3833_X64b9.lp_nperetto"
    plain_group = imaging / "group.uid___A001_X3833_X64b9.lp_nperetto"
    aux_group.mkdir(parents=True)
    plain_group.mkdir(parents=True)

    # 12m7m pbcor under aux/ — should be picked up
    (aux_group / "src.AG.12m7m.86-86.GHz.cube.pbcor.fits").write_bytes(b"")
    # 12m7mTP sibling under aux/ — same setup but different combo, must NOT match
    (aux_group / "src.AG.12m7mTP.86-86.GHz.cube.pbcor.fits").write_bytes(b"")
    # 12m7m pbcor outside aux/ — must NOT match (lives in plain group)
    (plain_group / "src.AG.12m7m.86-86.GHz.cube.pbcor.fits").write_bytes(b"")
    # mask/residual/pb under aux/ — must not appear as 'cubes'
    (aux_group / "src.AG.12m7m.86-86.GHz.cube.mask.fits").write_bytes(b"")

    found = discover_cubes(imaging)
    assert [p.name for p in found] == ["src.AG.12m7m.86-86.GHz.cube.pbcor.fits"]
    assert all(p.parent == aux_group for p in found)


def test_resolve_siblings_returns_four_paths(tmp_path: Path):
    from panta_rei.analysis.clean_diagnostics import resolve_siblings

    aux_group = tmp_path / "aux" / "group.uid___A001_X3833_X.lp_nperetto"
    aux_group.mkdir(parents=True)
    base = aux_group / "s.AG.12m7m.86-86.GHz.cube"
    for kind in ("pbcor", "mask", "residual", "pb"):
        (Path(f"{base}.{kind}.fits")).write_bytes(b"")
    pbcor = Path(f"{base}.pbcor.fits")
    sibs = resolve_siblings(pbcor)
    assert set(sibs) == {"pbcor", "mask", "residual", "pb"}
    for kind, p in sibs.items():
        assert p.name.endswith(f".cube.{kind}.fits")


def test_resolve_siblings_raises_when_pb_missing(tmp_path: Path):
    from panta_rei.analysis.clean_diagnostics import MissingSibling, resolve_siblings

    aux_group = tmp_path / "aux" / "group.uid___A001_X3833_X.lp_nperetto"
    aux_group.mkdir(parents=True)
    base = aux_group / "s.AG.12m7m.86-86.GHz.cube"
    # All siblings EXCEPT pb
    for kind in ("pbcor", "mask", "residual"):
        (Path(f"{base}.{kind}.fits")).write_bytes(b"")
    with pytest.raises(MissingSibling) as excinfo:
        resolve_siblings(Path(f"{base}.pbcor.fits"))
    assert "pb" in str(excinfo.value)


# ---------- synthetic 4D quad ----------


def _write_4d_cube(
    path: Path, data_3d: np.ndarray, *, bunit: str, with_beam: bool,
) -> None:
    """Write a (1, n_chan, ny, nx) FITS cube with FREQ axis 3 + STOKES axis 4."""
    data_4d = data_3d[np.newaxis, ...].astype(np.float32)  # add Stokes axis
    n_chan, ny, nx = data_3d.shape

    hdr = fits.Header()
    hdr["NAXIS"] = 4
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = n_chan
    hdr["NAXIS4"] = 1
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CTYPE3"] = "FREQ"
    hdr["CTYPE4"] = "STOKES"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CUNIT3"] = "Hz"
    hdr["CRVAL1"] = 0.0
    hdr["CRVAL2"] = 0.0
    hdr["CRVAL3"] = 86.0e9
    hdr["CRVAL4"] = 1.0
    hdr["CDELT1"] = -0.0001
    hdr["CDELT2"] = 0.0001
    hdr["CDELT3"] = 1.0e6  # 1 MHz channels — dv ≈ 3.487 km/s at 86 GHz
    hdr["CDELT4"] = 1.0
    hdr["CRPIX1"] = 1.0
    hdr["CRPIX2"] = 1.0
    hdr["CRPIX3"] = 1.0
    hdr["CRPIX4"] = 1.0
    hdr["BUNIT"] = bunit
    if with_beam:
        hdr["BMAJ"] = 1.0e-4
        hdr["BMIN"] = 8.0e-5
        hdr["BPA"] = 0.0
    hdr["RESTFRQ"] = 86.0e9
    hdr["EQUINOX"] = 2000.0
    hdr["RADESYS"] = "ICRS"
    hdr["SPECSYS"] = "LSRK"
    fits.PrimaryHDU(data=data_4d, header=hdr).writeto(str(path))


@pytest.fixture
def quad_4d(tmp_path: Path) -> dict:
    """Synthetic 4-cube 4D-singleton-Stokes set under aux/<group>/."""
    pytest.importorskip("spectral_cube")

    n_chan, ny, nx = 8, 6, 6
    rng = np.random.default_rng(42)

    pbcor = rng.normal(0.0, 0.01, size=(n_chan, ny, nx)).astype(np.float32)
    # Inject signal at channel 3 in a 2×2 block
    pbcor[3, 2:4, 2:4] = 5.0
    # Residual is the unmodelled flux — small, with a small spike
    residual = rng.normal(0.0, 0.005, size=(n_chan, ny, nx)).astype(np.float32)
    residual[3, 2, 2] = 0.3
    # PB taper from centre — values in [0, 1]
    yy, xx = np.indices((ny, nx))
    r = np.sqrt((yy - (ny - 1) / 2) ** 2 + (xx - (nx - 1) / 2) ** 2)
    pb_plane = np.clip(1.0 - r / (1.5 * (ny / 2)), 0.0, 1.0).astype(np.float32)
    pb = np.broadcast_to(pb_plane, (n_chan, ny, nx)).copy()
    # Mask: 1 around the signal pixel only at the signal channel
    mask = np.zeros((n_chan, ny, nx), dtype=np.float32)
    mask[3, 2:4, 2:4] = 1.0

    imaging = tmp_path / "imaging" / "output"
    aux_group = imaging / "aux" / "group.uid___A001_X3833_TEST.lp_nperetto"
    aux_group.mkdir(parents=True)

    base = aux_group / "synth.AG.12m7m.86-86.GHz.cube"
    _write_4d_cube(Path(f"{base}.pbcor.fits"), pbcor, bunit="Jy/beam", with_beam=True)
    _write_4d_cube(Path(f"{base}.mask.fits"), mask, bunit="", with_beam=False)
    _write_4d_cube(Path(f"{base}.residual.fits"), residual, bunit="", with_beam=False)
    _write_4d_cube(Path(f"{base}.pb.fits"), pb, bunit="", with_beam=False)

    return {
        "imaging_dir": imaging,
        "analysis_dir": tmp_path / "analysis",
        "pbcor_path": Path(f"{base}.pbcor.fits"),
        "n_chan": n_chan, "ny": ny, "nx": nx,
    }


# ---------- compute / writer tests ----------


def test_process_cube_writes_seven_fits_six_png_two_summary(quad_4d):
    from panta_rei.analysis.clean_diagnostics import (
        PRODUCT_KINDS, SUMMARY_PLOT_KINDS, PAIRED_SPECTRUM_PLOT_KINDS,
        SINGLE_PANEL_MAP_PNG_KINDS,
        process_cube,
    )

    res = process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=True)

    # No failures.
    failures = [k for k, v in res.products.items() if v.startswith("failed:")]
    assert not failures, f"unexpected failures: {failures}"

    # 7 FITS + 5 single-panel map/mask PNGs + 1 paired-spectrum PNG + 2 summaries.
    expected_status_keys = set(PRODUCT_KINDS)
    expected_status_keys |= {f"plot:{k}" for k in SINGLE_PANEL_MAP_PNG_KINDS}
    expected_status_keys |= {f"plot:{k}" for k in PAIRED_SPECTRUM_PLOT_KINDS}
    expected_status_keys |= {f"plot:{k}" for k in SUMMARY_PLOT_KINDS}
    assert set(res.products) == expected_status_keys

    base = quad_4d["pbcor_path"].name[: -len(".cube.pbcor.fits")]
    group = quad_4d["pbcor_path"].parent.name
    fits_dir = quad_4d["analysis_dir"] / group / "fits"
    png_dir = quad_4d["analysis_dir"] / group / "png"
    for kind in PRODUCT_KINDS:
        assert (fits_dir / f"{base}.{kind}.fits").exists(), kind
    for kind in SINGLE_PANEL_MAP_PNG_KINDS + PAIRED_SPECTRUM_PLOT_KINDS + SUMMARY_PLOT_KINDS:
        assert (png_dir / f"{base}.{kind}.png").exists(), kind


def test_squeeze_stokes_via_open_cube_arrays(quad_4d):
    from contextlib import ExitStack

    from panta_rei.analysis.clean_diagnostics import (
        _open_cube_arrays, resolve_siblings,
    )

    sibs = resolve_siblings(quad_4d["pbcor_path"])
    with ExitStack() as stack:
        bundle = _open_cube_arrays(sibs, stack)
        assert bundle.pbcor.shape == (quad_4d["n_chan"], quad_4d["ny"], quad_4d["nx"])
        assert bundle.residual.shape == bundle.pbcor.shape
        assert bundle.mask.shape == bundle.pbcor.shape
        assert bundle.pb.shape == bundle.pbcor.shape


def test_shape_mismatch_detected(quad_4d, tmp_path: Path):
    """If residual has wrong n_chan, ShapeMismatch is raised."""
    from contextlib import ExitStack

    from panta_rei.analysis.clean_diagnostics import (
        ShapeMismatch, _open_cube_arrays, resolve_siblings,
    )

    # Overwrite the residual with a wrong-shape cube.
    sibs = resolve_siblings(quad_4d["pbcor_path"])
    bad = np.zeros((quad_4d["n_chan"] + 1, quad_4d["ny"], quad_4d["nx"]),
                   dtype=np.float32)
    sibs["residual"].unlink()
    _write_4d_cube(sibs["residual"], bad, bunit="", with_beam=False)

    with pytest.raises(ShapeMismatch):
        with ExitStack() as stack:
            _open_cube_arrays(sibs, stack)


def test_dv_kms_from_header_radio_convention():
    from panta_rei.analysis.clean_diagnostics import compute_dv_kms

    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["CTYPE3"] = "FREQ"
    hdr["CDELT3"] = 1.0e6
    hdr["RESTFRQ"] = 86.0e9
    # dv = c * |CDELT3| / RESTFRQ = 299792.458 * 1e6 / 86e9 = 3.4859... km/s
    assert compute_dv_kms(hdr) == pytest.approx(299792.458 * 1.0e6 / 86.0e9, rel=1e-6)


def test_dv_kms_handles_negative_cdelt():
    from panta_rei.analysis.clean_diagnostics import compute_dv_kms

    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["CTYPE3"] = "FREQ"
    hdr["CDELT3"] = -1.0e6  # flipped sign — must still give positive dv
    hdr["RESTFRQ"] = 86.0e9
    assert compute_dv_kms(hdr) > 0


def test_mask_is_uint8_zero_one_and_records_projection(quad_4d):
    from panta_rei.analysis.clean_diagnostics import process_cube

    process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)
    base = quad_4d["pbcor_path"].name[: -len(".cube.pbcor.fits")]
    group = quad_4d["pbcor_path"].parent.name
    out = quad_4d["analysis_dir"] / group / "fits" / f"{base}.clean_mask.fits"
    with fits.open(out) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
    assert data.dtype == np.uint8
    assert set(np.unique(data).tolist()) <= {0, 1}
    assert hdr["PROJ"] == "max(axis=0)"


def test_provenance_keywords_recorded(quad_4d):
    from panta_rei.analysis.clean_diagnostics import process_cube

    process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)
    base = quad_4d["pbcor_path"].name[: -len(".cube.pbcor.fits")]
    group = quad_4d["pbcor_path"].parent.name
    out = (
        quad_4d["analysis_dir"] / group / "fits"
        / f"{base}.peak_intensity_image.fits"
    )
    with fits.open(out) as hdul:
        hdr = hdul[0].header
    assert hdr["SRCFILE"] == quad_4d["pbcor_path"].name
    assert hdr["PRODUCT"] == "peak_intensity_image"
    assert hdr["SIB_PB"].endswith(".cube.pb.fits")
    assert hdr["SIB_RESI"].endswith(".cube.residual.fits")
    assert hdr["SIB_MASK"].endswith(".cube.mask.fits")
    assert "JYTOK" in hdr
    assert any("RECON_IMG" in str(line) for line in hdr.get("HISTORY", []))


def test_output_units_are_kelvin(quad_4d):
    from panta_rei.analysis.clean_diagnostics import process_cube

    process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)
    base = quad_4d["pbcor_path"].name[: -len(".cube.pbcor.fits")]
    group = quad_4d["pbcor_path"].parent.name
    fits_dir = quad_4d["analysis_dir"] / group / "fits"
    # Peak in K
    with fits.open(fits_dir / f"{base}.peak_intensity_image.fits") as hdul:
        assert hdul[0].header["BUNIT"] == "K"
    # Moment-0 in K km/s
    with fits.open(fits_dir / f"{base}.integrated_intensity_image.fits") as hdul:
        assert hdul[0].header["BUNIT"] == "K km/s"
    # Spectra FLUX column unit = K
    with fits.open(fits_dir / f"{base}.mean_spectrum_image_in_mask.fits") as hdul:
        flux_col = hdul["SPECTRUM"].columns["FLUX"]
        assert flux_col.unit == "K"


def test_idempotent_skip_on_rerun(quad_4d):
    from panta_rei.analysis.clean_diagnostics import process_cube

    process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)
    res2 = process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)
    # All FITS products report skipped (no failures, no rewrites).
    for kind, status in res2.products.items():
        assert status == "skipped", (kind, status)


def test_mtime_skip_uses_max_of_inputs(quad_4d):
    """Touching the residual must invalidate the existing outputs."""
    from panta_rei.analysis.clean_diagnostics import process_cube, resolve_siblings

    process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)
    sibs = resolve_siblings(quad_4d["pbcor_path"])
    # Advance the residual's mtime past the just-written outputs.
    future = time.time() + 5.0
    os.utime(sibs["residual"], (future, future))
    res2 = process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)
    # Outputs must regenerate because input clock advanced.
    assert any(s == "written" for s in res2.products.values())


def test_dry_run_writes_nothing(quad_4d):
    from panta_rei.analysis.clean_diagnostics import process_cube

    res = process_cube(
        quad_4d["pbcor_path"], quad_4d["analysis_dir"],
        plot=False, dry_run=True,
    )
    assert any(s == "dry-run" for s in res.products.values())
    assert not quad_4d["analysis_dir"].exists()


def test_force_regenerates(quad_4d):
    from panta_rei.analysis.clean_diagnostics import process_cube

    process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)
    res2 = process_cube(
        quad_4d["pbcor_path"], quad_4d["analysis_dir"],
        plot=False, force=True,
    )
    # Every FITS product reports written (none skipped).
    assert all(s == "written" for s in res2.products.values()), res2.products


def test_missing_required_sibling_surfaces_per_product(quad_4d):
    """pb absent → every product fails with missing_sibling (pb is required)."""
    from panta_rei.analysis.clean_diagnostics import process_cube, resolve_siblings

    sibs = resolve_siblings(quad_4d["pbcor_path"])
    sibs["pb"].unlink()
    res = process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)
    assert any(s.startswith("failed:missing_sibling") for s in res.products.values())


def test_missing_mask_synthesises_empty_mask(quad_4d):
    """mask absent → process continues; mask FITS is all-zero, spectra all-NaN.

    Mirrors the real-world case where tclean's auto-masking found
    nothing to mask (no detectable emission). The 4 intensity maps
    remain valuable QA so we still emit them.
    """
    from panta_rei.analysis.clean_diagnostics import (
        PRODUCT_KINDS, process_cube, resolve_siblings,
    )

    # Delete the mask sibling before processing.
    sibs = resolve_siblings(quad_4d["pbcor_path"])
    sibs["mask"].unlink()

    res = process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)

    # No failures; all 7 FITS produced.
    failures = [k for k, v in res.products.items() if v.startswith("failed:")]
    assert not failures, f"unexpected failures: {failures}"

    base = quad_4d["pbcor_path"].name[: -len(".cube.pbcor.fits")]
    group = quad_4d["pbcor_path"].parent.name
    fits_dir = quad_4d["analysis_dir"] / group / "fits"
    for kind in PRODUCT_KINDS:
        assert (fits_dir / f"{base}.{kind}.fits").exists(), kind

    # 2D mask is all-zero.
    with fits.open(fits_dir / f"{base}.clean_mask.fits") as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
    assert data.dtype == np.uint8
    assert data.sum() == 0
    assert hdr["SIB_MASK"].startswith("(absent")

    # Image moment / peak maps still finite where signal lived.
    with fits.open(fits_dir / f"{base}.peak_intensity_image.fits") as hdul:
        peak = hdul[0].data
    assert np.isfinite(peak).any()

    # Spectra are all-NaN (mask is empty).
    with fits.open(fits_dir / f"{base}.mean_spectrum_image_in_mask.fits") as hdul:
        flux = hdul["SPECTRUM"].data["FLUX"]
    assert np.all(np.isnan(flux))


def test_peak_image_value_is_positive_in_signal_pixel(quad_4d):
    """Sanity check: the synthetic signal at chan 3, pixel (2,2) drives a
    positive K value at that pixel in the peak_intensity_image map."""
    from panta_rei.analysis.clean_diagnostics import process_cube

    process_cube(quad_4d["pbcor_path"], quad_4d["analysis_dir"], plot=False)
    base = quad_4d["pbcor_path"].name[: -len(".cube.pbcor.fits")]
    group = quad_4d["pbcor_path"].parent.name
    out = (
        quad_4d["analysis_dir"] / group / "fits"
        / f"{base}.peak_intensity_image.fits"
    )
    with fits.open(out) as hdul:
        data = hdul[0].data
    # signal lives at (y=2,x=2); pb taper attenuates but keeps it well above zero.
    assert np.isfinite(data[2, 2])
    assert data[2, 2] > 0
