"""Tests for ``panta_rei.analysis.moments``."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from panta_rei.analysis.moments import (
    PRODUCT_KINDS,
    derive_output_paths,
    discover_cubes,
    needs_regeneration,
    process_cube,
)


# ---------- pure-Python helpers ----------

def test_derive_output_paths_default_products(tmp_path: Path):
    cube = (
        tmp_path / "imaging" / "output"
        / "group.uid___A001_X3833_X64b9.lp_nperetto"
        / "group.uid___A001_X3833_X64b9.lp_nperetto.AG221.12m7m.102.5-102.6GHz.cube.pbcor.fits"
    )
    cube.parent.mkdir(parents=True)
    cube.write_bytes(b"")  # placeholder; only the path matters here

    analysis = tmp_path / "analysis"
    out = derive_output_paths(cube, analysis)

    assert set(out) == set(PRODUCT_KINDS)
    expected_dir = (
        analysis / "group.uid___A001_X3833_X64b9.lp_nperetto" / "fits"
    )
    base = (
        "group.uid___A001_X3833_X64b9.lp_nperetto.AG221.12m7m."
        "102.5-102.6GHz.cube.pbcor"
    )
    for kind in PRODUCT_KINDS:
        assert out[kind] == expected_dir / f"{base}.{kind}.fits"


def test_derive_output_paths_subset(tmp_path: Path):
    cube = tmp_path / "g" / "x.fits"
    cube.parent.mkdir()
    cube.write_bytes(b"")
    out = derive_output_paths(cube, tmp_path / "a", products=["peak_intensity"])
    assert list(out) == ["peak_intensity"]
    assert out["peak_intensity"].name == "x.peak_intensity.fits"


def test_derive_output_paths_rejects_non_fits(tmp_path: Path):
    bad = tmp_path / "g" / "x.image"
    bad.parent.mkdir()
    bad.write_bytes(b"")
    with pytest.raises(ValueError):
        derive_output_paths(bad, tmp_path / "a")


def test_needs_regeneration_missing(tmp_path: Path):
    src = tmp_path / "a.fits"
    src.write_bytes(b"x")
    dst = tmp_path / "b.fits"
    assert needs_regeneration(src, dst, force=False) is True


def test_needs_regeneration_newer(tmp_path: Path):
    src = tmp_path / "a.fits"
    dst = tmp_path / "b.fits"
    src.write_bytes(b"x")
    time.sleep(0.01)
    dst.write_bytes(b"y")
    assert needs_regeneration(src, dst, force=False) is False


def test_needs_regeneration_stale(tmp_path: Path):
    src = tmp_path / "a.fits"
    dst = tmp_path / "b.fits"
    dst.write_bytes(b"y")
    time.sleep(0.01)
    src.write_bytes(b"x")
    assert needs_regeneration(src, dst, force=False) is True


def test_needs_regeneration_force(tmp_path: Path):
    src = tmp_path / "a.fits"
    dst = tmp_path / "b.fits"
    src.write_bytes(b"x")
    dst.write_bytes(b"y")
    # dst newer than src, but force overrides
    os.utime(dst, (time.time() + 60, time.time() + 60))
    assert needs_regeneration(src, dst, force=True) is True


def test_discover_cubes_filters(tmp_path: Path):
    imaging = tmp_path / "imaging" / "output"
    g1 = imaging / "group.uid___A001_X3833_X64b9.lp_nperetto"
    g2 = imaging / "group.uid___A001_X3833_X64c0.lp_nperetto"
    g1.mkdir(parents=True)
    g2.mkdir(parents=True)
    (g1 / "a.12m7mTP.cube.pbcor.fits").write_bytes(b"")
    (g1 / "b.12m7mTP.cube.pbcor.fits").write_bytes(b"")
    (g2 / "c.12m7mTP.cube.pbcor.fits").write_bytes(b"")
    # ignored: not a *.cube.pbcor.fits
    (g1 / "a.12m7mTP.cube.residual.fits").write_bytes(b"")

    all_cubes = discover_cubes(imaging)
    assert len(all_cubes) == 3

    just_g1 = discover_cubes(imaging, group_filters=["X64b9"])
    assert {c.parent.name for c in just_g1} == {g1.name}
    assert len(just_g1) == 2


def test_discover_cubes_array_combo_default_excludes_bare_12m7m(tmp_path: Path):
    """Default filter keeps 12m7mTP but rejects the bare 12m7m product."""
    imaging = tmp_path / "imaging" / "output"
    g = imaging / "group.uid___A001_X3833_X64b9.lp_nperetto"
    g.mkdir(parents=True)
    tp = g / "src.AG.12m7mTP.86-86.GHz.cube.pbcor.fits"
    bare = g / "src.AG.12m7m.86-86.GHz.cube.pbcor.fits"
    tp.write_bytes(b"")
    bare.write_bytes(b"")

    found = discover_cubes(imaging)
    assert [c.name for c in found] == [tp.name]


def test_discover_cubes_array_combo_override(tmp_path: Path):
    """Explicit override lets callers select the bare 12m7m product."""
    imaging = tmp_path / "imaging" / "output"
    g = imaging / "group.uid___A001_X3833_X64b9.lp_nperetto"
    g.mkdir(parents=True)
    tp = g / "src.AG.12m7mTP.86-86.GHz.cube.pbcor.fits"
    bare = g / "src.AG.12m7m.86-86.GHz.cube.pbcor.fits"
    tp.write_bytes(b"")
    bare.write_bytes(b"")

    only_bare = discover_cubes(imaging, array_combos=["12m7m"])
    assert [c.name for c in only_bare] == [bare.name]

    both = discover_cubes(imaging, array_combos=["12m7m", "12m7mTP"])
    assert {c.name for c in both} == {tp.name, bare.name}

    no_filter = discover_cubes(imaging, array_combos=None)
    assert {c.name for c in no_filter} == {tp.name, bare.name}


# ---------- end-to-end with a synthetic cube ----------

def _write_synthetic_cube(path: Path) -> tuple[np.ndarray, dict]:
    """Build a tiny FREQ cube with known signal and write to FITS.

    Returns ``(data, meta)`` where data has shape (n_chan, ny, nx) with
    the spectral axis first as ``np.ndarray``-style ordering.
    """
    n_chan, ny, nx = 8, 4, 4
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 0.1, size=(n_chan, ny, nx)).astype(np.float32)
    # Inject a strong signal at channel 3, pixel (1, 2)
    data[3, 1, 2] = 100.0
    # Inject a weaker negative spike at channel 5
    data[5, 2, 3] = -10.0

    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = n_chan
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CTYPE3"] = "FREQ"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CUNIT3"] = "Hz"
    hdr["CRVAL1"] = 0.0
    hdr["CRVAL2"] = 0.0
    hdr["CRVAL3"] = 86.0e9
    hdr["CDELT1"] = -0.0001
    hdr["CDELT2"] = 0.0001
    hdr["CDELT3"] = 1.0e6  # 1 MHz channels
    hdr["CRPIX1"] = 1.0
    hdr["CRPIX2"] = 1.0
    hdr["CRPIX3"] = 1.0
    hdr["BUNIT"] = "Jy/beam"
    hdr["BMAJ"] = 1.0e-4
    hdr["BMIN"] = 8.0e-5
    hdr["BPA"] = 0.0
    hdr["RESTFRQ"] = 86.0e9
    hdr["EQUINOX"] = 2000.0
    hdr["RADESYS"] = "ICRS"
    hdr["SPECSYS"] = "LSRK"
    fits.PrimaryHDU(data=data, header=hdr).writeto(str(path))
    return data, dict(n_chan=n_chan, ny=ny, nx=nx)


@pytest.fixture
def synthetic_cube(tmp_path: Path) -> dict:
    pytest.importorskip("spectral_cube")
    imaging = tmp_path / "imaging" / "output"
    group_dir = imaging / "group.uid___A001_X3833_TEST.lp_nperetto"
    group_dir.mkdir(parents=True)
    cube_path = group_dir / "synth.AG.12m7m.86-86.GHz.cube.pbcor.fits"
    data, meta = _write_synthetic_cube(cube_path)
    return {
        "imaging_dir": imaging,
        "analysis_dir": tmp_path / "analysis",
        "cube_path": cube_path,
        "data": data,
        "meta": meta,
    }


def test_process_cube_writes_three_products(synthetic_cube):
    res = process_cube(
        synthetic_cube["cube_path"],
        synthetic_cube["analysis_dir"],
        plot=False,
    )
    assert set(res.products) == set(PRODUCT_KINDS)
    assert all(s == "written" for s in res.products.values()), res.products

    for kind in PRODUCT_KINDS:
        out = (
            synthetic_cube["analysis_dir"] / synthetic_cube["cube_path"].parent.name
            / "fits"
            / f"{synthetic_cube['cube_path'].name[:-len('.fits')]}.{kind}.fits"
        )
        assert out.exists(), f"missing: {out}"


def test_process_cube_writes_plot_pngs(synthetic_cube):
    res = process_cube(
        synthetic_cube["cube_path"],
        synthetic_cube["analysis_dir"],
        plot=True,
    )
    # All FITS + all plot entries written.
    expected = set(PRODUCT_KINDS) | {f"plot:{k}" for k in PRODUCT_KINDS}
    assert set(res.products) == expected, res.products
    assert all(s == "written" for s in res.products.values()), res.products

    png_dir = (
        synthetic_cube["analysis_dir"]
        / synthetic_cube["cube_path"].parent.name / "png"
    )
    base = synthetic_cube["cube_path"].name[: -len(".fits")]
    for kind in PRODUCT_KINDS:
        png = png_dir / f"{base}.{kind}.png"
        assert png.exists(), f"missing plot: {png}"
        # PNG signature is the first eight bytes of the file.
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_process_cube_idempotent_skip_on_rerun(synthetic_cube):
    process_cube(
        synthetic_cube["cube_path"], synthetic_cube["analysis_dir"], plot=False,
    )
    res2 = process_cube(
        synthetic_cube["cube_path"], synthetic_cube["analysis_dir"], plot=False,
    )
    assert all(s == "skipped" for s in res2.products.values()), res2.products


def test_process_cube_force_rewrites(synthetic_cube):
    process_cube(
        synthetic_cube["cube_path"], synthetic_cube["analysis_dir"], plot=False,
    )
    res2 = process_cube(
        synthetic_cube["cube_path"],
        synthetic_cube["analysis_dir"],
        plot=False,
        force=True,
    )
    assert all(s == "written" for s in res2.products.values()), res2.products


def test_process_cube_dry_run_writes_nothing(synthetic_cube):
    res = process_cube(
        synthetic_cube["cube_path"],
        synthetic_cube["analysis_dir"],
        plot=False,
        dry_run=True,
    )
    assert all(s == "dry-run" for s in res.products.values())
    assert not synthetic_cube["analysis_dir"].exists()


def test_peak_and_moment_have_correct_shapes_and_values(synthetic_cube):
    process_cube(
        synthetic_cube["cube_path"], synthetic_cube["analysis_dir"], plot=False,
    )
    base = synthetic_cube["cube_path"].name[:-len(".fits")]
    out_dir = (
        synthetic_cube["analysis_dir"]
        / synthetic_cube["cube_path"].parent.name / "fits"
    )

    peak = fits.getdata(out_dir / f"{base}.peak_intensity.fits")
    m0 = fits.getdata(out_dir / f"{base}.integrated_intensity.fits")
    meta = synthetic_cube["meta"]

    assert peak.shape == (meta["ny"], meta["nx"])
    assert m0.shape == (meta["ny"], meta["nx"])
    # Peak at injected pixel should be ~100 Jy/beam
    assert np.isfinite(peak[1, 2])
    assert peak[1, 2] > 50.0
    # Moment 0 integrates: signal * channel_width (km/s after auto velocity).
    # We just check the same pixel dominates.
    assert m0[1, 2] > 0.5 * np.nanmax(m0)


def test_mean_spectrum_has_correct_length_and_columns(synthetic_cube):
    process_cube(
        synthetic_cube["cube_path"], synthetic_cube["analysis_dir"], plot=False,
    )
    base = synthetic_cube["cube_path"].name[:-len(".fits")]
    spec_path = (
        synthetic_cube["analysis_dir"]
        / synthetic_cube["cube_path"].parent.name / "fits"
        / f"{base}.mean_spectrum.fits"
    )
    with fits.open(spec_path) as hdul:
        assert "SPECTRUM" in [h.name for h in hdul]
        bin_hdu = hdul["SPECTRUM"]
        assert set(bin_hdu.columns.names) == {"SPECTRAL", "FLUX"}
        assert len(bin_hdu.data) == synthetic_cube["meta"]["n_chan"]
        # Channel 3 is the high-signal channel (1 strong pixel out of 16);
        # mean over 16 pixels is dominated by that 100 spike: ~ 100/16 = 6.25
        idx = bin_hdu.data["FLUX"].argmax()
        assert idx == 3
        assert bin_hdu.data["FLUX"][idx] > 1.0


def test_provenance_keywords_in_outputs(synthetic_cube):
    process_cube(
        synthetic_cube["cube_path"], synthetic_cube["analysis_dir"], plot=False,
    )
    base = synthetic_cube["cube_path"].name[:-len(".fits")]
    out_dir = (
        synthetic_cube["analysis_dir"]
        / synthetic_cube["cube_path"].parent.name / "fits"
    )
    for kind in ("integrated_intensity", "peak_intensity"):
        with fits.open(out_dir / f"{base}.{kind}.fits") as hdul:
            hdr = hdul[0].header
            assert hdr["SRCFILE"] == synthetic_cube["cube_path"].name
            assert hdr["PRODUCT"] == kind
