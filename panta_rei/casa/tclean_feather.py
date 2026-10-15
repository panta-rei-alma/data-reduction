#!/usr/bin/env python
"""
Standalone CASA script for parallel tclean + feather.

Runs inside monolithic CASA via mpicasa. Does NOT import from panta_rei.
All configuration comes from a job.json written by the pipeline.

Usage::

    mpicasa -n 8 casa --nologger --nogui -c tclean_feather.py --args job.json

The script reads the job spec, runs tclean (with parallel=True),
exports the tclean pbcor to FITS, imports and regrids the TP cube,
runs feather, exports the feathered product to FITS, and writes
a result.json for the pipeline to read.

Author: Generated for Dan Walker
Project: Panta Rei ALMA Large Program (2025.1.00383.L)
"""

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

try:
    from casatasks import tclean, feather, exportfits, importfits, imregrid, imtrans
    from casatools import image as iatool
except ImportError:
    print("ERROR: This script must be run inside CASA.")
    print("Usage: mpicasa -n N casa --nologger --nogui -c tclean_feather.py --args job.json")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("tclean_feather")


# Headline fields of tclean's return dict copied into result.json; the full
# (JSON-safe) dict goes to the cleaning record (tclean_summary.json).
SUMMARY_KEYS = ("stopcode", "stopDescription", "iterdone", "nmajordone")


def casa_version_string():
    """Version of the CASA actually executing this script."""
    try:
        import casatools
        return casatools.version_string()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def json_safe(obj):
    """Recursively convert tclean's return value into JSON-serialisable data.

    numpy scalars/arrays become Python numbers/lists; anything else that
    json cannot handle is stored as its repr rather than failing the job.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "tolist"):  # numpy array or scalar
        try:
            return json_safe(obj.tolist())
        except Exception:  # noqa: BLE001
            pass
    return repr(obj)


# Cleaning record: the single source of truth for who cleaned the products.
# Written straight after tclean returns (completed=True) or raises
# (completed=False), before any export/TP/feather step, so a later failure
# or resume can neither lose nor misattribute it.  A fresh tclean deletes
# the previous record first, so it is always bound to the current
# generation.  A prior result.json casa_version is never trusted (legacy
# stamps there may be exporter versions).
CLEANING_RECORD = "tclean_summary.json"


def write_cleaning_record(run_dir, ret, casa_version, completed):
    """Write {completed, casa_version, summary} for this tclean generation."""
    record = {
        "completed": completed,
        "casa_version": casa_version,
        "summary": json_safe(ret) if ret is not None else None,
    }
    path = run_dir / CLEANING_RECORD
    try:
        path.write_text(json.dumps(record, indent=2))
    except Exception as e:  # noqa: BLE001 - provenance must not fail the job
        log.warning("could not write %s: %s", path.name, e)


def apply_cleaning_record(run_dir, result):
    """Copy a *completed* cleaning record's version + headline into result.

    Returns False (result untouched) if there is no completed record.
    """
    path = run_dir / CLEANING_RECORD
    try:
        record = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - missing/corrupt → no provenance
        return False
    if not isinstance(record, dict) or record.get("completed") is not True:
        return False
    result["casa_version"] = record.get("casa_version") or "unknown"
    summary = record.get("summary")
    if isinstance(summary, dict):
        result["tclean_summary"] = {k: summary.get(k) for k in SUMMARY_KEYS}
    result["tclean_summary_json"] = str(path)
    return True


def run_tclean_recorded(tclean_task, params, run_dir, result, casa_version):
    """Run a fresh tclean generation and record its cleaning provenance."""
    path = run_dir / CLEANING_RECORD
    if path.exists():
        path.unlink()  # new generation: never inherit an old record
    result["casa_version"] = casa_version
    try:
        ret = tclean_task(**params)
    except Exception:
        write_cleaning_record(run_dir, None, casa_version, completed=False)
        raise
    write_cleaning_record(run_dir, ret, casa_version, completed=True)
    apply_cleaning_record(run_dir, result)


def ensure_axis_order(casa_image):
    """Transpose to [RA, DEC, Stokes, Freq] if needed."""
    ia = iatool()
    try:
        ia.open(casa_image)
        csys = ia.coordsys()
        axis_names = [csys.axiscoordinatetypes()[i] for i in range(csys.naxes())]
        ia.close()
        csys.done()
    except Exception:
        ia.close()
        raise

    if len(axis_names) >= 4 and axis_names[3] == "Spectral":
        return casa_image

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
    log.info("Transposing TP axes %s with order='%s'", axis_names, order_str)

    transposed = casa_image + ".transposed"
    imtrans(imagename=casa_image, outfile=transposed, order=order_str)
    shutil.rmtree(casa_image)
    shutil.move(transposed, casa_image)
    return casa_image


def main():
    ap = argparse.ArgumentParser(description="tclean + feather (parallel CASA)")
    ap.add_argument("job_json", help="Path to job.json")
    args = ap.parse_args()

    job_path = Path(args.job_json)
    if not job_path.exists():
        print(f"ERROR: job.json not found: {job_path}")
        sys.exit(1)

    spec = json.loads(job_path.read_text())
    run_dir = Path(spec["run_dir"])
    params = spec["params"]
    canonical = spec["canonical_paths"]
    unit = spec["unit"]

    # Aux QA products to also export to FITS alongside the science cubes.
    # Each entry maps a CASA image suffix (relative to ``imagename``) to
    # the canonical FITS suffix used in the published file name.  Best
    # effort: a missing or failing export does NOT fail the whole job.
    aux_products = spec.get("aux_products") or ["mask", "residual", "pb"]

    result = {
        "success": False,
        "feathered_fits": None,
        "tclean_fits": None,
        "aux_fits": {},
        "tclean_summary": None,
        # CASA that cleaned the products (from the cleaning record; unknown
        # on a resume without one); export_casa_version is this CASA.
        "casa_version": "unknown",
        "export_casa_version": casa_version_string(),
        "error_message": None,
    }
    result_path = run_dir / "result.json"

    try:
        # Ensure parallel=True for mpicasa execution
        params["parallel"] = True

        imagename = params["imagename"]
        pbcor_image = f"{imagename}.image.pbcor"

        # Skip tclean if products already exist (resume from feather)
        if Path(pbcor_image).exists():
            log.info("tclean products already exist, skipping to feather: %s", Path(pbcor_image).name)
            apply_cleaning_record(run_dir, result)
        else:
            log.info("Running tclean: %s", Path(imagename).name)
            run_tclean_recorded(
                tclean, params, run_dir, result, result["export_casa_version"],
            )

            if not Path(pbcor_image).exists():
                raise FileNotFoundError(f"tclean did not produce {pbcor_image}")

        # Export tclean pbcor to FITS (12m7m product)
        tclean_fits_name = Path(canonical["tclean_only"]).name
        local_tclean_fits = str(run_dir / tclean_fits_name)
        log.info("Exporting tclean FITS: %s", tclean_fits_name)
        Path(local_tclean_fits).parent.mkdir(parents=True, exist_ok=True)
        exportfits(imagename=pbcor_image, fitsimage=local_tclean_fits, overwrite=True)
        result["tclean_fits"] = local_tclean_fits

        # Prepare TP for feathering
        tp_fits = unit["sdimage"]
        tp_work = run_dir / "tp"
        tp_work.mkdir(parents=True, exist_ok=True)

        tp_basename = Path(tp_fits).stem + ".image"
        tp_casa = str(tp_work / tp_basename)
        if not Path(tp_casa).exists():
            log.info("Importing TP: %s", Path(tp_fits).name)
            importfits(fitsimage=tp_fits, imagename=tp_casa, overwrite=True)

        # Regrid TP to match tclean spatial/spectral grid
        tclean_image = f"{imagename}.image"
        tp_regridded = tp_casa + ".regridded"
        if Path(tp_regridded).exists():
            shutil.rmtree(tp_regridded)
        log.info("Regridding TP to match tclean: %s", Path(tclean_image).name)
        imregrid(imagename=tp_casa, output=tp_regridded, template=tclean_image, overwrite=True)

        # Transpose regridded TP to match tclean axis order
        # [RA, DEC, Stokes, Freq]. Done AFTER imregrid to avoid
        # corrupting beam sets. Safe with restoringbeam='common'.
        tp_regridded = ensure_axis_order(tp_regridded)

        # Feather
        feathered_image = f"{imagename}.image.pbcor.feather"
        if Path(feathered_image).exists():
            shutil.rmtree(feathered_image)
        log.info("Feathering: %s + %s", Path(pbcor_image).name, Path(tp_regridded).name)
        feather(imagename=feathered_image, highres=pbcor_image, lowres=tp_regridded)

        # Export feathered to FITS (12m7mTP product)
        feathered_fits_name = Path(canonical["feathered"]).name
        local_feathered_fits = str(run_dir / feathered_fits_name)
        log.info("Exporting feathered FITS: %s", feathered_fits_name)
        exportfits(imagename=feathered_image, fitsimage=local_feathered_fits, overwrite=True)
        result["feathered_fits"] = local_feathered_fits

        # Best-effort export of QA aux products (mask/residual/pb).  The
        # canonical paths come from the spec (so the runner controls
        # filenames); we just produce the local FITS in run_dir and
        # report each path under ``aux_fits[kind]``.  Failure of one
        # aux export must not fail the job.
        aux_canonical = spec.get("canonical_paths", {}).get("aux", {})
        for kind in aux_products:
            casa_image = f"{imagename}.{kind}"
            if not Path(casa_image).exists():
                log.warning("aux %s: CASA image not found at %s — skipping",
                            kind, casa_image)
                continue
            canonical_aux = aux_canonical.get(kind)
            if not canonical_aux:
                log.warning("aux %s: no canonical path in spec — skipping",
                            kind)
                continue
            local_aux_fits = str(run_dir / Path(canonical_aux).name)
            try:
                log.info("Exporting aux FITS: %s", Path(local_aux_fits).name)
                exportfits(
                    imagename=casa_image,
                    fitsimage=local_aux_fits,
                    overwrite=True,
                )
                result["aux_fits"][kind] = local_aux_fits
            except Exception as aux_e:
                log.warning("aux %s export failed: %s", kind, aux_e)

        result["success"] = True
        log.info("SUCCESS: %s", feathered_fits_name)

    except Exception as e:
        result["error_message"] = str(e)
        log.error("FAILED: %s", e)
        traceback.print_exc()

    result_path.write_text(json.dumps(result, indent=2))
    log.info("Wrote result: %s", result_path)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
