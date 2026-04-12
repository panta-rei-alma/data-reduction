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

    result = {
        "success": False,
        "feathered_fits": None,
        "tclean_fits": None,
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
        else:
            log.info("Running tclean: %s", Path(imagename).name)
            tclean(**params)

            if not Path(pbcor_image).exists():
                raise FileNotFoundError(f"tclean did not produce {pbcor_image}")

        # Export tclean pbcor to FITS (12m7m product)
        tclean_fits_name = Path(canonical["tclean_only"]).name
        local_tclean_fits = str(run_dir / tclean_fits_name)
        log.info("Exporting tclean FITS: %s", tclean_fits_name)
        Path(local_tclean_fits).parent.mkdir(parents=True, exist_ok=True)
        exportfits(imagename=pbcor_image, fitsimage=local_tclean_fits, overwrite=True)
        result["tclean_fits"] = local_tclean_fits

        # Ensure common beam for tclean
        params["restoringbeam"] = "common"

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
