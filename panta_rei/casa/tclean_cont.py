#!/usr/bin/env python
"""
Standalone CASA script for parallel 12m+7m continuum tclean (mtmfs).

Runs inside monolithic CASA via mpicasa. Does NOT import from panta_rei.
All configuration comes from a job.json written by the pipeline.

Usage::

    mpicasa -n 8 casa --nologger --nogui -c tclean_cont.py --args job.json

Unlike ``tclean_feather.py`` there is **no single-dish / feather step** —
ALMA TP is line-only, so the joint 12m+7m ``tclean`` image IS the continuum
science product. Imaging is always multi-term MFS (``deconvolver='mtmfs'``,
``nterms=2``), so the products are term images: the script exports
``.image.tt0.pbcor`` (science), ``.image.tt1.pbcor``, and ``.alpha`` (when
present), plus QA aux (mask / residual.tt0 / pb.tt0). It writes a result.json
for the pipeline to read.

Author: Generated for Dan Walker
Project: Panta Rei ALMA Large Program (2025.1.00383.L)
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

try:
    from casatasks import tclean, exportfits
except ImportError:
    print("ERROR: This script must be run inside CASA.")
    print("Usage: mpicasa -n N casa --nologger --nogui -c tclean_cont.py --args job.json")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("tclean_cont")


# CASA image suffix (relative to imagename) for each aux/diagnostic product
# under multi-term MFS.  ``mask`` is not term-split; the others are (we keep
# the tt0 term for residual/pb/psf).  ``tt0_image``/``tt1_image`` are the flat,
# non-pb-corrected term images.
AUX_SUFFIX = {
    "tt0_image": ".image.tt0",
    "tt1_image": ".image.tt1",
    "mask": ".mask",
    "residual": ".residual.tt0",
    "pb": ".pb.tt0",
    "psf": ".psf.tt0",
}


def _export(casa_image, canonical_path, run_dir, label):
    """Export a CASA image to a local FITS in run_dir; return the path or None.

    Best-effort: a missing image or a failed export is logged and returns
    None rather than raising (the caller decides whether it is fatal).
    """
    if not Path(casa_image).exists():
        log.warning("%s: CASA image not found at %s — skipping", label, casa_image)
        return None
    local = str(run_dir / Path(canonical_path).name)
    try:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        log.info("Exporting %s FITS: %s", label, Path(local).name)
        exportfits(imagename=casa_image, fitsimage=local, overwrite=True)
        return local
    except Exception as e:  # noqa: BLE001 - aux export must not crash the job
        log.warning("%s export failed: %s", label, e)
        return None


def main():
    ap = argparse.ArgumentParser(description="12m+7m continuum tclean (parallel CASA)")
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
    aux_products = spec.get("aux_products") or ["mask", "residual", "pb"]

    result = {
        "success": False,
        "tt0_fits": None,
        "tt1_fits": None,
        "alpha_fits": None,
        "aux_fits": {},
        "error_message": None,
    }
    result_path = run_dir / "result.json"

    try:
        # Ensure parallel=True for mpicasa execution
        params["parallel"] = True

        imagename = params["imagename"]
        # Multi-term MFS science product is the tt0 pbcor image.
        tt0_pbcor = f"{imagename}.image.tt0.pbcor"

        # Resume: skip tclean if the science product already exists.
        if Path(tt0_pbcor).exists():
            log.info("tclean tt0 product already exists, skipping tclean: %s",
                     Path(tt0_pbcor).name)
        else:
            log.info("Running continuum tclean (mtmfs): %s", Path(imagename).name)
            tclean(**params)
            if not Path(tt0_pbcor).exists():
                raise FileNotFoundError(f"tclean did not produce {tt0_pbcor}")

        # Science tt0 pbcor (12m7m continuum) — fatal if it cannot export.
        local_tt0 = _export(tt0_pbcor, canonical["tt0"], run_dir, "tt0")
        if local_tt0 is None:
            raise RuntimeError(f"failed to export science tt0 FITS from {tt0_pbcor}")
        result["tt0_fits"] = local_tt0

        # tt1 pbcor — present whenever nterms>=2.
        if canonical.get("tt1"):
            result["tt1_fits"] = _export(
                f"{imagename}.image.tt1.pbcor", canonical["tt1"], run_dir, "tt1"
            )

        # alpha (spectral index) — only present for nterms>1.
        if canonical.get("alpha"):
            result["alpha_fits"] = _export(
                f"{imagename}.alpha", canonical["alpha"], run_dir, "alpha"
            )

        # Best-effort QA aux products (mask / residual.tt0 / pb.tt0).
        aux_canonical = canonical.get("aux", {})
        for kind in aux_products:
            suffix = AUX_SUFFIX.get(kind)
            canonical_aux = aux_canonical.get(kind)
            if not suffix or not canonical_aux:
                continue
            local_aux = _export(
                f"{imagename}{suffix}", canonical_aux, run_dir, f"aux:{kind}"
            )
            if local_aux is not None:
                result["aux_fits"][kind] = local_aux

        result["success"] = True
        log.info("SUCCESS: %s", Path(canonical["tt0"]).name)

    except Exception as e:  # noqa: BLE001 - report via result.json
        result["error_message"] = str(e)
        log.error("FAILED: %s", e)
        traceback.print_exc()

    result_path.write_text(json.dumps(result, indent=2))
    log.info("Wrote result: %s", result_path)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
