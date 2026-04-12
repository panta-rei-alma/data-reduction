#!/usr/bin/env python
"""
Continuum subtraction remediation for Panta Rei MOUSs.

Produces *_targets_line.ms for MOUSs where ScriptForPI skipped the contsub
branch (because selfcal was unsuccessful or absent).  Mirrors exactly what
ScriptForPI's docontsub branch does:

    h_init() + hifa_restoredata()     → restore pipeline state
    copy cont.dat                     → provide continuum channel info
    hif_mstransform()                 → split to _targets.ms / _targets_line.ms
    hifa_flagtargets()                → flag target data
    hif_uvcontsub()                   → subtract continuum

Usage:
    casa --nologger --nogui --pipeline -c contsub_remediation.py \
         --member-dir /path/to/member.uid___A001_X3833_X64bc

    casa --nologger --nogui --pipeline -c contsub_remediation.py \
         --member-dir /path/to/member.uid___... --dry-run

Author: Generated for Dan Walker
Project: Panta Rei ALMA Large Program (2025.1.00383.L)
"""

import argparse
import ast
import glob
import logging
import os
import re
import shutil
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("contsub_remediation")


# =============================================================================
# Helpers
# =============================================================================

def find_base_ms_files(working_dir):
    """Return base calibrated MS dirs (not _targets*, .flagversions, .tbl).

    These are the uid___A002_*.ms directories that hif_mstransform will
    split into *_targets.ms and *_targets_line.ms.
    """
    all_ms = sorted(glob.glob(os.path.join(working_dir, "uid___A002_*.ms")))
    base = []
    for p in all_ms:
        name = os.path.basename(p)
        if "_targets" in name:
            continue
        if name.endswith(".ms.flagversions"):
            continue
        if name.endswith(".tbl"):
            continue
        if not os.path.isdir(p):
            continue
        base.append(p)
    return base


def check_per_eb_completeness(working_dir):
    """Check each base MS for its corresponding *_targets_line.ms.

    Returns (all_present, missing_list, existing_list).
    """
    base_ms = find_base_ms_files(working_dir)
    if not base_ms:
        return False, [], []

    missing = []
    existing = []
    for ms_path in base_ms:
        stem = ms_path  # e.g. .../uid___A002_X12fb842_X4e4e.ms
        expected = stem.replace(".ms", "_targets_line.ms")
        if os.path.isdir(expected):
            existing.append(expected)
        else:
            missing.append(expected)

    return len(missing) == 0, missing, existing


def parse_restoredata_args(member_dir):
    """Extract vis and session lists from the shipped piperestorescript.

    Parses the hifa_restoredata() call deterministically.  Fails with a
    clear error if the script is missing or unparseable.
    """
    script_dir = os.path.join(member_dir, "script")
    candidates = glob.glob(
        os.path.join(script_dir, "*casa_piperestorescript.py")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No *casa_piperestorescript.py found in {script_dir}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple piperestorescript files in {script_dir}: {candidates}"
        )

    restore_script = candidates[0]
    log.info("Parsing restoredata args from %s", restore_script)

    with open(restore_script) as f:
        content = f.read()

    # Extract vis=[...] and session=[...]
    vis_match = re.search(r"vis\s*=\s*(\[.*?\])", content)
    session_match = re.search(r"session\s*=\s*(\[.*?\])", content)

    if not vis_match:
        raise ValueError(
            f"Cannot parse vis=[...] from {restore_script}"
        )
    if not session_match:
        raise ValueError(
            f"Cannot parse session=[...] from {restore_script}"
        )

    vis = ast.literal_eval(vis_match.group(1))
    session = ast.literal_eval(session_match.group(1))

    log.info("  vis=%s", vis)
    log.info("  session=%s", session)
    return vis, session


def check_has_uvcontfit(member_dir):
    """Replicate ScriptForPI's has_uvcontfit detection.

    Returns True only if the pipescript contains hif_uvcontfit AND the
    task is available in the current CASA session.
    """
    script_dir = os.path.join(member_dir, "script")
    pipescripts = glob.glob(
        os.path.join(script_dir, "*cal*casa_pipescript.py")
    )
    if not pipescripts:
        return False

    with open(pipescripts[0]) as f:
        content = f.read()

    if "hif_uvcontfit" not in content:
        return False

    # Check if the task is available in the CASA session
    return "hif_uvcontfit" in dir()


def validate_output(working_dir):
    """Post-creation data-level validation of _targets_line.ms files.

    Opens each with msmd to verify it has target fields and science SPWs.
    Returns (ok, messages).
    """
    targets_line = sorted(glob.glob(
        os.path.join(working_dir, "*_targets_line.ms")
    ))
    if not targets_line:
        return False, ["No _targets_line.ms files found after contsub"]

    messages = []
    ok = True
    for ms_path in targets_line:
        name = os.path.basename(ms_path)
        try:
            msmd.open(ms_path)
            fields = msmd.fieldnames()
            target_spws = msmd.spwsforintent("OBSERVE_TARGET*")
            msmd.close()
            messages.append(
                f"  {name}: fields={fields}, target_spws={list(target_spws)}"
            )
            if len(fields) == 0:
                messages.append(f"  WARNING: {name} has no fields")
                ok = False
            if len(target_spws) == 0:
                messages.append(f"  WARNING: {name} has no target SPWs")
                ok = False
        except Exception as e:
            messages.append(f"  ERROR validating {name}: {e}")
            ok = False

    return ok, messages


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run continuum subtraction for a single MOUS"
    )
    parser.add_argument(
        "--member-dir", required=True,
        help="Path to member.uid___A001_... directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without executing",
    )

    # CASA passes extra args we need to ignore
    args, _unknown = parser.parse_known_args()
    member_dir = os.path.abspath(args.member_dir)
    dry_run = args.dry_run

    log.info("=" * 70)
    log.info("Contsub remediation for %s", os.path.basename(member_dir))
    log.info("=" * 70)

    working_dir = os.path.join(member_dir, "calibrated", "working")
    if not os.path.isdir(working_dir):
        log.error("No calibrated/working/ directory: %s", working_dir)
        sys.exit(1)

    # --- Per-EB idempotency check ---
    all_present, missing, existing = check_per_eb_completeness(working_dir)
    base_ms = find_base_ms_files(working_dir)
    log.info("Base MS files: %d", len(base_ms))
    log.info("Existing _targets_line.ms: %d", len(existing))
    log.info("Missing _targets_line.ms: %d", len(missing))

    if all_present:
        log.info("All _targets_line.ms already exist. Nothing to do.")
        sys.exit(0)

    if not base_ms:
        log.error("No base calibrated MS files in %s", working_dir)
        sys.exit(1)

    # --- Find and copy cont.dat ---
    contfiles = glob.glob(os.path.join(member_dir, "calibration", "*cont.dat"))
    if len(contfiles) == 0:
        log.error("No *cont.dat found in %s/calibration/", member_dir)
        sys.exit(1)
    if len(contfiles) > 1:
        log.error("Multiple *cont.dat found: %s", contfiles)
        sys.exit(1)

    cont_dat_src = contfiles[0]
    cont_dat_dst = os.path.join(working_dir, "cont.dat")
    log.info("cont.dat source: %s", cont_dat_src)

    # --- Parse vis/session from piperestorescript ---
    try:
        vis, session = parse_restoredata_args(member_dir)
    except (FileNotFoundError, ValueError) as e:
        log.error("Failed to parse restoredata args: %s", e)
        sys.exit(1)

    # --- Check has_uvcontfit ---
    has_uvcontfit = check_has_uvcontfit(member_dir)
    if has_uvcontfit:
        log.info("Pipeline uses hif_uvcontfit — will use 4-stage sequence")
    else:
        log.info("Pipeline uses hif_findcont — will use 3-stage sequence")

    # --- Find saved context file for h_resume ---
    context_files = sorted(glob.glob(
        os.path.join(working_dir, "pipeline-*.context")
    ))

    # --- Dry run ---
    if dry_run:
        log.info("DRY RUN — would execute:")
        log.info("  1. os.chdir(%s)", working_dir)
        log.info("  2. cp %s -> %s", cont_dat_src, cont_dat_dst)
        if context_files:
            log.info("  3. h_resume(%s)", context_files[0])
        else:
            log.info("  3. h_init() + hifa_restoredata(vis=%s, session=%s)",
                     vis, session)
        if has_uvcontfit:
            log.info("  4. hif_mstransform(pipelinemode='automatic')")
            log.info("  5. hifa_flagtargets(pipelinemode='automatic')")
            log.info("  6. hif_uvcontfit(pipelinemode='automatic')")
            log.info("  7. hif_uvcontsub(pipelinemode='automatic')")
        else:
            log.info("  4. hif_mstransform()")
            log.info("  5. hifa_flagtargets()")
            log.info("  6. hif_uvcontsub()")
        log.info("  N. h_save()")
        log.info("DRY RUN complete. Exiting.")
        sys.exit(0)

    # --- Execute ---
    os.chdir(working_dir)
    log.info("Working directory: %s", os.getcwd())

    # Copy cont.dat into working dir
    shutil.copy2(cont_dat_src, cont_dat_dst)
    log.info("Copied cont.dat to working dir")

    # Restore pipeline state — try h_resume first (fast), fall back to
    # h_init + hifa_restoredata (slow but always works).
    resumed = False
    if context_files:
        context_path = os.path.basename(context_files[0])
        log.info("Trying h_resume(%s)...", context_path)
        try:
            h_resume(context_path)
            resumed = True
            log.info("Pipeline context resumed successfully (fast path)")
        except Exception as e:
            log.warning("h_resume failed: %s — falling back to hifa_restoredata", e)

    if not resumed:
        log.info("Initializing pipeline context with h_init + hifa_restoredata...")
        h_init()
        hifa_restoredata(vis=vis, session=session, ocorr_mode='ca')
        log.info("Pipeline state restored (slow path)")

    # Run contsub stages
    log.info("Running hif_mstransform...")
    if has_uvcontfit:
        hif_mstransform(pipelinemode="automatic")
        log.info("Running hifa_flagtargets...")
        hifa_flagtargets(pipelinemode="automatic")
        log.info("Running hif_uvcontfit...")
        hif_uvcontfit(pipelinemode="automatic")
        log.info("Running hif_uvcontsub...")
        hif_uvcontsub(pipelinemode="automatic")
    else:
        hif_mstransform()
        log.info("Running hifa_flagtargets...")
        hifa_flagtargets()
        log.info("Running hif_uvcontsub...")
        hif_uvcontsub()

    # Save context
    log.info("Saving pipeline context...")
    h_save()

    # --- Post-check ---
    all_present, missing, existing = check_per_eb_completeness(working_dir)
    log.info("Post-check: %d _targets_line.ms created", len(existing))

    if missing:
        log.error("FAILED: Missing _targets_line.ms after contsub:")
        for m in missing:
            log.error("  %s", m)
        sys.exit(1)

    # --- Data-level validation ---
    log.info("Validating output...")
    valid, messages = validate_output(working_dir)
    for msg in messages:
        log.info(msg)

    if not valid:
        log.error("FAILED: Data validation failed")
        sys.exit(1)

    log.info("SUCCESS: All %d _targets_line.ms created and validated",
             len(existing))
    sys.exit(0)


if __name__ == "__main__":
    main()
