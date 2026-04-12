#!/usr/bin/env python
"""
Feather script for combining 7M (SM) and TP ALMA data cubes for the Panta Rei project.

This script reads a CSV file containing observation IDs and uses CASA's feather task
to combine the interferometric (7M/SM) and single-dish (TP) data cubes.

SM and TP cubes are matched by frequency rather than by SPW ID, making the script
robust to variations in SPW numbering across different observations.

Usage:
    # Dry run (show what would be done)
    casa -c feather_sm_tp.py --base-dir /path/to/data --output-dir /path/to/output --dry-run

    # Actually run feathering
    casa -c feather_sm_tp.py --base-dir /path/to/data --output-dir /path/to/output

    # Limit to first N combinations (for testing)
    casa -c feather_sm_tp.py --base-dir /path/to/data --output-dir /path/to/output --limit 2

Author: Generated for Dan Walker
Project: Panta Rei ALMA Large Program (2025.1.00383.L)
"""

import argparse
import csv
import glob
import logging
import os
import re
import shutil
import sys
from collections import defaultdict

# CASA imports - this script must be run within CASA
try:
    from casatasks import feather, importfits, exportfits
    from astropy.io import fits
    import numpy as np
except ImportError:
    print("ERROR: This script must be run within CASA (casatasks) and requires astropy.")
    print("Run with: casa -c feather_sm_tp.py --base-dir ... --output-dir ...")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("feather")

# =============================================================================
# Constants
# =============================================================================

# Expected number of SPW matches per line group (for validation)
EXPECTED_SPW_COUNTS = {
    'N2H+': 4,
    'HCO+': 6,
}

# Default frequency matching tolerance in GHz
DEFAULT_FREQ_TOLERANCE_GHZ = 0.05


# =============================================================================
# Helper Functions
# =============================================================================

def build_product_dir_path(base_dir, sgous_id, gous_id, mous_id):
    """
    Build the path to the product directory for a given observation.

    Parameters
    ----------
    base_dir : str
        Base directory containing extracted ALMA data
    sgous_id : str
        Science goal UID suffix (e.g., 'X3833_X64b8')
    gous_id : str
        Group UID suffix (e.g., 'X3833_X64b9')
    mous_id : str
        Member UID suffix (e.g., 'X3833_X64bc')

    Returns
    -------
    str
        Full path to the product directory, or None if sgous_id is empty
    """
    if not sgous_id:
        return None

    science_goal_dir = f"science_goal.uid___A001_{sgous_id}"
    group_dir = f"group.uid___A001_{gous_id}"
    member_dir = f"member.uid___A001_{mous_id}"

    return os.path.join(base_dir, science_goal_dir, group_dir, member_dir, 'product')


def sanitize_source_name_for_glob(source_name):
    """
    Convert source name to match ALMA pipeline filename conventions.
    
    The ALMA pipeline replaces '+' with 'p' in filenames.
    
    Parameters
    ----------
    source_name : str
        Original source name (e.g., 'AG342.0584+0.4213')
    
    Returns
    -------
    str
        Sanitized source name for glob matching (e.g., 'AG342.0584p0.4213')
    """
    return source_name.replace('+', 'p')


def sanitize_source_name(source_name):
    """
    Convert source name for use in filenames.
    
    Replaces '+' with 'p' and '-' with 'm' (ALMA archive requirement).
    
    Parameters
    ----------
    source_name : str
        Original source name (e.g., 'AG221.9599-1.9932')
    
    Returns
    -------
    str
        Sanitized source name (e.g., 'AG221.9599m1.9932')
    """
    return source_name.replace('+', 'p').replace('-', 'm')


def get_freq_bounds_from_fits(fits_file):
    """
    Extract the frequency bounds from a FITS file header.
    
    Parameters
    ----------
    fits_file : str
        Path to the FITS file
    
    Returns
    -------
    tuple
        (freq_min_hz, freq_max_hz, spw_id) where spw_id is extracted from filename
    """
    with fits.open(fits_file) as hdul:
        header = hdul[0].header
        
        # Find the frequency axis (usually CTYPE3 or CTYPE4)
        freq_axis = None
        for i in range(1, 5):
            ctype = header.get(f'CTYPE{i}', '')
            if 'FREQ' in ctype.upper():
                freq_axis = i
                break
        
        if freq_axis is None:
            raise ValueError(f"Could not find frequency axis in {fits_file}")
        
        # Get frequency axis parameters
        crval = header[f'CRVAL{freq_axis}']  # Reference value (Hz)
        cdelt = header[f'CDELT{freq_axis}']  # Channel width (Hz)
        crpix = header[f'CRPIX{freq_axis}']  # Reference pixel
        naxis = header[f'NAXIS{freq_axis}']  # Number of channels
        
        # Calculate frequency at first and last channel
        freq_first = crval + (1 - crpix) * cdelt
        freq_last = crval + (naxis - crpix) * cdelt
        
        freq_min = min(freq_first, freq_last)
        freq_max = max(freq_first, freq_last)
    
    # Extract SPW ID from filename for logging purposes
    spw_match = re.search(r'spw(\d+)', os.path.basename(fits_file))
    spw_id = int(spw_match.group(1)) if spw_match else None
    
    return (freq_min, freq_max, spw_id)


def get_freq_range_string(freq_min_hz, freq_max_hz):
    """
    Format frequency bounds as a string for filenames.
    
    Parameters
    ----------
    freq_min_hz : float
        Minimum frequency in Hz
    freq_max_hz : float
        Maximum frequency in Hz
    
    Returns
    -------
    str
        Frequency range string in the format "XX.X-YY.Y"
    """
    freq_min_ghz = freq_min_hz / 1e9
    freq_max_ghz = freq_max_hz / 1e9
    return f"{freq_min_ghz:.1f}-{freq_max_ghz:.1f}"


def find_all_cubes(product_dir, source_name, cube_type):
    """
    Find all cube files of a given type, returning frequency info for each.
    
    Parameters
    ----------
    product_dir : str
        Path to the product directory
    source_name : str
        Source name to search for
    cube_type : str
        Type of cube: 'SM' for interferometric or 'TP' for single-dish
    
    Returns
    -------
    list of dict
        Each dict contains: {'file': path, 'freq_min': Hz, 'freq_max': Hz, 'spw': int}
    """
    if not product_dir or not os.path.exists(product_dir):
        return []
    
    source_glob = sanitize_source_name_for_glob(source_name)
    files = []
    
    if cube_type == 'SM':
        # For SM: prefer selfcal, fall back to regcal
        # First, find all selfcal files
        selfcal_pattern = os.path.join(
            product_dir,
            f"*{source_glob}*spw*.cube.I.selfcal.pbcor.fits"
        )
        selfcal_files = glob.glob(selfcal_pattern)
        
        # Extract SPW IDs from selfcal files
        selfcal_spws = set()
        for f in selfcal_files:
            spw_match = re.search(r'spw(\d+)', os.path.basename(f))
            if spw_match:
                selfcal_spws.add(int(spw_match.group(1)))
        
        # Find regcal files for SPWs not covered by selfcal
        regcal_pattern = os.path.join(
            product_dir,
            f"*{source_glob}*spw*.cube.I.pbcor.fits"
        )
        regcal_files = glob.glob(regcal_pattern)
        
        # Filter regcal to exclude selfcal files and SPWs already covered
        for f in regcal_files:
            if 'selfcal' in f:
                continue
            spw_match = re.search(r'spw(\d+)', os.path.basename(f))
            if spw_match and int(spw_match.group(1)) not in selfcal_spws:
                files.append(f)
        
        # Add all selfcal files
        files.extend(selfcal_files)
        
    elif cube_type == 'TP':
        pattern = os.path.join(
            product_dir,
            f"*{source_glob}*spw*.cube.I.sd.fits"
        )
        files = glob.glob(pattern)
    
    else:
        raise ValueError(f"Unknown cube_type: {cube_type}. Expected 'SM' or 'TP'")
    
    # Extract frequency info from each file
    cubes = []
    for f in files:
        try:
            freq_min, freq_max, spw_id = get_freq_bounds_from_fits(f)
            cubes.append({
                'file': f,
                'freq_min': freq_min,
                'freq_max': freq_max,
                'spw': spw_id,
            })
        except Exception as e:
            log.warning(f"    Could not read frequency from {os.path.basename(f)}: {e}")
    
    # Sort by frequency for consistent ordering
    cubes.sort(key=lambda x: x['freq_min'])
    
    return cubes


def match_cubes_by_frequency(sm_cubes, tp_cubes, tolerance_ghz=DEFAULT_FREQ_TOLERANCE_GHZ):
    """
    Match SM and TP cubes by overlapping frequency range.
    
    Uses centre frequency matching with a tolerance to handle small
    differences in exact frequency bounds between SM and TP data.
    
    Parameters
    ----------
    sm_cubes : list of dict
        SM cube info from find_all_cubes()
    tp_cubes : list of dict
        TP cube info from find_all_cubes()
    tolerance_ghz : float
        Tolerance in GHz for considering frequencies as matching
    
    Returns
    -------
    list of tuple
        List of (sm_cube_dict, tp_cube_dict) pairs, sorted by frequency
    """
    tolerance_hz = tolerance_ghz * 1e9
    matched = []
    used_tp = set()
    
    for sm in sm_cubes:
        sm_center = (sm['freq_min'] + sm['freq_max']) / 2
        sm_bw = sm['freq_max'] - sm['freq_min']
        
        best_tp = None
        best_diff = float('inf')
        best_idx = None
        
        for i, tp in enumerate(tp_cubes):
            if i in used_tp:
                continue
            
            tp_center = (tp['freq_min'] + tp['freq_max']) / 2
            diff = abs(sm_center - tp_center)
            
            if diff < best_diff and diff < tolerance_hz:
                best_diff = diff
                best_tp = tp
                best_idx = i
        
        if best_tp is not None:
            # Check for bandwidth mismatch and warn if significant
            tp_bw = best_tp['freq_max'] - best_tp['freq_min']
            if abs(sm_bw - tp_bw) / sm_bw > 0.5:  # >50% difference
                log.warning(f"    Bandwidth mismatch for SPW {sm['spw']}/{best_tp['spw']} - "
                            f"SM: {sm_bw/1e9:.3f} GHz, TP: {tp_bw/1e9:.3f} GHz")
            
            matched.append((sm, best_tp))
            used_tp.add(best_idx)
    
    # Sort by frequency for consistent output ordering
    matched.sort(key=lambda x: x[0]['freq_min'])
    
    return matched


def read_csv_and_group_observations(csv_file):
    """
    Read the CSV file and group SM and TP observations by source and line group.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file
    
    Returns
    -------
    dict
        Nested dictionary: {(source_name, line_group): {'SM': row_data, 'TP': row_data}}
    """
    observations = defaultdict(dict)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_name = row['source_name']
            array = row['array']
            line_group = row['Line group']
            
            key = (source_name, line_group)
            observations[key][array] = {
                'sgous_id': row['sgous_id'],
                'gous_id': row['gous_id'],
                'mous_ids': row['mous_ids'],
                'sb_name': row['sb_name'],
            }
    
    return observations


def import_fits_to_casa_image(fits_file, dry_run=True):
    """
    Import a FITS file to CASA image format.
    
    The output image is placed in the same directory as the FITS file,
    with the extension changed from .fits to .image
    
    Parameters
    ----------
    fits_file : str
        Path to the FITS file
    dry_run : bool
        If True, only print what would be done
    
    Returns
    -------
    str
        Path to the CASA image, or None if failed
    """
    # Generate output image path
    casa_image = fits_file.replace('.fits', '.image')
    
    # Check if already exists
    if os.path.exists(casa_image):
        log.debug(f"      CASA image already exists: {os.path.basename(casa_image)}")
        return casa_image

    if dry_run:
        log.info(f"      [DRY-RUN] Would import: {os.path.basename(fits_file)} -> {os.path.basename(casa_image)}")
        return casa_image

    try:
        log.info(f"      Importing: {os.path.basename(fits_file)} -> {os.path.basename(casa_image)}")
        importfits(
            fitsimage=fits_file,
            imagename=casa_image,
            overwrite=True,
        )
        return casa_image
    except Exception as e:
        log.error(f"      FAILED to import {fits_file}: {e}")
        return None


def run_feather(sm_cube, tp_cube, output_file, dry_run=True):
    """
    Run CASA feather task to combine SM and TP cubes.
    
    This function first imports the FITS files to CASA image format using
    importfits, then runs feather on the CASA images, and finally exports
    the result back to FITS format.
    
    Parameters
    ----------
    sm_cube : str
        Path to the SM (7M) FITS cube (high-resolution)
    tp_cube : str
        Path to the TP FITS cube (low-resolution)
    output_file : str
        Path for the output feathered FITS cube
    dry_run : bool
        If True, only print what would be done without actually feathering
    
    Returns
    -------
    bool
        True if successful (or would be successful in dry-run), False otherwise
    """
    try:
        log.info("  Feathering:")
        log.info(f"    High-res (SM): {sm_cube}")
        log.info(f"    Low-res (TP):  {tp_cube}")
        log.info(f"    Output:        {output_file}")

        # Step 1: Import FITS files to CASA image format
        log.info("    Step 1: Import FITS to CASA images")
        sm_image = import_fits_to_casa_image(sm_cube, dry_run=dry_run)
        tp_image = import_fits_to_casa_image(tp_cube, dry_run=dry_run)

        if sm_image is None or tp_image is None:
            log.error("    FAILED: Could not import FITS files")
            return False

        # Step 2: Run feather
        # Output as CASA image first, then export to FITS
        output_image = output_file.replace('.fits', '.image')

        log.info("    Step 2: Feather")
        if dry_run:
            log.info(f"      [DRY-RUN] Would feather: {os.path.basename(sm_image)} + {os.path.basename(tp_image)}")
            log.info(f"      [DRY-RUN] Would create: {os.path.basename(output_image)}")
        else:
            # Remove output image if it exists (feather doesn't overwrite)
            if os.path.exists(output_image):
                shutil.rmtree(output_image)

            # CASA feather: highres is the interferometer, lowres is single-dish
            feather(
                imagename=output_image,
                highres=sm_image,
                lowres=tp_image,
            )
            log.info(f"      Created: {os.path.basename(output_image)}")

        # Step 3: Export to FITS
        log.info("    Step 3: Export to FITS")
        if dry_run:
            log.info(f"      [DRY-RUN] Would export: {os.path.basename(output_image)} -> {os.path.basename(output_file)}")
        else:
            exportfits(
                imagename=output_image,
                fitsimage=output_file,
                overwrite=True,
            )
            log.info(f"      Exported: {os.path.basename(output_file)}")

            # Clean up the intermediate CASA image in output directory
            if os.path.exists(output_image):
                shutil.rmtree(output_image)
                log.info(f"      Cleaned up: {os.path.basename(output_image)}")

        log.info("    SUCCESS")
        return True

    except Exception as e:
        log.error(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main Processing
# =============================================================================

def run_feathering(base_dir, output_dir, csv_file, dry_run=True, limit=None, freq_tolerance=DEFAULT_FREQ_TOLERANCE_GHZ):
    """
    Main processing function for feathering SM and TP cubes.

    Parameters
    ----------
    base_dir : str
        Base directory containing extracted ALMA data (with science_goal.* subdirs)
    output_dir : str
        Output directory for feathered cubes
    csv_file : str
        Path to targets_by_array.csv
    dry_run : bool
        If True, only show what would be done
    limit : int or None
        Maximum number of source/line combinations to process
    freq_tolerance : float
        Frequency matching tolerance in GHz
    """
    log.info("=" * 70)
    log.info("Panta Rei Feathering Script")
    log.info("=" * 70)
    log.info(f"Base directory: {base_dir}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"CSV file: {csv_file}")
    log.info("-" * 70)
    log.info(f"DRY-RUN MODE: {dry_run}")
    log.info(f"LIMIT: {limit if limit is not None else 'None (process all)'}")
    log.info(f"Frequency matching tolerance: {freq_tolerance} GHz")
    if dry_run:
        log.info("*** DRY-RUN: No files will be created or modified ***")
    log.info("=" * 70)

    # Check paths exist
    if not os.path.exists(base_dir):
        log.error(f"Base directory does not exist: {base_dir}")
        return 1

    if not os.path.exists(csv_file):
        log.error(f"CSV file does not exist: {csv_file}")
        return 1

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        if dry_run:
            log.info(f"[DRY-RUN] Would create output directory: {output_dir}")
        else:
            log.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

    # Read and group observations
    log.info("Reading CSV file and grouping observations...")
    observations = read_csv_and_group_observations(csv_file)
    log.info(f"Found {len(observations)} source/line-group combinations")

    if limit is not None:
        log.info(f"Processing limited to first {limit} combination(s)")

    # Statistics
    stats = {
        'total_combinations': len(observations),
        'processed': 0,
        'feathered': 0,
        'skipped_missing_data': 0,
        'skipped_no_sgous': 0,
        'skipped_existing': 0,
        'skipped_no_match': 0,
        'failed': 0,
        'spw_count_warnings': 0,
    }

    # Counter for limit
    processed_count = 0

    # Process each source/line-group combination
    for (source_name, line_group), arrays in observations.items():
        # Check limit
        if limit is not None and processed_count >= limit:
            log.info(f"*** Limit of {limit} reached, stopping ***")
            break

        log.info("=" * 70)
        log.info(f"Processing: {source_name} ({line_group})")

        # Check we have both SM and TP data
        if 'SM' not in arrays or 'TP' not in arrays:
            log.info(f"  SKIPPED: Missing {'SM' if 'SM' not in arrays else 'TP'} data")
            stats['skipped_missing_data'] += 1
            continue

        sm_info = arrays['SM']
        tp_info = arrays['TP']

        # Check for empty sgous_id (some rows have this)
        if not sm_info['sgous_id'] or not tp_info['sgous_id']:
            log.info(f"  SKIPPED: Missing sgous_id (SM: '{sm_info['sgous_id']}', TP: '{tp_info['sgous_id']}')")
            stats['skipped_no_sgous'] += 1
            continue

        # Build product directory paths
        sm_product_dir = build_product_dir_path(
            base_dir, sm_info['sgous_id'], sm_info['gous_id'], sm_info['mous_ids']
        )
        tp_product_dir = build_product_dir_path(
            base_dir, tp_info['sgous_id'], tp_info['gous_id'], tp_info['mous_ids']
        )

        log.info(f"  SM product dir: {sm_product_dir}")
        log.info(f"  TP product dir: {tp_product_dir}")

        # Check directories exist
        if not os.path.exists(sm_product_dir):
            log.info("  SKIPPED: SM product directory does not exist")
            stats['skipped_missing_data'] += 1
            continue

        if not os.path.exists(tp_product_dir):
            log.info("  SKIPPED: TP product directory does not exist")
            stats['skipped_missing_data'] += 1
            continue

        # Find all cube files (frequency-based discovery)
        sm_cubes = find_all_cubes(sm_product_dir, source_name, 'SM')
        tp_cubes = find_all_cubes(tp_product_dir, source_name, 'TP')

        log.info(f"  Found {len(sm_cubes)} SM cubes and {len(tp_cubes)} TP cubes")

        if not sm_cubes:
            log.info("  SKIPPED: No SM cubes found")
            stats['skipped_missing_data'] += 1
            continue

        if not tp_cubes:
            log.info("  SKIPPED: No TP cubes found")
            stats['skipped_missing_data'] += 1
            continue

        # Match cubes by frequency
        matched_pairs = match_cubes_by_frequency(sm_cubes, tp_cubes, tolerance_ghz=freq_tolerance)
        log.info(f"  Matched {len(matched_pairs)} SM/TP pairs by frequency")

        if not matched_pairs:
            log.info("  SKIPPED: No frequency matches found between SM and TP cubes")
            stats['skipped_no_match'] += 1
            continue

        # Validate expected SPW count for this line group
        expected_count = EXPECTED_SPW_COUNTS.get(line_group)
        if expected_count is not None and len(matched_pairs) != expected_count:
            log.warning(f"  Expected {expected_count} SPW matches for {line_group}, but found {len(matched_pairs)}")
            stats['spw_count_warnings'] += 1

        stats['processed'] += 1
        processed_count += 1

        # Create output subdirectory
        # Use the SM gous_id for the output directory name
        gous_id = sm_info['gous_id']
        output_subdir = os.path.join(output_dir, f"group.uid___A001_{gous_id}.lp_nperetto")

        if not os.path.exists(output_subdir):
            if dry_run:
                log.info(f"  [DRY-RUN] Would create output directory: {output_subdir}")
            else:
                log.info(f"  Creating output directory: {output_subdir}")
                os.makedirs(output_subdir)

        # Feather each matched pair
        for sm_cube_info, tp_cube_info in matched_pairs:
            sm_cube = sm_cube_info['file']
            tp_cube = tp_cube_info['file']
            sm_spw = sm_cube_info['spw']
            tp_spw = tp_cube_info['spw']

            # Get frequency range string
            freq_range = get_freq_range_string(sm_cube_info['freq_min'], sm_cube_info['freq_max'])

            # Build output filename
            sanitized_source = sanitize_source_name(source_name)
            output_filename = (
                f"group.uid___A001_{gous_id}.lp_nperetto."
                f"{sanitized_source}.7mTP.{freq_range}GHz.cube.pbcor.fits"
            )
            output_file = os.path.join(output_subdir, output_filename)

            # Skip if output already exists
            if os.path.exists(output_file):
                log.info(f"  SPW {sm_spw}/{tp_spw} ({freq_range}GHz): Output already exists, skipping")
                stats['skipped_existing'] += 1
                continue

            # Run feather
            log.info(f"  SPW {sm_spw} (SM) + SPW {tp_spw} (TP) -> {freq_range}GHz")
            success = run_feather(sm_cube, tp_cube, output_file, dry_run=dry_run)

            if success:
                stats['feathered'] += 1
            else:
                stats['failed'] += 1

    # Print summary
    log.info("=" * 70)
    log.info("SUMMARY")
    if dry_run:
        log.info("*** DRY-RUN MODE - No files were created ***")
    log.info("=" * 70)
    log.info(f"Total source/line combinations: {stats['total_combinations']}")
    if limit is not None:
        log.info(f"Limit applied:                  {limit}")
    log.info(f"Successfully processed:         {stats['processed']}")
    log.info(f"Cubes {'would be ' if dry_run else ''}feathered:        {stats['feathered']}")
    log.info(f"Skipped (missing data):         {stats['skipped_missing_data']}")
    log.info(f"Skipped (no sgous_id):          {stats['skipped_no_sgous']}")
    log.info(f"Skipped (no freq match):        {stats['skipped_no_match']}")
    log.info(f"Skipped (already exists):       {stats['skipped_existing']}")
    log.info(f"SPW count warnings:             {stats['spw_count_warnings']}")
    log.info(f"Failed:                         {stats['failed']}")
    log.info("=" * 70)
    if dry_run:
        log.info("To run for real, remove the --dry-run flag.")

    return 0 if stats['failed'] == 0 else 1


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Feather SM (7M) and TP ALMA data cubes for the Panta Rei project."
    )
    ap.add_argument(
        "--base-dir", required=True,
        help="Base directory containing extracted ALMA data (with science_goal.* subdirs)"
    )
    ap.add_argument(
        "--output-dir", required=True,
        help="Output directory for feathered cubes"
    )
    ap.add_argument(
        "--csv",
        help="Path to targets_by_array.csv (default: <base-dir>/../targets_by_array.csv)"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without actually feathering"
    )
    ap.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Maximum number of source/line-group combinations to process"
    )
    ap.add_argument(
        "--freq-tolerance", type=float, default=DEFAULT_FREQ_TOLERANCE_GHZ,
        help=f"Frequency matching tolerance in GHz (default: {DEFAULT_FREQ_TOLERANCE_GHZ})"
    )
    args = ap.parse_args()

    # Resolve paths
    base_dir = os.path.abspath(args.base_dir)
    output_dir = os.path.abspath(args.output_dir)

    # Default CSV path: sibling to base_dir
    if args.csv:
        csv_file = os.path.abspath(args.csv)
    else:
        csv_file = os.path.join(os.path.dirname(base_dir), "targets_by_array.csv")

    return run_feathering(
        base_dir=base_dir,
        output_dir=output_dir,
        csv_file=csv_file,
        dry_run=args.dry_run,
        limit=args.limit,
        freq_tolerance=args.freq_tolerance,
    )


if __name__ == '__main__':
    sys.exit(main())
