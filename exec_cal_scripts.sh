#!/bin/bash
# Execute calibration scripts for panta-rei
# Paths are loaded from .env file

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the .env file
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a  # Automatically export all variables
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "Error: .env file not found at $SCRIPT_DIR/.env"
    echo "Please copy .env.example to .env and configure your paths."
    exit 1
fi

# Construct paths from environment variables
OBS_CSV="${PANTA_REI_BASE}/${PROJECT_CODE}/targets_by_array.csv"
CASA_CMD="${CASA_PATH}/bin/casa --nologger --nogui --pipeline"

"${PYTHON_ENV}/bin/python" -m panta_rei.cli.run_calibration \
    --skip-tp \
    --obs-csv "$OBS_CSV" \
    --base-dir "${PANTA_REI_BASE}/${PROJECT_CODE}" \
    --casa-cmd "${CASA_CMD} -c \"{script}\""
calibration_rc=$?

if [ "$calibration_rc" -ne 0 ]; then
    echo "ERROR: Calibration failed with exit code $calibration_rc. Skipping contsub."
    exit "$calibration_rc"
fi

# Contsub remediation — produce _targets_line.ms for MOUSs where
# ScriptForPI skipped continuum subtraction.  Safe to run unconditionally:
# MOUSs that already have _targets_line.ms are skipped automatically.
"${PYTHON_ENV}/bin/python" -m panta_rei.cli.run_contsub \
    --obs-csv "$OBS_CSV" \
    --base-dir "${PANTA_REI_BASE}/${PROJECT_CODE}" \
    --casa-cmd "${CASA_CMD}"
