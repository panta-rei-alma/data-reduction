#!/bin/bash
# Wrapper script for alma-retrieve systemd service
# Sources environment configuration and runs the retrieval script
# This allows using environment variables for paths that systemd doesn't expand

set -euo pipefail

# Source the centralized path configuration
ENV_FILE="$(dirname "$0")/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: Environment file not found: $ENV_FILE" >&2
    exit 1
fi
source "$ENV_FILE"

# Ensure log directory exists
mkdir -p "${CRON_LOG_DIR}"

# Change to working directory
cd "${PANTA_REI_BASE}"

# Run the retrieval pipeline (refactored package)
exec "${PYTHON_ENV}/bin/python" -m panta_rei.cli.run_pipeline "${ALMA_USER}" \
    --project-code "${PROJECT_CODE}" \
    --base-dir "${PANTA_REI_BASE}/${PROJECT_CODE}" \
    --non-interactive \
    >> "${CRON_LOG_DIR}/alma_systemd.out" 2>> "${CRON_LOG_DIR}/alma_systemd.err"
