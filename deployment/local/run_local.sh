#!/usr/bin/env bash
# Local development startup for the SENSE backend.
# Usage:  bash deployment/local/run_local.sh
set -euo pipefail

cd "$(dirname "$0")/../.."   # -> project root (sense-ai-demo)

export SENSE_HOST="${SENSE_HOST:-127.0.0.1}"
export SENSE_PORT="${SENSE_PORT:-8000}"
export SENSE_PIPELINE_MODE="${SENSE_PIPELINE_MODE:-hybrid}"

echo "Starting SENSE backend on http://${SENSE_HOST}:${SENSE_PORT}"
echo "Pipeline mode: ${SENSE_PIPELINE_MODE}"
echo "Docs: http://${SENSE_HOST}:${SENSE_PORT}/docs"

uvicorn backend.main:app --host "${SENSE_HOST}" --port "${SENSE_PORT}" --reload
