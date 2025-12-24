#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${ROOT_DIR}/../echo-vmat-venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "echo-vmat-venv not found at ${VENV_PY}"
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}"
if [[ -z "${MOSEKLM_LICENSE_FILE:-}" ]] && [[ -f "${ROOT_DIR}/../mosek.lic" ]]; then
  export MOSEKLM_LICENSE_FILE="${ROOT_DIR}/../mosek.lic"
fi

"${VENV_PY}" -m uvicorn backend.main:app --reload \
  --no-access-log \
  --reload-dir "${ROOT_DIR}/backend" \
  --reload-exclude '**/runs*' \
  --reload-exclude '**/PortPy/**' \
  --reload-exclude '**/node_modules/**'
