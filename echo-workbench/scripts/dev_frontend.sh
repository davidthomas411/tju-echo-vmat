#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"

if [[ ! -d "${FRONTEND_DIR}" ]]; then
  echo "frontend/ not found at ${FRONTEND_DIR}"
  exit 1
fi

cd "${FRONTEND_DIR}"
if [[ ! -d node_modules ]]; then
  npm install
fi
exec npm run dev
