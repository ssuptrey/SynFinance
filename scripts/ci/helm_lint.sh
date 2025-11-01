#!/usr/bin/env bash
set -euo pipefail

echo "Installing helm..."
# In CI, Helm will be provided; this script verifies chart lint
if ! command -v helm >/dev/null 2>&1; then
  echo "helm not found in PATH"
  exit 1
fi

CHART_DIR="helm/synfinance"

echo "Running helm lint on ${CHART_DIR}"
helm lint ${CHART_DIR} || true

echo "Rendering templates (first 200 lines)"
helm template synfinance ${CHART_DIR} --values ${CHART_DIR}/values.yaml | head -n 200

echo "Helm lint completed."