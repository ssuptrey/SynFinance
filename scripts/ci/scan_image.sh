#!/usr/bin/env bash
set -euo pipefail

# Placeholder script for image scanning with Trivy or another scanner.
# Configure scanner in CI and set SCANNER variable or replace with scanner command.

IMAGE="$1"
if [ -z "${IMAGE:-}" ]; then
  echo "Usage: $0 <image>"
  exit 2
fi

if command -v trivy >/dev/null 2>&1; then
  echo "Running Trivy scan on ${IMAGE}"
  trivy image --severity HIGH,CRITICAL ${IMAGE} || true
else
  echo "Trivy not installed; skipping real scan. Install Trivy in CI or run locally."
  echo "Placeholder: would scan ${IMAGE}"
fi

exit 0
