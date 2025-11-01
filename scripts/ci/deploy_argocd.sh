#!/usr/bin/env bash
set -euo pipefail

# Helper to trigger ArgoCD sync for the production application using the argocd CLI.
# Usage: deploy_argocd.sh [APP_NAME] [ARGOCD_SERVER] [ARGOCD_USERNAME] [ARGOCD_PASSWORD]
# Requires: argocd CLI installed in CI runner or environment.

APP_NAME="${1:-synfinance-production}"
ARGOCD_SERVER="${2:-argocd.example.com}"
ARGOCD_USERNAME="${3:-admin}"
ARGOCD_PASSWORD="${4:-}" # recommend using token or ARGOCd_PASSWORD env

if [ -z "${ARGOCD_PASSWORD}" ]; then
  echo "Provide ARGOCD_PASSWORD as argument or via env var. Exiting."
  exit 2
fi

# Login
argocd login ${ARGOCD_SERVER} --username ${ARGOCD_USERNAME} --password-stdin <<< "${ARGOCD_PASSWORD}" --insecure || true

# Sync app
argocd app sync ${APP_NAME}

# Optionally wait for health
argocd app wait ${APP_NAME} --health --timeout 300s

echo "Triggered ArgoCD sync for ${APP_NAME} on ${ARGOCD_SERVER}" 
