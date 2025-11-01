# Week 9 Day 3: CI/CD, GitOps, Monitoring and Observability

Date: November 2, 2025
Status: STARTING

Overview

Day 3 focuses on automating builds, tests and deployments. Deliverables include GitHub Actions CI workflows, Helm linting and testing, image scanning placeholders, and a GitOps (ArgoCD) application manifest to drive deployments from the repository.

Objectives

- Implement GitHub Actions CI workflow that runs lint, unit tests, manifest tests, helm lint, and a Docker build/publish placeholder.
- Add Helm CI workflow for chart linting and chart-testing placeholders.
- Provide small cross-platform CI scripts used by workflows.
- Add ArgoCD Application manifest in `k8s/overlays/production/argocd-app.yaml` to bootstrap GitOps.
- Add documentation for configuring CI secrets, registry credentials, and ArgoCD access.
- Add placeholders for Trivy/Clair/Snyk scanning and cosign signing.
- Run local manifest unit tests to validate generated files.

Success criteria

- CI workflow files present at `.github/workflows/` (ci.yml, helm-ci.yml)
- Scripts present at `scripts/ci/` and executable in CI runners
- ArgoCD app manifest present and documented
- Unit manifest tests pass locally (existing tests run successfully)

Timeline (estimated)

- CI workflows + scripts: 45-60 minutes
- ArgoCD manifest + docs: 30 minutes
- Local verification (run manifest tests): 10 minutes

Notes / Assumptions

- Publishing images requires registry credentials stored in GitHub Secrets (REGISTRY, REGISTRY_USERNAME, REGISTRY_PASSWORD). Workflows include placeholders and comments.
- ArgoCD requires cluster and credentials; the created manifest is a template that must be adapted to production cluster and ArgoCD instance.

Next steps

1. Create workflows and scripts.
2. Add ArgoCD application manifest.
3. Run local manifest tests and report results.
