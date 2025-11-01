Day 3 — next steps (CI/CD & GitOps)

Summary
- Added: ArgoCD Application manifest (k8s/overlays/production/argocd-app.yaml), an additive CI workflow to lint manifests (.github/workflows/ci-manifest.yml), and helper scripts under scripts/ci/ (deploy_argocd.sh, scan_image.sh).

Required secrets (for full pipeline):
- DOCKER_REGISTRY (e.g. ghcr.io or docker.io)
- DOCKER_USERNAME / DOCKER_PASSWORD or GITHUB_TOKEN for GitHub Package Registry
- IMAGE_SIGNING_KEY (or cosign key) and IMAGE_SIGNING_PASSWORD (if using cosign)
- ARGOCD_SERVER, ARGOCD_USERNAME, ARGOCD_PASSWORD (or an argocd API token) for triggering sync (optional; ArgoCD can auto-sync)

Decisions you need to make
1) Keep existing main CI workflow or replace it?
   - I detected an existing `.github/workflows/ci.yml` in your repo. I created `ci-manifest.yml` as a small, additive workflow that focuses on manifest lint/tests so we don't overwrite anything.
   - If you want a unified pipeline (build -> scan -> sign -> push -> notify ArgoCD), I can either:
     A) Add a new `ci-build-and-push.yml` that builds images and pushes to your registry (requires secrets).
     B) Modify/extend the existing `ci.yml` (I will propose a non-destructive PR patch and wait for your confirmation).

2) How would you like GitOps sync to be performed?
   - ArgoCD automated sync (manifest contains automated.syncPolicy: true) — recommended for continuous delivery.
   - Or manual sync triggered from CI (I included `deploy_argocd.sh` to call argocd CLI), which is useful if you want image signing gates.

Next concrete steps I can take (pick any):
- Create a GitHub Actions workflow that builds the Docker image, runs Trivy scan, signs with cosign, and pushes to registry (I will include conditionals to skip push if secrets missing). I will not store secrets in the repo.
- Extend the existing `ci.yml` instead of adding new files (I will open a patch and show details first).
- Create a minimal ArgoCD Application for staging as well (k8s/overlays/staging/argocd-app.yaml).
- Configure CI to notify ArgoCD via the CLI or via a webhook to auto-sync.

If you want me to proceed, tell me which option you prefer for the build/push workflow:
- "add-build-push" — add a separate `ci-build-and-push.yml` (recommended)
- "extend-existing" — update the existing `.github/workflows/ci.yml` (I will show the proposed patch)

Also confirm the repository URL to place in `argocd-app.yaml` (currently a placeholder). If you want, I can also open a PR-style patch for the existing workflow instead of adding a new one.

Notes
- I kept all changes non-destructive: I didn't overwrite your existing CI workflow.
- The ArgoCD manifest uses Git path `k8s/overlays/production` and will create namespace `synfinance-production` on sync.
- Full image build/push/sign requires CI secrets; I'll add conditional steps so the workflow is safe to run without them.
