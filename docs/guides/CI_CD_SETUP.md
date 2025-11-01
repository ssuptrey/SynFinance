# CI/CD Configuration Guide

## Overview
SynFinance uses a multi-workflow CI/CD pipeline with GitOps deployment via ArgoCD.

## Workflows

### 1. `ci-manifest.yml` - Fast Feedback
**Triggers:** On every push/PR affecting `k8s/`, `helm/`, or `tests/deployment/`
**Purpose:** Quick validation of manifests and Helm charts
**Steps:**
- YAML linting
- Helm lint and template validation
- Manifest unit tests
- Fast test suite

**No secrets required** - runs immediately on every commit.

---

### 2. `ci-build-push.yml` - Build & Deploy Pipeline
**Triggers:** Push to `main`, `release/**` branches, or version tags
**Purpose:** Build, scan, sign, and publish container images
**Steps:**
1. Build Docker image with BuildKit cache
2. Trivy vulnerability scan (fails on >10 CRITICAL vulns)
3. Push to GitHub Container Registry (ghcr.io)
4. (Optional) Sign with Cosign
5. (Optional) Generate SBOM
6. (Optional) Trigger ArgoCD sync

---

## Required Secrets & Variables

### Auto-Provided (No Action Needed)
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions
  - Used for: pushing to ghcr.io, accessing repo

### Optional - Image Signing (Cosign)
If you want signed images:

1. **Generate signing key:**
   ```bash
   cosign generate-key-pair
   ```
   This creates `cosign.key` (private) and `cosign.pub` (public)

2. **Add to GitHub Secrets:**
   - `COSIGN_PRIVATE_KEY` - Contents of `cosign.key`
   - `COSIGN_PASSWORD` - Password you set during key generation

3. **Enable in GitHub Variables:**
   - `ENABLE_IMAGE_SIGNING` = `true`

**Alternative:** Use keyless signing (OIDC) - no secrets needed, just enable the variable.

---

### Optional - ArgoCD Auto-Sync
If you want CI to trigger ArgoCD deployments:

1. **Get ArgoCD auth token:**
   ```bash
   # Login to ArgoCD
   argocd login argocd.yourdomain.com
   
   # Generate token
   argocd account generate-token --account ci-pipeline
   ```

2. **Add to GitHub Secrets:**
   - `ARGOCD_SERVER` - e.g., `argocd.example.com`
   - `ARGOCD_AUTH_TOKEN` - Token from step 1

3. **Enable in GitHub Variables:**
   - `ENABLE_ARGOCD_SYNC` = `true`

**Note:** If you use ArgoCD's automated sync policy (default in our manifests), you don't need this - ArgoCD will detect changes automatically.

---

## GitHub Container Registry Setup

### 1. Enable GitHub Packages
Already configured in the workflow - images will push to:
```
ghcr.io/ssuptrey/synfinance:latest
ghcr.io/ssuptrey/synfinance:main-<sha>
ghcr.io/ssuptrey/synfinance:v1.2.3  (for tags)
```

### 2. Make Images Public (Optional)
- Go to: https://github.com/ssuptrey/SynFinance/packages
- Select the `synfinance` package
- Settings → Change visibility → Public

---

## ArgoCD Setup

### 1. Install ArgoCD in Cluster
```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

### 2. Access ArgoCD UI
```bash
# Port forward
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Get initial password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

### 3. Deploy Applications
```bash
# Apply staging application
kubectl apply -f k8s/overlays/staging/argocd-app.yaml

# Apply production application
kubectl apply -f k8s/overlays/production/argocd-app.yaml
```

ArgoCD will now:
- Monitor the GitHub repo
- Auto-sync on changes (if automated sync enabled)
- Deploy using Helm with environment-specific values

---

## Security Scanning Thresholds

### Current Configuration
The build fails if:
- **> 10 CRITICAL** vulnerabilities found by Trivy

### Adjust Thresholds
Edit `.github/workflows/ci-build-push.yml`:
```yaml
- name: Check for critical vulnerabilities
  run: |
    CRITICAL_COUNT=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL")] | length' trivy-report.json)
    if [ "${CRITICAL_COUNT}" -gt 5 ]; then  # Change threshold here
      exit 1
    fi
```

---

## Image Tag Strategy

### Automatic Tags
Every build creates multiple tags:
- `latest` - Only on main branch
- `main-abc1234` - Branch name + short SHA
- `v1.2.3` - Semantic version (for git tags)
- `1.2` - Major.minor (for git tags)

### Using Tags in Kubernetes
**Staging:** Uses `main-*` tags (latest builds)
**Production:** Uses `v*` tags (releases only)

---

## Rollback Procedures

### Option 1: Git Revert (Recommended)
```bash
# Revert the problematic commit
git revert <commit-sha>
git push origin main

# ArgoCD will auto-sync to the reverted state
```

### Option 2: ArgoCD Rollback
```bash
# View history
argocd app history synfinance-production

# Rollback to previous version
argocd app rollback synfinance-production <history-id>
```

### Option 3: Manual Image Rollback
```bash
# Update Helm values to use previous image tag
helm upgrade synfinance ./helm/synfinance \
  --set image.tag=v1.2.2 \
  --namespace synfinance-production
```

---

## Monitoring & Notifications

### Build Status
- Check: https://github.com/ssuptrey/SynFinance/actions
- Artifacts available: Trivy reports, SBOM

### Add Slack Notifications (Optional)
Add to workflow:
```yaml
- name: Slack Notification
  if: always()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Quick Reference

### Test Locally Before Push
```bash
# Lint manifests
yamllint k8s/

# Lint Helm
./scripts/ci/helm_lint.sh

# Run unit tests
./scripts/ci/run_tests.sh

# Build image locally
docker build -t synfinance:test .

# Scan locally
trivy image synfinance:test
```

### Force Workflow Run
```bash
# Trigger workflow manually
gh workflow run ci-build-push.yml
```

### Debug Failed Builds
1. Check Actions tab: https://github.com/ssuptrey/SynFinance/actions
2. Download artifacts (Trivy report, SBOM)
3. Review step logs
4. Test locally with same Docker version

---

## Common Issues

### Issue: Image push fails with 403
**Solution:** Ensure `packages: write` permission in workflow and GITHUB_TOKEN is valid

### Issue: Trivy scan fails
**Solution:** Check threshold settings, review vulnerability report artifact

### Issue: ArgoCD sync fails
**Solution:** 
- Verify repo URL in argocd-app.yaml
- Check ArgoCD has read access to repo
- Review ArgoCD application logs: `argocd app logs synfinance-production`

### Issue: Helm template errors
**Solution:** Run `./scripts/ci/helm_lint.sh` locally to debug template issues

---

## Next Steps

1. ✅ Workflows created and committed
2. ⏳ Push to GitHub to trigger first CI run
3. ⏳ Review build results and adjust thresholds if needed
4. ⏳ Install ArgoCD in cluster
5. ⏳ Apply ArgoCD application manifests
6. ⏳ (Optional) Configure image signing
7. ⏳ (Optional) Add Slack notifications

For detailed deployment checklist, see: `k8s/DEPLOYMENT_CHECKLIST.md`
