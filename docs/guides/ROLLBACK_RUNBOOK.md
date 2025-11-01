# Rollback Runbook

## Quick Rollback Decision Tree

```
Issue detected in production?
├─ Is it a code/config bug?
│  └─ → Use Git Revert (Option 1)
├─ Is it a bad deployment/manifest?
│  └─ → Use ArgoCD Rollback (Option 2)
└─ Is it a container image issue?
   └─ → Use Image Tag Rollback (Option 3)
```

---

## Option 1: Git Revert (Recommended)

**Use when:** Code changes caused the issue
**Time:** 2-5 minutes
**Impact:** Triggers automatic redeployment

### Steps

1. **Identify the problematic commit:**
   ```bash
   git log --oneline -n 10
   # Or check GitHub: https://github.com/ssuptrey/SynFinance/commits/main
   ```

2. **Revert the commit:**
   ```bash
   git revert <commit-sha>
   # This creates a new commit that undoes the changes
   ```

3. **Push revert:**
   ```bash
   git push origin main
   ```

4. **Monitor ArgoCD:**
   ```bash
   # ArgoCD will detect the revert and sync automatically
   argocd app get synfinance-production
   argocd app wait synfinance-production --health
   ```

5. **Verify:**
   ```bash
   kubectl get pods -n synfinance-production
   kubectl logs -n synfinance-production -l app=synfinance --tail=50
   ```

### Pros/Cons
✅ **Pros:** Clean Git history, auditable, triggers CI/CD automatically  
❌ **Cons:** Takes 2-5 minutes for CI/CD pipeline

---

## Option 2: ArgoCD Rollback

**Use when:** Need immediate rollback, bypass Git
**Time:** 30 seconds - 2 minutes
**Impact:** Instant rollback to previous known-good state

### Steps

1. **View deployment history:**
   ```bash
   argocd app history synfinance-production
   ```
   Output example:
   ```
   ID  DATE                           REVISION
   10  2025-11-02 14:30:00 +0000 UTC  abc1234 (HEAD -> main)
   9   2025-11-02 12:15:00 +0000 UTC  def5678
   8   2025-11-01 18:00:00 +0000 UTC  ghi9012
   ```

2. **Identify last good deployment:**
   - Look for the deployment ID before the issue started

3. **Rollback:**
   ```bash
   argocd app rollback synfinance-production 9
   # Replace '9' with the target deployment ID
   ```

4. **Monitor rollback:**
   ```bash
   argocd app wait synfinance-production --health --timeout 300
   ```

5. **Verify application health:**
   ```bash
   kubectl get pods -n synfinance-production
   curl https://synfinance.yourdomain.com/health
   ```

6. **⚠️ Important: Sync Git afterward**
   ```bash
   # The rollback is NOT in Git! You must either:
   # A) Revert the bad commit in Git (recommended)
   git revert <bad-commit-sha>
   git push origin main
   
   # B) Or accept that next sync will re-apply the bad change
   ```

### Pros/Cons
✅ **Pros:** Instant rollback, good for emergencies  
❌ **Cons:** Creates drift between Git and cluster, must sync Git manually

---

## Option 3: Image Tag Rollback

**Use when:** Container image is broken (vulnerabilities, bugs)
**Time:** 1-2 minutes
**Impact:** Redeploys with previous image tag

### Steps

1. **Find previous working image tag:**
   ```bash
   # Check GitHub Container Registry
   # Visit: https://github.com/ssuptrey/SynFinance/packages
   
   # Or list tags via CLI
   gh api /user/packages/container/synfinance/versions | jq -r '.[].metadata.container.tags[]'
   ```
   Example tags:
   ```
   v2.15.0
   v2.14.0  ← Previous stable
   main-abc1234
   latest
   ```

2. **Update Helm values with specific tag:**
   ```bash
   # Quick rollback via Helm CLI
   helm upgrade synfinance ./helm/synfinance \
     --set image.tag=v2.14.0 \
     --namespace synfinance-production \
     --wait
   ```

3. **OR update via ArgoCD parameters:**
   ```bash
   argocd app set synfinance-production \
     --helm-set-string image.tag=v2.14.0
   
   argocd app sync synfinance-production
   ```

4. **Monitor rollout:**
   ```bash
   kubectl rollout status deployment/synfinance-api -n synfinance-production
   ```

5. **Verify pods running old image:**
   ```bash
   kubectl describe pod -n synfinance-production -l app=synfinance | grep Image:
   # Should show: ghcr.io/ssuptrey/synfinance:v2.14.0
   ```

6. **⚠️ Update Git to match:**
   ```bash
   # Edit helm/synfinance/values-prod.yaml
   # Change image.tag to v2.14.0
   git add helm/synfinance/values-prod.yaml
   git commit -m "Rollback to v2.14.0 due to issue in v2.15.0"
   git push origin main
   ```

### Pros/Cons
✅ **Pros:** Fast, targeted rollback of just the image  
❌ **Cons:** Must remember to update Git, can cause confusion

---

## Option 4: Full Namespace Reset (Nuclear Option)

**Use when:** Everything is broken, need clean slate
**Time:** 5-10 minutes
**Impact:** Complete redeployment from scratch

### ⚠️ WARNING: This deletes all data!

### Steps

1. **Backup critical data first:**
   ```bash
   # Backup Postgres database
   kubectl exec -n synfinance-production postgres-0 -- pg_dump -U synfinance > backup.sql
   
   # Backup Redis data (if needed)
   kubectl exec -n synfinance-production redis-0 -- redis-cli SAVE
   kubectl cp synfinance-production/redis-0:/data/dump.rdb ./redis-backup.rdb
   ```

2. **Delete ArgoCD application:**
   ```bash
   argocd app delete synfinance-production --cascade
   # This deletes the app AND all Kubernetes resources
   ```

3. **Verify namespace is clean:**
   ```bash
   kubectl get all -n synfinance-production
   # Should show: No resources found
   ```

4. **Reapply ArgoCD application:**
   ```bash
   kubectl apply -f k8s/overlays/production/argocd-app.yaml
   ```

5. **Wait for full deployment:**
   ```bash
   argocd app wait synfinance-production --health --timeout 600
   ```

6. **Restore data (if needed):**
   ```bash
   # Restore Postgres
   kubectl exec -i -n synfinance-production postgres-0 -- psql -U synfinance < backup.sql
   ```

### Pros/Cons
✅ **Pros:** Guaranteed clean state  
❌ **Cons:** Data loss risk, downtime, slow

---

## Rollback Verification Checklist

After any rollback, verify:

- [ ] Pods are running and healthy
  ```bash
  kubectl get pods -n synfinance-production
  ```

- [ ] Health endpoint returns 200
  ```bash
  curl https://synfinance.yourdomain.com/health
  ```

- [ ] Application logs show no errors
  ```bash
  kubectl logs -n synfinance-production -l app=synfinance --tail=100
  ```

- [ ] Database connectivity is working
  ```bash
  kubectl exec -n synfinance-production postgres-0 -- psql -U synfinance -c "SELECT 1"
  ```

- [ ] Redis connectivity is working
  ```bash
  kubectl exec -n synfinance-production redis-0 -- redis-cli PING
  ```

- [ ] ArgoCD shows synced and healthy
  ```bash
  argocd app get synfinance-production
  ```

- [ ] Git matches deployed state (if applicable)
  ```bash
  git log -1
  # Should match what ArgoCD deployed
  ```

---

## Post-Rollback Actions

1. **Document the incident:**
   - What went wrong?
   - Which rollback method was used?
   - How long was the outage?
   - Root cause?

2. **Create GitHub issue:**
   ```bash
   gh issue create \
     --title "Production rollback on 2025-11-02" \
     --body "Rolled back due to X. Root cause: Y. Resolution: Z."
   ```

3. **Update runbook if needed:**
   - Did you encounter issues with this runbook?
   - Add notes for next time

4. **Schedule post-mortem:**
   - Review with team
   - Identify prevention measures
   - Update CI/CD gates if needed

---

## Prevention: Pre-Deployment Checks

Run these BEFORE deploying to production:

```bash
# 1. Test in staging first
argocd app sync synfinance-staging
argocd app wait synfinance-staging --health

# 2. Run integration tests against staging
pytest tests/integration/ --env=staging

# 3. Check image scan results
gh run list --workflow=ci-build-push.yml --limit 1
# Download and review Trivy report artifact

# 4. Review Helm diff
argocd app diff synfinance-production

# 5. Confirm with team
echo "All checks passed. Proceeding with production deployment."
```

---

## Emergency Contacts

**On-Call Rotation:** [Link to PagerDuty/Opsgenie]
**Slack Channel:** #synfinance-incidents
**Escalation:** [Manager contact]

---

## Rollback Time Estimates

| Method | Time | Downtime | Complexity |
|--------|------|----------|------------|
| Git Revert | 2-5 min | ~2 min | Low |
| ArgoCD Rollback | 30s-2 min | ~30s | Low |
| Image Tag Rollback | 1-2 min | ~1 min | Medium |
| Namespace Reset | 5-10 min | ~5 min | High |

**Choose based on:**
- Severity (critical = fastest method)
- Type of issue (code vs config vs image)
- Time available
- Risk tolerance
