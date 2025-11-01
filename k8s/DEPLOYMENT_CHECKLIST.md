# Kubernetes Deployment Verification Checklist

Use this checklist to verify successful deployment of SynFinance to Kubernetes.

## Pre-Deployment Verification

### Environment Check
- [ ] Kubernetes cluster is running and accessible
  ```bash
  kubectl cluster-info
  kubectl get nodes
  ```

- [ ] kubectl version is compatible (v1.24+)
  ```bash
  kubectl version --client
  ```

- [ ] Helm is installed (if using Helm deployment)
  ```bash
  helm version
  ```

- [ ] Docker image is available
  ```bash
  docker images | grep synfinance
  # OR for Minikube
  minikube image ls | grep synfinance
  ```

### Configuration Check
- [ ] Secrets are updated with production values (NOT defaults)
- [ ] ConfigMap settings are appropriate for environment
- [ ] Storage class is available in cluster
- [ ] Ingress controller is installed (if using ingress)
- [ ] Metrics server is installed (for HPA)

## Deployment Verification

### Step 1: Namespace
```bash
kubectl get namespace synfinance
```
- [ ] Namespace exists
- [ ] Labels are correct (app.kubernetes.io/name=synfinance)

### Step 2: ConfigMaps and Secrets
```bash
kubectl get configmap -n synfinance
kubectl get secret -n synfinance
```
- [ ] synfinance-config exists (50+ keys)
- [ ] postgres-config exists
- [ ] redis-config exists
- [ ] synfinance-secrets exists
- [ ] postgres-secrets exists
- [ ] redis-secrets exists

### Step 3: RBAC
```bash
kubectl get serviceaccount -n synfinance
kubectl get role -n synfinance
kubectl get rolebinding -n synfinance
```
- [ ] synfinance-api ServiceAccount exists
- [ ] postgres ServiceAccount exists
- [ ] redis ServiceAccount exists
- [ ] synfinance-api-role exists
- [ ] synfinance-api-rolebinding exists

### Step 4: Storage
```bash
kubectl get storageclass
kubectl get pvc -n synfinance
```
- [ ] Storage classes exist (synfinance-fast-ssd, synfinance-standard)
- [ ] PVCs are created (when StatefulSets start)
- [ ] PVCs are bound to volumes

### Step 5: Services
```bash
kubectl get svc -n synfinance
```
- [ ] synfinance-api service exists (ClusterIP)
- [ ] postgres service exists (Headless)
- [ ] redis service exists (Headless)
- [ ] Services have correct ports (8000, 5432, 6379)

### Step 6: Deployments and StatefulSets
```bash
kubectl get deployment -n synfinance
kubectl get statefulset -n synfinance
```
- [ ] synfinance-api deployment exists
- [ ] postgres statefulset exists
- [ ] redis statefulset exists

### Step 7: Pods
```bash
kubectl get pods -n synfinance -o wide
```
- [ ] All pods are in Running state
- [ ] No CrashLoopBackOff errors
- [ ] No ImagePullBackOff errors
- [ ] Pods are distributed across nodes (if multi-node)

Expected pods:
- [ ] synfinance-api-XXXXXX (3 pods)
- [ ] postgres-0 (1 pod)
- [ ] redis-0 (1 pod)

### Step 8: Pod Health
```bash
kubectl get pods -n synfinance
```
- [ ] All containers show READY (e.g., 1/1)
- [ ] RESTARTS count is 0 or low
- [ ] AGE shows pods are stable

### Step 9: Autoscaling
```bash
kubectl get hpa -n synfinance
kubectl get pdb -n synfinance
```
- [ ] synfinance-api-hpa exists
- [ ] HPA shows current/target metrics
- [ ] HPA min/max replicas correct (3-20)
- [ ] PodDisruptionBudgets exist for all components

### Step 10: Ingress and Network
```bash
kubectl get ingress -n synfinance
kubectl get networkpolicy -n synfinance
```
- [ ] Ingress exists (if enabled)
- [ ] Ingress has address/hostname
- [ ] NetworkPolicies exist (3 policies)

### Step 11: Resource Limits
```bash
kubectl get resourcequota -n synfinance
kubectl get limitrange -n synfinance
```
- [ ] ResourceQuota is applied
- [ ] LimitRange is applied
- [ ] No pods exceed quotas

## Functional Verification

### Step 12: Pod Logs
```bash
kubectl logs -n synfinance -l app.kubernetes.io/name=synfinance-api --tail=50
kubectl logs -n synfinance postgres-0 --tail=50
kubectl logs -n synfinance redis-0 --tail=50
```
- [ ] No ERROR messages in API logs
- [ ] PostgreSQL shows "database system is ready"
- [ ] Redis shows "Ready to accept connections"
- [ ] No connection errors
- [ ] Migrations completed successfully

### Step 13: Database Connectivity
```bash
# PostgreSQL
kubectl exec -it postgres-0 -n synfinance -- psql -U synfinance_trey -d synfinance -c "SELECT version();"

# Redis
kubectl exec -it redis-0 -n synfinance -- redis-cli -a "$REDIS_PASSWORD" PING
```
- [ ] PostgreSQL responds with version
- [ ] Redis responds with PONG
- [ ] No authentication errors

### Step 14: API Health Endpoints
```bash
kubectl port-forward -n synfinance svc/synfinance-api 8000:8000 &
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/detailed
```
- [ ] /health returns 200 OK
- [ ] /health/ready returns 200 OK
- [ ] /health/detailed shows database and cache connected
- [ ] Response time is reasonable (<500ms)

### Step 15: API Documentation
```bash
curl http://localhost:8000/docs
```
- [ ] /docs endpoint accessible
- [ ] Swagger UI loads correctly
- [ ] API endpoints are listed

### Step 16: Resource Usage
```bash
kubectl top nodes
kubectl top pods -n synfinance
```
- [ ] Nodes have available resources
- [ ] Pod CPU usage within limits
- [ ] Pod memory usage within limits
- [ ] No resource exhaustion warnings

### Step 17: Events
```bash
kubectl get events -n synfinance --sort-by='.lastTimestamp' | tail -20
```
- [ ] No Warning events
- [ ] No Error events
- [ ] No FailedScheduling events
- [ ] Only Normal events present

## Load Testing (Optional)

### Step 18: Scale Testing
```bash
# Generate load
kubectl run -it --rm load-test --image=williamyeh/hey --restart=Never -- \
  -z 60s -c 10 http://synfinance-api.synfinance.svc.cluster.local:8000/health

# Watch HPA
kubectl get hpa -n synfinance -w
```
- [ ] API responds to all requests
- [ ] No 500 errors under load
- [ ] HPA scales up when CPU/Memory increases
- [ ] New pods start successfully
- [ ] HPA scales down after load decreases

### Step 19: Resilience Testing
```bash
# Delete a pod
kubectl delete pod -n synfinance -l app.kubernetes.io/name=synfinance-api --force --grace-period=0

# Watch recovery
kubectl get pods -n synfinance -w
```
- [ ] Deployment creates replacement pod
- [ ] New pod reaches Running state
- [ ] Service continues routing traffic
- [ ] No downtime observed

### Step 20: Rolling Update Testing
```bash
# Trigger rolling update
kubectl rollout restart deployment/synfinance-api -n synfinance

# Watch rollout
kubectl rollout status deployment/synfinance-api -n synfinance
```
- [ ] Rolling update completes successfully
- [ ] No more than 1 pod unavailable at a time
- [ ] All pods reach Running state
- [ ] Service maintains availability

## Security Verification

### Step 21: Security Contexts
```bash
kubectl get pod POD_NAME -n synfinance -o jsonpath='{.spec.containers[0].securityContext}'
```
- [ ] runAsNonRoot: true
- [ ] allowPrivilegeEscalation: false
- [ ] capabilities.drop includes ALL

### Step 22: Network Policies
```bash
kubectl describe networkpolicy -n synfinance
```
- [ ] Ingress policies are restrictive
- [ ] Egress policies are restrictive
- [ ] Only required connections allowed

### Step 23: RBAC Permissions
```bash
kubectl auth can-i --list --as=system:serviceaccount:synfinance:synfinance-api -n synfinance
```
- [ ] Minimal permissions granted
- [ ] No cluster-wide permissions
- [ ] Read-only for most resources

## Helm-Specific Verification (If using Helm)

### Step 24: Helm Release
```bash
helm list -n synfinance
helm status synfinance -n synfinance
helm get values synfinance -n synfinance
```
- [ ] Release shows DEPLOYED status
- [ ] Revision number is correct
- [ ] Values are as expected

### Step 25: Helm Tests
```bash
helm test synfinance -n synfinance
```
- [ ] All Helm tests pass (if defined)

## Post-Deployment Actions

### Step 26: Documentation
- [ ] Deployment date and time recorded
- [ ] Deployment method documented (kubectl/Helm)
- [ ] Configuration values documented
- [ ] Any issues encountered documented

### Step 27: Monitoring Setup
- [ ] Prometheus scraping endpoints (if configured)
- [ ] Grafana dashboards configured (if applicable)
- [ ] Alerts configured (if applicable)

### Step 28: Backup Verification
- [ ] Database backup process tested
- [ ] Backup schedule configured
- [ ] Restore procedure documented

### Step 29: Disaster Recovery
- [ ] Rollback procedure tested
- [ ] Disaster recovery plan documented
- [ ] Contact information updated

### Step 30: Handoff
- [ ] Operations team notified
- [ ] Runbook provided
- [ ] Access credentials shared securely
- [ ] On-call rotation updated

## Sign-Off

- [ ] All critical checks passed
- [ ] All warnings addressed or documented
- [ ] Deployment verified by second engineer
- [ ] Go-live approval obtained

**Deployment Verified By:** _______________  
**Date:** _______________  
**Environment:** [ ] Dev [ ] Staging [ ] Production  
**Version:** 2.15.0  

## Troubleshooting Reference

If any check fails, refer to:
- k8s/README.md - Comprehensive troubleshooting guide
- k8s/QUICKSTART.md - Quick fixes for common issues
- tests/deployment/test_kubernetes.py - Automated verification

## Emergency Contacts

- Kubernetes Admin: _______________
- Database Admin: _______________
- Security Team: _______________
- On-Call Engineer: _______________

---

**Note:** This checklist should be completed for every deployment to every environment. Keep completed checklists for audit and compliance purposes.
