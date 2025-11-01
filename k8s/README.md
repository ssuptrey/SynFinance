# SynFinance Kubernetes Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Deployment](#local-development-deployment)
3. [Production Deployment](#production-deployment)
4. [Helm Deployment](#helm-deployment)
5. [Verification and Testing](#verification-and-testing)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Troubleshooting](#troubleshooting)
8. [Rollback Procedures](#rollback-procedures)
9. [Scaling and Performance](#scaling-and-performance)
10. [Security Best Practices](#security-best-practices)

---

## Prerequisites

### Required Software

- **Kubernetes Cluster:** v1.24+ (Minikube, Kind, EKS, GKE, AKS, etc.)
- **kubectl:** v1.24+
- **Helm:** v3.10+ (optional, for Helm-based deployment)
- **Docker:** v20.10+ (for building images)

### Kubernetes Cluster Setup

#### Local Development (Minikube)

```bash
# Install Minikube
# Windows: choco install minikube
# Mac: brew install minikube
# Linux: curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64

# Start Minikube with adequate resources
minikube start --memory=8192 --cpus=4 --disk-size=50g

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server
minikube addons enable storage-provisioner

# Verify cluster
kubectl cluster-info
kubectl get nodes
```

#### Local Development (Kind)

```bash
# Install Kind
# Windows: choco install kind
# Mac: brew install kind
# Linux: curl -Lo ./kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64

# Create cluster with custom configuration
cat <<EOF | kind create cluster --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
- role: worker
- role: worker
EOF

# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# Install metrics-server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Docker Image Preparation

```bash
# Build the Docker image
docker build -t synfinance:2.15.0 .

# For Minikube: Load image into Minikube
minikube image load synfinance:2.15.0

# For Kind: Load image into Kind
kind load docker-image synfinance:2.15.0

# For production: Push to registry
docker tag synfinance:2.15.0 your-registry.com/synfinance:2.15.0
docker push your-registry.com/synfinance:2.15.0
```

---

## Local Development Deployment

### Option 1: Direct kubectl Deployment

#### Step 1: Deploy Base Resources

```bash
# Navigate to k8s directory
cd k8s/base

# Create namespace
kubectl apply -f namespace.yaml

# Verify namespace
kubectl get namespace synfinance
```

#### Step 2: Deploy Configuration and Secrets

```bash
# Apply ConfigMaps
kubectl apply -f configmap.yaml

# Update secrets with actual values (IMPORTANT: Never commit real secrets)
# Edit secrets.yaml and replace all CHANGE_ME_IN_PRODUCTION values
kubectl apply -f secrets.yaml

# Verify
kubectl get configmap -n synfinance
kubectl get secret -n synfinance
```

#### Step 3: Deploy Storage

```bash
# Apply storage class
kubectl apply -f storage-class.yaml

# Verify
kubectl get storageclass
```

#### Step 4: Deploy Databases

```bash
# Deploy PostgreSQL
kubectl apply -f postgres-statefulset.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgres -n synfinance --timeout=300s

# Check PostgreSQL logs
kubectl logs -n synfinance -l app.kubernetes.io/name=postgres --tail=50

# Deploy Redis
kubectl apply -f redis-statefulset.yaml

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n synfinance --timeout=300s

# Check Redis logs
kubectl logs -n synfinance -l app.kubernetes.io/name=redis --tail=50
```

#### Step 5: Deploy API

```bash
# Apply RBAC
kubectl apply -f rbac.yaml

# Deploy API
kubectl apply -f api-deployment.yaml

# Wait for API pods to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=synfinance-api -n synfinance --timeout=300s

# Check API logs
kubectl logs -n synfinance -l app.kubernetes.io/name=synfinance-api --tail=50
```

#### Step 6: Deploy Autoscaling and Policies

```bash
# Apply HPA and PodDisruptionBudgets
kubectl apply -f hpa.yaml

# Apply resource limits
kubectl apply -f resource-limits.yaml

# Apply network policies (if supported by your cluster)
kubectl apply -f ingress.yaml
```

#### Step 7: Verify Deployment

```bash
# Check all resources
kubectl get all -n synfinance

# Check pods status
kubectl get pods -n synfinance -o wide

# Check services
kubectl get svc -n synfinance

# Check persistent volumes
kubectl get pvc -n synfinance

# Check HPA
kubectl get hpa -n synfinance

# Check events for any issues
kubectl get events -n synfinance --sort-by='.lastTimestamp'
```

#### Step 8: Test API Access

```bash
# Port forward to API service
kubectl port-forward -n synfinance svc/synfinance-api 8000:8000

# In another terminal, test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/detailed
curl http://localhost:8000/docs
```

### Option 2: Kustomize Deployment

```bash
# Create kustomization.yaml
cat <<EOF > k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: synfinance

resources:
  - namespace.yaml
  - storage-class.yaml
  - configmap.yaml
  - secrets.yaml
  - rbac.yaml
  - postgres-statefulset.yaml
  - redis-statefulset.yaml
  - api-deployment.yaml
  - hpa.yaml
  - ingress.yaml
  - resource-limits.yaml
EOF

# Deploy with kustomize
kubectl apply -k k8s/base

# Verify
kubectl get all -n synfinance
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Kubernetes cluster is production-ready (HA, multiple nodes)
- [ ] Docker images are pushed to production registry
- [ ] Secrets are updated with production values
- [ ] Database backups are configured
- [ ] Monitoring and alerting are set up
- [ ] SSL/TLS certificates are configured
- [ ] DNS records are configured
- [ ] Load balancer is configured
- [ ] Network policies are tested
- [ ] Disaster recovery plan is documented

### Production Secrets Management

```bash
# Generate strong random secrets
# For SECRET_KEY (64 characters)
openssl rand -hex 32

# For JWT_SECRET (64 characters)
openssl rand -hex 32

# For ENCRYPTION_KEY (32 bytes, base64 encoded)
openssl rand -base64 32

# For database passwords (32 characters)
openssl rand -base64 24

# Update secrets.yaml with generated values
# NEVER commit real production secrets to version control

# For production, use external secret management
# Option 1: Kubernetes External Secrets Operator
# Option 2: HashiCorp Vault
# Option 3: AWS Secrets Manager / Azure Key Vault / GCP Secret Manager
```

### Production Deployment Steps

```bash
# 1. Create production namespace
kubectl apply -f k8s/base/namespace.yaml

# 2. Create secrets from external secret manager
# Example with kubectl (use proper secret management in production)
kubectl create secret generic synfinance-secrets \
  --from-literal=DATABASE_URL="postgresql://user:password@postgres:5432/synfinance" \
  --from-literal=REDIS_URL="redis://:password@redis:6379/0" \
  --from-literal=SECRET_KEY="$(openssl rand -hex 32)" \
  --from-literal=JWT_SECRET="$(openssl rand -hex 32)" \
  -n synfinance

# 3. Apply all configurations
kubectl apply -f k8s/base/configmap.yaml
kubectl apply -f k8s/base/storage-class.yaml
kubectl apply -f k8s/base/rbac.yaml
kubectl apply -f k8s/base/resource-limits.yaml

# 4. Deploy databases with production settings
kubectl apply -f k8s/base/postgres-statefulset.yaml
kubectl apply -f k8s/base/redis-statefulset.yaml

# Wait for databases
kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=database -n synfinance --timeout=600s

# 5. Deploy API
kubectl apply -f k8s/base/api-deployment.yaml

# Wait for API
kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=api -n synfinance --timeout=600s

# 6. Deploy autoscaling and ingress
kubectl apply -f k8s/base/hpa.yaml
kubectl apply -f k8s/base/ingress.yaml

# 7. Verify production deployment
kubectl get all -n synfinance
kubectl get pvc -n synfinance
kubectl get ingress -n synfinance
```

---

## Helm Deployment

### Install with Helm (Recommended for Production)

#### Development Environment

```bash
# Install Helm chart for development
helm upgrade --install synfinance ./helm/synfinance \
  --namespace synfinance \
  --create-namespace \
  --values helm/synfinance/values-dev.yaml \
  --wait \
  --timeout 10m

# Check deployment
helm list -n synfinance
helm status synfinance -n synfinance

# Get values
helm get values synfinance -n synfinance
```

#### Staging Environment

```bash
# Install Helm chart for staging
helm upgrade --install synfinance ./helm/synfinance \
  --namespace synfinance \
  --create-namespace \
  --values helm/synfinance/values-staging.yaml \
  --wait \
  --timeout 10m
```

#### Production Environment

```bash
# Create production secrets first
kubectl create namespace synfinance

kubectl create secret generic synfinance-secrets \
  --from-literal=DATABASE_URL="$PROD_DATABASE_URL" \
  --from-literal=REDIS_URL="$PROD_REDIS_URL" \
  --from-literal=SECRET_KEY="$PROD_SECRET_KEY" \
  --from-literal=JWT_SECRET="$PROD_JWT_SECRET" \
  -n synfinance

# Install Helm chart for production
helm upgrade --install synfinance ./helm/synfinance \
  --namespace synfinance \
  --create-namespace \
  --values helm/synfinance/values-prod.yaml \
  --set global.imageRegistry="your-registry.com/" \
  --set api.image.tag="2.15.0" \
  --wait \
  --timeout 15m

# Verify production deployment
helm test synfinance -n synfinance
```

#### Custom Values Override

```bash
# Create custom values file
cat <<EOF > custom-values.yaml
api:
  replicaCount: 5
  resources:
    requests:
      cpu: 1000m
      memory: 2Gi
postgres:
  persistence:
    size: 100Gi
EOF

# Install with custom values
helm upgrade --install synfinance ./helm/synfinance \
  --namespace synfinance \
  --values helm/synfinance/values-prod.yaml \
  --values custom-values.yaml
```

---

## Verification and Testing

### Health Check Verification

```bash
# Port forward to API
kubectl port-forward -n synfinance svc/synfinance-api 8000:8000 &

# Test health endpoints
curl http://localhost:8000/health
# Expected: {"status":"healthy"}

curl http://localhost:8000/health/ready
# Expected: {"status":"ready"}

curl http://localhost:8000/health/detailed
# Expected: Detailed status with database and cache info

curl http://localhost:8000/docs
# Expected: API documentation page
```

### Database Connectivity Test

```bash
# Test PostgreSQL connection
kubectl exec -it -n synfinance postgres-0 -- psql -U synfinance_trey -d synfinance -c "SELECT version();"

# Test Redis connection
kubectl exec -it -n synfinance redis-0 -- redis-cli -a "$REDIS_PASSWORD" PING
```

### Load Testing

```bash
# Install Apache Bench or use k6
# Using kubectl run with temporary pod
kubectl run -it --rm load-test --image=williamyeh/hey --restart=Never -- \
  -z 60s -c 10 http://synfinance-api.synfinance.svc.cluster.local:8000/health

# Monitor HPA during load test
kubectl get hpa -n synfinance -w
```

### Run Automated Tests

```bash
# Run Kubernetes tests
pytest tests/deployment/test_kubernetes.py -v

# Run integration tests
pytest tests/deployment/test_kubernetes.py::TestKubernetesDeployment -v -m integration
```

---

## Monitoring and Observability

### Pod Monitoring

```bash
# Watch pod status
kubectl get pods -n synfinance -w

# View pod logs
kubectl logs -n synfinance -l app.kubernetes.io/name=synfinance-api -f

# View previous pod logs (if pod crashed)
kubectl logs -n synfinance POD_NAME --previous

# Describe pod for events
kubectl describe pod -n synfinance POD_NAME
```

### Resource Usage Monitoring

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n synfinance

# Check HPA metrics
kubectl get hpa -n synfinance
kubectl describe hpa synfinance-api-hpa -n synfinance

# Check resource quotas
kubectl get resourcequota -n synfinance
kubectl describe resourcequota synfinance-resource-quota -n synfinance
```

### Event Monitoring

```bash
# Watch events
kubectl get events -n synfinance --watch

# Get recent events
kubectl get events -n synfinance --sort-by='.lastTimestamp' | tail -20
```

---

## Troubleshooting

### Common Issues and Solutions

#### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n synfinance

# Describe pod to see events
kubectl describe pod POD_NAME -n synfinance

# Common causes:
# 1. Image pull errors
# 2. Insufficient resources
# 3. Failed health checks
# 4. Configuration errors

# Check pod events
kubectl get events -n synfinance --field-selector involvedObject.name=POD_NAME
```

#### ImagePullBackOff Error

```bash
# Verify image exists
docker images | grep synfinance

# For Minikube: Load image
minikube image load synfinance:2.15.0

# For Kind: Load image
kind load docker-image synfinance:2.15.0

# Verify image in cluster
kubectl get pod POD_NAME -n synfinance -o jsonpath='{.spec.containers[0].image}'
```

#### CrashLoopBackOff Error

```bash
# View logs
kubectl logs POD_NAME -n synfinance

# View previous logs
kubectl logs POD_NAME -n synfinance --previous

# Common causes:
# 1. Application errors
# 2. Database connection failures
# 3. Missing environment variables
# 4. Failed migrations
```

#### Database Connection Issues

```bash
# Check PostgreSQL pod
kubectl get pod -n synfinance -l app.kubernetes.io/name=postgres

# Check PostgreSQL logs
kubectl logs -n synfinance postgres-0

# Test connection from API pod
kubectl exec -it POD_NAME -n synfinance -- sh
# Inside pod:
nc -zv postgres 5432
psql postgresql://synfinance_trey:password@postgres:5432/synfinance
```

#### Network Policy Issues

```bash
# Check network policies
kubectl get networkpolicy -n synfinance

# Describe network policy
kubectl describe networkpolicy POLICY_NAME -n synfinance

# Temporarily disable network policy for debugging
kubectl delete networkpolicy POLICY_NAME -n synfinance
```

#### Persistent Volume Issues

```bash
# Check PVC status
kubectl get pvc -n synfinance

# Describe PVC
kubectl describe pvc PVC_NAME -n synfinance

# Check PV
kubectl get pv

# For local development, ensure storage provisioner is enabled
minikube addons enable storage-provisioner
```

#### HPA Not Scaling

```bash
# Check metrics-server is running
kubectl get deployment metrics-server -n kube-system

# Check HPA status
kubectl describe hpa synfinance-api-hpa -n synfinance

# Check metrics
kubectl top pods -n synfinance

# If metrics not available, install metrics-server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Debugging Commands

```bash
# Execute shell in running pod
kubectl exec -it POD_NAME -n synfinance -- /bin/sh

# Copy files from pod
kubectl cp synfinance/POD_NAME:/app/logs/app.log ./app.log

# Port forward to pod
kubectl port-forward POD_NAME -n synfinance 8000:8000

# Get pod YAML
kubectl get pod POD_NAME -n synfinance -o yaml

# Get all resources with labels
kubectl get all -n synfinance -l app.kubernetes.io/part-of=synfinance
```

---

## Rollback Procedures

### Helm Rollback

```bash
# List releases
helm list -n synfinance

# Check release history
helm history synfinance -n synfinance

# Rollback to previous version
helm rollback synfinance -n synfinance

# Rollback to specific revision
helm rollback synfinance 2 -n synfinance
```

### kubectl Rollback

```bash
# View deployment rollout history
kubectl rollout history deployment/synfinance-api -n synfinance

# Rollback to previous revision
kubectl rollout undo deployment/synfinance-api -n synfinance

# Rollback to specific revision
kubectl rollout undo deployment/synfinance-api -n synfinance --to-revision=2

# Check rollout status
kubectl rollout status deployment/synfinance-api -n synfinance
```

### Emergency Procedures

```bash
# Scale down deployment
kubectl scale deployment/synfinance-api --replicas=0 -n synfinance

# Delete problematic resources
kubectl delete pod POD_NAME -n synfinance --force --grace-period=0

# Restart all pods in deployment
kubectl rollout restart deployment/synfinance-api -n synfinance
```

---

## Scaling and Performance

### Manual Scaling

```bash
# Scale API deployment
kubectl scale deployment/synfinance-api --replicas=5 -n synfinance

# Scale StatefulSet (use with caution)
kubectl scale statefulset/postgres --replicas=1 -n synfinance
```

### Autoscaling Configuration

```bash
# Update HPA
kubectl patch hpa synfinance-api-hpa -n synfinance -p '{"spec":{"minReplicas":5,"maxReplicas":30}}'

# Disable autoscaling
kubectl delete hpa synfinance-api-hpa -n synfinance

# Re-enable autoscaling
kubectl apply -f k8s/base/hpa.yaml
```

### Performance Optimization

1. **Resource Requests/Limits Tuning**
   - Monitor actual usage with `kubectl top`
   - Adjust requests/limits based on observed usage
   - Set limits 1.5-2x higher than requests

2. **Database Optimization**
   - Increase PostgreSQL connection pool
   - Enable query caching
   - Add read replicas for scaling

3. **Cache Optimization**
   - Increase Redis memory
   - Enable Redis persistence
   - Use Redis clustering for scale

4. **Network Optimization**
   - Enable HTTP/2 in ingress
   - Configure connection pooling
   - Use service mesh for advanced routing

---

## Security Best Practices

### Secrets Management

1. **Never commit secrets to version control**
2. **Use external secret management** (Vault, AWS Secrets Manager, etc.)
3. **Rotate secrets regularly**
4. **Use RBAC to restrict secret access**
5. **Enable encryption at rest** for secrets

### Network Security

1. **Enable NetworkPolicies** in production
2. **Use TLS for all external communications**
3. **Restrict ingress to specific IPs** if possible
4. **Enable pod security standards**

### Container Security

1. **Run containers as non-root** (already configured)
2. **Use read-only root filesystem** where possible
3. **Drop all capabilities** and add only required ones
4. **Scan images for vulnerabilities** regularly
5. **Use minimal base images** (alpine)

### Access Control

1. **Use RBAC** for fine-grained permissions
2. **Disable service account token auto-mounting** (already configured)
3. **Enable audit logging**
4. **Use pod security admission**

---

## Maintenance

### Regular Maintenance Tasks

```bash
# Update deployments
kubectl set image deployment/synfinance-api api=synfinance:2.16.0 -n synfinance

# Restart deployments
kubectl rollout restart deployment/synfinance-api -n synfinance

# Clean up completed pods
kubectl delete pod --field-selector=status.phase==Succeeded -n synfinance

# Check resource usage
kubectl top nodes
kubectl top pods -n synfinance
```

### Backup Procedures

```bash
# Backup PostgreSQL
kubectl exec postgres-0 -n synfinance -- pg_dump -U synfinance_trey synfinance > backup.sql

# Backup Redis
kubectl exec redis-0 -n synfinance -- redis-cli -a "$REDIS_PASSWORD" SAVE
kubectl cp synfinance/redis-0:/data/dump.rdb ./redis-backup.rdb
```

### Upgrade Procedures

1. **Test in staging environment first**
2. **Backup databases before upgrade**
3. **Use rolling updates** (already configured)
4. **Monitor health checks during upgrade**
5. **Keep previous version for quick rollback**

---

## Additional Resources

- [Kubernetes Official Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Production Checklist](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/)
