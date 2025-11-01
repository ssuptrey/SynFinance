# Week 9 Day 2: Kubernetes Orchestration

**Date:** November 2, 2025  
**Status:** 🚀 **STARTING**  
**Focus:** Production Kubernetes deployment with autoscaling and monitoring

---

## Overview

Building on our Docker containerization from Day 1, we'll now deploy SynFinance to Kubernetes with production-ready configurations including deployments, services, autoscaling, and ingress.

---

## Prerequisites (From Day 1) ✅

- ✅ Docker image built: `synfinance:2.15.0`
- ✅ Docker Compose tested and working
- ✅ Health endpoints implemented
- ✅ All 21/22 tests passing
- ✅ 3 services running (API, PostgreSQL, Redis)

---

## Objectives

### Primary Goals
1. Create Kubernetes Deployment manifests
2. Configure Service discovery and load balancing
3. Implement Horizontal Pod Autoscaling (HPA)
4. Set up ConfigMaps and Secrets
5. Configure Ingress for external access
6. Add Persistent Volume Claims (PVCs)
7. Implement resource limits and requests
8. Set up liveness and readiness probes
9. Create Helm charts for package management

### Success Criteria
- ✅ All services deploy to Kubernetes successfully
- ✅ Pods are healthy and passing readiness checks
- ✅ Autoscaling responds to load
- ✅ Persistent data survives pod restarts
- ✅ External access works through Ingress
- ✅ Resource limits prevent resource exhaustion
- ✅ Helm charts simplify deployment
- ✅ Rolling updates work without downtime

---

## Implementation Plan

### Phase 1: Kubernetes Manifests (Basic)

#### 1.1 Namespace
**File:** `k8s/namespace.yaml`

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: synfinance
  labels:
    app: synfinance
    environment: production
```

#### 1.2 ConfigMap
**File:** `k8s/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: synfinance-config
  namespace: synfinance
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  LOG_LEVEL: "INFO"
  ENABLE_GRAPHQL: "true"
  ENABLE_WEBSOCKET: "true"
  ENABLE_MULTITENANCY: "true"
```

#### 1.3 Secrets
**File:** `k8s/secrets.yaml`

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: synfinance-secrets
  namespace: synfinance
type: Opaque
stringData:
  DATABASE_URL: "postgresql://synfinance_trey:password@postgres:5432/synfinance"
  REDIS_URL: "redis://redis:6379/0"
  SECRET_KEY: "your-secret-key-change-in-production"
  JWT_SECRET: "your-jwt-secret-change-in-production"
```

---

### Phase 2: Database & Cache (StatefulSets)

#### 2.1 PostgreSQL StatefulSet
**File:** `k8s/postgres-statefulset.yaml`

**Components:**
- StatefulSet for PostgreSQL
- Persistent Volume Claim (10GB)
- Service (ClusterIP)
- Init container for database setup

**Key Features:**
- Ordered pod deployment
- Stable network identity
- Persistent storage
- Health checks

#### 2.2 Redis StatefulSet
**File:** `k8s/redis-statefulset.yaml`

**Components:**
- StatefulSet for Redis
- Persistent Volume Claim (1GB)
- Service (ClusterIP)
- Memory limits (256MB)

---

### Phase 3: API Deployment

#### 3.1 API Deployment
**File:** `k8s/api-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: synfinance-api
  namespace: synfinance
spec:
  replicas: 3  # Start with 3 pods
  selector:
    matchLabels:
      app: synfinance-api
  template:
    metadata:
      labels:
        app: synfinance-api
        version: "2.15.0"
    spec:
      containers:
      - name: api
        image: synfinance:2.15.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        
        # Environment from ConfigMap
        envFrom:
        - configMapRef:
            name: synfinance-config
        
        # Secrets
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: synfinance-secrets
              key: DATABASE_URL
        
        # Resource limits
        resources:
          requests:
            cpu: 250m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 2Gi
        
        # Liveness probe
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        # Readiness probe
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
```

**Key Features:**
- 3 replicas for high availability
- Resource requests and limits
- ConfigMap and Secret integration
- Liveness and readiness probes
- Rolling update strategy

---

### Phase 4: Services

#### 4.1 API Service
**File:** `k8s/api-service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: synfinance-api
  namespace: synfinance
  labels:
    app: synfinance-api
spec:
  type: ClusterIP
  selector:
    app: synfinance-api
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http
  sessionAffinity: ClientIP  # Sticky sessions
```

#### 4.2 PostgreSQL Service
**File:** `k8s/postgres-service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: synfinance
spec:
  type: ClusterIP
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

#### 4.3 Redis Service
**File:** `k8s/redis-service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: synfinance
spec:
  type: ClusterIP
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

---

### Phase 5: Autoscaling

#### 5.1 Horizontal Pod Autoscaler
**File:** `k8s/hpa.yaml`

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: synfinance-api-hpa
  namespace: synfinance
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: synfinance-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
```

**Features:**
- Min 3 replicas, max 10
- CPU target: 70%
- Memory target: 80%
- Controlled scale-up (50% increase)
- Gradual scale-down (1 pod at a time)

---

### Phase 6: Ingress

#### 6.1 Ingress Controller
**File:** `k8s/ingress.yaml`

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: synfinance-ingress
  namespace: synfinance
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.synfinance.com
    secretName: synfinance-tls
  rules:
  - host: api.synfinance.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: synfinance-api
            port:
              number: 8000
```

**Features:**
- NGINX ingress controller
- TLS/SSL termination
- Domain-based routing
- Path-based routing support

---

### Phase 7: Persistent Volumes

#### 7.1 Storage Class
**File:** `k8s/storage-class.yaml`

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: synfinance-storage
provisioner: kubernetes.io/aws-ebs  # Or appropriate for your cloud
parameters:
  type: gp3
  iopsPerGB: "10"
  encrypted: "true"
reclaimPolicy: Retain
allowVolumeExpansion: true
```

#### 7.2 Persistent Volume Claims
**File:** `k8s/pvcs.yaml`

```yaml
---
# PostgreSQL PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: synfinance
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: synfinance-storage
  resources:
    requests:
      storage: 10Gi
---
# Redis PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: synfinance
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: synfinance-storage
  resources:
    requests:
      storage: 1Gi
```

---

### Phase 8: Helm Charts

#### 8.1 Chart Structure
```
helm/
└── synfinance/
    ├── Chart.yaml
    ├── values.yaml
    ├── values-dev.yaml
    ├── values-prod.yaml
    └── templates/
        ├── namespace.yaml
        ├── configmap.yaml
        ├── secrets.yaml
        ├── api-deployment.yaml
        ├── api-service.yaml
        ├── postgres-statefulset.yaml
        ├── postgres-service.yaml
        ├── redis-statefulset.yaml
        ├── redis-service.yaml
        ├── hpa.yaml
        ├── ingress.yaml
        └── pvcs.yaml
```

#### 8.2 Chart.yaml
**File:** `helm/synfinance/Chart.yaml`

```yaml
apiVersion: v2
name: synfinance
description: SynFinance Fraud Detection Platform
type: application
version: 2.15.0
appVersion: "2.15.0"
keywords:
  - fraud-detection
  - machine-learning
  - financial-analytics
maintainers:
  - name: SynFinance Team
    email: team@synfinance.com
```

#### 8.3 values.yaml
**File:** `helm/synfinance/values.yaml`

```yaml
# Global settings
global:
  environment: production
  namespace: synfinance

# API settings
api:
  replicaCount: 3
  image:
    repository: synfinance
    tag: "2.15.0"
    pullPolicy: IfNotPresent
  
  resources:
    requests:
      cpu: 250m
      memory: 512Mi
    limits:
      cpu: 1000m
      memory: 2Gi
  
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80

# PostgreSQL settings
postgres:
  enabled: true
  image: postgres:14-alpine
  storage: 10Gi
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 2Gi

# Redis settings
redis:
  enabled: true
  image: redis:7-alpine
  storage: 1Gi
  maxMemory: 256mb
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 256Mi

# Ingress settings
ingress:
  enabled: true
  className: nginx
  host: api.synfinance.com
  tls:
    enabled: true
    secretName: synfinance-tls
```

---

## Testing Strategy

### 1. Unit Tests
**File:** `tests/deployment/test_kubernetes.py`

```python
class TestKubernetesManifests:
    def test_namespace_yaml_valid(self):
        """Test namespace manifest is valid YAML"""
        
    def test_deployment_yaml_valid(self):
        """Test deployment manifests are valid"""
        
    def test_service_yaml_valid(self):
        """Test service manifests are valid"""
        
    def test_configmap_yaml_valid(self):
        """Test ConfigMap is valid"""
        
    def test_secrets_yaml_valid(self):
        """Test Secrets manifest is valid"""
        
    def test_hpa_yaml_valid(self):
        """Test HPA manifest is valid"""
        
    def test_ingress_yaml_valid(self):
        """Test Ingress manifest is valid"""
```

### 2. Integration Tests
```python
class TestKubernetesDeployment:
    def test_namespace_created(self):
        """Test namespace is created"""
        
    def test_pods_running(self):
        """Test all pods are in Running state"""
        
    def test_services_accessible(self):
        """Test services are accessible"""
        
    def test_health_checks_passing(self):
        """Test liveness and readiness probes pass"""
        
    def test_autoscaling_works(self):
        """Test HPA scales pods based on load"""
        
    def test_persistent_data(self):
        """Test data persists across pod restarts"""
        
    def test_rolling_update(self):
        """Test rolling updates work without downtime"""
```

---

## Deployment Steps

### Step 1: Setup (Local - Minikube or Kind)

```bash
# Install Minikube (for local testing)
# Windows: choco install minikube
# Mac: brew install minikube

# Start Minikube
minikube start --memory=8192 --cpus=4

# Enable ingress addon
minikube addons enable ingress
minikube addons enable metrics-server

# Load Docker image to Minikube
minikube image load synfinance:2.15.0
```

### Step 2: Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply ConfigMap and Secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Deploy database and cache
kubectl apply -f k8s/postgres-statefulset.yaml
kubectl apply -f k8s/postgres-service.yaml
kubectl apply -f k8s/redis-statefulset.yaml
kubectl apply -f k8s/redis-service.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n synfinance --timeout=300s

# Deploy API
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml

# Apply autoscaling
kubectl apply -f k8s/hpa.yaml

# Apply ingress
kubectl apply -f k8s/ingress.yaml
```

### Step 3: Verify Deployment

```bash
# Check pods
kubectl get pods -n synfinance

# Check services
kubectl get svc -n synfinance

# Check HPA
kubectl get hpa -n synfinance

# Check ingress
kubectl get ingress -n synfinance

# View logs
kubectl logs -f deployment/synfinance-api -n synfinance

# Test health endpoint
kubectl port-forward svc/synfinance-api 8000:8000 -n synfinance
curl http://localhost:8000/health
```

### Step 4: Helm Deployment (Alternative)

```bash
# Install/upgrade with Helm
helm upgrade --install synfinance ./helm/synfinance \
  --namespace synfinance \
  --create-namespace \
  --values helm/synfinance/values-prod.yaml

# Check release
helm list -n synfinance

# Rollback if needed
helm rollback synfinance -n synfinance
```

---

## File Structure

```
e:\SynFinance\
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── postgres-statefulset.yaml
│   ├── postgres-service.yaml
│   ├── redis-statefulset.yaml
│   ├── redis-service.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   ├── storage-class.yaml
│   └── pvcs.yaml
│
├── helm/
│   └── synfinance/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-dev.yaml
│       ├── values-prod.yaml
│       └── templates/
│           └── [all K8s manifests]
│
└── tests/
    └── deployment/
        └── test_kubernetes.py
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Pod Startup | < 60s | `kubectl get pods` |
| Health Check | 100% | Liveness/readiness probes |
| Autoscaling | 3-10 pods | HPA response to load |
| Uptime | 99.9% | Rolling updates |
| API Response | < 200ms | /health endpoint |
| Resource Usage | < 80% | kubectl top |

---

## Timeline

- **Kubernetes Manifests:** 2 hours
- **Helm Charts:** 1.5 hours
- **Testing:** 1.5 hours
- **Documentation:** 1 hour
- **Deployment & Verification:** 1 hour

**Total Estimated Time:** 6-7 hours

---

## Next Steps (Day 3)

After Day 2 completion:
1. CI/CD pipeline integration
2. GitOps with ArgoCD/FluxCD
3. Monitoring with Prometheus/Grafana
4. Logging with ELK stack
5. Service mesh (Istio) integration

---

**Day 2 Status:** 🚀 **READY TO START**

**Let's deploy to Kubernetes!**
