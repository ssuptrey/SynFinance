# Istio Installation Guide for SynFinance

## Prerequisites

- Kubernetes cluster running (v1.25+)
- `kubectl` configured
- `helm` installed (v3.0+)
- Cluster has at least 4 CPU cores and 8GB RAM available

---

## Step 1: Install Istio CLI

### Download Istio

```bash
# Download latest Istio (1.20.x)
curl -L https://istio.io/downloadIstio | sh -

# Move to Istio directory
cd istio-1.20.*

# Add istioctl to PATH
export PATH=$PWD/bin:$PATH

# Verify installation
istioctl version
```

### Windows (PowerShell)
```powershell
# Download Istio
curl -L https://istio.io/downloadIstio | sh -

# Add to PATH or use full path
.\istio-1.20.*\bin\istioctl.exe version
```

---

## Step 2: Install Istio

### Option A: Using istioctl (Recommended)

```bash
# Install with production profile
istioctl install --set profile=production -y

# Verify installation
istioctl verify-install

# Check components
kubectl get pods -n istio-system
```

**Expected output:**
```
NAME                                   READY   STATUS    RESTARTS   AGE
istiod-xxx                             1/1     Running   0          2m
istio-ingressgateway-xxx               1/1     Running   0          2m
istio-egressgateway-xxx                1/1     Running   0          2m
```

### Option B: Using Helm

```bash
# Add Istio Helm repository
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update

# Create namespace
kubectl create namespace istio-system

# Install Istio base
helm install istio-base istio/base -n istio-system

# Install istiod
helm install istiod istio/istiod -n istio-system --wait

# Install ingress gateway
helm install istio-ingressgateway istio/gateway -n istio-system
```

---

## Step 3: Enable Sidecar Injection

### Label Namespace for Automatic Injection

```bash
# For production
kubectl label namespace synfinance-production istio-injection=enabled

# For staging
kubectl label namespace synfinance-staging istio-injection=enabled

# Verify label
kubectl get namespace -L istio-injection
```

### Restart Pods to Inject Sidecars

```bash
# Restart API pods to get Envoy sidecars
kubectl rollout restart deployment/synfinance-api -n synfinance-production

# Wait for rollout
kubectl rollout status deployment/synfinance-api -n synfinance-production

# Verify pods now have 2 containers (app + envoy)
kubectl get pods -n synfinance-production
```

**Expected output:**
```
NAME                              READY   STATUS    RESTARTS   AGE
synfinance-api-xxx                2/2     Running   0          1m
```

---

## Step 4: Apply Istio Configurations

```bash
# Apply all Istio manifests
kubectl apply -f k8s/istio/

# Or apply individually:
kubectl apply -f k8s/istio/gateway.yaml
kubectl apply -f k8s/istio/virtualservice.yaml
kubectl apply -f k8s/istio/destinationrule.yaml
kubectl apply -f k8s/istio/peer-authentication.yaml
kubectl apply -f k8s/istio/authorization-policy.yaml
```

---

## Step 5: Verify Installation

### Check Istio Status

```bash
# Check control plane status
istioctl analyze -n synfinance-production

# Verify mTLS
istioctl authn tls-check synfinance-api.synfinance-production.svc.cluster.local

# Check proxy status
istioctl proxy-status
```

### Test Traffic Flow

```bash
# Get ingress gateway external IP
kubectl get svc istio-ingressgateway -n istio-system

# Test API endpoint
export GATEWAY_URL=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$GATEWAY_URL/health

# Or port-forward for testing
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
curl http://localhost:8080/health
```

---

## Step 6: Install Observability Add-ons

### Kiali (Service Mesh Dashboard)

```bash
# Install Kiali
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/kiali.yaml

# Access Kiali
istioctl dashboard kiali
# Or port-forward
kubectl port-forward -n istio-system svc/kiali 20001:20001
# Visit: http://localhost:20001
```

### Grafana (Istio Dashboards)

```bash
# Install Grafana with Istio dashboards
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/grafana.yaml

# Access Grafana
istioctl dashboard grafana
# Or port-forward
kubectl port-forward -n istio-system svc/grafana 3000:3000
# Visit: http://localhost:3000
```

### Jaeger (Distributed Tracing)

```bash
# Install Jaeger
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/jaeger.yaml

# Access Jaeger
istioctl dashboard jaeger
# Or port-forward
kubectl port-forward -n istio-system svc/tracing 16686:80
# Visit: http://localhost:16686
```

### Prometheus (Metrics)

```bash
# Install Prometheus (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/prometheus.yaml

# Verify scraping Istio metrics
kubectl port-forward -n istio-system svc/prometheus 9090:9090
# Visit: http://localhost:9090
# Query: istio_requests_total
```

---

## Step 7: Validate Setup

### Check All Components

```bash
# All Istio pods running
kubectl get pods -n istio-system

# Sidecars injected
kubectl get pods -n synfinance-production

# Services discovered
kubectl get svc -n synfinance-production

# VirtualServices
kubectl get virtualservice -n synfinance-production

# DestinationRules
kubectl get destinationrule -n synfinance-production

# PeerAuthentication
kubectl get peerauthentication -n synfinance-production
```

### Generate Traffic and View in Kiali

```bash
# Generate some traffic
for i in {1..100}; do curl -s http://$GATEWAY_URL/health; done

# Open Kiali and view service graph
istioctl dashboard kiali
# Navigate to: Graph → Select namespace: synfinance-production
```

---

## Troubleshooting

### Sidecar Not Injecting

```bash
# Check namespace label
kubectl get namespace synfinance-production --show-labels

# Manually inject (for testing)
istioctl kube-inject -f k8s/base/api-deployment.yaml | kubectl apply -f -
```

### mTLS Connection Errors

```bash
# Check mTLS status
istioctl authn tls-check synfinance-api.synfinance-production.svc.cluster.local

# View proxy logs
kubectl logs -n synfinance-production synfinance-api-xxx -c istio-proxy

# Check certificates
istioctl proxy-config secret synfinance-api-xxx -n synfinance-production
```

### Gateway Not Accessible

```bash
# Check gateway status
kubectl get gateway -n synfinance-production

# Check ingress gateway pods
kubectl get pods -n istio-system -l app=istio-ingressgateway

# View gateway logs
kubectl logs -n istio-system -l app=istio-ingressgateway
```

### High Latency After Istio

```bash
# Check proxy overhead
kubectl top pods -n synfinance-production

# Disable some features if needed (edit istio configmap)
kubectl edit configmap istio -n istio-system
# Set: accessLogFile: ""  # Disable access logs if too verbose
```

---

## Uninstallation (if needed)

```bash
# Delete Istio manifests
kubectl delete -f k8s/istio/

# Uninstall Istio
istioctl uninstall --purge -y

# Delete namespace
kubectl delete namespace istio-system

# Remove namespace labels
kubectl label namespace synfinance-production istio-injection-
```

---

## Production Checklist

- [ ] Istio installed with production profile
- [ ] Sidecars injected in all application pods
- [ ] mTLS enabled (STRICT mode)
- [ ] VirtualService configured for routing
- [ ] DestinationRule configured for resilience
- [ ] AuthorizationPolicy in place
- [ ] Kiali accessible for visualization
- [ ] Prometheus scraping Istio metrics
- [ ] Jaeger collecting traces
- [ ] Gateway external IP/DNS configured
- [ ] Resource limits set on sidecars
- [ ] Monitoring alerts configured

---

## Next Steps

After installation:
1. Configure traffic management (canary, A/B testing)
2. Set up security policies (fine-grained access control)
3. Add resilience patterns (circuit breakers, retries)
4. Monitor service mesh in Kiali
5. Analyze traffic patterns in Grafana

---

**Installation Time:** ~30-45 minutes  
**Difficulty:** Intermediate  
**Prerequisites Met:** ✅ All Week 9 Days 1-4 complete
