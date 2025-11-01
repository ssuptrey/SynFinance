# Istio Service Mesh Guide for SynFinance

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Traffic Management](#traffic-management)
4. [Security](#security)
5. [Resilience](#resilience)
6. [Observability](#observability)
7. [Operations](#operations)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Overview

Istio provides a service mesh layer that adds:
- **Traffic Management**: Canary deployments, A/B testing, traffic splitting
- **Security**: mTLS, authorization policies, authentication
- **Resilience**: Circuit breakers, retries, timeouts, rate limiting
- **Observability**: Distributed tracing, metrics, service topology

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   Istio Control Plane                   │
│                      (istiod)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │  Pilot   │  │ Citadel  │  │  Galley (Config) │    │
│  │(Traffic) │  │ (mTLS)   │  │                  │    │
│  └──────────┘  └──────────┘  └──────────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │ xDS Protocol
       ┌─────────────┴──────────────┐
       │                            │
┌──────▼──────┐            ┌────────▼────────┐
│   Ingress   │            │   Data Plane    │
│   Gateway   │            │    (Envoy)      │
│   (Envoy)   │            │                 │
└──────┬──────┘            └────────┬────────┘
       │                            │
       │ HTTP/HTTPS                 │
       │                            │
┌──────▼────────────────────────────▼──────┐
│          Application Pods                │
│  ┌────────┐  ┌────────┐  ┌────────┐    │
│  │  App   │  │  App   │  │  App   │    │
│  │ +Envoy │  │ +Envoy │  │ +Envoy │    │
│  └────────┘  └────────┘  └────────┘    │
└──────────────────────────────────────────┘
```

### Components

- **istiod**: Control plane managing configuration
- **Envoy Proxy**: Data plane sidecar for traffic control
- **Ingress Gateway**: Entry point for external traffic
- **Egress Gateway**: Exit point for external calls
- **Kiali**: Service mesh visualization
- **Jaeger**: Distributed tracing

---

## Traffic Management

### 1. Gateway Configuration

**Purpose**: Define entry points for external traffic.

```yaml
# k8s/istio/gateway.yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: synfinance-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    hosts:
    - "api.synfinance.com"
    tls:
      mode: SIMPLE
      credentialName: synfinance-tls-cert
```

**Access the gateway:**
```bash
# Get external IP
kubectl get svc istio-ingressgateway -n istio-system

# Or port-forward for testing
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
curl http://localhost:8080/health
```

### 2. VirtualService (Routing Rules)

**Purpose**: Define how traffic is routed to services.

#### Canary Deployment (10% new version)

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: synfinance-api
spec:
  hosts:
  - api.synfinance.com
  http:
  - route:
    - destination:
        host: synfinance-api
        subset: stable
      weight: 90
    - destination:
        host: synfinance-api
        subset: canary
      weight: 10
```

**Apply and test:**
```bash
kubectl apply -f k8s/istio/virtualservice.yaml

# Generate traffic
for i in {1..100}; do curl http://$GATEWAY_URL/api/v1/health; done

# View traffic distribution in Kiali
istioctl dashboard kiali
```

#### Header-Based Routing (A/B Testing)

```yaml
http:
- match:
  - headers:
      x-beta-user:
        exact: "true"
  route:
  - destination:
      host: synfinance-api
      subset: canary

- route:
  - destination:
      host: synfinance-api
      subset: stable
```

**Test:**
```bash
# Beta users get canary version
curl -H "x-beta-user: true" http://$GATEWAY_URL/api/v1/users

# Regular users get stable version
curl http://$GATEWAY_URL/api/v1/users
```

### 3. DestinationRule (Load Balancing & Resilience)

**Purpose**: Configure load balancing, circuit breakers, and connection pools.

#### Load Balancing Strategies

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: synfinance-api
spec:
  host: synfinance-api
  trafficPolicy:
    loadBalancer:
      simple: LEAST_REQUEST  # Options: ROUND_ROBIN, RANDOM, PASSTHROUGH, LEAST_REQUEST
```

**Verify:**
```bash
# Check which backend receives requests
kubectl logs -n synfinance-production -l app=synfinance-api --tail=10
```

### 4. Traffic Mirroring (Shadow Traffic)

**Purpose**: Send copy of production traffic to canary for testing without impact.

```yaml
http:
- route:
  - destination:
      host: synfinance-api
      subset: stable
  mirror:
    host: synfinance-api
    subset: canary
  mirrorPercentage:
    value: 10  # Mirror 10% of traffic
```

---

## Security

### 1. mTLS (Mutual TLS)

**Purpose**: Encrypt all service-to-service communication.

#### Enable Strict mTLS

```yaml
# k8s/istio/peer-authentication.yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: synfinance-production
spec:
  mtls:
    mode: STRICT  # All traffic must use mTLS
```

**Apply and verify:**
```bash
kubectl apply -f k8s/istio/peer-authentication.yaml

# Check mTLS status
istioctl authn tls-check synfinance-api.synfinance-production.svc.cluster.local

# Expected output:
# HOST:PORT                                               STATUS     SERVER     CLIENT     AUTHN POLICY     DESTINATION RULE
# synfinance-api.synfinance-production.svc.cluster.local  OK         STRICT     ISTIO_MUTUAL    default/synfinance-production     synfinance-api/synfinance-production
```

#### View Certificates

```bash
# Check certificate details
istioctl proxy-config secret synfinance-api-xxx -n synfinance-production

# Verify certificate rotation (automatic every 24h)
kubectl logs -n istio-system istiod-xxx | grep "CSR signed"
```

### 2. Authorization Policies

**Purpose**: Control which services can communicate.

#### Default Deny All

```yaml
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata:
  name: deny-all
  namespace: synfinance-production
spec:
  {}  # Empty = deny all
```

#### Allow Specific Services

```yaml
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata:
  name: allow-api-to-postgres
spec:
  selector:
    matchLabels:
      app: postgres
  action: ALLOW
  rules:
  - from:
    - source:
        principals:
        - cluster.local/ns/synfinance-production/sa/synfinance-api
    to:
    - operation:
        ports:
        - "5432"
```

**Apply and test:**
```bash
kubectl apply -f k8s/istio/authorization-policy.yaml

# Test blocked access (should fail)
kubectl run test-pod --rm -it --image=postgres:15 -- psql -h postgres -U synfinance

# Test allowed access from API pod (should succeed)
kubectl exec -it synfinance-api-xxx -- curl postgres:5432
```

#### JWT Authentication

```yaml
apiVersion: security.istio.io/v1
kind: RequestAuthentication
metadata:
  name: jwt-auth
spec:
  selector:
    matchLabels:
      app: synfinance-api
  jwtRules:
  - issuer: "https://auth.synfinance.com"
    jwksUri: "https://auth.synfinance.com/.well-known/jwks.json"
```

**Test:**
```bash
# Without JWT (should fail on protected endpoints)
curl http://$GATEWAY_URL/api/v1/users

# With JWT
curl -H "Authorization: Bearer $JWT_TOKEN" http://$GATEWAY_URL/api/v1/users
```

---

## Resilience

### 1. Circuit Breaker

**Purpose**: Prevent cascading failures by stopping requests to failing services.

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: synfinance-api
spec:
  host: synfinance-api
  trafficPolicy:
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

**How it works:**
- After 5 consecutive 5xx errors, the backend is ejected
- Ejection lasts for 30 seconds (baseEjectionTime)
- Check interval: every 30 seconds
- Max 50% of backends can be ejected

**Test circuit breaker:**
```bash
# Simulate failures in canary version
kubectl exec -it synfinance-api-canary-xxx -- kill 1

# Generate traffic
for i in {1..100}; do curl http://$GATEWAY_URL/api/v1/health; sleep 0.1; done

# Check ejected backends
istioctl proxy-config endpoint synfinance-api-xxx | grep canary
# Ejected backends show: HEALTHY:FAILED_ACTIVE_HC
```

### 2. Retries

**Purpose**: Automatically retry failed requests.

```yaml
http:
- route:
  - destination:
      host: synfinance-api
  retries:
    attempts: 3
    perTryTimeout: 2s
    retryOn: 5xx,reset,connect-failure
```

**Retry conditions:**
- `5xx`: Server errors
- `reset`: Connection reset
- `connect-failure`: Connection failed
- `refused-stream`: HTTP/2 refused stream
- `retriable-4xx`: 409 Conflict, 429 Too Many Requests

### 3. Timeouts

**Purpose**: Prevent slow requests from blocking resources.

```yaml
http:
- route:
  - destination:
      host: synfinance-api
  timeout: 30s  # Overall request timeout
  retries:
    attempts: 3
    perTryTimeout: 5s  # Per-retry timeout
```

### 4. Connection Pool

**Purpose**: Limit concurrent connections to prevent overload.

```yaml
trafficPolicy:
  connectionPool:
    tcp:
      maxConnections: 100
      connectTimeout: 3s
    http:
      http1MaxPendingRequests: 50
      http2MaxRequests: 100
      maxRequestsPerConnection: 2
      maxRetries: 3
```

**Monitor connection pools:**
```bash
# View connection pool metrics
kubectl exec -it synfinance-api-xxx -c istio-proxy -- curl localhost:15000/stats | grep upstream_cx
```

### 5. Rate Limiting (Advanced)

**Purpose**: Limit requests per second to protect services.

*Note: Requires EnvoyFilter or external rate limiter. Example:*

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: ratelimit-filter
spec:
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: SIDECAR_INBOUND
    patch:
      operation: INSERT_BEFORE
      value:
        name: envoy.filters.http.local_ratelimit
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit
          stat_prefix: http_local_rate_limiter
          token_bucket:
            max_tokens: 100
            tokens_per_fill: 10
            fill_interval: 1s
```

---

## Observability

### 1. Distributed Tracing

Istio automatically generates spans for all HTTP traffic.

**View traces in Jaeger:**
```bash
istioctl dashboard jaeger

# Or port-forward
kubectl port-forward -n istio-system svc/tracing 16686:80
# Visit: http://localhost:16686
```

**Search for traces:**
1. Service: `synfinance-api.synfinance-production`
2. Operation: `GET /api/v1/users`
3. Tags: `http.status_code=200`
4. Lookback: Last hour

**Trace includes:**
- Request headers (sanitized)
- Response code
- Duration per service
- Parent-child relationships

### 2. Metrics

Istio generates metrics for all traffic:

**Key metrics:**
- `istio_requests_total{destination_service="synfinance-api"}`
- `istio_request_duration_milliseconds`
- `istio_request_bytes` / `istio_response_bytes`
- `istio_tcp_connections_opened_total`

**Query in Prometheus:**
```bash
kubectl port-forward -n istio-system svc/prometheus 9090:9090
# Visit: http://localhost:9090
```

**Example queries:**
```promql
# Request rate by service
rate(istio_requests_total{destination_service="synfinance-api"}[5m])

# Error rate
rate(istio_requests_total{destination_service="synfinance-api",response_code=~"5.."}[5m])

# P95 latency
histogram_quantile(0.95, rate(istio_request_duration_milliseconds_bucket[5m]))

# Traffic by version
sum(rate(istio_requests_total{destination_service="synfinance-api"}[5m])) by (destination_version)
```

### 3. Kiali (Service Graph)

Kiali visualizes service mesh topology and traffic flow.

**Access Kiali:**
```bash
istioctl dashboard kiali
# Visit: http://localhost:20001
```

**Features:**
- **Graph**: Real-time traffic flow visualization
- **Applications**: List of apps with health status
- **Workloads**: Pods with sidecar status
- **Services**: Service list with VirtualServices
- **Istio Config**: Validation of all Istio configs

**Graph Views:**
- **App Graph**: By application
- **Versioned App Graph**: Show versions (stable/canary)
- **Workload Graph**: By workload/deployment
- **Service Graph**: By service

**Traffic Metrics in Graph:**
- Request rate (req/sec)
- Error rate (%)
- Response time (P50, P95, P99)

### 4. Envoy Admin Interface

**Access Envoy admin:**
```bash
kubectl exec -it synfinance-api-xxx -c istio-proxy -- curl localhost:15000/help

# View configuration
kubectl exec -it synfinance-api-xxx -c istio-proxy -- curl localhost:15000/config_dump

# View stats
kubectl exec -it synfinance-api-xxx -c istio-proxy -- curl localhost:15000/stats

# View clusters (backends)
kubectl exec -it synfinance-api-xxx -c istio-proxy -- curl localhost:15000/clusters
```

---

## Operations

### 1. Deploy Canary Version

**Step 1: Deploy canary pods**
```bash
# Update deployment with version=canary label
kubectl set image deployment/synfinance-api-canary api=synfinance:2.16.0 -n synfinance-production
kubectl label deployment synfinance-api-canary version=canary
```

**Step 2: Send 10% traffic to canary**
```yaml
# Already configured in virtualservice.yaml
http:
- route:
  - destination:
      host: synfinance-api
      subset: stable
    weight: 90
  - destination:
      host: synfinance-api
      subset: canary
    weight: 10
```

**Step 3: Monitor canary metrics**
```promql
# Error rate comparison
rate(istio_requests_total{destination_version="stable",response_code=~"5.."}[5m])
rate(istio_requests_total{destination_version="canary",response_code=~"5.."}[5m])

# Latency comparison
histogram_quantile(0.95, rate(istio_request_duration_milliseconds_bucket{destination_version="stable"}[5m]))
histogram_quantile(0.95, rate(istio_request_duration_milliseconds_bucket{destination_version="canary"}[5m]))
```

**Step 4: Promote or rollback**

**Promote canary (if successful):**
```bash
# Increase traffic to 50%
kubectl patch virtualservice synfinance-api --type=json -p='[{"op": "replace", "path": "/spec/http/0/route/0/weight", "value": 50}]'

# Monitor...

# Full promotion: 100% to canary
kubectl patch virtualservice synfinance-api --type=json -p='[{"op": "replace", "path": "/spec/http/0/route/0/weight", "value": 0}]'
kubectl patch virtualservice synfinance-api --type=json -p='[{"op": "replace", "path": "/spec/http/0/route/1/weight", "value": 100}]'

# Update stable deployment
kubectl set image deployment/synfinance-api api=synfinance:2.16.0 -n synfinance-production

# Revert canary traffic
kubectl patch virtualservice synfinance-api --type=json -p='[{"op": "replace", "path": "/spec/http/0/route/0/weight", "value": 90}]'
kubectl patch virtualservice synfinance-api --type=json -p='[{"op": "replace", "path": "/spec/http/0/route/1/weight", "value": 10}]'
```

**Rollback canary (if issues):**
```bash
# Remove all canary traffic
kubectl patch virtualservice synfinance-api --type=json -p='[{"op": "replace", "path": "/spec/http/0/route/1/weight", "value": 0}]'

# Delete canary deployment
kubectl delete deployment synfinance-api-canary -n synfinance-production
```

### 2. Blue-Green Deployment

**Step 1: Deploy green version**
```bash
# Create green deployment
kubectl create deployment synfinance-api-green --image=synfinance:2.16.0 -n synfinance-production
kubectl label deployment synfinance-api-green app=synfinance-api version=green
```

**Step 2: Switch all traffic to green**
```yaml
http:
- route:
  - destination:
      host: synfinance-api
      subset: green
    weight: 100
```

**Step 3: Verify and delete blue**
```bash
# After verification
kubectl delete deployment synfinance-api-blue -n synfinance-production
```

### 3. Fault Injection (Testing)

**Inject delays:**
```yaml
http:
- fault:
    delay:
      percentage:
        value: 10
      fixedDelay: 5s
  route:
  - destination:
      host: synfinance-api
```

**Inject errors:**
```yaml
http:
- fault:
    abort:
      percentage:
        value: 5
      httpStatus: 503
  route:
  - destination:
      host: synfinance-api
```

**Test:**
```bash
# Some requests will fail or be slow
for i in {1..100}; do curl -w "@curl-format.txt" http://$GATEWAY_URL/api/v1/health; done
```

---

## Troubleshooting

### 1. Sidecar Not Injecting

**Symptoms:** Pods have 1/1 containers instead of 2/2.

**Check:**
```bash
# Verify namespace label
kubectl get namespace synfinance-production --show-labels

# Check admission webhook
kubectl get mutatingwebhookconfiguration istio-sidecar-injector -o yaml

# View injection policy
kubectl get deployment synfinance-api -o yaml | grep sidecar.istio.io/inject
```

**Fix:**
```bash
# Label namespace
kubectl label namespace synfinance-production istio-injection=enabled

# Restart pods
kubectl rollout restart deployment/synfinance-api -n synfinance-production
```

### 2. mTLS Connection Errors

**Symptoms:** `503 Service Unavailable`, logs show `TLS error`.

**Check:**
```bash
# Verify mTLS status
istioctl authn tls-check synfinance-api.synfinance-production.svc.cluster.local

# Check PeerAuthentication
kubectl get peerauthentication -n synfinance-production

# View proxy logs
kubectl logs synfinance-api-xxx -c istio-proxy -n synfinance-production
```

**Fix:**
```bash
# If mismatch, set to PERMISSIVE temporarily
kubectl patch peerauthentication default -n synfinance-production --type=merge -p '{"spec":{"mtls":{"mode":"PERMISSIVE"}}}'

# Investigate which service doesn't have sidecar
kubectl get pods -n synfinance-production -o wide

# Fix and switch back to STRICT
kubectl patch peerauthentication default -n synfinance-production --type=merge -p '{"spec":{"mtls":{"mode":"STRICT"}}}'
```

### 3. High Latency After Istio

**Symptoms:** Requests are slower after Istio installation.

**Check:**
```bash
# View Envoy overhead
kubectl top pods -n synfinance-production

# Check access logs (can be disabled)
kubectl logs synfinance-api-xxx -c istio-proxy --tail=100

# View stats
kubectl exec synfinance-api-xxx -c istio-proxy -- curl localhost:15000/stats/prometheus | grep latency
```

**Optimize:**
```bash
# Disable access logs if too verbose
kubectl edit configmap istio -n istio-system
# Set: accessLogFile: ""

# Reduce telemetry sampling
kubectl edit configmap istio -n istio-system
# Set: enablePrometheusMerge: false

# Adjust sidecar resources
kubectl patch deployment synfinance-api --type=json -p='[
  {"op": "add", "path": "/spec/template/metadata/annotations/sidecar.istio.io~1proxyCPU", "value": "200m"},
  {"op": "add", "path": "/spec/template/metadata/annotations/sidecar.istio.io~1proxyMemory", "value": "256Mi"}
]'
```

### 4. Gateway Not Accessible

**Symptoms:** Can't reach services via ingress gateway.

**Check:**
```bash
# Gateway status
kubectl get gateway -n synfinance-production

# Ingress gateway pods
kubectl get pods -n istio-system -l app=istio-ingressgateway

# Logs
kubectl logs -n istio-system -l app=istio-ingressgateway --tail=50

# Check VirtualService binding
kubectl get virtualservice synfinance-api -o yaml | grep -A5 gateways
```

**Fix:**
```bash
# Verify hosts match
kubectl get gateway synfinance-gateway -o yaml | grep hosts
kubectl get virtualservice synfinance-api -o yaml | grep hosts

# Check external IP/LoadBalancer
kubectl get svc istio-ingressgateway -n istio-system

# If pending, use NodePort or port-forward
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
```

### 5. Authorization Policy Blocking Traffic

**Symptoms:** `RBAC: access denied` in logs, 403 responses.

**Check:**
```bash
# List all AuthorizationPolicies
kubectl get authorizationpolicy -n synfinance-production

# Check logs
kubectl logs synfinance-api-xxx -c istio-proxy | grep RBAC

# Analyze config
istioctl analyze -n synfinance-production
```

**Fix:**
```bash
# Temporarily disable deny-all to debug
kubectl delete authorizationpolicy deny-all -n synfinance-production

# Test access
curl http://$GATEWAY_URL/api/v1/health

# Re-enable with proper allow policies
kubectl apply -f k8s/istio/authorization-policy.yaml
```

---

## Best Practices

### 1. Start with PERMISSIVE mTLS

During migration, use PERMISSIVE mode to allow both mTLS and plain text:

```yaml
spec:
  mtls:
    mode: PERMISSIVE  # Allows both mTLS and plaintext
```

After all services have sidecars, switch to STRICT.

### 2. Use Subsets for Versioning

Always label deployments with `version` and define subsets in DestinationRule:

```yaml
labels:
  version: stable  # or canary, blue, green, v1, v2, etc.
```

### 3. Set Resource Limits on Sidecars

Prevent sidecar resource spikes:

```yaml
annotations:
  sidecar.istio.io/proxyCPU: "100m"
  sidecar.istio.io/proxyMemory: "128Mi"
  sidecar.istio.io/proxyCPULimit: "200m"
  sidecar.istio.io/proxyMemoryLimit: "256Mi"
```

### 4. Use Circuit Breakers for External Services

Protect against slow/failing external APIs:

```yaml
outlierDetection:
  consecutive5xxErrors: 3
  interval: 10s
  baseEjectionTime: 60s
```

### 5. Monitor Istio Metrics

Key alerts:
- High error rate: `rate(istio_requests_total{response_code=~"5.."}[5m]) > 0.01`
- High latency: `histogram_quantile(0.95, rate(istio_request_duration_milliseconds_bucket[5m])) > 1000`
- Circuit breaker tripped: `envoy_cluster_upstream_rq_pending_overflow > 0`

### 6. Use Telemetry v2 for Better Performance

Istio 1.18+ uses Telemetry v2 by default (more efficient).

### 7. Limit Access Logs in Production

Access logs can be verbose. Disable or sample:

```yaml
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: mesh-default
spec:
  accessLogging:
  - providers:
    - name: envoy
    disabled: true  # Disable completely
    # Or sample 1%
    # filter:
    #   expression: random(100) < 1
```

### 8. Use Locality-Based Load Balancing

Route traffic to nearby instances first:

```yaml
trafficPolicy:
  loadBalancer:
    localityLbSetting:
      enabled: true
      failover:
      - from: us-east-1
        to: us-west-2
```

### 9. Test with Fault Injection

Before production, test resilience:

```yaml
fault:
  delay:
    percentage:
      value: 10
    fixedDelay: 5s
  abort:
    percentage:
      value: 5
    httpStatus: 503
```

### 10. Version Your Istio Configs

Store all Istio manifests in Git and use GitOps (ArgoCD).

---

## Summary

**Traffic Management:**
- Gateway → Entry point
- VirtualService → Routing rules
- DestinationRule → Load balancing, resilience

**Security:**
- PeerAuthentication → mTLS
- AuthorizationPolicy → Access control
- RequestAuthentication → JWT validation

**Resilience:**
- Circuit breaker → Prevent cascading failures
- Retries → Auto-retry failed requests
- Timeouts → Prevent hanging requests
- Connection pools → Limit concurrent connections

**Observability:**
- Kiali → Service graph
- Jaeger → Distributed tracing
- Prometheus → Metrics
- Grafana → Dashboards

---

**Next Steps:**
1. Install Istio following [INSTALL.md](./INSTALL.md)
2. Apply manifests: `kubectl apply -f k8s/istio/`
3. Verify with Kiali: `istioctl dashboard kiali`
4. Run canary deployment test
5. Monitor metrics in Grafana

**Questions? See [Troubleshooting](#troubleshooting) or:** https://istio.io/latest/docs/
