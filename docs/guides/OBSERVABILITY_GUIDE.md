# SynFinance Observability Guide

## Overview

SynFinance includes a comprehensive observability stack built on industry-standard open-source tools. This guide covers monitoring, logging, and distributed tracing.

## Architecture

```
┌─────────────────────────────────────┐
│     SynFinance API (FastAPI)        │
│  ┌──────────┐  ┌─────────────────┐ │
│  │ /metrics │  │  JSON Logs      │ │
│  │ endpoint │  │  (structured)   │ │
│  └────┬─────┘  └────────┬────────┘ │
│       │                  │          │
│       │  OpenTelemetry Traces      │
│       │         (W3C)    │          │
└───────┼──────────────────┼──────────┘
        │                  │
        ▼                  ▼
┌───────────────┐  ┌──────────────┐
│  Prometheus   │  │   Promtail   │
│   (metrics)   │  │ (log shipper)│
└───────┬───────┘  └──────┬───────┘
        │                  │
        │                  ▼
        │          ┌──────────────┐
        │          │     Loki     │
        │          │ (log storage)│
        │          └──────┬───────┘
        │                  │
        └──────────────────┴───────┐
                           │        │
                           ▼        ▼
                    ┌──────────────────┐
                    │     Grafana      │
                    │  (visualization) │
                    └──────────────────┘
                             ▲
                             │
                    ┌──────────────────┐
                    │   Jaeger/Tempo   │
                    │     (traces)     │
                    └──────────────────┘
```

---

## Metrics (Prometheus)

### Accessing Metrics

**Direct endpoint:**
```bash
curl http://localhost:8000/metrics
```

**Prometheus UI:**
```
http://prometheus.yourdomain.com
```

**Grafana dashboards:**
```
http://grafana.yourdomain.com
```

### Available Metrics

#### Business Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `synfinance_transactions_total` | Counter | Total transactions processed |
| `synfinance_fraud_detections_total` | Counter | Fraud detections by pattern type |
| `synfinance_fraud_detection_rate` | Gauge | Current fraud rate (0-1) |
| `synfinance_generation_rate_per_second` | Gauge | Transaction generation rate |

#### Performance Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `synfinance_api_request_duration_seconds` | Histogram | API request latency |
| `synfinance_ml_inference_duration_seconds` | Histogram | ML model inference time |
| `synfinance_db_query_duration_seconds` | Histogram | Database query duration |

#### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `synfinance_memory_usage_bytes` | Gauge | Memory usage |
| `synfinance_cpu_usage_percent` | Gauge | CPU usage |
| `synfinance_active_connections` | Gauge | WebSocket connections |
| `synfinance_db_connections_active` | Gauge | Active DB connections |

#### Error Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `synfinance_errors_total` | Counter | Errors by type |
| `synfinance_http_requests_total` | Counter | HTTP requests by status |
| `synfinance_validation_failures_total` | Counter | Validation failures |

### Example Queries

**Request rate:**
```promql
sum(rate(synfinance_http_requests_total[5m]))
```

**Error rate:**
```promql
sum(rate(synfinance_http_requests_total{status_code=~"5.."}[5m])) 
  / 
sum(rate(synfinance_http_requests_total[5m]))
```

**P95 latency:**
```promql
histogram_quantile(0.95, 
  sum(rate(synfinance_api_request_duration_seconds_bucket[5m])) by (le)
)
```

**Fraud detection rate:**
```promql
synfinance_fraud_detection_rate
```

---

## Logging

### Log Format

All logs are structured JSON for easy parsing:

```json
{
  "timestamp": 1698916800.123,
  "level": "INFO",
  "logger": "synfinance.api",
  "module": "api_server",
  "function": "predict_fraud",
  "line": 123,
  "message": "Fraud prediction completed",
  "service": "synfinance-api",
  "version": "2.15.0",
  "environment": "production",
  "request_id": "abc123-def456",
  "trace_id": "789ghi",
  "user_id": "user_123",
  "tenant_id": "tenant_456",
  "http_method": "POST",
  "http_path": "/predict",
  "http_status_code": 200,
  "duration_ms": 45.2
}
```

### Log Levels

- **DEBUG**: Development details, verbose output
- **INFO**: Business events (transaction created, fraud detected)
- **WARNING**: Degraded performance, retries
- **ERROR**: Errors requiring attention
- **CRITICAL**: Service-impacting issues

### Viewing Logs

**Kubernetes (kubectl):**
```bash
# Tail logs from all API pods
kubectl logs -n synfinance-production -l app=synfinance-api --tail=100 -f

# Logs from specific pod
kubectl logs -n synfinance-production synfinance-api-xyz123 -f

# Logs with grep
kubectl logs -n synfinance-production -l app=synfinance-api | grep "ERROR"
```

**Grafana (Loki):**
```
{app="synfinance-api"} |= "ERROR"
{app="synfinance-api"} | json | level="INFO"
{app="synfinance-api"} | json | request_id="abc123"
```

### Common Log Queries

**All errors in last hour:**
```
{app="synfinance-api"} | json | level="ERROR"
```

**Requests for specific user:**
```
{app="synfinance-api"} | json | user_id="user_123"
```

**Slow requests (>1s):**
```
{app="synfinance-api"} | json | duration_ms > 1000
```

**Trace a specific request:**
```
{app="synfinance-api"} | json | request_id="abc123-def456"
```

---

## Distributed Tracing

### Accessing Traces

**Jaeger UI:**
```
http://jaeger.yourdomain.com
```

**Tempo (via Grafana):**
```
http://grafana.yourdomain.com/explore
Select: Tempo datasource
```

### Trace Structure

Each HTTP request creates a trace with multiple spans:

```
Request Trace (root span)
├── HTTP Request Handling
├── Request Validation
├── Database Query
│   ├── Connection Acquisition
│   └── Query Execution
├── ML Model Inference
│   ├── Feature Extraction
│   ├── Model Prediction
│   └── Post-processing
└── Response Serialization
```

### Finding Traces

**By trace ID:**
- Copy `trace_id` from logs
- Search in Jaeger UI

**By operation:**
- Service: `synfinance-api`
- Operation: `POST /predict`

**By tag:**
- `http.status_code=500` (errors)
- `user_id=user_123`
- `fraud_detected=true`

### Span Attributes

Common attributes in spans:
- `http.method`
- `http.url`
- `http.status_code`
- `db.system` (postgresql)
- `db.statement` (SQL query)
- `ml.model_name`
- `user.id`
- `tenant.id`

---

## Dashboards

### Application Overview
**Location:** Grafana > Dashboards > SynFinance - Application Overview

**Panels:**
- Requests per minute
- Error rate
- P95 latency
- Active WebSocket connections
- Request rate by endpoint
- Latency percentiles (p50, p95, p99)

**Use cases:**
- Monitor overall API health
- Identify performance degradation
- Track request patterns

### Fraud Analytics
**Location:** Grafana > Dashboards > SynFinance - Fraud Analytics

**Panels:**
- Fraud detection rate
- Detections by pattern type
- ML model inference time
- Fraud trends over time

**Use cases:**
- Monitor fraud detection performance
- Analyze fraud patterns
- ML model latency tracking

---

## Alerts

### Critical Alerts

#### High Error Rate
```yaml
alert: HighErrorRate
expr: |
  sum(rate(synfinance_http_requests_total{status_code=~"5.."}[5m])) 
  / 
  sum(rate(synfinance_http_requests_total[5m])) > 0.05
for: 5m
annotations:
  summary: "Error rate above 5% for 5 minutes"
```

#### Slow API Response
```yaml
alert: SlowAPIResponse
expr: |
  histogram_quantile(0.95, 
    sum(rate(synfinance_api_request_duration_seconds_bucket[5m])) by (le)
  ) > 1.0
for: 5m
annotations:
  summary: "P95 latency above 1 second"
```

#### Database Connection Pool Exhausted
```yaml
alert: DatabaseConnectionPoolExhausted
expr: synfinance_db_connections_active >= 95
for: 2m
annotations:
  summary: "Database connection pool near capacity"
```

### Warning Alerts

#### High Memory Usage
```yaml
alert: HighMemoryUsage
expr: |
  synfinance_memory_usage_bytes 
  / 
  (4 * 1024 * 1024 * 1024) > 0.8
for: 10m
annotations:
  summary: "Memory usage above 80%"
```

#### High Fraud Rate
```yaml
alert: HighFraudRate
expr: synfinance_fraud_detection_rate > 0.20
for: 15m
annotations:
  summary: "Fraud detection rate above 20%"
```

---

## Troubleshooting

### High Latency

1. **Check metrics:**
   ```promql
   synfinance_api_request_duration_seconds{quantile="0.95"}
   ```

2. **Identify slow endpoints:**
   ```promql
   topk(5, 
     histogram_quantile(0.95, 
       sum(rate(synfinance_api_request_duration_seconds_bucket[5m])) 
       by (le, endpoint)
     )
   )
   ```

3. **Check traces:**
   - Find slow request in Jaeger
   - Identify bottleneck span (DB, ML model, etc.)

4. **Check logs:**
   ```
   {app="synfinance-api"} | json | duration_ms > 1000
   ```

### High Error Rate

1. **Check error distribution:**
   ```promql
   sum(rate(synfinance_errors_total[5m])) by (error_type)
   ```

2. **View error logs:**
   ```
   {app="synfinance-api"} | json | level="ERROR"
   ```

3. **Find failing traces:**
   - Search Jaeger for `error=true`
   - Examine error spans

### Memory Leak

1. **Check memory trend:**
   ```promql
   synfinance_memory_usage_bytes
   ```

2. **Review logs for OOM:**
   ```
   {app="synfinance-api"} |= "OutOfMemory"
   ```

3. **Profile application** (if available)

### Database Issues

1. **Check connection pool:**
   ```promql
   synfinance_db_connections_active
   ```

2. **Check query latency:**
   ```promql
   histogram_quantile(0.95, 
     sum(rate(synfinance_db_query_duration_seconds_bucket[5m])) 
     by (le, operation)
   )
   ```

3. **Review slow queries in logs:**
   ```
   {app="synfinance-api"} | json | db.statement != "" | duration_ms > 500
   ```

---

## Best Practices

### For Developers

1. **Add custom metrics** for business events
2. **Use structured logging** with contextual fields
3. **Add spans** for important operations
4. **Include request_id** in error responses
5. **Log at appropriate levels** (avoid DEBUG in production)

### For Operations

1. **Set up alerts** for critical conditions
2. **Review dashboards** regularly
3. **Investigate** spikes in error rate/latency
4. **Correlate** metrics, logs, and traces
5. **Use retention policies** to manage storage

### Performance Impact

Observability overhead:
- **Metrics**: <0.5% CPU, minimal memory
- **Logging**: ~1-2% CPU (JSON serialization)
- **Tracing** (100% sampling): ~2-5% CPU
- **Total**: <5% overhead (acceptable for production)

**Recommendation**: Use sampling for tracing in high-traffic production (10-20% sample rate).

---

## Configuration

### Environment Variables

```bash
# Logging
LOG_LEVEL=INFO
ENVIRONMENT=production

# Tracing
JAEGER_ENDPOINT=jaeger-collector:14268
OTLP_ENDPOINT=tempo:4317
OTEL_SERVICE_NAME=synfinance-api
OTEL_RESOURCE_ATTRIBUTES=service.version=2.15.0

# Metrics (auto-configured)
```

### Kubernetes ConfigMap

See `k8s/base/configmap.yaml` for full configuration.

---

## Support

For issues or questions:
- **Metrics not showing**: Check Prometheus targets (`/targets`)
- **Logs missing**: Check Promtail status
- **Traces not appearing**: Verify OTLP endpoint connectivity
- **Dashboard empty**: Check datasource configuration in Grafana

---

**Last Updated:** November 2, 2025  
**Version:** 2.15.0
