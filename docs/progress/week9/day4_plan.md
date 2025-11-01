# Week 9 Day 4: Monitoring & Observability

**Date:** November 2, 2025  
**Focus:** Production-grade monitoring, metrics, logging, and tracing

---

## Objectives

Build comprehensive observability stack for SynFinance to enable:
- Real-time monitoring of application health and performance
- Fraud detection metrics and analytics
- Centralized logging with correlation
- Distributed tracing for request flows
- Alerting on critical conditions

---

## Deliverables

### 1. Prometheus Metrics
- **Metrics endpoint** (`/metrics`) in FastAPI app
- **Custom metrics:**
  - Transaction generation rate (txn/sec)
  - Fraud detection rate (fraud/total)
  - API request latency (histogram)
  - Database query performance
  - ML model inference time
  - Error rates by endpoint
  - Active connections (WebSocket)
  
- **System metrics:**
  - CPU/Memory usage
  - Request duration
  - HTTP status codes
  - Database pool utilization

### 2. Grafana Dashboards
- **Application Overview Dashboard:**
  - Transaction throughput
  - Error rates
  - Response times (p50, p95, p99)
  - Active users/connections
  
- **Fraud Analytics Dashboard:**
  - Fraud patterns detected
  - Pattern distribution
  - Detection accuracy metrics
  - Model performance
  
- **System Performance Dashboard:**
  - Resource utilization
  - Database performance
  - Cache hit rates
  - Queue depths

### 3. Structured Logging
- **JSON logging format** with:
  - Timestamp (ISO 8601)
  - Log level
  - Service name
  - Trace ID / Request ID
  - User/tenant context
  - Structured fields (not just strings)
  
- **Log levels:**
  - DEBUG: Development details
  - INFO: Business events (transaction created, fraud detected)
  - WARNING: Degraded performance, retries
  - ERROR: Errors that need attention
  - CRITICAL: Service-impacting issues
  
- **Log aggregation ready:**
  - Compatible with ELK (Elasticsearch/Logstash/Kibana)
  - Compatible with Loki/Promtail
  - Compatible with cloud logging (CloudWatch, Stackdriver)

### 4. Distributed Tracing
- **OpenTelemetry integration:**
  - Automatic instrumentation for FastAPI
  - Database query spans
  - ML model inference spans
  - External API call spans
  
- **Trace context propagation:**
  - HTTP headers (W3C Trace Context)
  - Cross-service correlation
  
- **Exporters:**
  - Jaeger (development/on-prem)
  - Tempo (Grafana stack)
  - Cloud exporters (optional)

### 5. Kubernetes Integration
- **ServiceMonitor CRD** (Prometheus Operator)
- **Logging sidecar pattern** (Promtail/Fluent Bit)
- **Tracing collector** (OpenTelemetry Collector)
- **Updated manifests** with:
  - Prometheus annotations
  - Log volume mounts
  - Tracing environment variables

### 6. Alerting Rules
- **Critical alerts:**
  - API error rate >5% for 5 minutes
  - Response time p95 >1s for 5 minutes
  - Database connection pool exhausted
  - ML model inference failures >10%
  
- **Warning alerts:**
  - Memory usage >80%
  - Disk usage >85%
  - High fraud detection rate (>20%)
  - Unusual transaction patterns

### 7. Documentation
- **Monitoring guide:**
  - How to access Grafana
  - Dashboard overview
  - Common queries
  - Troubleshooting with metrics
  
- **Runbook updates:**
  - Using logs for debugging
  - Tracing request flows
  - Alert response procedures

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SynFinance Application                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   FastAPI   │  │  ML Models   │  │  PostgreSQL  │       │
│  │   /metrics  │  │              │  │              │       │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │               │
│         └─────────────────┴──────────────────┘               │
│                           │                                  │
│              ┌────────────┴────────────┐                     │
│              │                         │                     │
│         JSON Logs              Prometheus Metrics            │
│              │                         │                     │
└──────────────┼─────────────────────────┼─────────────────────┘
               │                         │
               ▼                         ▼
      ┌────────────────┐        ┌────────────────┐
      │   Promtail/    │        │   Prometheus   │
      │   Fluent Bit   │        │     Server     │
      └────────┬───────┘        └────────┬───────┘
               │                         │
               ▼                         ▼
      ┌────────────────┐        ┌────────────────┐
      │      Loki      │        │    Grafana     │◄─── Dashboards
      │ (Log Storage)  │        │  (Visualization)│
      └────────────────┘        └────────────────┘
               │                         ▲
               └─────────────────────────┘
                     Query Logs

      ┌─────────────────────────────────┐
      │   OpenTelemetry Collector       │
      │   ┌─────────┐   ┌─────────┐    │
      │   │ Jaeger  │   │  Tempo  │    │
      │   │ (Traces)│   │(Traces) │    │
      │   └─────────┘   └─────────┘    │
      └─────────────────────────────────┘
                  ▲
                  │
            Trace exports
```

---

## Technology Stack

### Core
- **Prometheus** - Time-series metrics database
- **Grafana** - Visualization and dashboards
- **Loki** - Log aggregation (Grafana stack)
- **Tempo** - Distributed tracing (Grafana stack)

### Python Libraries
- `prometheus-client` - Prometheus metrics for Python
- `opentelemetry-api` - OpenTelemetry API
- `opentelemetry-sdk` - OpenTelemetry SDK
- `opentelemetry-instrumentation-fastapi` - Auto-instrumentation
- `opentelemetry-exporter-jaeger` - Jaeger exporter
- `python-json-logger` - Structured JSON logging

### Kubernetes
- **Prometheus Operator** - Manages Prometheus in K8s
- **Grafana Operator** - Manages Grafana in K8s
- **Loki** - Deployed via Helm
- **Promtail** - Log shipper (DaemonSet)

---

## Implementation Plan

### Step 1: Add Metrics (1 hour)
- Install `prometheus-client`
- Create `/metrics` endpoint in FastAPI
- Add custom metrics (counters, histograms, gauges)
- Add middleware for automatic HTTP metrics

### Step 2: Create Grafana Dashboards (1 hour)
- Export dashboard JSON files
- Application metrics dashboard
- Fraud analytics dashboard
- System performance dashboard

### Step 3: Structured Logging (45 min)
- Install `python-json-logger`
- Configure JSON formatter
- Add request ID middleware
- Add context fields (user, tenant, trace_id)

### Step 4: Distributed Tracing (1 hour)
- Install OpenTelemetry packages
- Configure auto-instrumentation
- Add custom spans for ML inference
- Set up Jaeger exporter

### Step 5: Kubernetes Manifests (45 min)
- Add ServiceMonitor for Prometheus
- Add Promtail DaemonSet for logs
- Add OpenTelemetry Collector deployment
- Update app deployment with env vars

### Step 6: Alerting (30 min)
- Create PrometheusRule CRD
- Define alert conditions
- Configure notification channels (Slack, email)

### Step 7: Documentation (30 min)
- Monitoring setup guide
- Dashboard usage guide
- Troubleshooting procedures

---

## Success Metrics

- ✅ `/metrics` endpoint returns 50+ metrics
- ✅ Grafana dashboards display real-time data
- ✅ Logs are structured JSON with trace IDs
- ✅ Traces visible in Jaeger/Tempo UI
- ✅ Alerts fire correctly in test scenarios
- ✅ Documentation clear and actionable
- ✅ Zero performance impact (<1% overhead)

---

## Dependencies

- Week 9 Day 1: Docker setup ✅
- Week 9 Day 2: Kubernetes manifests ✅
- Week 9 Day 3: CI/CD pipeline ✅
- FastAPI app running ✅
- PostgreSQL database ✅

---

## Key Metrics to Track

### Business Metrics
- `synfinance_transactions_total` (counter)
- `synfinance_fraud_detections_total` (counter)
- `synfinance_fraud_patterns` (gauge by pattern type)
- `synfinance_customers_active` (gauge)

### Performance Metrics
- `synfinance_api_request_duration_seconds` (histogram)
- `synfinance_db_query_duration_seconds` (histogram)
- `synfinance_ml_inference_duration_seconds` (histogram)
- `synfinance_generation_rate_per_second` (gauge)

### System Metrics
- `synfinance_memory_usage_bytes` (gauge)
- `synfinance_cpu_usage_percent` (gauge)
- `synfinance_db_connections_active` (gauge)
- `synfinance_cache_hit_rate` (gauge)

### Error Metrics
- `synfinance_errors_total` (counter by type)
- `synfinance_http_requests_total` (counter by status code)
- `synfinance_validation_failures_total` (counter)

---

## Timeline

**Total Time:** ~5-6 hours

- 09:00-10:00: Prometheus metrics implementation
- 10:00-11:00: Grafana dashboards
- 11:00-11:45: Structured logging
- 11:45-12:00: Break
- 12:00-13:00: Distributed tracing
- 13:00-13:45: Kubernetes integration
- 13:45-14:15: Alerting rules
- 14:15-14:45: Documentation
- 14:45-15:00: Testing and verification

---

## Testing Plan

1. **Metrics test:**
   ```bash
   curl http://localhost:8000/metrics
   # Should return Prometheus format metrics
   ```

2. **Grafana test:**
   - Access dashboards
   - Verify data flowing
   - Test queries

3. **Logging test:**
   ```bash
   kubectl logs -n synfinance-production -l app=synfinance --tail=100
   # Should show JSON formatted logs
   ```

4. **Tracing test:**
   - Make API requests
   - View traces in Jaeger UI
   - Verify span hierarchy

5. **Alerting test:**
   - Trigger error condition
   - Verify alert fires
   - Check notification received

---

## Next Steps (Day 5)

After monitoring is complete:
- Optional: Service mesh (Istio/Linkerd)
- Optional: Advanced security (WAF, rate limiting)
- Week 9 wrap-up and documentation

---

**Status:** Ready to implement  
**Estimated Completion:** End of Day 4
