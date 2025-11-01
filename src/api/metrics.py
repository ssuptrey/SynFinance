"""
Prometheus metrics for SynFinance API monitoring.

Provides comprehensive metrics for:
- Transaction processing
- Fraud detection
- API performance
- System health
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry
)
from fastapi import Response
from typing import Dict, Any
import time
import psutil
import os

# Create registry
registry = CollectorRegistry()

# ============================================================================
# Business Metrics
# ============================================================================

# Transaction metrics
transactions_total = Counter(
    'synfinance_transactions_total',
    'Total number of transactions processed',
    ['status'],  # success, failed
    registry=registry
)

fraud_detections_total = Counter(
    'synfinance_fraud_detections_total',
    'Total number of fraud detections',
    ['pattern_type'],  # card_cloning, account_takeover, etc.
    registry=registry
)

fraud_detection_rate = Gauge(
    'synfinance_fraud_detection_rate',
    'Current fraud detection rate (fraud/total)',
    registry=registry
)

# ============================================================================
# Performance Metrics
# ============================================================================

api_request_duration = Histogram(
    'synfinance_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint', 'status_code'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    registry=registry
)

ml_inference_duration = Histogram(
    'synfinance_ml_inference_duration_seconds',
    'ML model inference duration in seconds',
    ['model_name'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry
)

db_query_duration = Histogram(
    'synfinance_db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],  # select, insert, update, delete
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry
)

generation_rate = Gauge(
    'synfinance_generation_rate_per_second',
    'Transaction generation rate (transactions/second)',
    registry=registry
)

# ============================================================================
# System Metrics
# ============================================================================

memory_usage_bytes = Gauge(
    'synfinance_memory_usage_bytes',
    'Current memory usage in bytes',
    registry=registry
)

cpu_usage_percent = Gauge(
    'synfinance_cpu_usage_percent',
    'Current CPU usage percentage',
    registry=registry
)

active_connections = Gauge(
    'synfinance_active_connections',
    'Number of active WebSocket connections',
    registry=registry
)

db_connections_active = Gauge(
    'synfinance_db_connections_active',
    'Number of active database connections',
    registry=registry
)

cache_hit_rate = Gauge(
    'synfinance_cache_hit_rate',
    'Cache hit rate (0-1)',
    registry=registry
)

# ============================================================================
# Error Metrics
# ============================================================================

errors_total = Counter(
    'synfinance_errors_total',
    'Total number of errors',
    ['error_type'],  # validation, database, ml_model, internal
    registry=registry
)

http_requests_total = Counter(
    'synfinance_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

validation_failures_total = Counter(
    'synfinance_validation_failures_total',
    'Total validation failures',
    ['field'],
    registry=registry
)

# ============================================================================
# Application Info
# ============================================================================

app_info = Info(
    'synfinance_app',
    'SynFinance application information',
    registry=registry
)

# Set app info
app_info.info({
    'version': '2.15.0',
    'environment': os.getenv('ENVIRONMENT', 'development'),
    'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
})

# ============================================================================
# Metric Helper Functions
# ============================================================================

def update_system_metrics():
    """Update system resource metrics."""
    try:
        # Memory usage
        process = psutil.Process(os.getpid())
        memory_usage_bytes.set(process.memory_info().rss)
        
        # CPU usage
        cpu_usage_percent.set(process.cpu_percent(interval=0.1))
    except Exception as e:
        # Don't fail on metrics collection errors
        pass


def record_transaction(fraud_detected: bool, pattern_type: str = None):
    """Record a transaction and fraud detection."""
    transactions_total.labels(status='success').inc()
    
    if fraud_detected and pattern_type:
        fraud_detections_total.labels(pattern_type=pattern_type).inc()
    
    # Update fraud rate
    try:
        total = transactions_total.labels(status='success')._value.get()
        fraud_count = sum(
            fraud_detections_total.labels(pattern_type=pt)._value.get()
            for pt in ['card_cloning', 'account_takeover', 'velocity_abuse', 'other']
        )
        if total > 0:
            fraud_detection_rate.set(fraud_count / total)
    except Exception:
        pass


def record_error(error_type: str):
    """Record an error."""
    errors_total.labels(error_type=error_type).inc()


def record_http_request(method: str, endpoint: str, status_code: int, duration: float):
    """Record an HTTP request with timing."""
    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).inc()
    
    api_request_duration.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).observe(duration)


def record_ml_inference(model_name: str, duration: float):
    """Record ML model inference time."""
    ml_inference_duration.labels(model_name=model_name).observe(duration)


def record_db_query(operation: str, duration: float):
    """Record database query time."""
    db_query_duration.labels(operation=operation).observe(duration)


# ============================================================================
# Metrics Endpoint
# ============================================================================

def get_metrics() -> Response:
    """
    Generate Prometheus metrics response.
    
    Returns:
        Response with Prometheus format metrics
    """
    # Update system metrics before serving
    update_system_metrics()
    
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )
