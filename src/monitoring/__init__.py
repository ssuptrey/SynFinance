"""
Monitoring module for SynFinance.

Provides comprehensive monitoring capabilities including:
- Prometheus metrics export
- Custom business metrics
- Metrics middleware for FastAPI
- Grafana dashboard configurations
"""

from .prometheus_exporter import (
    PrometheusMetricsExporter,
    MetricsRegistry,
    get_metrics_exporter
)
from .business_metrics import (
    FraudDetectionMetrics,
    PerformanceMetrics,
    DataQualityMetrics
)
from .metrics_middleware import MetricsMiddleware

__all__ = [
    'PrometheusMetricsExporter',
    'MetricsRegistry',
    'get_metrics_exporter',
    'FraudDetectionMetrics',
    'PerformanceMetrics',
    'DataQualityMetrics',
    'MetricsMiddleware',
]

__version__ = '0.8.0'
