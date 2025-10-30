"""
Prometheus metrics exporter for SynFinance.

This module provides comprehensive Prometheus metrics export capabilities including:
- Counter metrics (monotonically increasing)
- Gauge metrics (can increase/decrease)
- Histogram metrics (observations with buckets)
- Summary metrics (quantiles)
- Metrics registry for custom metrics
- Health check endpoint
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricDefinition:
    """Definition of a Prometheus metric."""
    
    name: str
    metric_type: str  # 'counter', 'gauge', 'histogram', 'summary'
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    
    def __post_init__(self):
        """Validate metric definition."""
        valid_types = ['counter', 'gauge', 'histogram', 'summary']
        if self.metric_type not in valid_types:
            raise ValueError(f"Invalid metric type: {self.metric_type}. Must be one of {valid_types}")


class MetricsRegistry:
    """
    Registry for managing Prometheus metrics.
    
    Provides centralized metric management with:
    - Metric registration and retrieval
    - Type-safe metric creation
    - Label validation
    - Auto-discovery of metrics
    """
    
    def __init__(self, namespace: str = "synfinance"):
        """
        Initialize metrics registry.
        
        Args:
            namespace: Prefix for all metric names
        """
        self.namespace = namespace
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.definitions: Dict[str, MetricDefinition] = {}
        
        logger.info(f"Initialized MetricsRegistry with namespace: {namespace}")
    
    def register_counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Counter:
        """
        Register a Counter metric.
        
        Counters are monotonically increasing values (e.g., total requests).
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional label names
            
        Returns:
            Counter metric instance
        """
        labels = labels or []
        full_name = f"{self.namespace}_{name}"
        
        if full_name in self.metrics:
            logger.warning(f"Counter {full_name} already registered, returning existing")
            return self.metrics[full_name]
        
        counter = Counter(
            full_name,
            description,
            labelnames=labels,
            registry=self.registry
        )
        
        self.metrics[full_name] = counter
        self.definitions[full_name] = MetricDefinition(
            name=full_name,
            metric_type='counter',
            description=description,
            labels=labels
        )
        
        logger.debug(f"Registered counter: {full_name}")
        return counter
    
    def register_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Gauge:
        """
        Register a Gauge metric.
        
        Gauges can increase or decrease (e.g., active requests, memory usage).
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional label names
            
        Returns:
            Gauge metric instance
        """
        labels = labels or []
        full_name = f"{self.namespace}_{name}"
        
        if full_name in self.metrics:
            logger.warning(f"Gauge {full_name} already registered, returning existing")
            return self.metrics[full_name]
        
        gauge = Gauge(
            full_name,
            description,
            labelnames=labels,
            registry=self.registry
        )
        
        self.metrics[full_name] = gauge
        self.definitions[full_name] = MetricDefinition(
            name=full_name,
            metric_type='gauge',
            description=description,
            labels=labels
        )
        
        logger.debug(f"Registered gauge: {full_name}")
        return gauge
    
    def register_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """
        Register a Histogram metric.
        
        Histograms track distribution of observations (e.g., request latency).
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional label names
            buckets: Custom bucket boundaries (default: [.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0, +Inf])
            
        Returns:
            Histogram metric instance
        """
        labels = labels or []
        full_name = f"{self.namespace}_{name}"
        
        if full_name in self.metrics:
            logger.warning(f"Histogram {full_name} already registered, returning existing")
            return self.metrics[full_name]
        
        # Default buckets suitable for latency (in seconds)
        if buckets is None:
            buckets = [.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0]
        
        histogram = Histogram(
            full_name,
            description,
            labelnames=labels,
            buckets=buckets,
            registry=self.registry
        )
        
        self.metrics[full_name] = histogram
        self.definitions[full_name] = MetricDefinition(
            name=full_name,
            metric_type='histogram',
            description=description,
            labels=labels,
            buckets=buckets
        )
        
        logger.debug(f"Registered histogram: {full_name}")
        return histogram
    
    def register_summary(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Summary:
        """
        Register a Summary metric.
        
        Summaries track quantiles of observations (e.g., p50, p95, p99 latency).
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional label names
            
        Returns:
            Summary metric instance
        """
        labels = labels or []
        full_name = f"{self.namespace}_{name}"
        
        if full_name in self.metrics:
            logger.warning(f"Summary {full_name} already registered, returning existing")
            return self.metrics[full_name]
        
        summary = Summary(
            full_name,
            description,
            labelnames=labels,
            registry=self.registry
        )
        
        self.metrics[full_name] = summary
        self.definitions[full_name] = MetricDefinition(
            name=full_name,
            metric_type='summary',
            description=description,
            labels=labels
        )
        
        logger.debug(f"Registered summary: {full_name}")
        return summary
    
    def get_metric(self, name: str) -> Optional[Any]:
        """
        Get a registered metric by name.
        
        Args:
            name: Metric name (with or without namespace prefix)
            
        Returns:
            Metric instance or None if not found
        """
        # Try name as-is first (in case it already has namespace)
        if name in self.metrics:
            return self.metrics[name]
        
        # Try with namespace prefix
        full_name = f"{self.namespace}_{name}"
        return self.metrics.get(full_name)
    
    def list_metrics(self) -> List[str]:
        """
        List all registered metric names.
        
        Returns:
            List of metric names
        """
        return list(self.metrics.keys())
    
    def get_definition(self, name: str) -> Optional[MetricDefinition]:
        """
        Get metric definition.
        
        Args:
            name: Metric name
            
        Returns:
            MetricDefinition or None if not found
        """
        # Try name as-is first (in case it already has namespace)
        if name in self.definitions:
            return self.definitions[name]
        
        # Try with namespace prefix
        full_name = f"{self.namespace}_{name}"
        return self.definitions.get(full_name)
    
    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Metrics data in Prometheus exposition format
        """
        return generate_latest(self.registry)
    
    def clear(self):
        """Clear all registered metrics."""
        self.metrics.clear()
        self.definitions.clear()
        logger.info("Cleared all metrics from registry")


class PrometheusMetricsExporter:
    """
    Main Prometheus metrics exporter for SynFinance.
    
    Provides comprehensive metrics export including:
    - Request/response metrics
    - Fraud detection metrics
    - Performance metrics
    - Data quality metrics
    - System health metrics
    """
    
    def __init__(self, namespace: str = "synfinance"):
        """
        Initialize Prometheus metrics exporter.
        
        Args:
            namespace: Metric namespace prefix
        """
        self.namespace = namespace
        self.registry = MetricsRegistry(namespace)
        
        # Initialize standard metrics
        self._init_request_metrics()
        self._init_fraud_metrics()
        self._init_performance_metrics()
        self._init_system_metrics()
        
        logger.info("Initialized PrometheusMetricsExporter")
    
    def _init_request_metrics(self):
        """Initialize HTTP request/response metrics."""
        # Counter: Total requests
        self.requests_total = self.registry.register_counter(
            'requests_total',
            'Total HTTP requests',
            labels=['method', 'endpoint', 'status']
        )
        
        # Counter: Total errors
        self.errors_total = self.registry.register_counter(
            'errors_total',
            'Total errors',
            labels=['type', 'endpoint']
        )
        
        # Gauge: Active requests
        self.active_requests = self.registry.register_gauge(
            'active_requests',
            'Number of active HTTP requests',
            labels=['endpoint']
        )
        
        # Histogram: Request latency
        self.request_latency = self.registry.register_histogram(
            'request_latency_seconds',
            'HTTP request latency in seconds',
            labels=['method', 'endpoint'],
            buckets=[.001, .005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0]
        )
        
        # Summary: Request size
        self.request_size = self.registry.register_summary(
            'request_size_bytes',
            'HTTP request size in bytes',
            labels=['method', 'endpoint']
        )
        
        # Summary: Response size
        self.response_size = self.registry.register_summary(
            'response_size_bytes',
            'HTTP response size in bytes',
            labels=['endpoint', 'status']
        )
    
    def _init_fraud_metrics(self):
        """Initialize fraud detection metrics."""
        # Counter: Total fraud detections
        self.fraud_detections_total = self.registry.register_counter(
            'fraud_detections_total',
            'Total fraud detections',
            labels=['fraud_type', 'severity']
        )
        
        # Gauge: Current fraud rate
        self.fraud_rate = self.registry.register_gauge(
            'fraud_rate',
            'Current fraud detection rate (0-1)',
            labels=['fraud_type']
        )
        
        # Histogram: Fraud confidence scores
        self.fraud_confidence = self.registry.register_histogram(
            'fraud_confidence',
            'Fraud detection confidence scores',
            labels=['fraud_type'],
            buckets=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.0]
        )
        
        # Counter: Anomaly detections
        self.anomaly_detections_total = self.registry.register_counter(
            'anomaly_detections_total',
            'Total anomaly detections',
            labels=['anomaly_type', 'severity']
        )
        
        # Gauge: Anomaly rate
        self.anomaly_rate = self.registry.register_gauge(
            'anomaly_rate',
            'Current anomaly detection rate (0-1)',
            labels=['anomaly_type']
        )
    
    def _init_performance_metrics(self):
        """Initialize performance metrics."""
        # Counter: Transactions generated
        self.transactions_generated = self.registry.register_counter(
            'transactions_generated_total',
            'Total transactions generated'
        )
        
        # Gauge: Generation rate
        self.generation_rate = self.registry.register_gauge(
            'generation_rate_txn_per_sec',
            'Transaction generation rate (txn/sec)'
        )
        
        # Histogram: Feature engineering time
        self.feature_engineering_time = self.registry.register_histogram(
            'feature_engineering_seconds',
            'Time spent on feature engineering in seconds',
            buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0, 2.5, 5.0]
        )
        
        # Histogram: Model prediction time
        self.prediction_time = self.registry.register_histogram(
            'prediction_seconds',
            'Model prediction time in seconds',
            buckets=[.001, .005, .01, .025, .05, .075, .1, .15, .2]
        )
        
        # Gauge: Cache hit rate
        self.cache_hit_rate = self.registry.register_gauge(
            'cache_hit_rate',
            'Cache hit rate (0-1)',
            labels=['cache_type']
        )
        
        # Counter: Cache hits
        self.cache_hits = self.registry.register_counter(
            'cache_hits_total',
            'Total cache hits',
            labels=['cache_type']
        )
        
        # Counter: Cache misses
        self.cache_misses = self.registry.register_counter(
            'cache_misses_total',
            'Total cache misses',
            labels=['cache_type']
        )
    
    def _init_system_metrics(self):
        """Initialize system health metrics."""
        # Gauge: Memory usage
        self.memory_usage = self.registry.register_gauge(
            'memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        # Gauge: CPU usage
        self.cpu_usage = self.registry.register_gauge(
            'cpu_usage_percent',
            'CPU usage percentage (0-100)'
        )
        
        # Gauge: Disk usage
        self.disk_usage = self.registry.register_gauge(
            'disk_usage_percent',
            'Disk usage percentage (0-100)'
        )
        
        # Gauge: Active connections
        self.active_connections = self.registry.register_gauge(
            'active_connections',
            'Number of active database/API connections',
            labels=['connection_type']
        )
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        latency: float,
        request_size: int = 0,
        response_size: int = 0
    ):
        """
        Record HTTP request metrics.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            status: HTTP status code
            latency: Request latency in seconds
            request_size: Request body size in bytes
            response_size: Response body size in bytes
        """
        self.requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_latency.labels(method=method, endpoint=endpoint).observe(latency)
        
        if request_size > 0:
            self.request_size.labels(method=method, endpoint=endpoint).observe(request_size)
        
        if response_size > 0:
            self.response_size.labels(endpoint=endpoint, status=status).observe(response_size)
    
    def record_error(self, error_type: str, endpoint: str):
        """
        Record error occurrence.
        
        Args:
            error_type: Type of error (validation, timeout, internal, etc.)
            endpoint: API endpoint where error occurred
        """
        self.errors_total.labels(type=error_type, endpoint=endpoint).inc()
    
    def record_fraud_detection(
        self,
        fraud_type: str,
        severity: str,
        confidence: float
    ):
        """
        Record fraud detection event.
        
        Args:
            fraud_type: Type of fraud detected
            severity: Severity level (low, medium, high, critical)
            confidence: Detection confidence (0-1)
        """
        self.fraud_detections_total.labels(fraud_type=fraud_type, severity=severity).inc()
        self.fraud_confidence.labels(fraud_type=fraud_type).observe(confidence)
    
    def update_fraud_rate(self, fraud_type: str, rate: float):
        """
        Update current fraud rate.
        
        Args:
            fraud_type: Type of fraud
            rate: Current fraud rate (0-1)
        """
        self.fraud_rate.labels(fraud_type=fraud_type).set(rate)
    
    def record_anomaly_detection(
        self,
        anomaly_type: str,
        severity: str
    ):
        """
        Record anomaly detection event.
        
        Args:
            anomaly_type: Type of anomaly detected
            severity: Severity level
        """
        self.anomaly_detections_total.labels(anomaly_type=anomaly_type, severity=severity).inc()
    
    def update_cache_metrics(self, cache_type: str, hits: int, misses: int):
        """
        Update cache metrics.
        
        Args:
            cache_type: Type of cache (customer, merchant, features, etc.)
            hits: Number of cache hits
            misses: Number of cache misses
        """
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        self.cache_hit_rate.labels(cache_type=cache_type).set(hit_rate)
    
    def export(self) -> bytes:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Metrics data in Prometheus exposition format
        """
        return self.registry.export_metrics()
    
    def get_content_type(self) -> str:
        """
        Get content type for metrics endpoint.
        
        Returns:
            Content type string
        """
        return CONTENT_TYPE_LATEST


# Global metrics exporter instances by namespace
_metrics_exporters: Dict[str, PrometheusMetricsExporter] = {}


def get_metrics_exporter(namespace: str = "synfinance") -> PrometheusMetricsExporter:
    """
    Get or create metrics exporter instance for the given namespace.
    
    Args:
        namespace: Metric namespace prefix
        
    Returns:
        PrometheusMetricsExporter instance
    """
    global _metrics_exporters
    
    if namespace not in _metrics_exporters:
        _metrics_exporters[namespace] = PrometheusMetricsExporter(namespace)
        logger.info(f"Created PrometheusMetricsExporter instance for namespace: {namespace}")
    
    return _metrics_exporters[namespace]
