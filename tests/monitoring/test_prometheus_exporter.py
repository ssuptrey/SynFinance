"""
Tests for Prometheus metrics exporter.

Tests:
- MetricsRegistry functionality
- Metric registration and retrieval
- PrometheusMetricsExporter standard metrics
- Request/error recording
- Fraud detection metrics
- Performance metrics
- Export functionality
"""

import pytest
import time
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary

from src.monitoring.prometheus_exporter import (
    MetricDefinition,
    MetricsRegistry,
    PrometheusMetricsExporter,
    get_metrics_exporter
)


class TestMetricDefinition:
    """Tests for MetricDefinition dataclass."""
    
    def test_valid_metric_definition(self):
        """Test creating valid metric definition."""
        metric_def = MetricDefinition(
            name="test_metric",
            metric_type="counter",
            description="Test metric",
            labels=["label1", "label2"]
        )
        
        assert metric_def.name == "test_metric"
        assert metric_def.metric_type == "counter"
        assert metric_def.description == "Test metric"
        assert metric_def.labels == ["label1", "label2"]
    
    def test_invalid_metric_type(self):
        """Test that invalid metric type raises error."""
        with pytest.raises(ValueError, match="Invalid metric type"):
            MetricDefinition(
                name="test_metric",
                metric_type="invalid_type",
                description="Test metric"
            )
    
    def test_metric_definition_with_buckets(self):
        """Test metric definition with histogram buckets."""
        buckets = [0.1, 0.5, 1.0, 5.0]
        metric_def = MetricDefinition(
            name="test_histogram",
            metric_type="histogram",
            description="Test histogram",
            buckets=buckets
        )
        
        assert metric_def.buckets == buckets


class TestMetricsRegistry:
    """Tests for MetricsRegistry class."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = MetricsRegistry(namespace="test")
        assert registry.namespace == "test"
        assert len(registry._metrics) == 0
        assert len(registry._definitions) == 0
    
    def test_register_counter(self):
        """Test registering a counter metric."""
        registry = MetricsRegistry(namespace="test")
        counter = registry.register_counter(
            name="test_counter",
            description="Test counter",
            labels=["status"]
        )
        
        assert "test_counter" in registry._metrics
        assert isinstance(counter, Counter)
        
        # Test increment
        counter.labels(status="success").inc()
        counter.labels(status="success").inc()
        
        assert counter.labels(status="success")._value._value == 2
    
    def test_register_gauge(self):
        """Test registering a gauge metric."""
        registry = MetricsRegistry(namespace="test")
        gauge = registry.register_gauge(
            name="test_gauge",
            description="Test gauge",
            labels=["type"]
        )
        
        assert "test_gauge" in registry._metrics
        assert isinstance(gauge, Gauge)
        
        # Test set and inc/dec
        gauge.labels(type="memory").set(100)
        gauge.labels(type="memory").inc(50)
        gauge.labels(type="memory").dec(25)
        
        assert gauge.labels(type="memory")._value._value == 125
    
    def test_register_histogram(self):
        """Test registering a histogram metric."""
        registry = MetricsRegistry(namespace="test")
        buckets = [0.1, 0.5, 1.0]
        histogram = registry.register_histogram(
            name="test_histogram",
            description="Test histogram",
            labels=["endpoint"],
            buckets=buckets
        )
        
        assert "test_histogram" in registry._metrics
        assert isinstance(histogram, Histogram)
        
        # Test observe
        histogram.labels(endpoint="/api/test").observe(0.3)
        histogram.labels(endpoint="/api/test").observe(0.7)
    
    def test_register_summary(self):
        """Test registering a summary metric."""
        registry = MetricsRegistry(namespace="test")
        summary = registry.register_summary(
            name="test_summary",
            description="Test summary",
            labels=["method"]
        )
        
        assert "test_summary" in registry._metrics
        assert isinstance(summary, Summary)
        
        # Test observe
        summary.labels(method="GET").observe(100)
        summary.labels(method="GET").observe(200)
    
    def test_get_metric(self):
        """Test retrieving registered metric."""
        registry = MetricsRegistry(namespace="test")
        counter = registry.register_counter(
            name="test_metric",
            description="Test"
        )
        
        retrieved = registry.get_metric("test_metric")
        assert retrieved is counter
    
    def test_get_nonexistent_metric(self):
        """Test retrieving non-existent metric returns None."""
        registry = MetricsRegistry(namespace="test")
        assert registry.get_metric("nonexistent") is None
    
    def test_list_metrics(self):
        """Test listing all metrics."""
        registry = MetricsRegistry(namespace="test")
        registry.register_counter("metric1", "Test 1")
        registry.register_gauge("metric2", "Test 2")
        registry.register_histogram("metric3", "Test 3")
        
        metrics = registry.list_metrics()
        assert len(metrics) == 3
        assert "metric1" in metrics
        assert "metric2" in metrics
        assert "metric3" in metrics
    
    def test_get_definition(self):
        """Test retrieving metric definition."""
        registry = MetricsRegistry(namespace="test")
        registry.register_counter(
            name="test_metric",
            description="Test counter",
            labels=["status"]
        )
        
        definition = registry.get_definition("test_metric")
        assert definition is not None
        assert definition.name == "test_metric"
        assert definition.metric_type == "counter"
        assert definition.description == "Test counter"
        assert definition.labels == ["status"]
    
    def test_clear_metrics(self):
        """Test clearing all metrics."""
        registry = MetricsRegistry(namespace="test")
        registry.register_counter("metric1", "Test 1")
        registry.register_gauge("metric2", "Test 2")
        
        assert len(registry._metrics) == 2
        
        registry.clear()
        
        assert len(registry._metrics) == 0
        assert len(registry._definitions) == 0


class TestPrometheusMetricsExporter:
    """Tests for PrometheusMetricsExporter class."""
    
    @pytest.fixture
    def exporter(self):
        """Create fresh exporter for each test."""
        return PrometheusMetricsExporter(namespace="test")
    
    def test_exporter_initialization(self, exporter):
        """Test exporter initializes all standard metrics."""
        # Check request metrics
        assert exporter.requests_total is not None
        assert exporter.errors_total is not None
        assert exporter.active_requests is not None
        assert exporter.request_latency is not None
        
        # Check fraud metrics
        assert exporter.fraud_detections_total is not None
        assert exporter.fraud_rate is not None
        assert exporter.fraud_confidence is not None
        
        # Check performance metrics
        assert exporter.transactions_generated is not None
        assert exporter.generation_rate is not None
        assert exporter.feature_engineering_time is not None
        assert exporter.prediction_time is not None
        
        # Check system metrics
        assert exporter.memory_usage is not None
        assert exporter.cpu_usage is not None
        assert exporter.disk_usage is not None
    
    def test_record_request(self, exporter):
        """Test recording HTTP request."""
        exporter.record_request(
            method="GET",
            endpoint="/api/transactions",
            status=200,
            latency=0.5,
            request_size=1024,
            response_size=2048
        )
        
        # Verify request counter incremented
        assert exporter.requests_total.labels(
            method="GET",
            endpoint="/api/transactions",
            status=200
        )._value._value == 1
    
    def test_record_error(self, exporter):
        """Test recording error."""
        exporter.record_error(
            error_type="ValueError",
            endpoint="/api/fraud"
        )
        
        # Verify error counter incremented
        assert exporter.errors_total.labels(
            type="ValueError",
            endpoint="/api/fraud"
        )._value._value == 1
    
    def test_record_fraud_detection(self, exporter):
        """Test recording fraud detection."""
        exporter.record_fraud_detection(
            fraud_type="card_cloning",
            severity="high",
            confidence=0.95
        )
        
        # Verify fraud counter incremented
        assert exporter.fraud_detections_total.labels(
            fraud_type="card_cloning",
            severity="high"
        )._value._value == 1
    
    def test_update_fraud_rate(self, exporter):
        """Test updating fraud rate."""
        exporter.update_fraud_rate(
            fraud_type="account_takeover",
            rate=0.03
        )
        
        # Verify fraud rate gauge updated
        assert exporter.fraud_rate.labels(
            fraud_type="account_takeover"
        )._value._value == 0.03
    
    def test_record_anomaly_detection(self, exporter):
        """Test recording anomaly detection."""
        exporter.record_anomaly_detection(
            anomaly_type="unusual_amount",
            severity="medium"
        )
        
        # Verify anomaly counter incremented
        assert exporter.anomaly_detections_total.labels(
            anomaly_type="unusual_amount",
            severity="medium"
        )._value._value == 1
    
    def test_update_cache_metrics(self, exporter):
        """Test updating cache metrics."""
        exporter.update_cache_metrics(
            cache_type="customer",
            hits=80,
            misses=20
        )
        
        # Verify cache hit rate (should be 0.8)
        hit_rate = exporter.cache_hit_rate.labels(
            cache_type="customer"
        )._value._value
        
        assert 0.79 < hit_rate < 0.81  # Allow for floating point precision
    
    def test_export_metrics(self, exporter):
        """Test exporting metrics in Prometheus format."""
        # Record some metrics
        exporter.record_request("GET", "/api/test", 200, 0.1, 100, 200)
        exporter.record_fraud_detection("phishing", "low", 0.7)
        
        # Export
        output = exporter.export()
        
        # Verify output contains metrics
        assert b"test_requests_total" in output
        assert b"test_fraud_detections_total" in output
    
    def test_get_content_type(self, exporter):
        """Test getting Prometheus content type."""
        from prometheus_client import CONTENT_TYPE_LATEST
        assert exporter.get_content_type() == CONTENT_TYPE_LATEST


class TestGetMetricsExporter:
    """Tests for get_metrics_exporter singleton function."""
    
    def test_singleton_pattern(self):
        """Test that get_metrics_exporter returns same instance."""
        exporter1 = get_metrics_exporter("test_singleton")
        exporter2 = get_metrics_exporter("test_singleton")
        
        assert exporter1 is exporter2
    
    def test_different_namespaces(self):
        """Test that different namespaces create different instances."""
        exporter1 = get_metrics_exporter("namespace1")
        exporter2 = get_metrics_exporter("namespace2")
        
        assert exporter1 is not exporter2
        assert exporter1.namespace == "namespace1"
        assert exporter2.namespace == "namespace2"


class TestMetricsIntegration:
    """Integration tests for metrics system."""
    
    def test_multiple_metrics_workflow(self):
        """Test complete workflow with multiple metrics."""
        exporter = PrometheusMetricsExporter(namespace="integration")
        
        # Simulate API requests
        for i in range(10):
            exporter.record_request(
                method="POST",
                endpoint="/api/detect",
                status=200,
                latency=0.05 + (i * 0.01),
                request_size=1024,
                response_size=512
            )
        
        # Simulate fraud detections
        exporter.record_fraud_detection("smurfing", "high", 0.92)
        exporter.record_fraud_detection("money_laundering", "critical", 0.98)
        exporter.update_fraud_rate("overall", 0.02)
        
        # Simulate anomaly detections
        exporter.record_anomaly_detection("velocity", "medium")
        
        # Verify metrics were recorded
        assert exporter.requests_total.labels(
            method="POST",
            endpoint="/api/detect",
            status=200
        )._value._value == 10
        
        assert exporter.fraud_detections_total.labels(
            fraud_type="smurfing",
            severity="high"
        )._value._value == 1
        
        # Export and verify output
        output = exporter.export()
        assert output is not None
        assert len(output) > 0
