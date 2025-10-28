"""
Tests for business metrics (fraud detection, performance, data quality).

Tests:
- FraudDetectionMetrics functionality
- PerformanceMetrics functionality
- DataQualityMetrics functionality
- Statistics calculation
- Metric updates and tracking
"""

import pytest
import time
from datetime import datetime, timedelta

from src.monitoring.business_metrics import (
    FraudStats,
    FraudDetectionMetrics,
    PerformanceStats,
    PerformanceMetrics,
    DataQualityStats,
    DataQualityMetrics
)


class TestFraudStats:
    """Tests for FraudStats dataclass."""
    
    def test_fraud_rate_calculation(self):
        """Test fraud rate calculation."""
        stats = FraudStats(
            total_transactions=1000,
            total_fraud_detected=50
        )
        
        assert stats.fraud_rate == 0.05  # 5%
    
    def test_precision_calculation(self):
        """Test precision calculation (TP / (TP + FP))."""
        stats = FraudStats(
            true_positives=80,
            false_positives=20
        )
        
        assert stats.precision == 0.8  # 80%
    
    def test_recall_calculation(self):
        """Test recall calculation (TP / (TP + FN))."""
        stats = FraudStats(
            true_positives=80,
            false_negatives=20
        )
        
        assert stats.recall == 0.8  # 80%
    
    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        stats = FraudStats(
            true_positives=80,
            false_positives=10,
            false_negatives=10
        )
        
        precision = 80 / 90  # 0.888...
        recall = 80 / 90     # 0.888...
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        assert abs(stats.f1_score - expected_f1) < 0.001
    
    def test_average_confidence(self):
        """Test average confidence calculation."""
        stats = FraudStats(
            confidence_scores=[0.7, 0.8, 0.9, 0.95, 0.85]
        )
        
        expected = sum(stats.confidence_scores) / len(stats.confidence_scores)
        assert abs(stats.average_confidence - expected) < 0.001


class TestFraudDetectionMetrics:
    """Tests for FraudDetectionMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Create fresh metrics for each test."""
        return FraudDetectionMetrics()
    
    def test_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.stats.total_transactions == 0
        assert metrics.stats.total_fraud_detected == 0
        assert len(metrics.recent_detections) == 0
    
    def test_record_transaction(self, metrics):
        """Test recording normal transaction."""
        metrics.record_transaction(is_fraud=False)
        metrics.record_transaction(is_fraud=True)
        
        assert metrics.stats.total_transactions == 2
        assert metrics.stats.total_fraud_detected == 1
    
    def test_record_fraud_detection(self, metrics):
        """Test recording fraud detection."""
        metrics.record_fraud_detection(
            fraud_type="card_cloning",
            severity="high",
            confidence=0.92,
            actual_fraud=True
        )
        
        assert metrics.stats.fraud_by_type["card_cloning"] == 1
        assert metrics.stats.fraud_by_severity["high"] == 1
        assert metrics.stats.true_positives == 1
        assert 0.92 in metrics.stats.confidence_scores
    
    def test_record_false_positive(self, metrics):
        """Test recording false positive."""
        metrics.record_fraud_detection(
            fraud_type="phishing",
            severity="medium",
            confidence=0.75,
            actual_fraud=False  # False positive
        )
        
        assert metrics.stats.false_positives == 1
        assert metrics.stats.true_positives == 0
    
    def test_record_missed_fraud(self, metrics):
        """Test recording missed fraud (false negative)."""
        metrics.record_missed_fraud("money_laundering")
        
        assert metrics.stats.false_negatives == 1
    
    def test_record_normal_transaction(self, metrics):
        """Test recording correctly classified normal transaction."""
        metrics.record_normal_transaction(correctly_classified=True)
        
        assert metrics.stats.true_negatives == 1
    
    def test_multiple_fraud_types(self, metrics):
        """Test tracking multiple fraud types."""
        metrics.record_fraud_detection("card_cloning", "high", 0.9, True)
        metrics.record_fraud_detection("card_cloning", "medium", 0.85, True)
        metrics.record_fraud_detection("phishing", "low", 0.7, True)
        
        assert metrics.stats.fraud_by_type["card_cloning"] == 2
        assert metrics.stats.fraud_by_type["phishing"] == 1
    
    def test_get_fraud_stats(self, metrics):
        """Test retrieving fraud statistics."""
        metrics.record_fraud_detection("smurfing", "critical", 0.95, True)
        metrics.record_transaction(is_fraud=True)
        
        stats = metrics.get_fraud_stats()
        
        assert isinstance(stats, FraudStats)
        assert stats.total_fraud_detected >= 1
    
    def test_reset_stats(self, metrics):
        """Test resetting statistics."""
        metrics.record_fraud_detection("test", "high", 0.9, True)
        metrics.record_transaction(is_fraud=True)
        
        assert metrics.stats.total_fraud_detected > 0
        
        metrics.reset_stats()
        
        assert metrics.stats.total_transactions == 0
        assert metrics.stats.total_fraud_detected == 0
        assert len(metrics.recent_detections) == 0


class TestPerformanceStats:
    """Tests for PerformanceStats dataclass."""
    
    def test_avg_generation_time(self):
        """Test average generation time calculation."""
        stats = PerformanceStats(
            transactions_generated=1000,
            total_generation_time=10.0  # 10 seconds
        )
        
        assert stats.avg_generation_time == 0.01  # 10ms per transaction
    
    def test_generation_rate(self):
        """Test generation rate calculation."""
        stats = PerformanceStats(
            transactions_generated=10000,
            total_generation_time=1.0  # 1 second
        )
        
        assert stats.generation_rate == 10000.0  # 10k txn/sec
    
    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        stats = PerformanceStats(
            cache_hits_by_type={"customer": 800},
            cache_misses_by_type={"customer": 200}
        )
        
        assert stats.get_cache_hit_rate("customer") == 0.8  # 80%


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Create fresh metrics for each test."""
        return PerformanceMetrics()
    
    def test_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.stats.transactions_generated == 0
        assert metrics.stats.total_generation_time == 0.0
    
    def test_record_generation(self, metrics):
        """Test recording transaction generation."""
        metrics.record_generation(count=1000, elapsed_time=0.5)
        
        assert metrics.stats.transactions_generated == 1000
        assert metrics.stats.total_generation_time == 0.5
    
    def test_record_feature_engineering(self, metrics):
        """Test recording feature engineering time."""
        metrics.record_feature_engineering(elapsed_time=0.3)
        
        assert metrics.stats.total_feature_engineering_time == 0.3
    
    def test_record_prediction(self, metrics):
        """Test recording prediction time."""
        metrics.record_prediction(elapsed_time=0.05)
        
        assert metrics.stats.total_prediction_time == 0.05
    
    def test_record_cache_hit(self, metrics):
        """Test recording cache hit."""
        metrics.record_cache_hit(cache_type="merchant")
        metrics.record_cache_hit(cache_type="merchant")
        
        assert metrics.stats.cache_hits_by_type["merchant"] == 2
    
    def test_record_cache_miss(self, metrics):
        """Test recording cache miss."""
        metrics.record_cache_miss(cache_type="features")
        
        assert metrics.stats.cache_misses_by_type["features"] == 1
    
    def test_cache_metrics_calculation(self, metrics):
        """Test cache hit rate calculation."""
        # 8 hits, 2 misses = 80% hit rate
        for _ in range(8):
            metrics.record_cache_hit("test_cache")
        for _ in range(2):
            metrics.record_cache_miss("test_cache")
        
        hit_rate = metrics.stats.get_cache_hit_rate("test_cache")
        assert 0.79 < hit_rate < 0.81  # Allow for floating point precision
    
    def test_get_performance_stats(self, metrics):
        """Test retrieving performance statistics."""
        metrics.record_generation(500, 0.25)
        
        stats = metrics.get_performance_stats()
        
        assert isinstance(stats, PerformanceStats)
        assert stats.transactions_generated == 500
    
    def test_reset_stats(self, metrics):
        """Test resetting statistics."""
        metrics.record_generation(1000, 1.0)
        metrics.record_cache_hit("test")
        
        assert metrics.stats.transactions_generated > 0
        
        metrics.reset_stats()
        
        assert metrics.stats.transactions_generated == 0
        assert metrics.stats.total_generation_time == 0.0


class TestDataQualityStats:
    """Tests for DataQualityStats dataclass."""
    
    def test_missing_rate_calculation(self):
        """Test missing value rate calculation."""
        stats = DataQualityStats(
            total_records=1000,
            missing_values_by_field={"amount": 50}
        )
        
        assert stats.get_missing_rate("amount") == 0.05  # 5%
    
    def test_outlier_rate_calculation(self):
        """Test outlier rate calculation."""
        stats = DataQualityStats(
            total_records=1000,
            outliers_by_field={"balance": 20}
        )
        
        assert stats.get_outlier_rate("balance") == 0.02  # 2%
    
    def test_overall_missing_rate(self):
        """Test overall missing value rate calculation."""
        stats = DataQualityStats(
            total_records=1000,
            missing_values_by_field={
                "field1": 50,
                "field2": 30,
                "field3": 20
            }
        )
        
        # 100 total missing / (1000 records * 3 fields) = 0.0333...
        expected = 100 / (1000 * 3)
        assert abs(stats.overall_missing_rate - expected) < 0.001


class TestDataQualityMetrics:
    """Tests for DataQualityMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Create fresh metrics for each test."""
        return DataQualityMetrics()
    
    def test_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.stats.total_records == 0
        assert len(metrics.stats.missing_values_by_field) == 0
    
    def test_record_dataset(self, metrics):
        """Test recording dataset size."""
        metrics.record_dataset(record_count=10000)
        
        assert metrics.stats.total_records == 10000
    
    def test_record_missing_values(self, metrics):
        """Test recording missing values."""
        metrics.record_dataset(1000)
        metrics.record_missing_values(field="email", count=50)
        
        assert metrics.stats.missing_values_by_field["email"] == 50
    
    def test_record_outliers(self, metrics):
        """Test recording outliers."""
        metrics.record_dataset(1000)
        metrics.record_outliers(field="amount", count=25)
        
        assert metrics.stats.outliers_by_field["amount"] == 25
    
    def test_record_schema_violation(self, metrics):
        """Test recording schema violation."""
        metrics.record_schema_violation()
        metrics.record_schema_violation()
        
        assert metrics.stats.schema_violations == 2
    
    def test_record_distribution_drift(self, metrics):
        """Test recording distribution drift."""
        metrics.record_distribution_drift(field="transaction_amount")
        
        assert metrics.stats.distribution_drifts == 1
    
    def test_get_quality_stats(self, metrics):
        """Test retrieving quality statistics."""
        metrics.record_dataset(5000)
        metrics.record_missing_values("field1", 100)
        
        stats = metrics.get_quality_stats()
        
        assert isinstance(stats, DataQualityStats)
        assert stats.total_records == 5000
    
    def test_get_quality_score(self, metrics):
        """Test calculating quality score."""
        metrics.record_dataset(1000)
        
        # Perfect data should score 100
        score = metrics.get_quality_score()
        assert score == 100.0
        
        # Add some issues
        metrics.record_missing_values("field1", 100)  # 10% missing
        metrics.record_schema_violation()
        
        score = metrics.get_quality_score()
        assert score < 100.0
        assert score > 0.0
    
    def test_reset_stats(self, metrics):
        """Test resetting statistics."""
        metrics.record_dataset(1000)
        metrics.record_missing_values("test", 50)
        metrics.record_schema_violation()
        
        assert metrics.stats.total_records > 0
        
        metrics.reset_stats()
        
        assert metrics.stats.total_records == 0
        assert len(metrics.stats.missing_values_by_field) == 0
        assert metrics.stats.schema_violations == 0


class TestBusinessMetricsIntegration:
    """Integration tests for business metrics."""
    
    def test_complete_fraud_workflow(self):
        """Test complete fraud detection workflow."""
        metrics = FraudDetectionMetrics()
        
        # Process batch of transactions
        for i in range(100):
            metrics.record_transaction(is_fraud=(i < 5))  # 5% fraud rate
        
        # Detect fraud cases
        metrics.record_fraud_detection("card_cloning", "high", 0.92, True)
        metrics.record_fraud_detection("card_cloning", "high", 0.88, True)
        metrics.record_fraud_detection("phishing", "medium", 0.75, True)
        metrics.record_fraud_detection("phishing", "low", 0.65, False)  # FP
        
        # Miss one fraud
        metrics.record_missed_fraud("account_takeover")
        
        # Record normal transactions
        for _ in range(95):
            metrics.record_normal_transaction(correctly_classified=True)
        
        stats = metrics.get_fraud_stats()
        
        # Verify stats
        assert stats.total_transactions == 100
        assert stats.total_fraud_detected == 5
        assert stats.fraud_rate == 0.05
        assert stats.true_positives == 3
        assert stats.false_positives == 1
        assert stats.false_negatives == 1
        assert stats.true_negatives == 95
        
        # Check precision and recall
        assert 0.74 < stats.precision < 0.76  # 3/(3+1) = 0.75
        assert 0.74 < stats.recall < 0.76     # 3/(3+1) = 0.75
    
    def test_complete_performance_workflow(self):
        """Test complete performance monitoring workflow."""
        metrics = PerformanceMetrics()
        
        # Generate transactions
        metrics.record_generation(count=10000, elapsed_time=1.0)
        
        # Record processing times
        for _ in range(100):
            metrics.record_feature_engineering(0.01)  # 10ms each
            metrics.record_prediction(0.001)  # 1ms each
        
        # Simulate cache behavior (80% hit rate)
        for _ in range(800):
            metrics.record_cache_hit("customer")
        for _ in range(200):
            metrics.record_cache_miss("customer")
        
        stats = metrics.get_performance_stats()
        
        # Verify stats
        assert stats.transactions_generated == 10000
        assert stats.generation_rate == 10000.0
        assert abs(stats.avg_feature_time - 0.01) < 0.001
        assert abs(stats.avg_prediction_time - 0.001) < 0.0001
        assert 0.79 < stats.get_cache_hit_rate("customer") < 0.81
    
    def test_complete_data_quality_workflow(self):
        """Test complete data quality monitoring workflow."""
        metrics = DataQualityMetrics()
        
        # Process dataset
        metrics.record_dataset(record_count=10000)
        
        # Record quality issues
        metrics.record_missing_values("email", 150)  # 1.5%
        metrics.record_missing_values("phone", 200)  # 2%
        metrics.record_outliers("amount", 50)  # 0.5%
        metrics.record_schema_violation()
        metrics.record_distribution_drift("transaction_time")
        
        stats = metrics.get_quality_stats()
        
        # Verify stats
        assert stats.total_records == 10000
        assert stats.get_missing_rate("email") == 0.015
        assert stats.get_missing_rate("phone") == 0.02
        assert stats.get_outlier_rate("amount") == 0.005
        
        # Check quality score
        score = metrics.get_quality_score()
        assert score > 80.0  # Should still be good quality
        assert score < 100.0  # But not perfect
