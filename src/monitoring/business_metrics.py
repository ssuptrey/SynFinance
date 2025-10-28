"""
Custom business metrics for SynFinance fraud detection.

Provides domain-specific metrics including:
- Fraud detection metrics (rates, types, confidence)
- Performance metrics (generation, prediction, caching)
- Data quality metrics (missing values, outliers, drift)
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from .prometheus_exporter import get_metrics_exporter

logger = logging.getLogger(__name__)


@dataclass
class FraudStats:
    """Statistics for fraud detection."""
    
    total_transactions: int = 0
    total_fraud_detected: int = 0
    fraud_by_type: Dict[str, int] = field(default_factory=dict)
    fraud_by_severity: Dict[str, int] = field(default_factory=dict)
    confidence_scores: List[float] = field(default_factory=list)
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    
    @property
    def fraud_rate(self) -> float:
        """Calculate overall fraud rate."""
        if self.total_transactions == 0:
            return 0.0
        return self.total_fraud_detected / self.total_transactions
    
    @property
    def precision(self) -> float:
        """Calculate precision (TP / (TP + FP))."""
        predicted_positive = self.true_positives + self.false_positives
        if predicted_positive == 0:
            return 0.0
        return self.true_positives / predicted_positive
    
    @property
    def recall(self) -> float:
        """Calculate recall (TP / (TP + FN))."""
        actual_positive = self.true_positives + self.false_negatives
        if actual_positive == 0:
            return 0.0
        return self.true_positives / actual_positive
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def average_confidence(self) -> float:
        """Calculate average fraud confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


class FraudDetectionMetrics:
    """
    Metrics for fraud detection performance.
    
    Tracks:
    - Fraud detection rates by type and severity
    - Precision, recall, F1 score
    - Confidence score distributions
    - Anomaly detection rates
    """
    
    def __init__(self):
        """Initialize fraud detection metrics."""
        self.exporter = get_metrics_exporter()
        self.stats = FraudStats()
        self.window_size = timedelta(minutes=5)  # Rolling window for rate calculations
        self.recent_detections: List[tuple] = []  # (timestamp, fraud_type, severity, confidence)
        
        logger.info("Initialized FraudDetectionMetrics")
    
    def record_transaction(self, is_fraud: bool = False):
        """
        Record a transaction.
        
        Args:
            is_fraud: Whether transaction is fraudulent
        """
        self.stats.total_transactions += 1
        if is_fraud:
            self.stats.total_fraud_detected += 1
    
    def record_fraud_detection(
        self,
        fraud_type: str,
        severity: str,
        confidence: float,
        actual_fraud: bool = True
    ):
        """
        Record a fraud detection event.
        
        Args:
            fraud_type: Type of fraud (card_cloning, account_takeover, etc.)
            severity: Severity level (low, medium, high, critical)
            confidence: Detection confidence (0-1)
            actual_fraud: Whether it was actually fraud (for precision/recall)
        """
        # Update Prometheus metrics
        self.exporter.record_fraud_detection(fraud_type, severity, confidence)
        
        # Update internal stats
        self.stats.fraud_by_type[fraud_type] = self.stats.fraud_by_type.get(fraud_type, 0) + 1
        self.stats.fraud_by_severity[severity] = self.stats.fraud_by_severity.get(severity, 0) + 1
        self.stats.confidence_scores.append(confidence)
        
        # Update confusion matrix
        if actual_fraud:
            self.stats.true_positives += 1
        else:
            self.stats.false_positives += 1
        
        # Add to recent detections window
        self.recent_detections.append((datetime.now(), fraud_type, severity, confidence))
        self._cleanup_old_detections()
        
        # Update fraud rates
        self._update_fraud_rates()
        
        logger.debug(f"Recorded fraud detection: {fraud_type}, severity: {severity}, confidence: {confidence:.3f}")
    
    def record_missed_fraud(self, fraud_type: str):
        """
        Record a false negative (fraud not detected).
        
        Args:
            fraud_type: Type of fraud that was missed
        """
        self.stats.false_negatives += 1
        logger.warning(f"Missed fraud detection: {fraud_type}")
    
    def record_normal_transaction(self, correctly_classified: bool = True):
        """
        Record a normal (non-fraud) transaction.
        
        Args:
            correctly_classified: Whether it was correctly classified as normal
        """
        if correctly_classified:
            self.stats.true_negatives += 1
    
    def _cleanup_old_detections(self):
        """Remove detections older than window size."""
        cutoff_time = datetime.now() - self.window_size
        self.recent_detections = [
            det for det in self.recent_detections
            if det[0] > cutoff_time
        ]
    
    def _update_fraud_rates(self):
        """Update fraud rate metrics."""
        # Overall fraud rate
        if self.stats.total_transactions > 0:
            overall_rate = self.stats.total_fraud_detected / self.stats.total_transactions
            self.exporter.update_fraud_rate('overall', overall_rate)
        
        # Per-type fraud rates
        for fraud_type, count in self.stats.fraud_by_type.items():
            if self.stats.total_transactions > 0:
                rate = count / self.stats.total_transactions
                self.exporter.update_fraud_rate(fraud_type, rate)
    
    def get_fraud_stats(self) -> FraudStats:
        """
        Get current fraud statistics.
        
        Returns:
            FraudStats with current metrics
        """
        return self.stats
    
    def get_recent_fraud_rate(self, fraud_type: Optional[str] = None) -> float:
        """
        Get fraud rate in recent window.
        
        Args:
            fraud_type: Specific fraud type, or None for all types
            
        Returns:
            Fraud rate in recent window
        """
        if not self.recent_detections:
            return 0.0
        
        if fraud_type:
            count = sum(1 for det in self.recent_detections if det[1] == fraud_type)
        else:
            count = len(self.recent_detections)
        
        # Estimate transactions in window (assuming constant rate)
        window_minutes = self.window_size.total_seconds() / 60
        estimated_transactions = self.stats.total_transactions / window_minutes if window_minutes > 0 else 0
        
        if estimated_transactions == 0:
            return 0.0
        
        return count / estimated_transactions
    
    def reset_stats(self):
        """Reset all statistics."""
        self.stats = FraudStats()
        self.recent_detections = []
        logger.info("Reset fraud detection statistics")


@dataclass
class PerformanceStats:
    """Statistics for performance metrics."""
    
    transactions_generated: int = 0
    total_generation_time: float = 0.0
    total_feature_engineering_time: float = 0.0
    total_prediction_time: float = 0.0
    cache_hits_by_type: Dict[str, int] = field(default_factory=dict)
    cache_misses_by_type: Dict[str, int] = field(default_factory=dict)
    
    @property
    def avg_generation_time(self) -> float:
        """Average time per transaction generation."""
        if self.transactions_generated == 0:
            return 0.0
        return self.total_generation_time / self.transactions_generated
    
    @property
    def generation_rate(self) -> float:
        """Transactions per second generation rate."""
        if self.total_generation_time == 0:
            return 0.0
        return self.transactions_generated / self.total_generation_time
    
    @property
    def avg_feature_time(self) -> float:
        """Average feature engineering time."""
        if self.transactions_generated == 0:
            return 0.0
        return self.total_feature_engineering_time / self.transactions_generated
    
    @property
    def avg_prediction_time(self) -> float:
        """Average prediction time."""
        if self.transactions_generated == 0:
            return 0.0
        return self.total_prediction_time / self.transactions_generated
    
    def get_cache_hit_rate(self, cache_type: str) -> float:
        """Get cache hit rate for specific cache type."""
        hits = self.cache_hits_by_type.get(cache_type, 0)
        misses = self.cache_misses_by_type.get(cache_type, 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0


class PerformanceMetrics:
    """
    Metrics for system performance.
    
    Tracks:
    - Transaction generation rate
    - Feature engineering performance
    - Model prediction latency
    - Cache hit rates
    """
    
    def __init__(self):
        """Initialize performance metrics."""
        self.exporter = get_metrics_exporter()
        self.stats = PerformanceStats()
        
        logger.info("Initialized PerformanceMetrics")
    
    def record_generation(self, count: int, elapsed_time: float):
        """
        Record transaction generation.
        
        Args:
            count: Number of transactions generated
            elapsed_time: Time taken in seconds
        """
        self.stats.transactions_generated += count
        self.stats.total_generation_time += elapsed_time
        
        # Update Prometheus metrics
        self.exporter.transactions_generated.inc(count)
        
        # Calculate and update rate
        rate = count / elapsed_time if elapsed_time > 0 else 0
        self.exporter.generation_rate.set(rate)
        
        logger.debug(f"Generated {count} transactions in {elapsed_time:.3f}s ({rate:.0f} txn/sec)")
    
    def record_feature_engineering(self, elapsed_time: float):
        """
        Record feature engineering time.
        
        Args:
            elapsed_time: Time taken in seconds
        """
        self.stats.total_feature_engineering_time += elapsed_time
        self.exporter.feature_engineering_time.observe(elapsed_time)
        
        logger.debug(f"Feature engineering took {elapsed_time:.4f}s")
    
    def record_prediction(self, elapsed_time: float):
        """
        Record model prediction time.
        
        Args:
            elapsed_time: Time taken in seconds
        """
        self.stats.total_prediction_time += elapsed_time
        self.exporter.prediction_time.observe(elapsed_time)
        
        logger.debug(f"Prediction took {elapsed_time:.4f}s")
    
    def record_cache_hit(self, cache_type: str):
        """
        Record cache hit.
        
        Args:
            cache_type: Type of cache (customer, merchant, features, etc.)
        """
        self.stats.cache_hits_by_type[cache_type] = self.stats.cache_hits_by_type.get(cache_type, 0) + 1
        self.exporter.cache_hits.labels(cache_type=cache_type).inc()
        
        # Update cache hit rate
        self._update_cache_metrics(cache_type)
    
    def record_cache_miss(self, cache_type: str):
        """
        Record cache miss.
        
        Args:
            cache_type: Type of cache
        """
        self.stats.cache_misses_by_type[cache_type] = self.stats.cache_misses_by_type.get(cache_type, 0) + 1
        self.exporter.cache_misses.labels(cache_type=cache_type).inc()
        
        # Update cache hit rate
        self._update_cache_metrics(cache_type)
    
    def _update_cache_metrics(self, cache_type: str):
        """Update cache metrics."""
        hits = self.stats.cache_hits_by_type.get(cache_type, 0)
        misses = self.stats.cache_misses_by_type.get(cache_type, 0)
        self.exporter.update_cache_metrics(cache_type, hits, misses)
    
    def get_performance_stats(self) -> PerformanceStats:
        """
        Get current performance statistics.
        
        Returns:
            PerformanceStats with current metrics
        """
        return self.stats
    
    def reset_stats(self):
        """Reset all statistics."""
        self.stats = PerformanceStats()
        logger.info("Reset performance statistics")


@dataclass
class DataQualityStats:
    """Statistics for data quality metrics."""
    
    total_records: int = 0
    missing_values_by_field: Dict[str, int] = field(default_factory=dict)
    outliers_by_field: Dict[str, int] = field(default_factory=dict)
    schema_violations: int = 0
    distribution_drifts: int = 0
    
    def get_missing_rate(self, field: str) -> float:
        """Get missing value rate for field."""
        if self.total_records == 0:
            return 0.0
        missing = self.missing_values_by_field.get(field, 0)
        return missing / self.total_records
    
    def get_outlier_rate(self, field: str) -> float:
        """Get outlier rate for field."""
        if self.total_records == 0:
            return 0.0
        outliers = self.outliers_by_field.get(field, 0)
        return outliers / self.total_records
    
    @property
    def overall_missing_rate(self) -> float:
        """Overall missing value rate across all fields."""
        if self.total_records == 0:
            return 0.0
        total_missing = sum(self.missing_values_by_field.values())
        total_fields = len(self.missing_values_by_field)
        if total_fields == 0:
            return 0.0
        return total_missing / (self.total_records * total_fields)


class DataQualityMetrics:
    """
    Metrics for data quality monitoring.
    
    Tracks:
    - Missing value rates per field
    - Outlier detection counts
    - Schema validation failures
    - Distribution drift detection
    """
    
    def __init__(self):
        """Initialize data quality metrics."""
        self.exporter = get_metrics_exporter()
        self.stats = DataQualityStats()
        
        # Register data quality specific metrics
        self.missing_values_gauge = self.exporter.registry.register_gauge(
            'data_missing_values_rate',
            'Missing value rate per field',
            labels=['field']
        )
        
        self.outliers_gauge = self.exporter.registry.register_gauge(
            'data_outliers_rate',
            'Outlier rate per field',
            labels=['field']
        )
        
        self.schema_violations_counter = self.exporter.registry.register_counter(
            'data_schema_violations_total',
            'Total schema validation violations'
        )
        
        self.drift_detections_counter = self.exporter.registry.register_counter(
            'data_distribution_drifts_total',
            'Total distribution drift detections',
            labels=['field']
        )
        
        logger.info("Initialized DataQualityMetrics")
    
    def record_dataset(self, record_count: int):
        """
        Record dataset size.
        
        Args:
            record_count: Number of records in dataset
        """
        self.stats.total_records += record_count
    
    def record_missing_values(self, field: str, count: int):
        """
        Record missing values for a field.
        
        Args:
            field: Field name
            count: Number of missing values
        """
        self.stats.missing_values_by_field[field] = self.stats.missing_values_by_field.get(field, 0) + count
        
        # Update gauge
        rate = self.stats.get_missing_rate(field)
        self.missing_values_gauge.labels(field=field).set(rate)
        
        logger.debug(f"Field '{field}' has {count} missing values ({rate:.2%} rate)")
    
    def record_outliers(self, field: str, count: int):
        """
        Record outliers for a field.
        
        Args:
            field: Field name
            count: Number of outliers detected
        """
        self.stats.outliers_by_field[field] = self.stats.outliers_by_field.get(field, 0) + count
        
        # Update gauge
        rate = self.stats.get_outlier_rate(field)
        self.outliers_gauge.labels(field=field).set(rate)
        
        logger.debug(f"Field '{field}' has {count} outliers ({rate:.2%} rate)")
    
    def record_schema_violation(self):
        """Record a schema validation violation."""
        self.stats.schema_violations += 1
        self.schema_violations_counter.inc()
        
        logger.warning("Schema validation violation detected")
    
    def record_distribution_drift(self, field: str):
        """
        Record distribution drift detection.
        
        Args:
            field: Field with detected drift
        """
        self.stats.distribution_drifts += 1
        self.drift_detections_counter.labels(field=field).inc()
        
        logger.warning(f"Distribution drift detected in field: {field}")
    
    def get_quality_stats(self) -> DataQualityStats:
        """
        Get current data quality statistics.
        
        Returns:
            DataQualityStats with current metrics
        """
        return self.stats
    
    def get_quality_score(self) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Returns:
            Quality score
        """
        # Penalize for missing values, outliers, violations, drifts
        penalties = 0.0
        
        # Missing values penalty (max 30 points)
        missing_penalty = min(self.stats.overall_missing_rate * 100, 30)
        penalties += missing_penalty
        
        # Schema violations penalty (max 20 points)
        if self.stats.total_records > 0:
            violation_rate = self.stats.schema_violations / self.stats.total_records
            violation_penalty = min(violation_rate * 100, 20)
            penalties += violation_penalty
        
        # Drift penalty (max 10 points)
        if self.stats.total_records > 0:
            drift_rate = self.stats.distribution_drifts / self.stats.total_records
            drift_penalty = min(drift_rate * 100, 10)
            penalties += drift_penalty
        
        # Calculate score
        score = max(0, 100 - penalties)
        
        return score
    
    def reset_stats(self):
        """Reset all statistics."""
        self.stats = DataQualityStats()
        logger.info("Reset data quality statistics")
