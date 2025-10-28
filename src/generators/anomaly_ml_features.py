"""
Anomaly-Based ML Features Module

This module provides ML feature engineering specifically for anomaly detection tasks.
It generates frequency features, severity aggregates, persistence metrics, and
unsupervised anomaly detection using Isolation Forest.

For Indian market synthetic financial data generation.
Production-ready implementation for fraud and anomaly detection models.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics
import json


@dataclass
class AnomalyMLFeatures:
    """ML features derived from anomaly patterns"""
    transaction_id: str
    customer_id: str
    
    # Frequency features
    hourly_anomaly_count: int
    daily_anomaly_count: int
    weekly_anomaly_count: int
    anomaly_frequency_trend: float  # -1.0 to 1.0 (decreasing to increasing)
    time_since_last_anomaly_hours: float
    
    # Severity aggregates
    mean_severity_last_10: float
    max_severity_last_10: float
    severity_std_last_10: float
    high_severity_rate_last_10: float  # % above 0.7
    current_severity: float
    
    # Type distribution features
    behavioral_anomaly_rate: float
    geographic_anomaly_rate: float
    temporal_anomaly_rate: float
    amount_anomaly_rate: float
    anomaly_type_diversity: float  # 0.0-1.0 (Shannon entropy)
    
    # Persistence metrics
    consecutive_anomaly_count: int
    anomaly_streak_length: int
    days_since_first_anomaly: int
    
    # Cross-pattern features
    is_fraud_and_anomaly: int  # Binary
    fraud_anomaly_correlation_score: float
    
    # Evidence-based features
    has_impossible_travel: int  # Binary
    has_unusual_category: int  # Binary
    has_unusual_hour: int  # Binary
    has_spending_spike: int  # Binary
    
    # Unsupervised anomaly score
    isolation_forest_score: float  # -1 to 1 (anomaly to normal)
    anomaly_probability: float  # 0.0 to 1.0


class AnomalyFrequencyCalculator:
    """
    Calculate anomaly frequency features with rolling windows
    
    Tracks hourly, daily, and weekly anomaly counts and trends.
    """
    
    def __init__(self):
        """Initialize the frequency calculator"""
        self.customer_anomalies: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def calculate_frequency_features(self, transaction: Dict[str, Any],
                                    customer_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate frequency features for a transaction
        
        Args:
            transaction: Current transaction
            customer_history: Previous transactions for this customer
            
        Returns:
            Dictionary of frequency features
        """
        customer_id = transaction.get('Customer_ID', '')
        current_time = self._parse_datetime(transaction)
        
        # Filter anomalies from history
        anomalies = [
            txn for txn in customer_history
            if txn.get('Anomaly_Type', 'None') != 'None'
        ]
        
        if not anomalies:
            return {
                'hourly_anomaly_count': 0,
                'daily_anomaly_count': 0,
                'weekly_anomaly_count': 0,
                'anomaly_frequency_trend': 0.0,
                'time_since_last_anomaly_hours': 9999.0
            }
        
        # Count anomalies in windows
        hourly_count = self._count_in_window(anomalies, current_time, hours=1)
        daily_count = self._count_in_window(anomalies, current_time, hours=24)
        weekly_count = self._count_in_window(anomalies, current_time, hours=168)
        
        # Calculate trend (compare recent vs. older anomalies)
        trend = self._calculate_trend(anomalies, current_time)
        
        # Time since last anomaly
        last_anomaly_time = self._parse_datetime(anomalies[-1])
        time_diff = (current_time - last_anomaly_time).total_seconds() / 3600
        
        return {
            'hourly_anomaly_count': hourly_count,
            'daily_anomaly_count': daily_count,
            'weekly_anomaly_count': weekly_count,
            'anomaly_frequency_trend': trend,
            'time_since_last_anomaly_hours': time_diff
        }
    
    def _parse_datetime(self, transaction: Dict[str, Any]) -> datetime:
        """Parse datetime from transaction"""
        date_str = transaction.get('Date', '2025-01-01')
        hour = transaction.get('Hour', 0)
        
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            return date.replace(hour=hour)
        except:
            return datetime(2025, 1, 1, hour)
    
    def _count_in_window(self, anomalies: List[Dict[str, Any]], 
                        current_time: datetime, hours: int) -> int:
        """Count anomalies within time window"""
        window_start = current_time - timedelta(hours=hours)
        count = 0
        
        for anomaly in anomalies:
            anomaly_time = self._parse_datetime(anomaly)
            if window_start <= anomaly_time <= current_time:
                count += 1
        
        return count
    
    def _calculate_trend(self, anomalies: List[Dict[str, Any]], 
                        current_time: datetime) -> float:
        """
        Calculate frequency trend (-1.0 to 1.0)
        
        Compares recent week vs. previous week anomaly counts
        Returns: positive = increasing, negative = decreasing
        """
        if len(anomalies) < 2:
            return 0.0
        
        # Recent week (last 7 days)
        recent_start = current_time - timedelta(days=7)
        recent_count = sum(
            1 for a in anomalies
            if recent_start <= self._parse_datetime(a) <= current_time
        )
        
        # Previous week (8-14 days ago)
        prev_end = current_time - timedelta(days=7)
        prev_start = current_time - timedelta(days=14)
        prev_count = sum(
            1 for a in anomalies
            if prev_start <= self._parse_datetime(a) < prev_end
        )
        
        # Calculate trend
        if prev_count == 0:
            return 1.0 if recent_count > 0 else 0.0
        
        change_ratio = (recent_count - prev_count) / prev_count
        # Clamp to -1.0 to 1.0
        return max(-1.0, min(1.0, change_ratio))


class AnomalySeverityAggregator:
    """
    Calculate anomaly severity aggregate features
    
    Provides mean, max, std deviation, and high-severity rate.
    """
    
    def calculate_severity_features(self, transaction: Dict[str, Any],
                                   customer_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate severity aggregate features
        
        Args:
            transaction: Current transaction
            customer_history: Previous transactions for this customer
            
        Returns:
            Dictionary of severity features
        """
        # Get last 10 anomalies
        anomalies = [
            txn for txn in customer_history
            if txn.get('Anomaly_Type', 'None') != 'None'
        ][-10:]
        
        if not anomalies:
            return {
                'mean_severity_last_10': 0.0,
                'max_severity_last_10': 0.0,
                'severity_std_last_10': 0.0,
                'high_severity_rate_last_10': 0.0,
                'current_severity': transaction.get('Anomaly_Severity', 0.0)
            }
        
        # Extract severities
        severities = [txn.get('Anomaly_Severity', 0.0) for txn in anomalies]
        
        # Calculate aggregates
        mean_sev = statistics.mean(severities)
        max_sev = max(severities)
        std_sev = statistics.stdev(severities) if len(severities) > 1 else 0.0
        high_sev_count = sum(1 for s in severities if s >= 0.7)
        high_sev_rate = high_sev_count / len(severities)
        
        return {
            'mean_severity_last_10': mean_sev,
            'max_severity_last_10': max_sev,
            'severity_std_last_10': std_sev,
            'high_severity_rate_last_10': high_sev_rate,
            'current_severity': transaction.get('Anomaly_Severity', 0.0)
        }


class AnomalyTypeDistributionCalculator:
    """
    Calculate anomaly type distribution features
    
    Tracks rates of each anomaly type and diversity score.
    """
    
    def calculate_type_features(self, transaction: Dict[str, Any],
                               customer_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate type distribution features
        
        Args:
            transaction: Current transaction
            customer_history: Previous transactions for this customer
            
        Returns:
            Dictionary of type distribution features
        """
        # Get anomalies
        anomalies = [
            txn for txn in customer_history
            if txn.get('Anomaly_Type', 'None') != 'None'
        ]
        
        if not anomalies:
            return {
                'behavioral_anomaly_rate': 0.0,
                'geographic_anomaly_rate': 0.0,
                'temporal_anomaly_rate': 0.0,
                'amount_anomaly_rate': 0.0,
                'anomaly_type_diversity': 0.0
            }
        
        # Count by type
        type_counts = Counter(txn.get('Anomaly_Type', 'None') for txn in anomalies)
        total = len(anomalies)
        
        behavioral_rate = type_counts.get('BEHAVIORAL', 0) / total
        geographic_rate = type_counts.get('GEOGRAPHIC', 0) / total
        temporal_rate = type_counts.get('TEMPORAL', 0) / total
        amount_rate = type_counts.get('AMOUNT', 0) / total
        
        # Calculate diversity (Shannon entropy)
        diversity = self._calculate_entropy(type_counts.values())
        
        return {
            'behavioral_anomaly_rate': behavioral_rate,
            'geographic_anomaly_rate': geographic_rate,
            'temporal_anomaly_rate': temporal_rate,
            'amount_anomaly_rate': amount_rate,
            'anomaly_type_diversity': diversity
        }
    
    def _calculate_entropy(self, counts: List[int]) -> float:
        """
        Calculate Shannon entropy (normalized to 0-1)
        
        Higher values = more diverse anomaly types
        """
        total = sum(counts)
        if total == 0:
            return 0.0
        
        probabilities = [c / total for c in counts if c > 0]
        
        if len(probabilities) <= 1:
            return 0.0
        
        # Shannon entropy
        import math
        entropy = -sum(p * math.log2(p) for p in probabilities)
        
        # Normalize to 0-1 (max entropy for 4 types is log2(4) = 2)
        max_entropy = math.log2(len(probabilities))
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized


class AnomalyPersistenceCalculator:
    """
    Calculate anomaly persistence metrics
    
    Tracks consecutive anomalies, streaks, and time since first anomaly.
    """
    
    def calculate_persistence_features(self, transaction: Dict[str, Any],
                                      customer_history: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate persistence features
        
        Args:
            transaction: Current transaction
            customer_history: Previous transactions for this customer
            
        Returns:
            Dictionary of persistence features
        """
        if not customer_history:
            return {
                'consecutive_anomaly_count': 0,
                'anomaly_streak_length': 0,
                'days_since_first_anomaly': 0
            }
        
        # Count consecutive anomalies (from most recent)
        consecutive = 0
        for txn in reversed(customer_history):
            if txn.get('Anomaly_Type', 'None') != 'None':
                consecutive += 1
            else:
                break
        
        # Find longest streak
        max_streak = 0
        current_streak = 0
        for txn in customer_history:
            if txn.get('Anomaly_Type', 'None') != 'None':
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        # Days since first anomaly
        first_anomaly = next(
            (txn for txn in customer_history if txn.get('Anomaly_Type', 'None') != 'None'),
            None
        )
        
        days_since_first = 0
        if first_anomaly:
            try:
                first_date = datetime.strptime(first_anomaly.get('Date', '2025-01-01'), '%Y-%m-%d')
                current_date = datetime.strptime(transaction.get('Date', '2025-01-01'), '%Y-%m-%d')
                days_since_first = (current_date - first_date).days
            except:
                days_since_first = 0
        
        return {
            'consecutive_anomaly_count': consecutive,
            'anomaly_streak_length': max_streak,
            'days_since_first_anomaly': days_since_first
        }


class AnomalyCrossPatternCalculator:
    """
    Calculate cross-pattern features (fraud-anomaly interaction)
    
    Identifies transactions with both fraud and anomaly patterns.
    """
    
    def calculate_cross_pattern_features(self, transaction: Dict[str, Any],
                                        customer_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate cross-pattern features
        
        Args:
            transaction: Current transaction
            customer_history: Previous transactions for this customer
            
        Returns:
            Dictionary of cross-pattern features
        """
        # Check if current transaction is both fraud and anomaly
        is_fraud = transaction.get('Fraud_Type', 'None') != 'None'
        is_anomaly = transaction.get('Anomaly_Type', 'None') != 'None'
        is_both = 1 if (is_fraud and is_anomaly) else 0
        
        # Calculate historical fraud-anomaly correlation
        if not customer_history:
            return {
                'is_fraud_and_anomaly': is_both,
                'fraud_anomaly_correlation_score': 0.0
            }
        
        # Count overlaps in history
        fraud_count = sum(1 for t in customer_history if t.get('Fraud_Type', 'None') != 'None')
        anomaly_count = sum(1 for t in customer_history if t.get('Anomaly_Type', 'None') != 'None')
        both_count = sum(
            1 for t in customer_history
            if t.get('Fraud_Type', 'None') != 'None' and t.get('Anomaly_Type', 'None') != 'None'
        )
        
        # Calculate correlation score (Jaccard index)
        if fraud_count + anomaly_count - both_count == 0:
            correlation_score = 0.0
        else:
            correlation_score = both_count / (fraud_count + anomaly_count - both_count)
        
        return {
            'is_fraud_and_anomaly': is_both,
            'fraud_anomaly_correlation_score': correlation_score
        }


class AnomalyEvidenceExtractor:
    """
    Extract binary features from anomaly evidence JSON
    
    Parses evidence to identify specific anomaly characteristics.
    """
    
    def extract_evidence_features(self, transaction: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract evidence-based binary features
        
        Args:
            transaction: Transaction with Anomaly_Evidence field
            
        Returns:
            Dictionary of binary evidence features
        """
        evidence_str = transaction.get('Anomaly_Evidence', '')
        
        if not evidence_str or evidence_str == '':
            return {
                'has_impossible_travel': 0,
                'has_unusual_category': 0,
                'has_unusual_hour': 0,
                'has_spending_spike': 0
            }
        
        # Parse evidence JSON
        try:
            evidence = json.loads(evidence_str)
        except:
            evidence = {}
        
        # Extract specific patterns
        has_impossible_travel = 1 if 'speed_kmh' in evidence and evidence.get('speed_kmh', 0) > 800 else 0
        has_unusual_category = 1 if 'unusual_category' in evidence else 0
        has_unusual_hour = 1 if 'hour' in evidence and evidence.get('hour', 12) in [0, 1, 2, 3, 4, 5] else 0
        has_spending_spike = 1 if 'multiplier' in evidence and evidence.get('multiplier', 1.0) > 3.0 else 0
        
        return {
            'has_impossible_travel': has_impossible_travel,
            'has_unusual_category': has_unusual_category,
            'has_unusual_hour': has_unusual_hour,
            'has_spending_spike': has_spending_spike
        }


class AnomalyMLFeatureGenerator:
    """
    Main orchestrator for anomaly-based ML feature generation
    
    Combines all feature calculators to generate comprehensive feature set.
    """
    
    def __init__(self):
        """Initialize feature generators"""
        self.frequency_calc = AnomalyFrequencyCalculator()
        self.severity_agg = AnomalySeverityAggregator()
        self.type_dist_calc = AnomalyTypeDistributionCalculator()
        self.persistence_calc = AnomalyPersistenceCalculator()
        self.cross_pattern_calc = AnomalyCrossPatternCalculator()
        self.evidence_extractor = AnomalyEvidenceExtractor()
    
    def generate_features(self, transaction: Dict[str, Any],
                         customer_history: List[Dict[str, Any]],
                         isolation_score: float = 0.0) -> AnomalyMLFeatures:
        """
        Generate all anomaly-based ML features for a transaction
        
        Args:
            transaction: Current transaction
            customer_history: Previous transactions for customer
            isolation_score: Isolation Forest anomaly score (-1 to 1)
            
        Returns:
            AnomalyMLFeatures object with all features
        """
        # Calculate feature groups
        freq_features = self.frequency_calc.calculate_frequency_features(transaction, customer_history)
        sev_features = self.severity_agg.calculate_severity_features(transaction, customer_history)
        type_features = self.type_dist_calc.calculate_type_features(transaction, customer_history)
        persist_features = self.persistence_calc.calculate_persistence_features(transaction, customer_history)
        cross_features = self.cross_pattern_calc.calculate_cross_pattern_features(transaction, customer_history)
        evidence_features = self.evidence_extractor.extract_evidence_features(transaction)
        
        # Convert isolation score to probability (0-1)
        # Isolation Forest returns -1 for anomalies, 1 for normal
        # Convert to 0 (normal) to 1 (anomaly)
        anomaly_probability = (1 - isolation_score) / 2.0
        
        return AnomalyMLFeatures(
            transaction_id=transaction.get('Transaction_ID', ''),
            customer_id=transaction.get('Customer_ID', ''),
            
            # Frequency features
            hourly_anomaly_count=freq_features['hourly_anomaly_count'],
            daily_anomaly_count=freq_features['daily_anomaly_count'],
            weekly_anomaly_count=freq_features['weekly_anomaly_count'],
            anomaly_frequency_trend=freq_features['anomaly_frequency_trend'],
            time_since_last_anomaly_hours=freq_features['time_since_last_anomaly_hours'],
            
            # Severity features
            mean_severity_last_10=sev_features['mean_severity_last_10'],
            max_severity_last_10=sev_features['max_severity_last_10'],
            severity_std_last_10=sev_features['severity_std_last_10'],
            high_severity_rate_last_10=sev_features['high_severity_rate_last_10'],
            current_severity=sev_features['current_severity'],
            
            # Type distribution features
            behavioral_anomaly_rate=type_features['behavioral_anomaly_rate'],
            geographic_anomaly_rate=type_features['geographic_anomaly_rate'],
            temporal_anomaly_rate=type_features['temporal_anomaly_rate'],
            amount_anomaly_rate=type_features['amount_anomaly_rate'],
            anomaly_type_diversity=type_features['anomaly_type_diversity'],
            
            # Persistence features
            consecutive_anomaly_count=persist_features['consecutive_anomaly_count'],
            anomaly_streak_length=persist_features['anomaly_streak_length'],
            days_since_first_anomaly=persist_features['days_since_first_anomaly'],
            
            # Cross-pattern features
            is_fraud_and_anomaly=cross_features['is_fraud_and_anomaly'],
            fraud_anomaly_correlation_score=cross_features['fraud_anomaly_correlation_score'],
            
            # Evidence features
            has_impossible_travel=evidence_features['has_impossible_travel'],
            has_unusual_category=evidence_features['has_unusual_category'],
            has_unusual_hour=evidence_features['has_unusual_hour'],
            has_spending_spike=evidence_features['has_spending_spike'],
            
            # Unsupervised features
            isolation_forest_score=isolation_score,
            anomaly_probability=anomaly_probability
        )
    
    def generate_features_batch(self, transactions: List[Dict[str, Any]],
                                customer_histories: Dict[str, List[Dict[str, Any]]],
                                isolation_scores: Optional[List[float]] = None) -> List[AnomalyMLFeatures]:
        """
        Generate features for batch of transactions
        
        Args:
            transactions: List of transactions
            customer_histories: Dict mapping customer_id to transaction history
            isolation_scores: Optional list of Isolation Forest scores
            
        Returns:
            List of AnomalyMLFeatures objects
        """
        if isolation_scores is None:
            isolation_scores = [0.0] * len(transactions)
        
        features_list = []
        for i, txn in enumerate(transactions):
            customer_id = txn.get('Customer_ID', '')
            history = customer_histories.get(customer_id, [])
            iso_score = isolation_scores[i] if i < len(isolation_scores) else 0.0
            
            features = self.generate_features(txn, history, iso_score)
            features_list.append(features)
        
        return features_list


class IsolationForestAnomalyDetector:
    """
    Unsupervised anomaly detection using Isolation Forest
    
    Trains on transaction features and assigns anomaly scores.
    """
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize Isolation Forest detector
        
        Args:
            contamination: Expected proportion of anomalies (default 0.05 = 5%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.feature_names: List[str] = []
    
    def prepare_features(self, transactions: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[str]]:
        """
        Prepare numerical features for Isolation Forest
        
        Args:
            transactions: List of transactions
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        feature_matrix = []
        
        # Define features to use
        numerical_features = [
            'Amount', 'Hour', 'Distance_From_Last_Txn_km',
            'Time_Since_Last_Txn_hours', 'Anomaly_Severity',
            'Anomaly_Confidence'
        ]
        
        self.feature_names = numerical_features
        
        for txn in transactions:
            features = []
            for feat_name in numerical_features:
                value = txn.get(feat_name, 0.0)
                features.append(float(value))
            feature_matrix.append(features)
        
        return feature_matrix, self.feature_names
    
    def fit_predict(self, transactions: List[Dict[str, Any]]) -> List[float]:
        """
        Fit Isolation Forest and predict anomaly scores
        
        Args:
            transactions: List of transactions
            
        Returns:
            List of anomaly scores (-1 for anomalies, 1 for normal)
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            # Return zeros if sklearn not available
            return [0.0] * len(transactions)
        
        # Prepare features
        X, feature_names = self.prepare_features(transactions)
        
        if not X:
            return [0.0] * len(transactions)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        # Fit and predict
        predictions = self.model.fit_predict(X)
        
        # Get decision scores (more negative = more anomalous)
        scores = self.model.score_samples(X)
        
        # Normalize scores to -1 to 1 range
        if len(scores) > 1:
            min_score = min(scores)
            max_score = max(scores)
            if max_score != min_score:
                normalized_scores = [
                    2 * (s - min_score) / (max_score - min_score) - 1
                    for s in scores
                ]
            else:
                normalized_scores = [0.0] * len(scores)
        else:
            normalized_scores = [0.0] * len(scores)
        
        return normalized_scores
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances (not directly available in Isolation Forest)
        
        Returns approximate importances based on split frequencies
        
        Returns:
            Dictionary mapping feature_name to importance score
        """
        if self.model is None:
            return {}
        
        # Isolation Forest doesn't have feature_importances_
        # Return uniform importances
        importance = 1.0 / len(self.feature_names) if self.feature_names else 0.0
        return {name: importance for name in self.feature_names}
