"""
Combined ML Features Module

This module integrates fraud-based and anomaly-based ML features into a unified
feature set with additional interaction features. It provides a comprehensive
feature engineering pipeline for production fraud detection models.

Features Include:
- 32 Fraud-based features (from ml_features.py)
- 26 Anomaly-based features (from anomaly_ml_features.py)
- 10 Interaction features (fraud-anomaly combinations)

Total: 68 comprehensive ML features for production models.

Author: SynFinance Development Team
Version: 0.7.0
Date: October 28, 2025
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
import math


@dataclass
class CombinedMLFeatures:
    """
    Unified ML feature set combining fraud and anomaly features
    
    Total: 68 features across 9 categories
    """
    
    # Transaction identifier
    transaction_id: str
    customer_id: str
    
    # ==================== FRAUD FEATURES (32) ====================
    
    # Aggregate Features (6)
    daily_txn_count: int
    weekly_txn_count: int
    daily_txn_amount: float
    weekly_txn_amount: float
    avg_daily_amount: float
    avg_weekly_amount: float
    
    # Velocity Features (6)
    txn_frequency_1h: int
    txn_frequency_6h: int
    txn_frequency_24h: int
    amount_velocity_1h: float
    amount_velocity_6h: float
    amount_velocity_24h: float
    
    # Geographic Features (5)
    distance_from_home: float
    avg_distance_last_10: float
    distance_variance: float
    unique_cities_7d: int
    travel_velocity_kmh: float
    
    # Temporal Features (5)
    is_unusual_hour: int
    is_weekend: int
    is_holiday: int
    hour_of_day: int
    day_of_week: int
    
    # Behavioral Features (6)
    category_diversity_score: float
    merchant_loyalty_score: float
    avg_merchant_reputation: float
    new_merchant_flag: int
    refund_rate_30d: float
    declined_rate_7d: float
    
    # Network Features (4)
    shared_merchant_count: int
    shared_location_count: int
    customer_proximity_score: float
    temporal_cluster_flag: int
    
    # ==================== ANOMALY FEATURES (26) ====================
    
    # Frequency Features (5)
    hourly_anomaly_count: int
    daily_anomaly_count: int
    weekly_anomaly_count: int
    anomaly_frequency_trend: float
    time_since_last_anomaly_hours: float
    
    # Severity Features (5)
    mean_severity_last_10: float
    max_severity_last_10: float
    severity_std_last_10: float
    high_severity_rate_last_10: float
    current_severity: float
    
    # Type Distribution Features (5)
    behavioral_anomaly_rate: float
    geographic_anomaly_rate: float
    temporal_anomaly_rate: float
    amount_anomaly_rate: float
    anomaly_type_diversity: float
    
    # Persistence Features (3)
    consecutive_anomaly_count: int
    anomaly_streak_length: int
    days_since_first_anomaly: int
    
    # Cross-Pattern Features (2)
    is_fraud_and_anomaly: int
    fraud_anomaly_correlation_score: float
    
    # Evidence-Based Features (4)
    has_impossible_travel: int
    has_unusual_category: int
    has_unusual_hour: int
    has_spending_spike: int
    
    # Unsupervised Features (2)
    isolation_forest_score: float
    anomaly_probability: float
    
    # ==================== INTERACTION FEATURES (10) ====================
    
    # Risk Amplification Features (3)
    high_risk_combination: int  # Fraud + anomaly + high velocity
    risk_amplification_score: float  # Combined fraud-anomaly risk
    compound_severity_score: float  # Fraud severity * anomaly severity
    
    # Behavioral Consistency Features (3)
    behavioral_consistency_score: float  # Agreement between fraud/anomaly signals
    pattern_alignment_score: float  # How well fraud and anomaly patterns align
    conflict_indicator: int  # Binary: fraud says yes, anomaly says no (or vice versa)
    
    # Temporal-Spatial Interaction (2)
    velocity_severity_product: float  # Travel velocity * anomaly severity
    geographic_risk_score: float  # Distance from home * geographic anomaly rate
    
    # Advanced Composite Features (2)
    weighted_risk_score: float  # Weighted combination of all risk signals
    ensemble_fraud_probability: float  # Ensemble prediction from all features
    
    # ==================== LABELS ====================
    
    is_fraud: int  # 0 or 1
    fraud_type: Optional[str] = None
    anomaly_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary for export"""
        return asdict(self)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names (excluding IDs and labels)"""
        exclude_fields = {
            'transaction_id', 'customer_id', 'is_fraud', 
            'fraud_type', 'anomaly_type'
        }
        return [k for k in asdict(self).keys() if k not in exclude_fields]
    
    def get_feature_values(self) -> List[float]:
        """Get list of all feature values (excluding IDs and labels)"""
        exclude_fields = {
            'transaction_id', 'customer_id', 'is_fraud', 
            'fraud_type', 'anomaly_type'
        }
        data = asdict(self)
        return [float(v) if v is not None else 0.0 
                for k, v in data.items() if k not in exclude_fields]


class InteractionFeatureCalculator:
    """
    Calculate interaction features between fraud and anomaly signals
    
    Generates 10 advanced composite features that combine fraud-based
    and anomaly-based signals for improved detection accuracy.
    """
    
    def __init__(self):
        """Initialize the interaction feature calculator"""
        self.feature_weights = {
            'fraud_velocity': 0.3,
            'fraud_geographic': 0.25,
            'fraud_behavioral': 0.20,
            'anomaly_severity': 0.15,
            'anomaly_frequency': 0.10
        }
    
    def calculate_interaction_features(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate all 10 interaction features
        
        Args:
            fraud_features: Dictionary of fraud-based features
            anomaly_features: Dictionary of anomaly-based features
            
        Returns:
            Dictionary of 10 interaction features
        """
        
        # Risk Amplification Features (3)
        high_risk_combination = self._calculate_high_risk_combination(
            fraud_features, anomaly_features
        )
        risk_amplification_score = self._calculate_risk_amplification(
            fraud_features, anomaly_features
        )
        compound_severity_score = self._calculate_compound_severity(
            fraud_features, anomaly_features
        )
        
        # Behavioral Consistency Features (3)
        behavioral_consistency = self._calculate_behavioral_consistency(
            fraud_features, anomaly_features
        )
        pattern_alignment = self._calculate_pattern_alignment(
            fraud_features, anomaly_features
        )
        conflict_indicator = self._calculate_conflict_indicator(
            fraud_features, anomaly_features
        )
        
        # Temporal-Spatial Interaction (2)
        velocity_severity_product = self._calculate_velocity_severity(
            fraud_features, anomaly_features
        )
        geographic_risk_score = self._calculate_geographic_risk(
            fraud_features, anomaly_features
        )
        
        # Advanced Composite Features (2)
        weighted_risk_score = self._calculate_weighted_risk(
            fraud_features, anomaly_features
        )
        ensemble_fraud_probability = self._calculate_ensemble_probability(
            fraud_features, anomaly_features
        )
        
        return {
            'high_risk_combination': high_risk_combination,
            'risk_amplification_score': risk_amplification_score,
            'compound_severity_score': compound_severity_score,
            'behavioral_consistency_score': behavioral_consistency,
            'pattern_alignment_score': pattern_alignment,
            'conflict_indicator': conflict_indicator,
            'velocity_severity_product': velocity_severity_product,
            'geographic_risk_score': geographic_risk_score,
            'weighted_risk_score': weighted_risk_score,
            'ensemble_fraud_probability': ensemble_fraud_probability
        }
    
    def _calculate_high_risk_combination(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> int:
        """
        Detect high-risk combination of fraud, anomaly, and high velocity
        
        Returns 1 if all three conditions are met, 0 otherwise
        """
        is_fraud = fraud_features.get('is_fraud', 0)
        has_anomaly = anomaly_features.get('is_fraud_and_anomaly', 0)
        high_velocity = fraud_features.get('txn_frequency_1h', 0) >= 5
        
        return 1 if (is_fraud and has_anomaly and high_velocity) else 0
    
    def _calculate_risk_amplification(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """
        Calculate risk amplification when fraud and anomaly signals combine
        
        Returns score from 0.0 to 1.0
        """
        # Normalize fraud signals
        velocity_score = min(fraud_features.get('txn_frequency_1h', 0) / 10.0, 1.0)
        distance_score = min(fraud_features.get('distance_from_home', 0) / 1000.0, 1.0)
        
        # Normalize anomaly signals
        severity_score = anomaly_features.get('current_severity', 0.0)
        frequency_score = min(anomaly_features.get('daily_anomaly_count', 0) / 10.0, 1.0)
        
        # Calculate amplification (non-linear interaction)
        fraud_signal = (velocity_score + distance_score) / 2.0
        anomaly_signal = (severity_score + frequency_score) / 2.0
        
        # Amplification is greater than linear sum
        amplification = (fraud_signal * anomaly_signal) + \
                       (fraud_signal + anomaly_signal) / 2.0
        
        return min(amplification, 1.0)
    
    def _calculate_compound_severity(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """
        Calculate compound severity score
        
        Multiplies fraud severity and anomaly severity for non-linear risk
        """
        # Estimate fraud severity from multiple signals
        fraud_severity = 0.0
        if fraud_features.get('is_fraud', 0) == 1:
            fraud_severity = min(
                (fraud_features.get('txn_frequency_1h', 0) * 0.1 +
                 fraud_features.get('distance_from_home', 0) / 1000.0 +
                 fraud_features.get('travel_velocity_kmh', 0) / 500.0) / 3.0,
                1.0
            )
        
        # Get anomaly severity
        anomaly_severity = anomaly_features.get('current_severity', 0.0)
        
        # Compound effect
        return fraud_severity * anomaly_severity
    
    def _calculate_behavioral_consistency(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """
        Measure consistency between fraud and anomaly behavioral signals
        
        Returns score from 0.0 (inconsistent) to 1.0 (highly consistent)
        """
        # Compare behavioral signals
        fraud_behavioral_score = fraud_features.get('category_diversity_score', 0.0)
        anomaly_behavioral_rate = anomaly_features.get('behavioral_anomaly_rate', 0.0)
        
        # Calculate agreement
        if fraud_behavioral_score == 0 and anomaly_behavioral_rate == 0:
            return 1.0  # Both agree: no behavioral issues
        
        # Normalize and compare
        normalized_fraud = min(fraud_behavioral_score, 1.0)
        normalized_anomaly = anomaly_behavioral_rate
        
        # Consistency is inverse of difference
        difference = abs(normalized_fraud - normalized_anomaly)
        consistency = 1.0 - difference
        
        return max(consistency, 0.0)
    
    def _calculate_pattern_alignment(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """
        Calculate how well fraud and anomaly patterns align
        
        Returns score from 0.0 (misaligned) to 1.0 (perfectly aligned)
        """
        # Check multiple pattern dimensions
        temporal_align = self._check_temporal_alignment(fraud_features, anomaly_features)
        geographic_align = self._check_geographic_alignment(fraud_features, anomaly_features)
        behavioral_align = self._check_behavioral_alignment(fraud_features, anomaly_features)
        
        # Average alignment across dimensions
        alignment = (temporal_align + geographic_align + behavioral_align) / 3.0
        
        return alignment
    
    def _check_temporal_alignment(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """Check temporal pattern alignment"""
        fraud_unusual_hour = fraud_features.get('is_unusual_hour', 0)
        anomaly_temporal_rate = anomaly_features.get('temporal_anomaly_rate', 0.0)
        
        if fraud_unusual_hour == 1 and anomaly_temporal_rate > 0.5:
            return 1.0  # Strong alignment
        elif fraud_unusual_hour == 0 and anomaly_temporal_rate < 0.5:
            return 1.0  # Both agree: normal temporal pattern
        else:
            return max(1.0 - abs(fraud_unusual_hour - anomaly_temporal_rate), 0.0)
    
    def _check_geographic_alignment(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """Check geographic pattern alignment"""
        distance_score = min(fraud_features.get('distance_from_home', 0) / 1000.0, 1.0)
        geographic_anomaly_rate = anomaly_features.get('geographic_anomaly_rate', 0.0)
        
        # High distance should align with high geographic anomaly rate
        alignment = 1.0 - abs(distance_score - geographic_anomaly_rate)
        return max(alignment, 0.0)
    
    def _check_behavioral_alignment(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """Check behavioral pattern alignment"""
        diversity_score = fraud_features.get('category_diversity_score', 0.0)
        behavioral_anomaly_rate = anomaly_features.get('behavioral_anomaly_rate', 0.0)
        
        # High diversity should align with behavioral anomalies
        alignment = 1.0 - abs(diversity_score - behavioral_anomaly_rate)
        return max(alignment, 0.0)
    
    def _calculate_conflict_indicator(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> int:
        """
        Detect conflicts between fraud and anomaly signals
        
        Returns 1 if signals conflict, 0 if aligned
        """
        is_fraud = fraud_features.get('is_fraud', 0)
        has_high_severity = anomaly_features.get('current_severity', 0.0) > 0.7
        
        # Check for major disagreement
        if is_fraud == 1 and anomaly_features.get('daily_anomaly_count', 0) == 0:
            return 1  # Fraud detected but no anomalies
        if is_fraud == 0 and has_high_severity:
            return 1  # No fraud but high anomaly severity
        
        return 0
    
    def _calculate_velocity_severity(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """
        Calculate product of travel velocity and anomaly severity
        
        High velocity + high severity = extreme risk
        """
        velocity_kmh = fraud_features.get('travel_velocity_kmh', 0.0)
        current_severity = anomaly_features.get('current_severity', 0.0)
        
        # Normalize velocity (500 km/h is max realistic)
        normalized_velocity = min(velocity_kmh / 500.0, 1.0)
        
        # Product creates non-linear risk
        product = normalized_velocity * current_severity
        
        return product
    
    def _calculate_geographic_risk(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """
        Calculate geographic risk combining distance and anomaly rate
        """
        distance_km = fraud_features.get('distance_from_home', 0.0)
        geographic_anomaly_rate = anomaly_features.get('geographic_anomaly_rate', 0.0)
        
        # Normalize distance (1000 km is high risk)
        normalized_distance = min(distance_km / 1000.0, 1.0)
        
        # Combine multiplicatively
        geographic_risk = normalized_distance * geographic_anomaly_rate
        
        return geographic_risk
    
    def _calculate_weighted_risk(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """
        Calculate weighted risk score combining all signals
        
        Uses predefined weights for different feature categories
        """
        # Fraud signals
        velocity_signal = min(fraud_features.get('txn_frequency_1h', 0) / 10.0, 1.0)
        geographic_signal = min(fraud_features.get('distance_from_home', 0) / 1000.0, 1.0)
        behavioral_signal = fraud_features.get('category_diversity_score', 0.0)
        
        # Anomaly signals
        severity_signal = anomaly_features.get('current_severity', 0.0)
        frequency_signal = min(anomaly_features.get('daily_anomaly_count', 0) / 10.0, 1.0)
        
        # Weighted combination
        weighted_score = (
            self.feature_weights['fraud_velocity'] * velocity_signal +
            self.feature_weights['fraud_geographic'] * geographic_signal +
            self.feature_weights['fraud_behavioral'] * behavioral_signal +
            self.feature_weights['anomaly_severity'] * severity_signal +
            self.feature_weights['anomaly_frequency'] * frequency_signal
        )
        
        return min(weighted_score, 1.0)
    
    def _calculate_ensemble_probability(
        self,
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> float:
        """
        Calculate ensemble fraud probability from all available signals
        
        Uses voting from multiple indicators
        """
        votes = []
        
        # Fraud indicators
        if fraud_features.get('is_fraud', 0) == 1:
            votes.append(1.0)
        if fraud_features.get('txn_frequency_1h', 0) >= 5:
            votes.append(0.8)
        if fraud_features.get('distance_from_home', 0) > 500:
            votes.append(0.7)
        if fraud_features.get('is_unusual_hour', 0) == 1:
            votes.append(0.6)
        
        # Anomaly indicators
        if anomaly_features.get('current_severity', 0.0) > 0.7:
            votes.append(0.9)
        if anomaly_features.get('daily_anomaly_count', 0) >= 3:
            votes.append(0.75)
        if anomaly_features.get('is_fraud_and_anomaly', 0) == 1:
            votes.append(1.0)
        if anomaly_features.get('has_impossible_travel', 0) == 1:
            votes.append(0.95)
        
        # Calculate ensemble probability
        if not votes:
            return 0.0
        
        # Average voting with threshold
        avg_vote = sum(votes) / len(votes)
        
        return min(avg_vote, 1.0)


class CombinedMLFeatureGenerator:
    """
    Main generator for combined ML features
    
    Orchestrates feature generation from fraud features, anomaly features,
    and interaction features into a unified feature set.
    """
    
    def __init__(self):
        """Initialize the combined feature generator"""
        self.interaction_calculator = InteractionFeatureCalculator()
    
    def generate_features(
        self,
        transaction: Dict[str, Any],
        fraud_features: Dict[str, Any],
        anomaly_features: Dict[str, Any]
    ) -> CombinedMLFeatures:
        """
        Generate combined ML features for a transaction
        
        Args:
            transaction: Current transaction data
            fraud_features: Fraud-based features (32 features)
            anomaly_features: Anomaly-based features (27 features)
            
        Returns:
            CombinedMLFeatures object with all 69 features
        """
        
        # Calculate interaction features (10 features)
        interaction_features = self.interaction_calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # Combine all features
        combined = CombinedMLFeatures(
            # Identifiers
            transaction_id=transaction.get('Transaction_ID', ''),
            customer_id=transaction.get('Customer_ID', ''),
            
            # Fraud features (32)
            daily_txn_count=fraud_features.get('daily_txn_count', 0),
            weekly_txn_count=fraud_features.get('weekly_txn_count', 0),
            daily_txn_amount=fraud_features.get('daily_txn_amount', 0.0),
            weekly_txn_amount=fraud_features.get('weekly_txn_amount', 0.0),
            avg_daily_amount=fraud_features.get('avg_daily_amount', 0.0),
            avg_weekly_amount=fraud_features.get('avg_weekly_amount', 0.0),
            txn_frequency_1h=fraud_features.get('txn_frequency_1h', 0),
            txn_frequency_6h=fraud_features.get('txn_frequency_6h', 0),
            txn_frequency_24h=fraud_features.get('txn_frequency_24h', 0),
            amount_velocity_1h=fraud_features.get('amount_velocity_1h', 0.0),
            amount_velocity_6h=fraud_features.get('amount_velocity_6h', 0.0),
            amount_velocity_24h=fraud_features.get('amount_velocity_24h', 0.0),
            distance_from_home=fraud_features.get('distance_from_home', 0.0),
            avg_distance_last_10=fraud_features.get('avg_distance_last_10', 0.0),
            distance_variance=fraud_features.get('distance_variance', 0.0),
            unique_cities_7d=fraud_features.get('unique_cities_7d', 0),
            travel_velocity_kmh=fraud_features.get('travel_velocity_kmh', 0.0),
            is_unusual_hour=fraud_features.get('is_unusual_hour', 0),
            is_weekend=fraud_features.get('is_weekend', 0),
            is_holiday=fraud_features.get('is_holiday', 0),
            hour_of_day=fraud_features.get('hour_of_day', 0),
            day_of_week=fraud_features.get('day_of_week', 0),
            category_diversity_score=fraud_features.get('category_diversity_score', 0.0),
            merchant_loyalty_score=fraud_features.get('merchant_loyalty_score', 0.0),
            avg_merchant_reputation=fraud_features.get('avg_merchant_reputation', 0.0),
            new_merchant_flag=fraud_features.get('new_merchant_flag', 0),
            refund_rate_30d=fraud_features.get('refund_rate_30d', 0.0),
            declined_rate_7d=fraud_features.get('declined_rate_7d', 0.0),
            shared_merchant_count=fraud_features.get('shared_merchant_count', 0),
            shared_location_count=fraud_features.get('shared_location_count', 0),
            customer_proximity_score=fraud_features.get('customer_proximity_score', 0.0),
            temporal_cluster_flag=fraud_features.get('temporal_cluster_flag', 0),
            
            # Anomaly features (27)
            hourly_anomaly_count=anomaly_features.get('hourly_anomaly_count', 0),
            daily_anomaly_count=anomaly_features.get('daily_anomaly_count', 0),
            weekly_anomaly_count=anomaly_features.get('weekly_anomaly_count', 0),
            anomaly_frequency_trend=anomaly_features.get('anomaly_frequency_trend', 0.0),
            time_since_last_anomaly_hours=anomaly_features.get('time_since_last_anomaly_hours', 0.0),
            mean_severity_last_10=anomaly_features.get('mean_severity_last_10', 0.0),
            max_severity_last_10=anomaly_features.get('max_severity_last_10', 0.0),
            severity_std_last_10=anomaly_features.get('severity_std_last_10', 0.0),
            high_severity_rate_last_10=anomaly_features.get('high_severity_rate_last_10', 0.0),
            current_severity=anomaly_features.get('current_severity', 0.0),
            behavioral_anomaly_rate=anomaly_features.get('behavioral_anomaly_rate', 0.0),
            geographic_anomaly_rate=anomaly_features.get('geographic_anomaly_rate', 0.0),
            temporal_anomaly_rate=anomaly_features.get('temporal_anomaly_rate', 0.0),
            amount_anomaly_rate=anomaly_features.get('amount_anomaly_rate', 0.0),
            anomaly_type_diversity=anomaly_features.get('anomaly_type_diversity', 0.0),
            consecutive_anomaly_count=anomaly_features.get('consecutive_anomaly_count', 0),
            anomaly_streak_length=anomaly_features.get('anomaly_streak_length', 0),
            days_since_first_anomaly=anomaly_features.get('days_since_first_anomaly', 0),
            is_fraud_and_anomaly=anomaly_features.get('is_fraud_and_anomaly', 0),
            fraud_anomaly_correlation_score=anomaly_features.get('fraud_anomaly_correlation_score', 0.0),
            has_impossible_travel=anomaly_features.get('has_impossible_travel', 0),
            has_unusual_category=anomaly_features.get('has_unusual_category', 0),
            has_unusual_hour=anomaly_features.get('has_unusual_hour', 0),
            has_spending_spike=anomaly_features.get('has_spending_spike', 0),
            isolation_forest_score=anomaly_features.get('isolation_forest_score', 0.0),
            anomaly_probability=anomaly_features.get('anomaly_probability', 0.0),
            
            # Interaction features (10)
            high_risk_combination=interaction_features['high_risk_combination'],
            risk_amplification_score=interaction_features['risk_amplification_score'],
            compound_severity_score=interaction_features['compound_severity_score'],
            behavioral_consistency_score=interaction_features['behavioral_consistency_score'],
            pattern_alignment_score=interaction_features['pattern_alignment_score'],
            conflict_indicator=interaction_features['conflict_indicator'],
            velocity_severity_product=interaction_features['velocity_severity_product'],
            geographic_risk_score=interaction_features['geographic_risk_score'],
            weighted_risk_score=interaction_features['weighted_risk_score'],
            ensemble_fraud_probability=interaction_features['ensemble_fraud_probability'],
            
            # Labels
            is_fraud=fraud_features.get('is_fraud', 0),
            fraud_type=fraud_features.get('fraud_type'),
            anomaly_type=anomaly_features.get('anomaly_type', 'None')
        )
        
        return combined
    
    def generate_batch_features(
        self,
        transactions: List[Dict[str, Any]],
        fraud_features_list: List[Dict[str, Any]],
        anomaly_features_list: List[Dict[str, Any]]
    ) -> List[CombinedMLFeatures]:
        """
        Generate features for batch of transactions
        
        Args:
            transactions: List of transactions
            fraud_features_list: List of fraud feature dictionaries
            anomaly_features_list: List of anomaly feature dictionaries
            
        Returns:
            List of CombinedMLFeatures objects
        """
        if len(transactions) != len(fraud_features_list) or \
           len(transactions) != len(anomaly_features_list):
            raise ValueError("All input lists must have the same length")
        
        combined_features = []
        
        for txn, fraud_feat, anomaly_feat in zip(
            transactions, fraud_features_list, anomaly_features_list
        ):
            combined = self.generate_features(txn, fraud_feat, anomaly_feat)
            combined_features.append(combined)
        
        return combined_features
    
    def export_to_dict_list(
        self,
        features_list: List[CombinedMLFeatures]
    ) -> List[Dict[str, Any]]:
        """
        Export list of features to list of dictionaries
        
        Args:
            features_list: List of CombinedMLFeatures objects
            
        Returns:
            List of feature dictionaries
        """
        return [f.to_dict() for f in features_list]
    
    def get_feature_statistics(
        self,
        features_list: List[CombinedMLFeatures]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for all features
        
        Args:
            features_list: List of CombinedMLFeatures objects
            
        Returns:
            Dictionary of statistics (mean, std, min, max) for each feature
        """
        if not features_list:
            return {}
        
        # Get feature names
        feature_names = features_list[0].get_feature_names()
        
        stats = {}
        for feature_name in feature_names:
            values = [
                float(getattr(f, feature_name))
                for f in features_list
                if getattr(f, feature_name) is not None
            ]
            
            if values:
                stats[feature_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'median': statistics.median(values)
                }
        
        return stats
