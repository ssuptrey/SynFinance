"""
Test Suite for Combined ML Features Module

Tests the unified feature generation system that combines fraud-based
and anomaly-based ML features with interaction features.

Coverage:
- CombinedMLFeatures dataclass (69 features)
- InteractionFeatureCalculator (10 interaction features)
- CombinedMLFeatureGenerator (batch processing)
- Feature statistics and export functions

Author: SynFinance Development Team
Version: 0.7.0
Date: October 28, 2025
"""

import pytest
from src.generators.combined_ml_features import (
    CombinedMLFeatures,
    InteractionFeatureCalculator,
    CombinedMLFeatureGenerator
)


class TestCombinedMLFeatures:
    """Test suite for CombinedMLFeatures dataclass"""
    
    def test_dataclass_initialization(self):
        """Test that CombinedMLFeatures can be initialized with all fields"""
        features = CombinedMLFeatures(
            transaction_id='TXN001',
            customer_id='CUST001',
            # Fraud features
            daily_txn_count=5,
            weekly_txn_count=20,
            daily_txn_amount=5000.0,
            weekly_txn_amount=25000.0,
            avg_daily_amount=1000.0,
            avg_weekly_amount=1250.0,
            txn_frequency_1h=2,
            txn_frequency_6h=8,
            txn_frequency_24h=15,
            amount_velocity_1h=2000.0,
            amount_velocity_6h=8000.0,
            amount_velocity_24h=15000.0,
            distance_from_home=150.0,
            avg_distance_last_10=120.0,
            distance_variance=45.0,
            unique_cities_7d=3,
            travel_velocity_kmh=75.0,
            is_unusual_hour=0,
            is_weekend=0,
            is_holiday=0,
            hour_of_day=14,
            day_of_week=2,
            category_diversity_score=0.65,
            merchant_loyalty_score=0.45,
            avg_merchant_reputation=4.2,
            new_merchant_flag=0,
            refund_rate_30d=0.02,
            declined_rate_7d=0.01,
            shared_merchant_count=15,
            shared_location_count=8,
            customer_proximity_score=0.35,
            temporal_cluster_flag=0,
            # Anomaly features
            hourly_anomaly_count=1,
            daily_anomaly_count=3,
            weekly_anomaly_count=12,
            anomaly_frequency_trend=0.15,
            time_since_last_anomaly_hours=2.5,
            mean_severity_last_10=0.45,
            max_severity_last_10=0.75,
            severity_std_last_10=0.15,
            high_severity_rate_last_10=0.20,
            current_severity=0.60,
            behavioral_anomaly_rate=0.30,
            geographic_anomaly_rate=0.25,
            temporal_anomaly_rate=0.15,
            amount_anomaly_rate=0.30,
            anomaly_type_diversity=0.55,
            consecutive_anomaly_count=2,
            anomaly_streak_length=3,
            days_since_first_anomaly=15,
            is_fraud_and_anomaly=1,
            fraud_anomaly_correlation_score=0.70,
            has_impossible_travel=0,
            has_unusual_category=1,
            has_unusual_hour=0,
            has_spending_spike=1,
            isolation_forest_score=-0.35,
            anomaly_probability=0.65,
            # Interaction features
            high_risk_combination=0,
            risk_amplification_score=0.55,
            compound_severity_score=0.40,
            behavioral_consistency_score=0.75,
            pattern_alignment_score=0.68,
            conflict_indicator=0,
            velocity_severity_product=0.30,
            geographic_risk_score=0.22,
            weighted_risk_score=0.58,
            ensemble_fraud_probability=0.62,
            # Labels
            is_fraud=0,
            fraud_type=None,
            anomaly_type='Behavioral'
        )
        
        assert features.transaction_id == 'TXN001'
        assert features.customer_id == 'CUST001'
        assert features.daily_txn_count == 5
        assert features.current_severity == 0.60
        assert features.risk_amplification_score == 0.55
    
    def test_to_dict_conversion(self):
        """Test conversion of features to dictionary"""
        features = CombinedMLFeatures(
            transaction_id='TXN002',
            customer_id='CUST002',
            daily_txn_count=3,
            weekly_txn_count=10,
            daily_txn_amount=3000.0,
            weekly_txn_amount=15000.0,
            avg_daily_amount=1000.0,
            avg_weekly_amount=1500.0,
            txn_frequency_1h=1,
            txn_frequency_6h=5,
            txn_frequency_24h=10,
            amount_velocity_1h=1000.0,
            amount_velocity_6h=5000.0,
            amount_velocity_24h=10000.0,
            distance_from_home=50.0,
            avg_distance_last_10=45.0,
            distance_variance=10.0,
            unique_cities_7d=2,
            travel_velocity_kmh=30.0,
            is_unusual_hour=1,
            is_weekend=0,
            is_holiday=0,
            hour_of_day=3,
            day_of_week=4,
            category_diversity_score=0.50,
            merchant_loyalty_score=0.60,
            avg_merchant_reputation=4.5,
            new_merchant_flag=1,
            refund_rate_30d=0.01,
            declined_rate_7d=0.005,
            shared_merchant_count=10,
            shared_location_count=5,
            customer_proximity_score=0.40,
            temporal_cluster_flag=1,
            hourly_anomaly_count=2,
            daily_anomaly_count=5,
            weekly_anomaly_count=15,
            anomaly_frequency_trend=0.25,
            time_since_last_anomaly_hours=1.5,
            mean_severity_last_10=0.55,
            max_severity_last_10=0.85,
            severity_std_last_10=0.20,
            high_severity_rate_last_10=0.30,
            current_severity=0.70,
            behavioral_anomaly_rate=0.40,
            geographic_anomaly_rate=0.20,
            temporal_anomaly_rate=0.35,
            amount_anomaly_rate=0.25,
            anomaly_type_diversity=0.60,
            consecutive_anomaly_count=3,
            anomaly_streak_length=4,
            days_since_first_anomaly=20,
            is_fraud_and_anomaly=1,
            fraud_anomaly_correlation_score=0.80,
            has_impossible_travel=1,
            has_unusual_category=0,
            has_unusual_hour=1,
            has_spending_spike=0,
            isolation_forest_score=-0.45,
            anomaly_probability=0.75,
            high_risk_combination=1,
            risk_amplification_score=0.70,
            compound_severity_score=0.55,
            behavioral_consistency_score=0.65,
            pattern_alignment_score=0.72,
            conflict_indicator=0,
            velocity_severity_product=0.42,
            geographic_risk_score=0.18,
            weighted_risk_score=0.68,
            ensemble_fraud_probability=0.75,
            is_fraud=1,
            fraud_type='Account_Takeover',
            anomaly_type='Temporal'
        )
        
        feature_dict = features.to_dict()
        
        assert isinstance(feature_dict, dict)
        assert feature_dict['transaction_id'] == 'TXN002'
        assert feature_dict['daily_txn_count'] == 3
        assert feature_dict['current_severity'] == 0.70
        assert feature_dict['high_risk_combination'] == 1
        assert feature_dict['is_fraud'] == 1
    
    def test_get_feature_names(self):
        """Test retrieving list of feature names"""
        features = self._create_minimal_features()
        
        feature_names = features.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) == 68  # Total features: 32 fraud + 26 anomaly + 10 interaction
        assert 'daily_txn_count' in feature_names
        assert 'current_severity' in feature_names
        assert 'risk_amplification_score' in feature_names
        assert 'transaction_id' not in feature_names
        assert 'is_fraud' not in feature_names
    
    def test_get_feature_values(self):
        """Test retrieving list of feature values"""
        features = self._create_minimal_features()
        
        feature_values = features.get_feature_values()
        
        assert isinstance(feature_values, list)
        assert len(feature_values) == 68  # Total features: 32 fraud + 26 anomaly + 10 interaction
        assert all(isinstance(v, float) for v in feature_values)
    
    def _create_minimal_features(self):
        """Helper to create minimal feature set for testing"""
        return CombinedMLFeatures(
            transaction_id='TXN_TEST',
            customer_id='CUST_TEST',
            daily_txn_count=0, weekly_txn_count=0,
            daily_txn_amount=0.0, weekly_txn_amount=0.0,
            avg_daily_amount=0.0, avg_weekly_amount=0.0,
            txn_frequency_1h=0, txn_frequency_6h=0, txn_frequency_24h=0,
            amount_velocity_1h=0.0, amount_velocity_6h=0.0, amount_velocity_24h=0.0,
            distance_from_home=0.0, avg_distance_last_10=0.0,
            distance_variance=0.0, unique_cities_7d=0, travel_velocity_kmh=0.0,
            is_unusual_hour=0, is_weekend=0, is_holiday=0,
            hour_of_day=0, day_of_week=0,
            category_diversity_score=0.0, merchant_loyalty_score=0.0,
            avg_merchant_reputation=0.0, new_merchant_flag=0,
            refund_rate_30d=0.0, declined_rate_7d=0.0,
            shared_merchant_count=0, shared_location_count=0,
            customer_proximity_score=0.0, temporal_cluster_flag=0,
            hourly_anomaly_count=0, daily_anomaly_count=0, weekly_anomaly_count=0,
            anomaly_frequency_trend=0.0, time_since_last_anomaly_hours=0.0,
            mean_severity_last_10=0.0, max_severity_last_10=0.0,
            severity_std_last_10=0.0, high_severity_rate_last_10=0.0,
            current_severity=0.0,
            behavioral_anomaly_rate=0.0, geographic_anomaly_rate=0.0,
            temporal_anomaly_rate=0.0, amount_anomaly_rate=0.0,
            anomaly_type_diversity=0.0,
            consecutive_anomaly_count=0, anomaly_streak_length=0,
            days_since_first_anomaly=0,
            is_fraud_and_anomaly=0, fraud_anomaly_correlation_score=0.0,
            has_impossible_travel=0, has_unusual_category=0,
            has_unusual_hour=0, has_spending_spike=0,
            isolation_forest_score=0.0, anomaly_probability=0.0,
            high_risk_combination=0, risk_amplification_score=0.0,
            compound_severity_score=0.0, behavioral_consistency_score=0.0,
            pattern_alignment_score=0.0, conflict_indicator=0,
            velocity_severity_product=0.0, geographic_risk_score=0.0,
            weighted_risk_score=0.0, ensemble_fraud_probability=0.0,
            is_fraud=0
        )


class TestInteractionFeatureCalculator:
    """Test suite for InteractionFeatureCalculator"""
    
    def test_high_risk_combination_detected(self):
        """Test detection of high-risk combination (fraud + anomaly + velocity)"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'is_fraud': 1,
            'txn_frequency_1h': 6
        }
        anomaly_features = {
            'is_fraud_and_anomaly': 1
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        assert interaction['high_risk_combination'] == 1
    
    def test_high_risk_combination_not_detected(self):
        """Test no high-risk combination when conditions not met"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'is_fraud': 0,
            'txn_frequency_1h': 2
        }
        anomaly_features = {
            'is_fraud_and_anomaly': 0
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        assert interaction['high_risk_combination'] == 0
    
    def test_risk_amplification_score(self):
        """Test risk amplification score calculation"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'txn_frequency_1h': 8,  # High velocity
            'distance_from_home': 800.0  # High distance
        }
        anomaly_features = {
            'current_severity': 0.85,  # High severity
            'daily_anomaly_count': 7  # High frequency
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # Should have high amplification
        assert 0.0 <= interaction['risk_amplification_score'] <= 1.0
        assert interaction['risk_amplification_score'] > 0.5
    
    def test_compound_severity_score(self):
        """Test compound severity calculation"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'is_fraud': 1,
            'txn_frequency_1h': 10,
            'distance_from_home': 1000.0,
            'travel_velocity_kmh': 500.0
        }
        anomaly_features = {
            'current_severity': 0.90
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # Compound severity should be product of individual severities
        assert 0.0 <= interaction['compound_severity_score'] <= 1.0
        assert interaction['compound_severity_score'] > 0.3
    
    def test_behavioral_consistency_high(self):
        """Test high behavioral consistency when signals agree"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'category_diversity_score': 0.75
        }
        anomaly_features = {
            'behavioral_anomaly_rate': 0.70
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # Should show high consistency (small difference)
        assert interaction['behavioral_consistency_score'] > 0.9
    
    def test_behavioral_consistency_low(self):
        """Test low behavioral consistency when signals disagree"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'category_diversity_score': 0.10
        }
        anomaly_features = {
            'behavioral_anomaly_rate': 0.90
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # Should show low consistency (large difference)
        assert interaction['behavioral_consistency_score'] < 0.3
    
    def test_pattern_alignment_score(self):
        """Test pattern alignment across temporal, geographic, behavioral dimensions"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'is_unusual_hour': 1,
            'distance_from_home': 800.0,
            'category_diversity_score': 0.80
        }
        anomaly_features = {
            'temporal_anomaly_rate': 0.85,
            'geographic_anomaly_rate': 0.75,
            'behavioral_anomaly_rate': 0.78
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # All dimensions aligned, should be high
        assert 0.0 <= interaction['pattern_alignment_score'] <= 1.0
        assert interaction['pattern_alignment_score'] > 0.6
    
    def test_conflict_indicator_fraud_no_anomaly(self):
        """Test conflict detection when fraud detected but no anomalies"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'is_fraud': 1
        }
        anomaly_features = {
            'daily_anomaly_count': 0,
            'current_severity': 0.1
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        assert interaction['conflict_indicator'] == 1
    
    def test_conflict_indicator_no_fraud_high_anomaly(self):
        """Test conflict detection when no fraud but high anomaly severity"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'is_fraud': 0
        }
        anomaly_features = {
            'daily_anomaly_count': 5,
            'current_severity': 0.85
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        assert interaction['conflict_indicator'] == 1
    
    def test_conflict_indicator_no_conflict(self):
        """Test no conflict when signals align"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'is_fraud': 1
        }
        anomaly_features = {
            'daily_anomaly_count': 5,
            'current_severity': 0.75
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        assert interaction['conflict_indicator'] == 0
    
    def test_velocity_severity_product(self):
        """Test velocity-severity interaction"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'travel_velocity_kmh': 400.0  # High velocity (80% of max)
        }
        anomaly_features = {
            'current_severity': 0.90  # High severity
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # Product should be significant
        assert interaction['velocity_severity_product'] > 0.5
    
    def test_geographic_risk_score(self):
        """Test geographic risk calculation"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'distance_from_home': 900.0  # Far from home
        }
        anomaly_features = {
            'geographic_anomaly_rate': 0.80  # High geographic anomaly rate
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # Should show significant geographic risk
        assert interaction['geographic_risk_score'] > 0.5
    
    def test_weighted_risk_score(self):
        """Test weighted risk score combining all signals"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'txn_frequency_1h': 8,
            'distance_from_home': 800.0,
            'category_diversity_score': 0.75
        }
        anomaly_features = {
            'current_severity': 0.85,
            'daily_anomaly_count': 6
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # All signals high, weighted score should be high
        assert 0.0 <= interaction['weighted_risk_score'] <= 1.0
        assert interaction['weighted_risk_score'] > 0.6
    
    def test_ensemble_fraud_probability_high_risk(self):
        """Test ensemble probability with many risk indicators"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'is_fraud': 1,
            'txn_frequency_1h': 7,
            'distance_from_home': 600.0,
            'is_unusual_hour': 1
        }
        anomaly_features = {
            'current_severity': 0.85,
            'daily_anomaly_count': 5,
            'is_fraud_and_anomaly': 1,
            'has_impossible_travel': 1
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # Multiple indicators, should be high
        assert interaction['ensemble_fraud_probability'] > 0.7
    
    def test_ensemble_fraud_probability_low_risk(self):
        """Test ensemble probability with few risk indicators"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'is_fraud': 0,
            'txn_frequency_1h': 1,
            'distance_from_home': 20.0,
            'is_unusual_hour': 0
        }
        anomaly_features = {
            'current_severity': 0.15,
            'daily_anomaly_count': 0,
            'is_fraud_and_anomaly': 0,
            'has_impossible_travel': 0
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        # Few indicators, should be low
        assert interaction['ensemble_fraud_probability'] < 0.3
    
    def test_all_interaction_features_present(self):
        """Test that all 10 interaction features are calculated"""
        calculator = InteractionFeatureCalculator()
        
        fraud_features = {
            'is_fraud': 0,
            'txn_frequency_1h': 3,
            'distance_from_home': 100.0,
            'category_diversity_score': 0.5,
            'travel_velocity_kmh': 50.0,
            'is_unusual_hour': 0
        }
        anomaly_features = {
            'current_severity': 0.4,
            'daily_anomaly_count': 2,
            'is_fraud_and_anomaly': 0,
            'behavioral_anomaly_rate': 0.3,
            'geographic_anomaly_rate': 0.2,
            'temporal_anomaly_rate': 0.1,
            'has_impossible_travel': 0
        }
        
        interaction = calculator.calculate_interaction_features(
            fraud_features, anomaly_features
        )
        
        expected_features = {
            'high_risk_combination',
            'risk_amplification_score',
            'compound_severity_score',
            'behavioral_consistency_score',
            'pattern_alignment_score',
            'conflict_indicator',
            'velocity_severity_product',
            'geographic_risk_score',
            'weighted_risk_score',
            'ensemble_fraud_probability'
        }
        
        assert set(interaction.keys()) == expected_features


class TestCombinedMLFeatureGenerator:
    """Test suite for CombinedMLFeatureGenerator"""
    
    def test_generate_features_complete(self):
        """Test generating complete feature set for single transaction"""
        generator = CombinedMLFeatureGenerator()
        
        transaction = {
            'Transaction_ID': 'TXN_GEN_001',
            'Customer_ID': 'CUST_GEN_001'
        }
        
        fraud_features = self._get_sample_fraud_features()
        anomaly_features = self._get_sample_anomaly_features()
        
        combined = generator.generate_features(
            transaction, fraud_features, anomaly_features
        )
        
        assert isinstance(combined, CombinedMLFeatures)
        assert combined.transaction_id == 'TXN_GEN_001'
        assert combined.customer_id == 'CUST_GEN_001'
        assert combined.daily_txn_count == 5
        assert combined.current_severity == 0.60
        assert 0.0 <= combined.risk_amplification_score <= 1.0
    
    def test_generate_batch_features(self):
        """Test batch feature generation for multiple transactions"""
        generator = CombinedMLFeatureGenerator()
        
        transactions = [
            {'Transaction_ID': f'TXN_{i}', 'Customer_ID': f'CUST_{i}'}
            for i in range(5)
        ]
        
        fraud_features_list = [self._get_sample_fraud_features() for _ in range(5)]
        anomaly_features_list = [self._get_sample_anomaly_features() for _ in range(5)]
        
        combined_features = generator.generate_batch_features(
            transactions, fraud_features_list, anomaly_features_list
        )
        
        assert len(combined_features) == 5
        assert all(isinstance(f, CombinedMLFeatures) for f in combined_features)
        assert combined_features[0].transaction_id == 'TXN_0'
        assert combined_features[4].transaction_id == 'TXN_4'
    
    def test_generate_batch_features_length_mismatch(self):
        """Test error handling when input lists have different lengths"""
        generator = CombinedMLFeatureGenerator()
        
        transactions = [{'Transaction_ID': 'TXN_1'}]
        fraud_features_list = [self._get_sample_fraud_features()]
        anomaly_features_list = []  # Empty list
        
        with pytest.raises(ValueError):
            generator.generate_batch_features(
                transactions, fraud_features_list, anomaly_features_list
            )
    
    def test_export_to_dict_list(self):
        """Test exporting features to list of dictionaries"""
        generator = CombinedMLFeatureGenerator()
        
        transactions = [
            {'Transaction_ID': f'TXN_{i}', 'Customer_ID': f'CUST_{i}'}
            for i in range(3)
        ]
        
        fraud_features_list = [self._get_sample_fraud_features() for _ in range(3)]
        anomaly_features_list = [self._get_sample_anomaly_features() for _ in range(3)]
        
        combined_features = generator.generate_batch_features(
            transactions, fraud_features_list, anomaly_features_list
        )
        
        dict_list = generator.export_to_dict_list(combined_features)
        
        assert len(dict_list) == 3
        assert all(isinstance(d, dict) for d in dict_list)
        assert dict_list[0]['transaction_id'] == 'TXN_0'
        assert 'daily_txn_count' in dict_list[0]
        assert 'risk_amplification_score' in dict_list[0]
    
    def test_get_feature_statistics(self):
        """Test calculating statistics for feature set"""
        generator = CombinedMLFeatureGenerator()
        
        transactions = [
            {'Transaction_ID': f'TXN_{i}', 'Customer_ID': f'CUST_{i}'}
            for i in range(10)
        ]
        
        fraud_features_list = [self._get_sample_fraud_features() for _ in range(10)]
        anomaly_features_list = [self._get_sample_anomaly_features() for _ in range(10)]
        
        combined_features = generator.generate_batch_features(
            transactions, fraud_features_list, anomaly_features_list
        )
        
        stats = generator.get_feature_statistics(combined_features)
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        # Check that statistics are calculated for features
        assert 'daily_txn_count' in stats
        assert 'mean' in stats['daily_txn_count']
        assert 'std' in stats['daily_txn_count']
        assert 'min' in stats['daily_txn_count']
        assert 'max' in stats['daily_txn_count']
        assert 'median' in stats['daily_txn_count']
    
    def test_get_feature_statistics_empty_list(self):
        """Test statistics calculation with empty feature list"""
        generator = CombinedMLFeatureGenerator()
        
        stats = generator.get_feature_statistics([])
        
        assert stats == {}
    
    def _get_sample_fraud_features(self):
        """Helper to create sample fraud features"""
        return {
            'daily_txn_count': 5,
            'weekly_txn_count': 20,
            'daily_txn_amount': 5000.0,
            'weekly_txn_amount': 25000.0,
            'avg_daily_amount': 1000.0,
            'avg_weekly_amount': 1250.0,
            'txn_frequency_1h': 2,
            'txn_frequency_6h': 8,
            'txn_frequency_24h': 15,
            'amount_velocity_1h': 2000.0,
            'amount_velocity_6h': 8000.0,
            'amount_velocity_24h': 15000.0,
            'distance_from_home': 150.0,
            'avg_distance_last_10': 120.0,
            'distance_variance': 45.0,
            'unique_cities_7d': 3,
            'travel_velocity_kmh': 75.0,
            'is_unusual_hour': 0,
            'is_weekend': 0,
            'is_holiday': 0,
            'hour_of_day': 14,
            'day_of_week': 2,
            'category_diversity_score': 0.65,
            'merchant_loyalty_score': 0.45,
            'avg_merchant_reputation': 4.2,
            'new_merchant_flag': 0,
            'refund_rate_30d': 0.02,
            'declined_rate_7d': 0.01,
            'shared_merchant_count': 15,
            'shared_location_count': 8,
            'customer_proximity_score': 0.35,
            'temporal_cluster_flag': 0,
            'is_fraud': 0,
            'fraud_type': None
        }
    
    def _get_sample_anomaly_features(self):
        """Helper to create sample anomaly features"""
        return {
            'hourly_anomaly_count': 1,
            'daily_anomaly_count': 3,
            'weekly_anomaly_count': 12,
            'anomaly_frequency_trend': 0.15,
            'time_since_last_anomaly_hours': 2.5,
            'mean_severity_last_10': 0.45,
            'max_severity_last_10': 0.75,
            'severity_std_last_10': 0.15,
            'high_severity_rate_last_10': 0.20,
            'current_severity': 0.60,
            'behavioral_anomaly_rate': 0.30,
            'geographic_anomaly_rate': 0.25,
            'temporal_anomaly_rate': 0.15,
            'amount_anomaly_rate': 0.30,
            'anomaly_type_diversity': 0.55,
            'consecutive_anomaly_count': 2,
            'anomaly_streak_length': 3,
            'days_since_first_anomaly': 15,
            'is_fraud_and_anomaly': 0,
            'fraud_anomaly_correlation_score': 0.50,
            'has_impossible_travel': 0,
            'has_unusual_category': 1,
            'has_unusual_hour': 0,
            'has_spending_spike': 1,
            'isolation_forest_score': -0.35,
            'anomaly_probability': 0.65,
            'anomaly_type': 'Behavioral'
        }
