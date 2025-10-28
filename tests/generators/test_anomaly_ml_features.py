"""
Tests for Anomaly ML Features Module

Comprehensive test suite for anomaly-based ML feature generation,
including frequency features, severity aggregates, type distribution,
persistence metrics, and Isolation Forest anomaly detection.
"""

import pytest
from datetime import datetime, timedelta
from src.generators.anomaly_ml_features import (
    AnomalyFrequencyCalculator,
    AnomalySeverityAggregator,
    AnomalyTypeDistributionCalculator,
    AnomalyPersistenceCalculator,
    AnomalyCrossPatternCalculator,
    AnomalyEvidenceExtractor,
    AnomalyMLFeatureGenerator,
    IsolationForestAnomalyDetector,
    AnomalyMLFeatures
)


class TestAnomalyFrequencyCalculator:
    """Test anomaly frequency feature calculation"""
    
    def test_no_anomalies_in_history(self):
        """Test frequency calculation with no anomalies"""
        calc = AnomalyFrequencyCalculator()
        
        transaction = {'Customer_ID': 'CUST001', 'Date': '2025-01-10', 'Hour': 14}
        history = [
            {'Customer_ID': 'CUST001', 'Date': '2025-01-09', 'Hour': 10, 'Anomaly_Type': 'None'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-08', 'Hour': 15, 'Anomaly_Type': 'None'}
        ]
        
        features = calc.calculate_frequency_features(transaction, history)
        
        assert features['hourly_anomaly_count'] == 0
        assert features['daily_anomaly_count'] == 0
        assert features['weekly_anomaly_count'] == 0
        assert features['anomaly_frequency_trend'] == 0.0
        assert features['time_since_last_anomaly_hours'] == 9999.0
    
    def test_hourly_window_counting(self):
        """Test anomaly counting in 1-hour window"""
        calc = AnomalyFrequencyCalculator()
        
        transaction = {'Customer_ID': 'CUST001', 'Date': '2025-01-10', 'Hour': 14}
        history = [
            {'Customer_ID': 'CUST001', 'Date': '2025-01-10', 'Hour': 13, 'Anomaly_Type': 'BEHAVIORAL'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-10', 'Hour': 14, 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-09', 'Hour': 10, 'Anomaly_Type': 'TEMPORAL'}
        ]
        
        features = calc.calculate_frequency_features(transaction, history)
        
        assert features['hourly_anomaly_count'] >= 1  # At least current hour
    
    def test_trend_calculation_increasing(self):
        """Test increasing trend detection"""
        calc = AnomalyFrequencyCalculator()
        
        transaction = {'Customer_ID': 'CUST001', 'Date': '2025-01-15', 'Hour': 14}
        history = []
        
        # Add 2 anomalies in recent week
        for i in range(2):
            history.append({
                'Customer_ID': 'CUST001',
                'Date': f'2025-01-{10+i}',
                'Hour': 14,
                'Anomaly_Type': 'BEHAVIORAL'
            })
        
        # Add 1 anomaly in previous week
        history.append({
            'Customer_ID': 'CUST001',
            'Date': '2025-01-03',
            'Hour': 14,
            'Anomaly_Type': 'BEHAVIORAL'
        })
        
        features = calc.calculate_frequency_features(transaction, history)
        
        # Should have positive trend (2 recent vs 1 previous)
        assert features['anomaly_frequency_trend'] > 0


class TestAnomalySeverityAggregator:
    """Test severity aggregate feature calculation"""
    
    def test_severity_with_no_anomalies(self):
        """Test severity aggregates with no anomalies in history"""
        agg = AnomalySeverityAggregator()
        
        transaction = {'Customer_ID': 'CUST001', 'Anomaly_Severity': 0.0}
        history = [
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'None'}
        ]
        
        features = agg.calculate_severity_features(transaction, history)
        
        assert features['mean_severity_last_10'] == 0.0
        assert features['max_severity_last_10'] == 0.0
        assert features['severity_std_last_10'] == 0.0
        assert features['high_severity_rate_last_10'] == 0.0
    
    def test_severity_aggregates(self):
        """Test severity aggregate calculations"""
        agg = AnomalySeverityAggregator()
        
        transaction = {'Customer_ID': 'CUST001', 'Anomaly_Severity': 0.5}
        history = [
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'BEHAVIORAL', 'Anomaly_Severity': 0.4},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'GEOGRAPHIC', 'Anomaly_Severity': 0.6},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'TEMPORAL', 'Anomaly_Severity': 0.8}
        ]
        
        features = agg.calculate_severity_features(transaction, history)
        
        assert abs(features['mean_severity_last_10'] - 0.6) < 0.01
        assert features['max_severity_last_10'] == 0.8
        assert features['current_severity'] == 0.5
    
    def test_high_severity_rate(self):
        """Test high severity rate calculation"""
        agg = AnomalySeverityAggregator()
        
        transaction = {'Customer_ID': 'CUST001', 'Anomaly_Severity': 0.5}
        history = [
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'BEHAVIORAL', 'Anomaly_Severity': 0.5},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'GEOGRAPHIC', 'Anomaly_Severity': 0.75},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'TEMPORAL', 'Anomaly_Severity': 0.85},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'AMOUNT', 'Anomaly_Severity': 0.6}
        ]
        
        features = agg.calculate_severity_features(transaction, history)
        
        # 2 out of 4 are high severity (>= 0.7)
        assert abs(features['high_severity_rate_last_10'] - 0.5) < 0.01


class TestAnomalyTypeDistributionCalculator:
    """Test type distribution feature calculation"""
    
    def test_type_rates(self):
        """Test anomaly type rate calculation"""
        calc = AnomalyTypeDistributionCalculator()
        
        transaction = {'Customer_ID': 'CUST001'}
        history = [
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'TEMPORAL'}
        ]
        
        features = calc.calculate_type_features(transaction, history)
        
        assert abs(features['behavioral_anomaly_rate'] - 0.5) < 0.01  # 2/4
        assert abs(features['geographic_anomaly_rate'] - 0.25) < 0.01  # 1/4
        assert abs(features['temporal_anomaly_rate'] - 0.25) < 0.01  # 1/4
        assert features['amount_anomaly_rate'] == 0.0
    
    def test_diversity_calculation(self):
        """Test Shannon entropy diversity calculation"""
        calc = AnomalyTypeDistributionCalculator()
        
        # High diversity (all 4 types)
        transaction1 = {'Customer_ID': 'CUST001'}
        history1 = [
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'TEMPORAL'},
            {'Customer_ID': 'CUST001', 'Anomaly_Type': 'AMOUNT'}
        ]
        
        features1 = calc.calculate_type_features(transaction1, history1)
        
        # Low diversity (single type)
        transaction2 = {'Customer_ID': 'CUST002'}
        history2 = [
            {'Customer_ID': 'CUST002', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Customer_ID': 'CUST002', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Customer_ID': 'CUST002', 'Anomaly_Type': 'BEHAVIORAL'}
        ]
        
        features2 = calc.calculate_type_features(transaction2, history2)
        
        # High diversity should be greater than low diversity
        assert features1['anomaly_type_diversity'] > features2['anomaly_type_diversity']


class TestAnomalyPersistenceCalculator:
    """Test persistence metric calculation"""
    
    def test_consecutive_anomalies(self):
        """Test consecutive anomaly counting"""
        calc = AnomalyPersistenceCalculator()
        
        transaction = {'Customer_ID': 'CUST001', 'Date': '2025-01-10'}
        history = [
            {'Customer_ID': 'CUST001', 'Date': '2025-01-08', 'Anomaly_Type': 'None'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-09', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-10', 'Anomaly_Type': 'GEOGRAPHIC'}
        ]
        
        features = calc.calculate_persistence_features(transaction, history)
        
        assert features['consecutive_anomaly_count'] == 2
    
    def test_streak_length(self):
        """Test longest streak detection"""
        calc = AnomalyPersistenceCalculator()
        
        transaction = {'Customer_ID': 'CUST001', 'Date': '2025-01-10'}
        history = [
            {'Customer_ID': 'CUST001', 'Date': '2025-01-01', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-02', 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-03', 'Anomaly_Type': 'TEMPORAL'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-04', 'Anomaly_Type': 'None'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-05', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-06', 'Anomaly_Type': 'GEOGRAPHIC'}
        ]
        
        features = calc.calculate_persistence_features(transaction, history)
        
        assert features['anomaly_streak_length'] == 3  # First 3 transactions
    
    def test_days_since_first_anomaly(self):
        """Test days since first anomaly calculation"""
        calc = AnomalyPersistenceCalculator()
        
        transaction = {'Customer_ID': 'CUST001', 'Date': '2025-01-10'}
        history = [
            {'Customer_ID': 'CUST001', 'Date': '2025-01-01', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Customer_ID': 'CUST001', 'Date': '2025-01-05', 'Anomaly_Type': 'None'}
        ]
        
        features = calc.calculate_persistence_features(transaction, history)
        
        assert features['days_since_first_anomaly'] == 9  # Jan 10 - Jan 1


class TestAnomalyCrossPatternCalculator:
    """Test cross-pattern feature calculation"""
    
    def test_fraud_and_anomaly_detection(self):
        """Test detection of transactions with both fraud and anomaly"""
        calc = AnomalyCrossPatternCalculator()
        
        # Transaction with both
        transaction1 = {
            'Customer_ID': 'CUST001',
            'Fraud_Type': 'Card Cloning',
            'Anomaly_Type': 'GEOGRAPHIC'
        }
        
        features1 = calc.calculate_cross_pattern_features(transaction1, [])
        assert features1['is_fraud_and_anomaly'] == 1
        
        # Transaction with only fraud
        transaction2 = {
            'Customer_ID': 'CUST002',
            'Fraud_Type': 'Account Takeover',
            'Anomaly_Type': 'None'
        }
        
        features2 = calc.calculate_cross_pattern_features(transaction2, [])
        assert features2['is_fraud_and_anomaly'] == 0
    
    def test_correlation_score_calculation(self):
        """Test Jaccard index correlation calculation"""
        calc = AnomalyCrossPatternCalculator()
        
        transaction = {'Customer_ID': 'CUST001', 'Fraud_Type': 'None', 'Anomaly_Type': 'None'}
        history = [
            {'Fraud_Type': 'Fraud1', 'Anomaly_Type': 'BEHAVIORAL'},  # Both
            {'Fraud_Type': 'Fraud2', 'Anomaly_Type': 'GEOGRAPHIC'},  # Both
            {'Fraud_Type': 'Fraud3', 'Anomaly_Type': 'None'},  # Fraud only
            {'Fraud_Type': 'None', 'Anomaly_Type': 'TEMPORAL'}  # Anomaly only
        ]
        
        features = calc.calculate_cross_pattern_features(transaction, history)
        
        # Jaccard: 2 / (3 + 3 - 2) = 2/4 = 0.5
        assert abs(features['fraud_anomaly_correlation_score'] - 0.5) < 0.01


class TestAnomalyEvidenceExtractor:
    """Test evidence feature extraction"""
    
    def test_evidence_parsing(self):
        """Test parsing of anomaly evidence JSON"""
        extractor = AnomalyEvidenceExtractor()
        
        transaction = {
            'Anomaly_Evidence': '{"speed_kmh": 1500, "unusual_category": "Jewelry", "hour": 3, "multiplier": 4.5}'
        }
        
        features = extractor.extract_evidence_features(transaction)
        
        assert features['has_impossible_travel'] == 1  # speed > 800
        assert features['has_unusual_category'] == 1
        assert features['has_unusual_hour'] == 1  # hour 3 is late night
        assert features['has_spending_spike'] == 1  # multiplier > 3
    
    def test_empty_evidence(self):
        """Test handling of empty evidence"""
        extractor = AnomalyEvidenceExtractor()
        
        transaction = {'Anomaly_Evidence': ''}
        
        features = extractor.extract_evidence_features(transaction)
        
        assert features['has_impossible_travel'] == 0
        assert features['has_unusual_category'] == 0
        assert features['has_unusual_hour'] == 0
        assert features['has_spending_spike'] == 0


class TestAnomalyMLFeatureGenerator:
    """Test ML feature generator orchestration"""
    
    def test_feature_generation(self):
        """Test complete feature generation"""
        generator = AnomalyMLFeatureGenerator()
        
        transaction = {
            'Transaction_ID': 'TXN001',
            'Customer_ID': 'CUST001',
            'Date': '2025-01-10',
            'Hour': 14,
            'Anomaly_Type': 'BEHAVIORAL',
            'Anomaly_Severity': 0.6,
            'Anomaly_Confidence': 0.7,
            'Fraud_Type': 'None',
            'Anomaly_Evidence': '{"unusual_category": "Jewelry"}'
        }
        
        history = [
            {
                'Customer_ID': 'CUST001',
                'Date': '2025-01-09',
                'Hour': 10,
                'Anomaly_Type': 'GEOGRAPHIC',
                'Anomaly_Severity': 0.5
            }
        ]
        
        features = generator.generate_features(transaction, history, isolation_score=-0.5)
        
        assert features.transaction_id == 'TXN001'
        assert features.customer_id == 'CUST001'
        assert features.current_severity == 0.6
        assert features.has_unusual_category == 1
        assert features.isolation_forest_score == -0.5
        assert 0.0 <= features.anomaly_probability <= 1.0
    
    def test_batch_feature_generation(self):
        """Test batch feature generation"""
        generator = AnomalyMLFeatureGenerator()
        
        transactions = [
            {
                'Transaction_ID': 'TXN001',
                'Customer_ID': 'CUST001',
                'Date': '2025-01-10',
                'Hour': 14,
                'Anomaly_Type': 'BEHAVIORAL',
                'Anomaly_Severity': 0.6,
                'Fraud_Type': 'None',
                'Anomaly_Evidence': ''
            },
            {
                'Transaction_ID': 'TXN002',
                'Customer_ID': 'CUST002',
                'Date': '2025-01-10',
                'Hour': 15,
                'Anomaly_Type': 'GEOGRAPHIC',
                'Anomaly_Severity': 0.7,
                'Fraud_Type': 'None',
                'Anomaly_Evidence': ''
            }
        ]
        
        customer_histories = {
            'CUST001': [],
            'CUST002': []
        }
        
        features_list = generator.generate_features_batch(transactions, customer_histories)
        
        assert len(features_list) == 2
        assert features_list[0].transaction_id == 'TXN001'
        assert features_list[1].transaction_id == 'TXN002'


class TestIsolationForestAnomalyDetector:
    """Test Isolation Forest anomaly detection"""
    
    def test_feature_preparation(self):
        """Test feature matrix preparation"""
        detector = IsolationForestAnomalyDetector()
        
        transactions = [
            {
                'Amount': 1000.0,
                'Hour': 14,
                'Distance_From_Last_Txn_km': 5.0,
                'Time_Since_Last_Txn_hours': 2.0,
                'Anomaly_Severity': 0.5,
                'Anomaly_Confidence': 0.6
            },
            {
                'Amount': 2000.0,
                'Hour': 15,
                'Distance_From_Last_Txn_km': 10.0,
                'Time_Since_Last_Txn_hours': 1.0,
                'Anomaly_Severity': 0.7,
                'Anomaly_Confidence': 0.8
            }
        ]
        
        X, feature_names = detector.prepare_features(transactions)
        
        assert len(X) == 2
        assert len(X[0]) == 6  # 6 features
        assert 'Amount' in feature_names
        assert 'Hour' in feature_names
    
    def test_fit_predict_basic(self):
        """Test basic fit and predict"""
        detector = IsolationForestAnomalyDetector(contamination=0.1, random_state=42)
        
        # Create transactions with one clear outlier
        transactions = [
            {'Amount': 1000.0, 'Hour': 14, 'Distance_From_Last_Txn_km': 5.0,
             'Time_Since_Last_Txn_hours': 2.0, 'Anomaly_Severity': 0.5, 'Anomaly_Confidence': 0.6}
            for _ in range(9)
        ] + [
            {'Amount': 10000.0, 'Hour': 3, 'Distance_From_Last_Txn_km': 500.0,
             'Time_Since_Last_Txn_hours': 0.5, 'Anomaly_Severity': 0.9, 'Anomaly_Confidence': 0.9}
        ]
        
        scores = detector.fit_predict(transactions)
        
        assert len(scores) == 10
        # All scores should be in -1 to 1 range
        for score in scores:
            assert -1.0 <= score <= 1.0


class TestAnomalyMLFeaturesDataclass:
    """Test AnomalyMLFeatures dataclass"""
    
    def test_dataclass_creation(self):
        """Test creation of AnomalyMLFeatures object"""
        features = AnomalyMLFeatures(
            transaction_id='TXN001',
            customer_id='CUST001',
            hourly_anomaly_count=2,
            daily_anomaly_count=5,
            weekly_anomaly_count=10,
            anomaly_frequency_trend=0.5,
            time_since_last_anomaly_hours=1.5,
            mean_severity_last_10=0.6,
            max_severity_last_10=0.8,
            severity_std_last_10=0.1,
            high_severity_rate_last_10=0.4,
            current_severity=0.7,
            behavioral_anomaly_rate=0.3,
            geographic_anomaly_rate=0.2,
            temporal_anomaly_rate=0.25,
            amount_anomaly_rate=0.25,
            anomaly_type_diversity=0.8,
            consecutive_anomaly_count=3,
            anomaly_streak_length=5,
            days_since_first_anomaly=30,
            is_fraud_and_anomaly=1,
            fraud_anomaly_correlation_score=0.5,
            has_impossible_travel=1,
            has_unusual_category=0,
            has_unusual_hour=1,
            has_spending_spike=0,
            isolation_forest_score=-0.5,
            anomaly_probability=0.75
        )
        
        assert features.transaction_id == 'TXN001'
        assert features.hourly_anomaly_count == 2
        assert features.anomaly_type_diversity == 0.8
        assert features.isolation_forest_score == -0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
