"""
Tests for ML Feature Engineering Module

Tests the MLFeatureEngineer class and feature calculation methods.
Validates that all 32 features are generated correctly.
"""

import pytest
from datetime import datetime, timedelta
from src.generators.ml_features import MLFeatureEngineer, MLFeatures


class TestMLFeatureEngineer:
    """Test suite for ML feature engineering."""
    
    @pytest.fixture
    def engineer(self):
        """Create a feature engineer instance."""
        return MLFeatureEngineer()
    
    @pytest.fixture
    def sample_transaction(self):
        """Create a sample transaction."""
        return {
            'Transaction_ID': 'TXN001',
            'Customer_ID': 'CUST001',
            'Date': '2025-01-15',
            'Time': '14:30:00',
            'Amount': 1500.0,
            'Category': 'Shopping',
            'Merchant': 'BigBazaar_MUM_001',
            'City': 'Mumbai',
            'Mode': 'UPI',
            'Is_Fraud': 0,
            'Fraud_Type': 'None'
        }
    
    @pytest.fixture
    def sample_customer(self):
        """Create a sample customer dict."""
        return {
            'Customer_ID': 'CUST001',
            'city': 'Mumbai',
            'Age': 30,
            'digital_savviness': 'High'
        }
    
    @pytest.fixture
    def sample_history(self):
        """Create sample transaction history."""
        base_time = datetime(2025, 1, 15, 10, 0, 0)
        history = []
        for i in range(5):
            txn_time = base_time + timedelta(hours=i)
            history.append({
                'Transaction_ID': f'TXN{i:03d}',
                'Customer_ID': 'CUST001',
                'Date': txn_time.strftime('%Y-%m-%d'),
                'Time': txn_time.strftime('%H:%M:%S'),
                'Amount': 500.0 + (i * 100),
                'Category': 'Shopping' if i % 2 == 0 else 'Food',
                'Merchant': f'Merchant_{i}',
                'City': 'Mumbai',
                'Mode': 'UPI',
                'Status': 'Approved'
            })
        return history


class TestFeatureGeneration(TestMLFeatureEngineer):
    """Test basic feature generation."""
    
    def test_engineer_features_returns_mlfeatures_object(self, engineer, sample_transaction, sample_customer):
        """Test that engineer_features returns an MLFeatures object."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        assert isinstance(features, MLFeatures)
    
    def test_all_32_features_present(self, engineer, sample_transaction, sample_customer):
        """Test that all 32 features are generated."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        feature_dict = features.to_dict()
        
        # Should have 32 features + transaction_id + labels
        assert len(feature_dict) >= 34  # 32 features + id + is_fraud + fraud_type
    
    def test_transaction_id_preserved(self, engineer, sample_transaction, sample_customer):
        """Test that transaction ID is preserved in features."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        assert features.transaction_id == 'TXN001'
    
    def test_fraud_label_preserved(self, engineer, sample_transaction, sample_customer):
        """Test that fraud labels are preserved."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        assert features.is_fraud == 0
        assert features.fraud_type == 'None'


class TestAggregateFeatures(TestMLFeatureEngineer):
    """Test aggregate feature calculations."""
    
    def test_daily_transaction_count_with_empty_history(self, engineer, sample_transaction, sample_customer):
        """Test daily count with no history."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        assert features.daily_txn_count == 0
    
    def test_daily_transaction_count_with_history(self, engineer, sample_transaction, sample_customer, sample_history):
        """Test daily count with transaction history."""
        features = engineer.engineer_features(sample_transaction, sample_customer, sample_history)
        # All 5 history transactions are within 24 hours
        assert features.daily_txn_count == 5
    
    def test_weekly_transaction_count(self, engineer, sample_transaction, sample_customer, sample_history):
        """Test weekly transaction count."""
        features = engineer.engineer_features(sample_transaction, sample_customer, sample_history)
        assert features.weekly_txn_count == 5
    
    def test_daily_transaction_amount(self, engineer, sample_transaction, sample_customer, sample_history):
        """Test daily transaction amount calculation."""
        features = engineer.engineer_features(sample_transaction, sample_customer, sample_history)
        # Sum: 500 + 600 + 700 + 800 + 900 = 3500
        assert features.daily_txn_amount == 3500.0
    
    def test_average_daily_amount(self, engineer, sample_transaction, sample_customer, sample_history):
        """Test average daily amount."""
        features = engineer.engineer_features(sample_transaction, sample_customer, sample_history)
        # Average: 3500 / 5 = 700
        assert features.avg_daily_amount == 700.0


class TestVelocityFeatures(TestMLFeatureEngineer):
    """Test velocity feature calculations."""
    
    def test_transaction_frequency_1h_empty_history(self, engineer, sample_transaction, sample_customer):
        """Test 1-hour frequency with no history."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        assert features.txn_frequency_1h == 0
    
    def test_transaction_frequency_calculation(self, engineer, sample_transaction, sample_customer, sample_history):
        """Test transaction frequency in different time windows."""
        features = engineer.engineer_features(sample_transaction, sample_customer, sample_history)
        
        # All 5 transactions within 6 hours
        assert features.txn_frequency_6h == 5
        assert features.txn_frequency_24h == 5
    
    def test_amount_velocity_1h(self, engineer, sample_transaction, sample_customer, sample_history):
        """Test amount velocity in 1-hour window."""
        # Create history with recent transaction
        recent_txn = sample_history[-1].copy()
        recent_txn['Time'] = '14:00:00'  # 30 minutes before current
        
        features = engineer.engineer_features(sample_transaction, sample_customer, [recent_txn])
        assert features.amount_velocity_1h == 900.0


class TestGeographicFeatures(TestMLFeatureEngineer):
    """Test geographic feature calculations."""
    
    def test_distance_from_home_same_city(self, engineer, sample_transaction, sample_customer):
        """Test distance from home when in same city."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        assert features.distance_from_home == 0.0
    
    def test_distance_from_home_different_city(self, engineer, sample_customer):
        """Test distance from home when in different city."""
        transaction = {
            'Transaction_ID': 'TXN001',
            'Customer_ID': 'CUST001',
            'Date': '2025-01-15',
            'Time': '14:30:00',
            'Amount': 1500.0,
            'Category': 'Shopping',
            'City': 'Delhi',  # Different from Mumbai
            'Is_Fraud': 0,
            'Fraud_Type': 'None'
        }
        
        features = engineer.engineer_features(transaction, sample_customer, [])
        assert features.distance_from_home > 0  # Should be ~1400 km
    
    def test_unique_cities_7d(self, engineer, sample_transaction, sample_customer):
        """Test unique cities count in 7 days."""
        history = [
            {'Date': '2025-01-14', 'Time': '10:00:00', 'City': 'Mumbai'},
            {'Date': '2025-01-14', 'Time': '11:00:00', 'City': 'Mumbai'},
            {'Date': '2025-01-13', 'Time': '10:00:00', 'City': 'Delhi'},
            {'Date': '2025-01-12', 'Time': '10:00:00', 'City': 'Bangalore'},
        ]
        
        features = engineer.engineer_features(sample_transaction, sample_customer, history)
        assert features.unique_cities_7d >= 3  # Mumbai, Delhi, Bangalore


class TestTemporalFeatures(TestMLFeatureEngineer):
    """Test temporal feature calculations."""
    
    def test_unusual_hour_detection_normal_hours(self, engineer, sample_transaction, sample_customer):
        """Test unusual hour flag during normal hours."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        # 14:30 is normal
        assert features.is_unusual_hour == 0
    
    def test_unusual_hour_detection_early_morning(self, engineer, sample_customer):
        """Test unusual hour flag during early morning (2-5 AM)."""
        transaction = {
            'Transaction_ID': 'TXN001',
            'Date': '2025-01-15',
            'Time': '03:30:00',  # 3:30 AM - unusual
            'City': 'Mumbai',
            'Is_Fraud': 0,
            'Fraud_Type': 'None'
        }
        
        features = engineer.engineer_features(transaction, sample_customer, [])
        assert features.is_unusual_hour == 1
    
    def test_weekend_detection_weekday(self, engineer, sample_customer):
        """Test weekend flag on weekday."""
        # 2025-01-15 is Wednesday
        transaction = {
            'Transaction_ID': 'TXN001',
            'Date': '2025-01-15',
            'Time': '14:30:00',
            'City': 'Mumbai',
            'Is_Fraud': 0,
            'Fraud_Type': 'None'
        }
        
        features = engineer.engineer_features(transaction, sample_customer, [])
        assert features.is_weekend == 0
    
    def test_weekend_detection_saturday(self, engineer, sample_customer):
        """Test weekend flag on Saturday."""
        # 2025-01-18 is Saturday
        transaction = {
            'Transaction_ID': 'TXN001',
            'Date': '2025-01-18',
            'Time': '14:30:00',
            'City': 'Mumbai',
            'Is_Fraud': 0,
            'Fraud_Type': 'None'
        }
        
        features = engineer.engineer_features(transaction, sample_customer, [])
        assert features.is_weekend == 1
    
    def test_hour_of_day_extraction(self, engineer, sample_transaction, sample_customer):
        """Test hour of day extraction."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        assert features.hour_of_day == 14
    
    def test_day_of_week_extraction(self, engineer, sample_transaction, sample_customer):
        """Test day of week extraction."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        # 2025-01-15 is Wednesday (day 2)
        assert features.day_of_week == 2


class TestBehavioralFeatures(TestMLFeatureEngineer):
    """Test behavioral feature calculations."""
    
    def test_category_diversity_single_category(self, engineer, sample_transaction, sample_customer):
        """Test category diversity with single category."""
        history = [
            {'Category': 'Shopping', 'Date': '2025-01-14', 'Time': '10:00:00'},
            {'Category': 'Shopping', 'Date': '2025-01-14', 'Time': '11:00:00'},
        ]
        
        features = engineer.engineer_features(sample_transaction, sample_customer, history)
        # Single category = low entropy (close to 0)
        assert features.category_diversity_score >= 0
    
    def test_category_diversity_multiple_categories(self, engineer, sample_transaction, sample_customer):
        """Test category diversity with multiple categories."""
        history = [
            {'Category': 'Shopping', 'Date': '2025-01-14', 'Time': '10:00:00'},
            {'Category': 'Food', 'Date': '2025-01-14', 'Time': '11:00:00'},
            {'Category': 'Travel', 'Date': '2025-01-14', 'Time': '12:00:00'},
            {'Category': 'Entertainment', 'Date': '2025-01-14', 'Time': '13:00:00'},
        ]
        
        features = engineer.engineer_features(sample_transaction, sample_customer, history)
        # Multiple categories = higher entropy
        assert features.category_diversity_score > 1.0
    
    def test_merchant_loyalty_score(self, engineer, sample_transaction, sample_customer):
        """Test merchant loyalty score calculation."""
        history = [
            {'Merchant': 'Merchant_A', 'Date': '2025-01-14', 'Time': '10:00:00'},
            {'Merchant': 'Merchant_A', 'Date': '2025-01-14', 'Time': '11:00:00'},
            {'Merchant': 'Merchant_B', 'Date': '2025-01-14', 'Time': '12:00:00'},
        ]
        
        features = engineer.engineer_features(sample_transaction, sample_customer, history)
        # Loyalty = 1 - (unique_merchants / total_transactions)
        # = 1 - (2 / 3) = 0.333...
        assert 0.0 <= features.merchant_loyalty_score <= 1.0
    
    def test_new_merchant_flag_first_time(self, engineer, sample_customer):
        """Test new merchant flag for first-time merchant."""
        transaction = {
            'Transaction_ID': 'TXN001',
            'Merchant': 'NewMerchant',
            'Date': '2025-01-15',
            'Time': '14:30:00',
            'City': 'Mumbai',
            'Is_Fraud': 0,
            'Fraud_Type': 'None'
        }
        
        history = [
            {'Merchant': 'OldMerchant', 'Date': '2025-01-14', 'Time': '10:00:00'},
        ]
        
        features = engineer.engineer_features(transaction, sample_customer, history)
        assert features.new_merchant_flag == 1
    
    def test_new_merchant_flag_repeat_merchant(self, engineer, sample_customer):
        """Test new merchant flag for repeat merchant."""
        transaction = {
            'Transaction_ID': 'TXN001',
            'Merchant': 'OldMerchant',
            'Date': '2025-01-15',
            'Time': '14:30:00',
            'City': 'Mumbai',
            'Is_Fraud': 0,
            'Fraud_Type': 'None'
        }
        
        history = [
            {'Merchant': 'OldMerchant', 'Date': '2025-01-14', 'Time': '10:00:00'},
        ]
        
        features = engineer.engineer_features(transaction, sample_customer, history)
        assert features.new_merchant_flag == 0


class TestNetworkFeatures(TestMLFeatureEngineer):
    """Test network feature calculations."""
    
    def test_network_features_initialized(self, engineer, sample_transaction, sample_customer):
        """Test that network features are calculated."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        
        assert features.shared_merchant_count >= 0
        assert features.shared_location_count >= 0
        assert 0.0 <= features.customer_proximity_score <= 1.0
        assert features.temporal_cluster_flag in [0, 1]


class TestFeatureMetadata(TestMLFeatureEngineer):
    """Test feature metadata generation."""
    
    def test_get_feature_metadata_structure(self, engineer):
        """Test feature metadata has correct structure."""
        metadata = engineer.get_feature_metadata()
        
        assert 'feature_count' in metadata
        assert 'features' in metadata
        assert metadata['feature_count'] == 32
    
    def test_feature_metadata_completeness(self, engineer):
        """Test that metadata describes all features."""
        metadata = engineer.get_feature_metadata()
        
        # Metadata includes 32 features + 2 labels (is_fraud, fraud_type) = 34
        assert len(metadata['features']) == 34
        
        # Check first feature structure
        first_feature = metadata['features'][0]
        assert 'name' in first_feature
        assert 'type' in first_feature
        assert 'description' in first_feature
    
    def test_feature_categories_in_metadata(self, engineer):
        """Test that all feature categories are represented."""
        metadata = engineer.get_feature_metadata()
        feature_names = [f['name'] for f in metadata['features']]
        
        # Check for features from each category
        assert any('daily' in name or 'weekly' in name for name in feature_names)  # Aggregate
        assert any('frequency' in name or 'velocity' in name for name in feature_names)  # Velocity
        assert any('distance' in name or 'cities' in name for name in feature_names)  # Geographic
        assert any('hour' in name or 'weekend' in name for name in feature_names)  # Temporal
        assert any('diversity' in name or 'loyalty' in name for name in feature_names)  # Behavioral
        assert any('shared' in name or 'proximity' in name for name in feature_names)  # Network


class TestEdgeCases(TestMLFeatureEngineer):
    """Test edge cases and error handling."""
    
    def test_empty_transaction_history(self, engineer, sample_transaction, sample_customer):
        """Test with empty transaction history."""
        features = engineer.engineer_features(sample_transaction, sample_customer, [])
        
        # Should not raise errors
        assert isinstance(features, MLFeatures)
        assert features.daily_txn_count == 0
        assert features.weekly_txn_count == 0
    
    def test_none_transaction_history(self, engineer, sample_transaction, sample_customer):
        """Test with None as transaction history."""
        features = engineer.engineer_features(sample_transaction, sample_customer, None)
        
        # Should handle None gracefully
        assert isinstance(features, MLFeatures)
    
    def test_missing_transaction_fields(self, engineer, sample_customer):
        """Test with minimal transaction fields."""
        minimal_transaction = {
            'Transaction_ID': 'TXN001',
            'Date': '2025-01-15',
            'Time': '14:30:00',
            'Is_Fraud': 0,
            'Fraud_Type': 'None'
        }
        
        features = engineer.engineer_features(minimal_transaction, sample_customer, [])
        
        # Should still generate features with defaults
        assert isinstance(features, MLFeatures)
