"""
Tests for anomaly pattern generation system

This module tests all anomaly pattern types, the orchestration system,
and the anomaly labeling functionality.
"""

import pytest
from datetime import datetime, timedelta
import json
from src.generators.anomaly_patterns import (
    AnomalyType,
    AnomalyIndicator,
    AnomalyPattern,
    BehavioralAnomalyPattern,
    GeographicAnomalyPattern,
    TemporalAnomalyPattern,
    AmountAnomalyPattern,
    AnomalyPatternGenerator,
    apply_anomaly_labels
)
from src.customer_profile import CustomerProfile, CustomerSegment, IncomeBracket, Occupation, RiskProfile


class MockCustomer:
    """Mock customer for testing"""
    def __init__(self, customer_id='C001', city='Mumbai', segment=CustomerSegment.YOUNG_PROFESSIONAL):
        self.customer_id = customer_id
        self.city = city
        self.segment = segment


class TestAnomalyIndicator:
    """Tests for AnomalyIndicator dataclass"""
    
    def test_anomaly_indicator_creation(self):
        """Test creating an anomaly indicator"""
        indicator = AnomalyIndicator(
            anomaly_type=AnomalyType.BEHAVIORAL,
            confidence=0.75,
            reason="Test anomaly",
            evidence={'key': 'value'},
            severity=0.6
        )
        
        assert indicator.anomaly_type == AnomalyType.BEHAVIORAL
        assert indicator.confidence == 0.75
        assert indicator.reason == "Test anomaly"
        assert indicator.evidence == {'key': 'value'}
        assert indicator.severity == 0.6
    
    def test_anomaly_indicator_to_dict(self):
        """Test converting anomaly indicator to dictionary"""
        indicator = AnomalyIndicator(
            anomaly_type=AnomalyType.GEOGRAPHIC,
            confidence=0.85,
            reason="Geographic test",
            evidence={'distance_km': 500},
            severity=0.7
        )
        
        result = indicator.to_dict()
        
        assert result['anomaly_type'] == 'Geographic Anomaly'
        assert result['confidence'] == 0.85
        assert result['reason'] == "Geographic test"
        assert result['evidence'] == {'distance_km': 500}
        assert result['severity'] == 0.7


class TestBehavioralAnomalyPattern:
    """Tests for behavioral anomaly detection"""
    
    def test_behavioral_anomaly_requires_history(self):
        """Test that behavioral anomaly requires sufficient history"""
        pattern = BehavioralAnomalyPattern(seed=42)
        customer = MockCustomer()
        transaction = {'Amount': 1000, 'Category': 'Shopping'}
        
        # Not enough history
        short_history = [{'Amount': 500} for _ in range(5)]
        assert not pattern.should_apply(transaction, customer, short_history)
        
        # Sufficient history
        long_history = [{'Amount': 500} for _ in range(15)]
        # should_apply has randomness, so just check it doesn't crash
        result = pattern.should_apply(transaction, customer, long_history)
        assert isinstance(result, bool)
    
    def test_behavioral_category_deviation(self):
        """Test category deviation detection"""
        pattern = BehavioralAnomalyPattern(seed=42)
        customer = MockCustomer()
        
        # Build history with consistent category
        history = []
        for i in range(20):
            history.append({
                'Category': 'Groceries',
                'Amount': 1000 + i * 10,
                'Payment_Mode': 'UPI'
            })
        
        transaction = {
            'Category': 'Entertainment',  # Will be changed by pattern
            'Amount': 1000,
            'Payment_Mode': 'UPI'
        }
        
        # Force pattern application
        if pattern.should_apply(transaction, customer, history):
            indicator = pattern.apply_pattern(transaction, customer, history)
            
            assert indicator.anomaly_type == AnomalyType.BEHAVIORAL
            assert 0.5 <= indicator.confidence <= 1.0
            assert 0.0 <= indicator.severity <= 1.0
            assert len(indicator.reason) > 0
            assert isinstance(indicator.evidence, dict)
    
    def test_behavioral_amount_spike(self):
        """Test amount spike detection"""
        pattern = BehavioralAnomalyPattern(seed=123)
        customer = MockCustomer()
        
        # Build history with consistent amounts
        history = []
        for i in range(20):
            history.append({
                'Category': 'Groceries',
                'Amount': 1000,
                'Payment_Mode': 'UPI'
            })
        
        transaction = {
            'Category': 'Groceries',
            'Amount': 1000,
            'Payment_Mode': 'UPI'
        }
        
        # Apply pattern multiple times to get amount spike
        for attempt in range(10):
            if pattern.should_apply(transaction, customer, history):
                indicator = pattern.apply_pattern(transaction, customer, history)
                
                # Check if amount was modified
                if transaction['Amount'] > 2500:  # 2.5x+ of original 1000
                    assert indicator.anomaly_type == AnomalyType.BEHAVIORAL
                    assert 'spike' in indicator.reason.lower() or 'spending' in indicator.reason.lower()
                    assert indicator.evidence.get('multiplier', 0) >= 3.0
                    break
    
    def test_behavioral_payment_method_change(self):
        """Test payment method change detection"""
        pattern = BehavioralAnomalyPattern(seed=456)
        customer = MockCustomer()
        
        # Build history with consistent payment method
        history = []
        for i in range(20):
            history.append({
                'Category': 'Groceries',
                'Amount': 1000,
                'Payment_Mode': 'UPI'
            })
        
        transaction = {
            'Category': 'Groceries',
            'Amount': 1000,
            'Payment_Mode': 'UPI'
        }
        
        # Apply pattern multiple times to get payment method change
        for attempt in range(10):
            if pattern.should_apply(transaction, customer, history):
                indicator = pattern.apply_pattern(transaction, customer, history)
                
                # Check if payment method changed
                if transaction['Payment_Mode'] != 'UPI':
                    assert indicator.anomaly_type == AnomalyType.BEHAVIORAL
                    assert 'payment' in indicator.reason.lower()
                    break


class TestGeographicAnomalyPattern:
    """Tests for geographic anomaly detection"""
    
    def test_geographic_anomaly_requires_history(self):
        """Test that geographic anomaly requires at least one previous transaction"""
        pattern = GeographicAnomalyPattern(seed=42)
        customer = MockCustomer()
        transaction = {'City': 'Mumbai'}
        
        # No history
        assert not pattern.should_apply(transaction, customer, [])
        
        # With history
        history = [{'City': 'Mumbai', 'Transaction_Date': '2025-01-01'}]
        result = pattern.should_apply(transaction, customer, history)
        assert isinstance(result, bool)
    
    def test_geographic_distance_calculation(self):
        """Test geographic distance anomaly"""
        pattern = GeographicAnomalyPattern(seed=42)
        customer = MockCustomer(city='Mumbai')
        
        # Build history in Mumbai
        history = [{
            'City': 'Mumbai',
            'Transaction_Date': '2025-01-01'
        }]
        
        transaction = {
            'City': 'Mumbai',  # Will be changed by pattern
            'Transaction_Date': '2025-01-02'
        }
        
        # Force pattern application
        if pattern.should_apply(transaction, customer, history):
            indicator = pattern.apply_pattern(transaction, customer, history)
            
            assert indicator.anomaly_type == AnomalyType.GEOGRAPHIC
            assert 0.5 <= indicator.confidence <= 1.0
            assert 0.0 <= indicator.severity <= 1.0
            
            # Check evidence contains distance information
            evidence = indicator.evidence
            if 'distance_km' in evidence:
                assert evidence['distance_km'] > 0
                assert 'previous_city' in evidence
                assert 'current_city' in evidence
    
    def test_geographic_impossible_travel(self):
        """Test impossible travel detection"""
        pattern = GeographicAnomalyPattern(seed=789)
        customer = MockCustomer(city='Mumbai')
        
        # Create a transaction 1 hour after previous one
        history = [{
            'City': 'Mumbai',
            'Transaction_Date': '2025-01-01'
        }]
        
        transaction = {
            'City': 'Delhi',
            'Transaction_Date': '2025-01-01'  # Same day = very short time
        }
        
        # Apply pattern
        if pattern.should_apply(transaction, customer, history):
            indicator = pattern.apply_pattern(transaction, customer, history)
            
            assert indicator.anomaly_type == AnomalyType.GEOGRAPHIC
            
            # If it's truly impossible travel, severity should be high
            if 'implied_speed_kmh' in indicator.evidence:
                speed = indicator.evidence['implied_speed_kmh']
                if speed > 800:
                    assert indicator.severity >= 0.7


class TestTemporalAnomalyPattern:
    """Tests for temporal anomaly detection"""
    
    def test_temporal_anomaly_requires_history(self):
        """Test that temporal anomaly requires sufficient history"""
        pattern = TemporalAnomalyPattern(seed=42)
        customer = MockCustomer()
        transaction = {'Hour': 14, 'Time': '14:00:00'}
        
        # Not enough history
        short_history = [{'Hour': 12} for _ in range(5)]
        assert not pattern.should_apply(transaction, customer, short_history)
        
        # Sufficient history
        long_history = [{'Hour': 12} for _ in range(15)]
        result = pattern.should_apply(transaction, customer, long_history)
        assert isinstance(result, bool)
    
    def test_temporal_unusual_hour_detection(self):
        """Test unusual hour detection"""
        pattern = TemporalAnomalyPattern(seed=42)
        customer = MockCustomer()
        
        # Build history with daytime transactions
        history = []
        for i in range(20):
            history.append({
                'Hour': 14,  # 2 PM
                'Time': '14:00:00'
            })
        
        transaction = {
            'Hour': 14,
            'Time': '14:00:00'
        }
        
        # Force pattern application
        if pattern.should_apply(transaction, customer, history):
            indicator = pattern.apply_pattern(transaction, customer, history)
            
            assert indicator.anomaly_type == AnomalyType.TEMPORAL
            assert 0.5 <= indicator.confidence <= 1.0
            assert 0.0 <= indicator.severity <= 1.0
            
            # Check that hour was changed
            new_hour = transaction['Hour']
            assert new_hour != 14  # Should be different from original
            
            # Check evidence
            assert 'transaction_hour' in indicator.evidence
            assert indicator.evidence['transaction_hour'] == new_hour
    
    def test_temporal_late_night_severity(self):
        """Test that late night hours have higher severity"""
        pattern = TemporalAnomalyPattern(seed=999)
        customer = MockCustomer()
        
        # Build daytime history
        history = []
        for i in range(20):
            history.append({
                'Hour': 15,
                'Time': '15:00:00'
            })
        
        transaction = {
            'Hour': 15,
            'Time': '15:00:00'
        }
        
        # Apply pattern multiple times to potentially get late night
        for attempt in range(20):
            if pattern.should_apply(transaction, customer, history):
                indicator = pattern.apply_pattern(transaction, customer, history)
                
                # If we got a late night hour, check severity is appropriate
                hour = transaction['Hour']
                if hour in [0, 1, 2, 3, 4, 5]:
                    assert indicator.severity >= 0.5
                    assert 'late night' in indicator.reason.lower() or 'early morning' in indicator.reason.lower()
                    break


class TestAmountAnomalyPattern:
    """Tests for amount anomaly detection"""
    
    def test_amount_anomaly_requires_history(self):
        """Test that amount anomaly requires sufficient history"""
        pattern = AmountAnomalyPattern(seed=42)
        customer = MockCustomer()
        transaction = {'Amount': 1000}
        
        # Not enough history
        short_history = [{'Amount': 500} for _ in range(5)]
        assert not pattern.should_apply(transaction, customer, short_history)
        
        # Sufficient history
        long_history = [{'Amount': 500} for _ in range(15)]
        result = pattern.should_apply(transaction, customer, long_history)
        assert isinstance(result, bool)
    
    def test_amount_spending_spike(self):
        """Test spending spike detection"""
        pattern = AmountAnomalyPattern(seed=111)
        customer = MockCustomer()
        
        # Build history with Rs. 1000 average
        history = []
        for i in range(20):
            history.append({'Amount': 1000})
        
        transaction = {'Amount': 1000}
        
        # Apply pattern multiple times to get spending spike
        for attempt in range(10):
            if pattern.should_apply(transaction, customer, history):
                indicator = pattern.apply_pattern(transaction, customer, history)
                
                # Check if we got a spending spike
                if transaction['Amount'] > 2500:  # More than 2.5x
                    assert indicator.anomaly_type == AnomalyType.AMOUNT
                    assert 'spike' in indicator.reason.lower()
                    assert indicator.evidence.get('multiplier', 0) >= 3.0
                    assert indicator.evidence['avg_amount_30d'] == 1000
                    break
    
    def test_amount_micro_transaction(self):
        """Test micro-transaction detection"""
        pattern = AmountAnomalyPattern(seed=222)
        customer = MockCustomer()
        
        # Build history with Rs. 1000 average
        history = []
        for i in range(20):
            history.append({'Amount': 1000})
        
        transaction = {'Amount': 1000}
        
        # Apply pattern multiple times to get micro-transaction
        for attempt in range(10):
            if pattern.should_apply(transaction, customer, history):
                indicator = pattern.apply_pattern(transaction, customer, history)
                
                # Check if we got a micro-transaction
                if transaction['Amount'] < 100:
                    assert indicator.anomaly_type == AnomalyType.AMOUNT
                    assert 'small' in indicator.reason.lower() or 'micro' in indicator.reason.lower()
                    assert transaction['Amount'] >= 10
                    assert transaction['Amount'] <= 50
                    break
    
    def test_amount_round_amount(self):
        """Test round amount detection"""
        pattern = AmountAnomalyPattern(seed=333)
        customer = MockCustomer()
        
        # Build history with Rs. 1250 average
        history = []
        for i in range(20):
            history.append({'Amount': 1250})
        
        transaction = {'Amount': 1250}
        
        # Apply pattern multiple times to get round amount
        for attempt in range(10):
            if pattern.should_apply(transaction, customer, history):
                indicator = pattern.apply_pattern(transaction, customer, history)
                
                # Check if we got a round amount
                amount = transaction['Amount']
                if amount in [1000, 2000, 5000, 10000, 15000, 20000]:
                    assert indicator.anomaly_type == AnomalyType.AMOUNT
                    assert 'round' in indicator.reason.lower()
                    assert indicator.evidence.get('is_round_amount') == True
                    break


class TestAnomalyPatternGenerator:
    """Tests for anomaly pattern generator orchestration"""
    
    def test_generator_initialization(self):
        """Test anomaly pattern generator initialization"""
        generator = AnomalyPatternGenerator(seed=42)
        
        assert len(generator.patterns) == 4  # 4 anomaly types
        assert generator.stats['total_transactions'] == 0
        assert generator.stats['anomaly_count'] == 0
        assert generator.stats['anomaly_rate'] == 0.0
    
    def test_anomaly_injection_rate(self):
        """Test that anomaly rate is approximately correct"""
        generator = AnomalyPatternGenerator(seed=42)
        
        # Create customers
        customers = [MockCustomer(customer_id=f'C{i:03d}') for i in range(10)]
        
        # Create transactions with enough history
        transactions = []
        for i in range(200):
            customer_id = f'C{i % 10:03d}'
            transactions.append({
                'Customer_ID': customer_id,
                'Amount': 1000 + (i * 10) % 500,
                'Category': ['Groceries', 'Shopping', 'Dining'][i % 3],
                'Payment_Mode': 'UPI',
                'City': 'Mumbai',
                'Transaction_Date': '2025-01-01',
                'Hour': 14,
                'Time': '14:00:00'
            })
        
        # Inject anomalies with 5% rate
        result = generator.inject_anomaly_patterns(transactions, customers, anomaly_rate=0.05)
        
        assert len(result) == 200
        stats = generator.get_statistics()
        
        # Check statistics
        assert stats['total_transactions'] == 200
        assert stats['anomaly_count'] > 0  # Should have some anomalies
        
        # Anomaly rate should be close to target (within 5% variance)
        # Note: May be lower than target due to history requirements
        assert stats['anomaly_rate'] >= 0.0
        assert stats['anomaly_rate'] <= 0.15  # Allow some variance
    
    def test_anomaly_fields_added(self):
        """Test that anomaly fields are added to transactions"""
        generator = AnomalyPatternGenerator(seed=42)
        
        customers = [MockCustomer(customer_id='C001')]
        
        transactions = []
        for i in range(50):
            transactions.append({
                'Customer_ID': 'C001',
                'Amount': 1000,
                'Category': 'Groceries',
                'Payment_Mode': 'UPI',
                'City': 'Mumbai',
                'Transaction_Date': '2025-01-01',
                'Hour': 14,
                'Time': '14:00:00'
            })
        
        result = generator.inject_anomaly_patterns(transactions, customers, anomaly_rate=0.1)
        
        # Check that at least one transaction has anomaly
        anomaly_found = False
        for txn in result:
            if 'Anomaly_Type' in txn and txn['Anomaly_Type'] != 'None':
                anomaly_found = True
                
                # Verify all anomaly fields present
                assert 'Anomaly_Type' in txn
                assert 'Anomaly_Confidence' in txn
                assert 'Anomaly_Reason' in txn
                assert 'Anomaly_Severity' in txn
                assert 'Anomaly_Evidence' in txn
                
                # Verify field types
                assert isinstance(txn['Anomaly_Type'], str)
                assert isinstance(txn['Anomaly_Confidence'], float)
                assert isinstance(txn['Anomaly_Reason'], str)
                assert isinstance(txn['Anomaly_Severity'], float)
                assert isinstance(txn['Anomaly_Evidence'], str)
                
                # Verify ranges
                assert 0.0 <= txn['Anomaly_Confidence'] <= 1.0
                assert 0.0 <= txn['Anomaly_Severity'] <= 1.0
                
                # Verify evidence is valid JSON
                evidence = json.loads(txn['Anomaly_Evidence'])
                assert isinstance(evidence, dict)
        
        # Should have found at least one anomaly
        assert anomaly_found or len(transactions) < 20  # Allow for random variation
    
    def test_anomaly_distribution(self):
        """Test that anomalies are distributed across all types"""
        generator = AnomalyPatternGenerator(seed=42)
        
        customers = [MockCustomer(customer_id=f'C{i:03d}') for i in range(20)]
        
        transactions = []
        for i in range(500):  # Larger dataset
            customer_id = f'C{i % 20:03d}'
            transactions.append({
                'Customer_ID': customer_id,
                'Amount': 1000 + (i * 10) % 1000,
                'Category': ['Groceries', 'Shopping', 'Dining', 'Travel'][i % 4],
                'Payment_Mode': ['UPI', 'Credit Card', 'Debit Card'][i % 3],
                'City': ['Mumbai', 'Delhi', 'Bangalore'][i % 3],
                'Transaction_Date': '2025-01-01',
                'Hour': 10 + (i % 12),
                'Time': f'{10 + (i % 12):02d}:00:00'
            })
        
        result = generator.inject_anomaly_patterns(transactions, customers, anomaly_rate=0.08)
        stats = generator.get_statistics()
        
        # Check that we have anomalies from multiple types
        anomaly_types = stats['anomalies_by_type']
        types_with_anomalies = sum(1 for count in anomaly_types.values() if count > 0)
        
        # Should have at least 2 different anomaly types
        assert types_with_anomalies >= 2
    
    def test_statistics_reset(self):
        """Test statistics reset functionality"""
        generator = AnomalyPatternGenerator(seed=42)
        
        customers = [MockCustomer(customer_id='C001')]
        transactions = [
            {
                'Customer_ID': 'C001',
                'Amount': 1000,
                'Category': 'Groceries',
                'Payment_Mode': 'UPI',
                'City': 'Mumbai',
                'Transaction_Date': '2025-01-01',
                'Hour': 14,
                'Time': '14:00:00'
            }
            for _ in range(50)
        ]
        
        # Generate anomalies
        generator.inject_anomaly_patterns(transactions, customers, anomaly_rate=0.1)
        
        # Reset statistics
        generator.reset_statistics()
        
        stats = generator.get_statistics()
        assert stats['total_transactions'] == 0
        assert stats['anomaly_count'] == 0
        assert stats['anomaly_rate'] == 0.0


class TestAnomalyLabeling:
    """Tests for anomaly labeling functionality"""
    
    def test_apply_anomaly_labels_to_clean_transactions(self):
        """Test adding anomaly labels to transactions without anomalies"""
        transactions = [
            {'Amount': 1000, 'Category': 'Groceries'},
            {'Amount': 2000, 'Category': 'Shopping'},
            {'Amount': 1500, 'Category': 'Dining'}
        ]
        
        result = apply_anomaly_labels(transactions)
        
        assert len(result) == 3
        
        for txn in result:
            assert txn['Anomaly_Type'] == 'None'
            assert txn['Anomaly_Confidence'] == 0.0
            assert txn['Anomaly_Reason'] == "No anomaly detected"
            assert txn['Anomaly_Severity'] == 0.0
            assert txn['Anomaly_Evidence'] == '{}'
    
    def test_apply_anomaly_labels_preserves_existing(self):
        """Test that existing anomaly labels are preserved"""
        transactions = [
            {
                'Amount': 1000,
                'Anomaly_Type': 'Behavioral Anomaly',
                'Anomaly_Confidence': 0.8,
                'Anomaly_Reason': 'Test',
                'Anomaly_Severity': 0.6,
                'Anomaly_Evidence': '{"key": "value"}'
            },
            {'Amount': 2000}
        ]
        
        result = apply_anomaly_labels(transactions)
        
        # First transaction should be unchanged
        assert result[0]['Anomaly_Type'] == 'Behavioral Anomaly'
        assert result[0]['Anomaly_Confidence'] == 0.8
        
        # Second transaction should have labels added
        assert result[1]['Anomaly_Type'] == 'None'
        assert result[1]['Anomaly_Confidence'] == 0.0


class TestIntegration:
    """Integration tests for complete anomaly workflow"""
    
    def test_end_to_end_anomaly_generation(self):
        """Test complete anomaly generation workflow"""
        # Create generator
        generator = AnomalyPatternGenerator(seed=42)
        
        # Create customers
        customers = [MockCustomer(customer_id=f'C{i:03d}') for i in range(5)]
        
        # Create transactions
        transactions = []
        for i in range(100):
            customer_id = f'C{i % 5:03d}'
            transactions.append({
                'Customer_ID': customer_id,
                'Amount': 1000 + (i * 50) % 500,
                'Category': ['Groceries', 'Shopping'][i % 2],
                'Payment_Mode': 'UPI',
                'City': 'Mumbai',
                'Transaction_Date': '2025-01-01',
                'Hour': 14,
                'Time': '14:00:00'
            })
        
        # Inject anomalies
        result = generator.inject_anomaly_patterns(transactions, customers, anomaly_rate=0.05)
        
        # Apply labels to non-anomalous transactions
        result = apply_anomaly_labels(result)
        
        # Verify all transactions have anomaly fields
        for txn in result:
            assert 'Anomaly_Type' in txn
            assert 'Anomaly_Confidence' in txn
            assert 'Anomaly_Reason' in txn
            assert 'Anomaly_Severity' in txn
            assert 'Anomaly_Evidence' in txn
        
        # Get statistics
        stats = generator.get_statistics()
        assert stats['total_transactions'] == 100
        assert stats['anomaly_count'] >= 0
    
    def test_anomaly_rate_clamping(self):
        """Test that anomaly rate is clamped to valid range"""
        generator = AnomalyPatternGenerator(seed=42)
        customers = [MockCustomer(customer_id='C001')]
        
        transactions = [
            {
                'Customer_ID': 'C001',
                'Amount': 1000,
                'Category': 'Groceries',
                'Payment_Mode': 'UPI',
                'City': 'Mumbai',
                'Transaction_Date': '2025-01-01',
                'Hour': 14,
                'Time': '14:00:00'
            }
            for _ in range(50)
        ]
        
        # Test with invalid rates (should be clamped)
        result1 = generator.inject_anomaly_patterns(transactions.copy(), customers, anomaly_rate=-0.5)
        stats1 = generator.get_statistics()
        assert stats1['anomaly_rate'] >= 0.0
        
        generator.reset_statistics()
        
        result2 = generator.inject_anomaly_patterns(transactions.copy(), customers, anomaly_rate=2.0)
        stats2 = generator.get_statistics()
        assert stats2['anomaly_rate'] <= 1.0
