"""Tests for cross-pattern statistics tracking."""

import pytest
from src.generators.fraud_patterns import FraudPatternGenerator, FraudType


class DummyCustomer:
    def __init__(self, customer_id='CUST-001'):
        self.customer_id = customer_id
        self.age = 35
        self.city = 'Mumbai'
        self.state = 'Maharashtra'
        self.segment = 'Premium'


def create_transaction(txn_id='TXN-001', amount=1000):
    return {
        'Transaction_ID': txn_id,
        'Customer_ID': 'CUST-001',
        'Date': '2025-10-26',
        'Time': '12:00:00',
        'Hour': 12,
        'Amount': amount,
        'Merchant_ID': 'MRCH-001',
        'City': 'Mumbai',
        'Category': 'Shopping',
        'Channel': 'Online',
        'Device_Type': 'Web',
        'Transaction_Status': 'Approved',
        'Is_Fraud': 0,
    }


def create_history():
    return [
        {'Date': '2025-10-25', 'Time': '11:00:00', 'Amount': 50, 'Merchant_ID': 'M1', 'Category': 'Entertainment', 'Channel': 'Online', 'City': 'Mumbai', 'Hour': 11},
        {'Date': '2025-10-25', 'Time': '13:00:00', 'Amount': 9000, 'Merchant_ID': 'M2', 'Category': 'Electronics', 'Channel': 'Online', 'City': 'Mumbai', 'Hour': 13},
        {'Date': '2025-10-24', 'Time': '10:00:00', 'Amount': 1000, 'Merchant_ID': 'M3', 'Category': 'Dining', 'Channel': 'POS', 'City': 'Mumbai', 'Hour': 10},
    ]


class TestCrossPatternStatistics:
    """Test cross-pattern statistics tracking."""
    
    def test_co_occurrence_tracking_initialization(self):
        """Test that co-occurrence tracking is initialized."""
        generator = FraudPatternGenerator(fraud_rate=0.02, seed=42)
        assert hasattr(generator, 'pattern_co_occurrences')
        assert hasattr(generator, 'pattern_isolation_stats')
    
    def test_get_co_occurrence_matrix(self):
        """Test getting co-occurrence matrix."""
        generator = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        customer = DummyCustomer()
        
        # Generate some transactions to populate co-occurrences
        for i in range(20):
            txn = create_transaction(f'TXN-{i}', amount=1000 + i*100)
            history = create_history()
            generator.maybe_apply_fraud(txn, customer, history)
        
        matrix = generator.get_pattern_co_occurrence_matrix()
        
        assert isinstance(matrix, dict)
        # Should have entries for all fraud types
        assert len(matrix) == len(FraudType)
        # Check that matrix is symmetric
        for ft1_name in matrix:
            for ft2_name in matrix[ft1_name]:
                assert matrix[ft1_name][ft2_name] == matrix[ft2_name][ft1_name]
    
    def test_pattern_isolation_stats(self):
        """Test getting pattern isolation statistics."""
        generator = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        customer = DummyCustomer()
        
        # Generate transactions
        for i in range(30):
            txn = create_transaction(f'TXN-{i}', amount=500 + i*50)
            history = create_history()
            generator.maybe_apply_fraud(txn, customer, history)
        
        isolation_stats = generator.get_pattern_isolation_stats()
        
        assert isinstance(isolation_stats, dict)
        # Should have entries for all fraud types
        for ft in FraudType:
            if ft.value in isolation_stats:
                stats = isolation_stats[ft.value]
                assert 'total_occurrences' in stats
                assert 'isolated_occurrences' in stats
                assert 'combined_occurrences' in stats
                assert 'isolation_rate' in stats
                assert 0.0 <= stats['isolation_rate'] <= 1.0
    
    def test_cross_pattern_statistics_comprehensive(self):
        """Test comprehensive cross-pattern statistics."""
        generator = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        customer = DummyCustomer()
        
        # Generate many transactions
        for i in range(50):
            txn = create_transaction(f'TXN-{i}', amount=100 + i*50)
            history = create_history()
            generator.maybe_apply_fraud(txn, customer, history)
        
        cross_stats = generator.get_cross_pattern_statistics()
        
        assert 'co_occurrence_matrix' in cross_stats
        assert 'isolation_stats' in cross_stats
        assert 'overall_isolation_rate' in cross_stats
        assert 'total_isolated_patterns' in cross_stats
        assert 'total_combined_patterns' in cross_stats
        assert 'most_common_combinations' in cross_stats
        assert 'patterns_meeting_isolation_target' in cross_stats
        
        # Check that overall isolation rate is reasonable
        assert 0.0 <= cross_stats['overall_isolation_rate'] <= 1.0
    
    def test_isolation_rate_calculation(self):
        """Test that isolation rate is calculated correctly."""
        generator = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        customer = DummyCustomer()
        
        # Generate transactions
        for i in range(40):
            txn = create_transaction(f'TXN-{i}', amount=200 + i*100)
            history = create_history()
            generator.maybe_apply_fraud(txn, customer, history)
        
        isolation_stats = generator.get_pattern_isolation_stats()
        
        # For each pattern that was used, verify isolation rate calculation
        for ft_name, stats in isolation_stats.items():
            if stats['total_occurrences'] > 0:
                expected_rate = stats['isolated_occurrences'] / stats['total_occurrences']
                assert abs(stats['isolation_rate'] - expected_rate) < 0.001
    
    def test_most_common_combinations(self):
        """Test that most common combinations are tracked."""
        generator = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        customer = DummyCustomer()
        
        # Generate many transactions
        for i in range(60):
            txn = create_transaction(f'TXN-{i}', amount=300 + i*75)
            history = create_history()
            generator.maybe_apply_fraud(txn, customer, history)
        
        cross_stats = generator.get_cross_pattern_statistics()
        combinations = cross_stats['most_common_combinations']
        
        assert isinstance(combinations, list)
        # Should be sorted by count (descending)
        for i in range(len(combinations) - 1):
            assert combinations[i]['count'] >= combinations[i+1]['count']
        
        # Each combination should have pattern_1, pattern_2, count
        for combo in combinations:
            assert 'pattern_1' in combo
            assert 'pattern_2' in combo
            assert 'count' in combo
            assert combo['count'] > 0
    
    def test_reset_statistics_clears_cross_pattern_data(self):
        """Test that reset_statistics clears cross-pattern tracking."""
        generator = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        customer = DummyCustomer()
        
        # Generate transactions
        for i in range(20):
            txn = create_transaction(f'TXN-{i}', amount=500)
            history = create_history()
            generator.maybe_apply_fraud(txn, customer, history)
        
        # Verify data was tracked
        cross_stats_before = generator.get_cross_pattern_statistics()
        assert cross_stats_before['total_isolated_patterns'] > 0 or cross_stats_before['total_combined_patterns'] > 0
        
        # Reset
        generator.reset_statistics()
        
        # Verify data was cleared
        cross_stats_after = generator.get_cross_pattern_statistics()
        assert cross_stats_after['total_isolated_patterns'] == 0
        assert cross_stats_after['total_combined_patterns'] == 0
        assert cross_stats_after['overall_isolation_rate'] == 0.0
    
    def test_pattern_isolation_target_95_percent(self):
        """Test that patterns meeting 95% isolation target are identified."""
        generator = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        customer = DummyCustomer()
        
        # Generate many transactions to get reliable statistics
        for i in range(100):
            txn = create_transaction(f'TXN-{i}', amount=100 + i*20)
            history = create_history()
            generator.maybe_apply_fraud(txn, customer, history)
        
        cross_stats = generator.get_cross_pattern_statistics()
        patterns_meeting_target = cross_stats['patterns_meeting_isolation_target']
        
        assert isinstance(patterns_meeting_target, list)
        # Verify each pattern in the list actually has 95%+ isolation
        isolation_stats = cross_stats['isolation_stats']
        for pattern_name in patterns_meeting_target:
            if pattern_name in isolation_stats:
                assert isolation_stats[pattern_name]['isolation_rate'] >= 0.95
    
    def test_co_occurrence_matrix_symmetry(self):
        """Test that co-occurrence matrix is symmetric."""
        generator = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        customer = DummyCustomer()
        
        # Generate transactions
        for i in range(30):
            txn = create_transaction(f'TXN-{i}', amount=800)
            history = create_history()
            generator.maybe_apply_fraud(txn, customer, history)
        
        matrix = generator.get_pattern_co_occurrence_matrix()
        
        # Check symmetry: matrix[A][B] == matrix[B][A]
        for ft1_name in matrix:
            for ft2_name in matrix[ft1_name]:
                assert matrix[ft1_name][ft2_name] == matrix[ft2_name][ft1_name]
    
    def test_combined_occurrences_calculation(self):
        """Test that combined occurrences are calculated correctly."""
        generator = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        customer = DummyCustomer()
        
        # Generate transactions
        for i in range(50):
            txn = create_transaction(f'TXN-{i}', amount=400 + i*30)
            history = create_history()
            generator.maybe_apply_fraud(txn, customer, history)
        
        isolation_stats = generator.get_pattern_isolation_stats()
        
        # For each pattern, combined = total - isolated
        for ft_name, stats in isolation_stats.items():
            expected_combined = stats['total_occurrences'] - stats['isolated_occurrences']
            assert stats['combined_occurrences'] == expected_combined
