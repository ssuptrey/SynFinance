"""
Tests for Fraud Pattern Generator

Tests all 10 fraud pattern types, fraud injection system, and fraud labeling.

Author: SynFinance Team
Version: 0.4.0
"""

import pytest
from datetime import datetime, timedelta
from src.generators.fraud_patterns import (
    FraudType, FraudIndicator, FraudPattern,
    CardCloningPattern, AccountTakeoverPattern, MerchantCollusionPattern,
    VelocityAbusePattern, AmountManipulationPattern, RefundFraudPattern,
    StolenCardPattern, SyntheticIdentityPattern, FirstPartyFraudPattern,
    FriendlyFraudPattern, FraudPatternGenerator, apply_fraud_labels,
    inject_fraud_into_dataset
)
from src.customer_profile import (
    CustomerProfile, CustomerSegment, IncomeBracket, DigitalSavviness,
    RiskProfile, Occupation
)
from src.customer_generator import CustomerGenerator


@pytest.fixture
def sample_customer():
    """Create a sample customer for testing."""
    return CustomerProfile(
        customer_id="CUST0000001",
        age=35,
        gender="Male",
        city="Mumbai",
        state="Maharashtra",
        region="West",
        occupation=Occupation.PROFESSIONAL,
        monthly_income=120000.0,
        income_bracket=IncomeBracket.UPPER_MIDDLE,
        segment=CustomerSegment.YOUNG_PROFESSIONAL,
        digital_savviness=DigitalSavviness.HIGH,
        preferred_categories=["Electronics", "Food & Dining", "Travel"],
        preferred_payment_modes=["UPI", "Credit Card"],
        avg_transaction_amount=5000.0,
        monthly_transaction_count=25,
        risk_profile=RiskProfile.MODERATE,
        preferred_shopping_hours=[10, 11, 12, 13, 14, 15, 16, 17, 18],
        weekend_shopper=False,
        merchant_loyalty=0.6,
        brand_conscious=True,
        impulse_buyer=False,
        travels_frequently=True,
        online_shopping_preference=0.7
    )


@pytest.fixture
def sample_transaction():
    """Create a sample transaction for testing."""
    return {
        'Transaction_ID': 'TXN_20251021_000001',
        'Customer_ID': 'CUST0000001',
        'Date': '2025-10-21',
        'Time': '14:30:00',
        'Amount': 5000.0,
        'Category': 'Electronics',
        'City': 'Mumbai',
        'Merchant_ID': 'MER_ELEC_MUM_001',
        'Merchant_Name': 'Croma Electronics',
        'Merchant_Rating': 4.2,
        'Merchant_Years_Operating': 10,
        'Payment_Mode': 'Credit Card',
        'Channel': 'POS',
        'Device_Type': 'POS',
        'Distance_From_Home_km': 5.0,
        'Is_Fraud': 0,
        'Transaction_Status': 'Approved',
        'Hour': 14,
        'Is_First_Transaction_with_Merchant': False
    }


@pytest.fixture
def customer_history():
    """Create sample customer history for testing."""
    history = []
    base_date = datetime(2025, 10, 1)
    
    for i in range(20):
        txn_date = base_date + timedelta(days=i)
        history.append({
            'Transaction_ID': f'TXN_20251001_{i:06d}',
            'Customer_ID': 'CUST0000001',
            'Date': txn_date.strftime('%Y-%m-%d'),
            'Time': f'{10 + (i % 12):02d}:00:00',
            'Amount': 3000.0 + (i * 100),
            'Category': ['Groceries', 'Food & Dining', 'Shopping'][i % 3],
            'City': 'Mumbai',
            'Distance_From_Home_km': 2.0,
            'Channel': 'Online' if i % 3 == 0 else 'POS',
            'Merchant_ID': f'MER_RET_MUM_{i % 5:03d}'
        })
    
    return history


class TestFraudIndicator:
    """Test FraudIndicator dataclass."""
    
    def test_fraud_indicator_creation(self):
        """Test creating fraud indicator."""
        indicator = FraudIndicator(
            fraud_type=FraudType.CARD_CLONING,
            confidence=0.85,
            reason="Impossible travel time",
            evidence={'distance': 1000, 'time_hours': 1},
            severity='high'
        )
        
        assert indicator.fraud_type == FraudType.CARD_CLONING
        assert indicator.confidence == 0.85
        assert indicator.severity == 'high'
        assert 'distance' in indicator.evidence
    
    def test_fraud_indicator_to_dict(self):
        """Test converting fraud indicator to dictionary."""
        indicator = FraudIndicator(
            fraud_type=FraudType.VELOCITY_ABUSE,
            confidence=0.75,
            reason="Too many transactions",
            evidence={'count': 10},
            severity='medium'
        )
        
        result = indicator.to_dict()
        assert result['fraud_type'] == 'Velocity Abuse'
        assert result['confidence'] == 0.75
        assert result['severity'] == 'medium'
        assert isinstance(result['evidence'], dict)


class TestCardCloningPattern:
    """Test Card Cloning fraud pattern."""
    
    def test_impossible_travel_detection(self, sample_customer, sample_transaction, customer_history):
        """Test detection of impossible travel time."""
        pattern = CardCloningPattern(seed=42)
        
        # Apply pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, customer_history
        )
        
        # Verify fraud applied
        assert modified_txn['Is_Fraud'] == 1
        assert modified_txn['Fraud_Type'] == 'Card Cloning'
        assert fraud_indicator.fraud_type == FraudType.CARD_CLONING
        assert fraud_indicator.confidence > 0.0
        assert fraud_indicator.confidence <= 1.0
        assert 'impossible_travel' in fraud_indicator.evidence
    
    def test_card_cloning_confidence_calculation(self):
        """Test confidence calculation for card cloning."""
        pattern = CardCloningPattern()
        
        # High speed (impossible)
        evidence = {
            'travel_speed_kmh': 2500,
            'distance_km': 1500,
            'time_diff_hours': 0.6,
            'amount_pattern': 'round'
        }
        confidence = pattern.calculate_confidence(evidence)
        assert confidence >= 0.9
        
        # Medium speed
        evidence['travel_speed_kmh'] = 900
        confidence = pattern.calculate_confidence(evidence)
        assert 0.6 <= confidence <= 1.0  # Can reach 1.0 with all indicators


class TestAccountTakeoverPattern:
    """Test Account Takeover fraud pattern."""
    
    def test_behavioral_change_detection(self, sample_customer, sample_transaction, customer_history):
        """Test detection of sudden behavioral changes."""
        pattern = AccountTakeoverPattern(seed=42)
        
        # Ensure sufficient history
        assert len(customer_history) >= pattern.min_history_length
        
        # Apply pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, customer_history
        )
        
        # Verify fraud applied
        assert modified_txn['Is_Fraud'] == 1
        assert modified_txn['Fraud_Type'] == 'Account Takeover'
        assert fraud_indicator.fraud_type == FraudType.ACCOUNT_TAKEOVER
        
        # Check for behavioral changes
        assert 'amount_multiplier' in fraud_indicator.evidence
        assert fraud_indicator.evidence['amount_multiplier'] >= 3.0
    
    def test_account_takeover_requires_history(self, sample_customer, sample_transaction):
        """Test that account takeover requires sufficient history."""
        pattern = AccountTakeoverPattern()
        
        # Short history should not apply
        should_apply = pattern.should_apply(sample_customer, sample_transaction, [])
        assert should_apply == False


class TestMerchantCollusionPattern:
    """Test Merchant Collusion fraud pattern."""
    
    def test_just_below_threshold_detection(self, sample_customer, sample_transaction, customer_history):
        """Test detection of amounts just below reporting thresholds."""
        pattern = MerchantCollusionPattern(seed=42)
        
        # Apply pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, customer_history
        )
        
        # Verify fraud applied
        assert modified_txn['Is_Fraud'] == 1
        assert modified_txn['Fraud_Type'] == 'Merchant Collusion'
        
        # Check for suspicious amount
        assert modified_txn['Amount'] in pattern.suspicious_round_amounts
        assert fraud_indicator.evidence['round_amount'] == True
    
    def test_merchant_collusion_low_rating(self, sample_customer, sample_transaction, customer_history):
        """Test that merchant collusion uses low-reputation merchants."""
        pattern = MerchantCollusionPattern(seed=42)
        
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, customer_history
        )
        
        # Should use low-reputation merchant
        assert modified_txn['Merchant_Rating'] < 3.0
        assert modified_txn['Merchant_Years_Operating'] <= 2


class TestVelocityAbusePattern:
    """Test Velocity Abuse fraud pattern."""
    
    def test_high_velocity_detection(self, sample_customer, sample_transaction, customer_history):
        """Test detection of high transaction velocity."""
        pattern = VelocityAbusePattern(seed=42)
        
        # Create recent transaction cluster
        current_time = datetime.fromisoformat(sample_transaction['Date'] + ' ' + sample_transaction['Time'])
        recent_history = []
        
        for i in range(8):
            recent_time = current_time - timedelta(minutes=30 - i * 3)
            recent_history.append({
                'Date': recent_time.strftime('%Y-%m-%d'),
                'Time': recent_time.strftime('%H:%M:%S'),
                'Amount': 500.0
            })
        
        # Apply pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, recent_history
        )
        
        # Verify fraud applied
        assert modified_txn['Is_Fraud'] == 1
        assert modified_txn['Fraud_Type'] == 'Velocity Abuse'
        assert fraud_indicator.evidence['transactions_in_hour'] > pattern.velocity_threshold


class TestAmountManipulationPattern:
    """Test Amount Manipulation fraud pattern."""
    
    def test_structuring_detection(self, sample_customer, sample_transaction, customer_history):
        """Test detection of amount structuring."""
        pattern = AmountManipulationPattern(seed=42)
        
        # Apply pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, customer_history
        )
        
        # Verify fraud applied
        assert modified_txn['Is_Fraud'] == 1
        assert modified_txn['Fraud_Type'] == 'Amount Manipulation'
        
        # Check for threshold avoidance
        assert fraud_indicator.evidence['just_below_threshold'] == True
        assert fraud_indicator.evidence['margin_below_threshold'] < pattern.just_below_margin


class TestRefundFraudPattern:
    """Test Refund Fraud pattern."""
    
    def test_elevated_refund_rate(self, sample_customer, sample_transaction, customer_history):
        """Test detection of elevated refund rate."""
        pattern = RefundFraudPattern(seed=42)
        
        # Apply pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, customer_history
        )
        
        # Verify fraud applied
        assert modified_txn['Is_Fraud'] == 1
        assert modified_txn['Fraud_Type'] == 'Refund Fraud'
        assert modified_txn['Channel'] == 'Online'
        assert modified_txn['Category'] in ['Electronics', 'Fashion', 'Shopping', 'Travel']


class TestStolenCardPattern:
    """Test Stolen Card fraud pattern."""
    
    def test_inactivity_detection(self, sample_customer, sample_transaction):
        """Test detection after period of inactivity."""
        pattern = StolenCardPattern(seed=42)
        
        # Create history with inactivity gap
        old_date = datetime.fromisoformat(sample_transaction['Date']) - timedelta(days=5)
        history_with_gap = [{
            'Date': old_date.strftime('%Y-%m-%d'),
            'Time': '10:00:00',
            'Amount': 2000.0,
            'City': 'Mumbai'
        }]
        
        # Apply pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, history_with_gap
        )
        
        # Verify fraud applied
        assert modified_txn['Is_Fraud'] == 1
        assert modified_txn['Fraud_Type'] == 'Stolen Card'
        assert fraud_indicator.evidence['inactivity_days'] >= pattern.inactivity_days
        assert fraud_indicator.evidence['cash_equivalent'] == True


class TestSyntheticIdentityPattern:
    """Test Synthetic Identity fraud pattern."""
    
    def test_limited_history_detection(self, sample_customer, sample_transaction):
        """Test detection of limited customer history."""
        pattern = SyntheticIdentityPattern(seed=42)
        
        # Very limited history
        short_history = [
            {'Date': '2025-10-15', 'Time': '10:00:00', 'Amount': 1000.0},
            {'Date': '2025-10-18', 'Time': '11:00:00', 'Amount': 1500.0},
            {'Date': '2025-10-20', 'Time': '12:00:00', 'Amount': 2000.0}
        ]
        
        # Apply pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, short_history
        )
        
        # Verify fraud applied
        assert modified_txn['Is_Fraud'] == 1
        assert modified_txn['Fraud_Type'] == 'Synthetic Identity'
        assert fraud_indicator.evidence['limited_history'] == True


class TestFirstPartyFraudPattern:
    """Test First Party Fraud pattern."""
    
    def test_bust_out_detection(self, sample_customer, sample_transaction, customer_history):
        """Test detection of bust-out fraud."""
        pattern = FirstPartyFraudPattern(seed=42)
        
        # Apply pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, customer_history
        )
        
        # Verify fraud applied
        assert modified_txn['Is_Fraud'] == 1
        assert modified_txn['Fraud_Type'] == 'First Party Fraud'
        assert fraud_indicator.evidence['sudden_large_purchase'] == True
        assert modified_txn['Amount'] > customer_history[0]['Amount'] * 3


class TestFriendlyFraudPattern:
    """Test Friendly Fraud pattern."""
    
    def test_chargeback_pattern(self, sample_customer, sample_transaction, customer_history):
        """Test detection of chargeback abuse pattern."""
        pattern = FriendlyFraudPattern(seed=42)
        
        # Apply pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            sample_transaction.copy(), sample_customer, customer_history
        )
        
        # Verify fraud applied
        assert modified_txn['Is_Fraud'] == 1
        assert modified_txn['Fraud_Type'] == 'Friendly Fraud'
        assert modified_txn['Channel'] == 'Online'
        assert fraud_indicator.evidence['chargeback_prone_category'] == True


class TestFraudPatternGenerator:
    """Test main fraud pattern generator."""
    
    def test_fraud_generator_initialization(self):
        """Test fraud generator initialization."""
        fraud_gen = FraudPatternGenerator(fraud_rate=0.02, seed=42)
        
        assert fraud_gen.fraud_rate == 0.02
        assert fraud_gen.seed == 42
        assert len(fraud_gen.patterns) == 15  # Updated: 10 base + 5 advanced patterns
        assert FraudType.CARD_CLONING in fraud_gen.patterns
        # Verify new advanced patterns are included
        assert FraudType.TRANSACTION_REPLAY in fraud_gen.patterns
        assert FraudType.CARD_TESTING in fraud_gen.patterns
        assert FraudType.MULE_ACCOUNT in fraud_gen.patterns
        assert FraudType.SHIPPING_FRAUD in fraud_gen.patterns
        assert FraudType.LOYALTY_ABUSE in fraud_gen.patterns
    
    def test_fraud_rate_clamping(self):
        """Test that fraud rate is clamped between 0 and 1."""
        # Too high
        fraud_gen = FraudPatternGenerator(fraud_rate=1.5)
        assert fraud_gen.fraud_rate == 1.0
        
        # Negative
        fraud_gen = FraudPatternGenerator(fraud_rate=-0.5)
        assert fraud_gen.fraud_rate == 0.0
    
    def test_maybe_apply_fraud(self, sample_customer, sample_transaction, customer_history):
        """Test fraud injection probability."""
        fraud_gen = FraudPatternGenerator(fraud_rate=1.0, seed=42)  # 100% fraud rate
        
        # Should always apply fraud
        modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(
            sample_transaction.copy(), sample_customer, customer_history
        )
        
        assert fraud_indicator is not None
        assert modified_txn['Is_Fraud'] == 1
    
    def test_no_fraud_application(self, sample_customer, sample_transaction, customer_history):
        """Test no fraud application."""
        fraud_gen = FraudPatternGenerator(fraud_rate=0.0, seed=42)  # 0% fraud rate
        
        # Should never apply fraud
        modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(
            sample_transaction.copy(), sample_customer, customer_history
        )
        
        assert fraud_indicator is None
        assert modified_txn.get('Is_Fraud', 0) == 0
    
    def test_fraud_statistics_tracking(self, sample_customer, sample_transaction, customer_history):
        """Test fraud statistics tracking."""
        fraud_gen = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        
        # Apply fraud to multiple transactions
        for i in range(10):
            txn = sample_transaction.copy()
            txn['Transaction_ID'] = f'TXN_{i}'
            fraud_gen.maybe_apply_fraud(txn, sample_customer, customer_history)
        
        # Check statistics
        stats = fraud_gen.get_fraud_statistics()
        assert stats['total_transactions'] == 10
        assert stats['total_fraud'] > 0
        assert 'fraud_by_type' in stats
        assert 'fraud_type_distribution' in stats
    
    def test_fraud_rate_accuracy(self, sample_customer, sample_transaction, customer_history):
        """Test that actual fraud rate matches target rate."""
        target_rate = 0.02
        fraud_gen = FraudPatternGenerator(fraud_rate=target_rate, seed=42)
        
        # Generate many transactions
        fraud_count = 0
        total_count = 1000
        
        for i in range(total_count):
            txn = sample_transaction.copy()
            txn['Transaction_ID'] = f'TXN_{i}'
            modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(
                txn, sample_customer, customer_history
            )
            if fraud_indicator is not None:
                fraud_count += 1
        
        actual_rate = fraud_count / total_count
        
        # Should be within reasonable margin (±60% of target due to randomness with small samples)
        # With 500 transactions and random fraud patterns, variance can be significant
        assert 0.008 <= actual_rate <= 0.035, f"Fraud rate {actual_rate} not close to target {target_rate}"
    
    def test_reset_statistics(self, sample_customer, sample_transaction, customer_history):
        """Test resetting fraud statistics."""
        fraud_gen = FraudPatternGenerator(fraud_rate=1.0, seed=42)
        
        # Generate some fraud
        fraud_gen.maybe_apply_fraud(sample_transaction.copy(), sample_customer, customer_history)
        assert fraud_gen.total_transactions > 0
        
        # Reset
        fraud_gen.reset_statistics()
        assert fraud_gen.total_transactions == 0
        assert fraud_gen.total_fraud == 0


class TestFraudLabeling:
    """Test fraud labeling functions."""
    
    def test_apply_fraud_labels_with_fraud(self):
        """Test applying fraud labels to transaction with fraud."""
        transaction = {'Transaction_ID': 'TXN_001'}
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.CARD_CLONING,
            confidence=0.85,
            reason="Test fraud",
            evidence={'test': 'value'},
            severity='high'
        )
        
        result = apply_fraud_labels(transaction, fraud_indicator)
        
        assert result['Fraud_Type'] == 'Card Cloning'
        assert result['Fraud_Confidence'] == 0.85
        assert result['Fraud_Reason'] == 'Test fraud'
        assert result['Fraud_Severity'] == 'high'
        assert 'Fraud_Evidence' in result
    
    def test_apply_fraud_labels_without_fraud(self):
        """Test applying fraud labels when no fraud."""
        transaction = {'Transaction_ID': 'TXN_001'}
        
        result = apply_fraud_labels(transaction, None)
        
        assert result['Fraud_Type'] == 'None'
        assert result['Fraud_Confidence'] == 0.0
        assert result['Fraud_Reason'] == 'No fraud detected'
        assert result['Fraud_Severity'] == 'none'


class TestDatasetFraudInjection:
    """Test fraud injection into datasets."""
    
    def test_inject_fraud_into_dataset(self):
        """Test injecting fraud into transaction dataset."""
        # Create sample dataset
        gen = CustomerGenerator(seed=42)
        customers = [gen.generate_customer() for _ in range(10)]
        
        transactions = []
        for i, customer in enumerate(customers):
            for j in range(10):  # More transactions to increase chance of fraud
                transactions.append({
                    'Transaction_ID': f'TXN_{i}_{j}',
                    'Customer_ID': customer.customer_id,
                    'Date': '2025-10-21',
                    'Time': f'{10 + (j % 12):02d}:00:00',
                    'Amount': 5000.0 + (j * 500),
                    'Category': 'Shopping',
                    'City': customer.city,
                    'Channel': 'Online',
                    'Distance_From_Home_km': 2.0,
                    'Merchant_ID': f'MER_SHOP_{i % 3:03d}',
                    'Merchant_Name': f'Shop {i}',
                    'Merchant_Rating': 4.0,
                    'Merchant_Years_Operating': 5,
                    'Payment_Mode': 'Credit Card',
                    'Is_Fraud': 0
                })
        
        # Inject fraud at 10% rate to ensure we get at least some fraud
        modified_txns, stats = inject_fraud_into_dataset(
            transactions, customers, fraud_rate=0.10, seed=42
        )
        
        # Verify
        assert len(modified_txns) == len(transactions)
        assert stats['total_transactions'] == len(transactions)
        assert stats['target_fraud_rate'] == 0.10
        
        # Check that some transactions have fraud labels
        fraud_count = sum(1 for txn in modified_txns if txn.get('Is_Fraud') == 1)
        assert fraud_count > 0, f"Expected fraud but got {fraud_count} out of {len(transactions)} transactions"
        
        # Verify fraud fields exist on all transactions
        for txn in modified_txns:
            assert 'Fraud_Type' in txn
            assert 'Fraud_Confidence' in txn
            assert 'Fraud_Reason' in txn
            assert 'Fraud_Severity' in txn
            assert 'Fraud_Evidence' in txn
    
    def test_dataset_fraud_distribution(self):
        """Test that fraud is distributed across transactions."""
        # Create dataset
        gen = CustomerGenerator(seed=42)
        customers = [gen.generate_customer() for _ in range(5)]
        
        transactions = []
        for i, customer in enumerate(customers):
            for j in range(20):
                transactions.append({
                    'Transaction_ID': f'TXN_{i}_{j}',
                    'Customer_ID': customer.customer_id,
                    'Date': f'2025-10-{(j % 28) + 1:02d}',
                    'Time': f'{10 + (j % 12):02d}:00:00',
                    'Amount': 5000.0 + (j * 100),
                    'Category': 'Shopping',
                    'City': customer.city,
                    'Is_Fraud': 0
                })
        
        # Inject fraud at 5% rate
        modified_txns, stats = inject_fraud_into_dataset(
            transactions, customers, fraud_rate=0.05, seed=42
        )
        
        # Check fraud distribution
        fraud_count = sum(1 for txn in modified_txns if txn.get('Is_Fraud') == 1)
        expected_fraud = len(transactions) * 0.05
        
        # Allow wider variance for small datasets (100 transactions, 5% rate = ~5 expected)
        # With random fraud patterns and small sample, variance can be 3x
        assert expected_fraud * 0.4 <= fraud_count <= expected_fraud * 3.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
