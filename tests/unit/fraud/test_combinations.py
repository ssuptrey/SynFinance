import pytest
from src.generators.fraud_patterns import (
    FraudCombinationGenerator,
    TransactionReplayPattern,
    CardTestingPattern,
    AccountTakeoverPattern,
    VelocityAbusePattern,
    MuleAccountPattern,
    ShippingFraudPattern,
    FraudType,
)


class DummyCustomer:
    def __init__(self):
        self.customer_id = 'CUST-COMB-001'
        self.age = 40
        self.city = 'Mumbai'
        self.state = 'Maharashtra'
        self.segment = None


def create_transaction():
    return {
        'Transaction_ID': 'TXN-COMB-001',
        'Customer_ID': 'CUST-COMB-001',
        'Date': '2025-10-26',
        'Time': '12:00:00',
        'Hour': 12,
        'Amount': 1000,
        'Merchant_ID': 'MRCH-TEST-001',
        'City': 'Mumbai',
        'Category': 'Shopping',
        'Channel': 'Online',
        'Device_Type': 'Web',
        'Transaction_Status': 'Approved',
        'Is_Fraud': 0
    }


def create_history():
    # Recent history with a few transactions to give patterns something to analyze
    return [
        {'Date': '2025-10-25', 'Time': '11:00:00', 'Amount': 50, 'Merchant_ID': 'M1', 'Category': 'Entertainment', 'Channel': 'Online'},
        {'Date': '2025-10-25', 'Time': '11:05:00', 'Amount': 30, 'Merchant_ID': 'M2', 'Category': 'Entertainment', 'Channel': 'Online'},
        {'Date': '2025-10-25', 'Time': '11:07:00', 'Amount': 20, 'Merchant_ID': 'M3', 'Category': 'Entertainment', 'Channel': 'Online'},
        {'Date': '2025-10-25', 'Time': '13:00:00', 'Amount': 9000, 'Merchant_ID': 'M4', 'Category': 'Electronics', 'Channel': 'Online'},
        {'Date': '2025-10-24', 'Time': '10:00:00', 'Amount': 1000, 'City': 'Mumbai', 'Merchant_ID': 'M5', 'Category': 'Dining', 'Channel': 'POS'},
        {'Date': '2025-10-24', 'Time': '14:00:00', 'Amount': 800, 'City': 'Mumbai', 'Merchant_ID': 'M6', 'Category': 'Shopping', 'Channel': 'Online'},
        {'Date': '2025-10-23', 'Time': '09:00:00', 'Amount': 1200, 'City': 'Mumbai', 'Merchant_ID': 'M7', 'Category': 'Groceries', 'Channel': 'POS'},
        {'Date': '2025-10-23', 'Time': '16:00:00', 'Amount': 500, 'City': 'Mumbai', 'Merchant_ID': 'M8', 'Category': 'Shopping', 'Channel': 'Online'},
        {'Date': '2025-10-22', 'Time': '11:30:00', 'Amount': 1500, 'City': 'Mumbai', 'Merchant_ID': 'M9', 'Category': 'Fashion', 'Channel': 'Online'},
        {'Date': '2025-10-22', 'Time': '15:00:00', 'Amount': 700, 'City': 'Mumbai', 'Merchant_ID': 'M10', 'Category': 'Dining', 'Channel': 'POS'},
        {'Date': '2025-10-21', 'Time': '12:00:00', 'Amount': 900, 'City': 'Mumbai', 'Merchant_ID': 'M11', 'Category': 'Shopping', 'Channel': 'Online'},
    ]


def test_combine_two_patterns_confidence_and_type():
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    # Use CardTesting and TransactionReplay as example combination
    p1 = CardTestingPattern(seed=42)
    p2 = TransactionReplayPattern(seed=42)

    combiner = FraudCombinationGenerator(seed=42)

    modified_txn, combined_indicator = combiner.combine_and_apply(txn.copy(), [p1, p2], customer, history)

    assert combined_indicator is not None
    assert combined_indicator.fraud_type == FraudType.COMBINED
    assert 0.0 <= combined_indicator.confidence <= 1.0
    assert modified_txn['Is_Fraud'] == 1


def test_combined_confidence_at_least_max_individual():
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    p1 = AccountTakeoverPattern(seed=42)
    p2 = VelocityAbusePattern(seed=42)

    # Individual application
    txn1, ind1 = p1.apply_pattern(txn.copy(), customer, history)
    txn2, ind2 = p2.apply_pattern(txn.copy(), customer, history)

    combiner = FraudCombinationGenerator(seed=42)
    txn_comb, combined = combiner.combine_and_apply(txn.copy(), [p1, p2], customer, history)

    assert combined.confidence >= max(ind1.confidence, ind2.confidence)
    assert combined.fraud_type == FraudType.COMBINED
    assert combined.severity in ['none', 'low', 'medium', 'high', 'critical']


def test_chained_fraud_pattern():
    """Test chained fraud: account takeover followed by velocity abuse."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    # Simulate account takeover -> velocity abuse chain
    p1 = AccountTakeoverPattern(seed=42)
    p2 = VelocityAbusePattern(seed=42)

    combiner = FraudCombinationGenerator(seed=42)
    modified_txn, chained_indicator = combiner.apply_chained(txn.copy(), [p1, p2], customer, history)

    assert chained_indicator is not None
    assert 'combination_type' in chained_indicator.evidence
    assert chained_indicator.evidence['combination_type'] == 'chained'
    assert chained_indicator.evidence['chain_length'] == 2
    assert '[CHAINED FRAUD]' in chained_indicator.reason
    assert modified_txn['Is_Fraud'] == 1


def test_chained_fraud_confidence_boost():
    """Test that chained fraud has boosted confidence compared to regular combination."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    p1 = CardTestingPattern(seed=42)
    p2 = TransactionReplayPattern(seed=42)

    combiner = FraudCombinationGenerator(seed=42)
    
    # Regular combination
    _, regular_indicator = combiner.combine_and_apply(txn.copy(), [p1, p2], customer, history)
    
    # Chained combination
    _, chained_indicator = combiner.apply_chained(txn.copy(), [p1, p2], customer, history)

    # Chained should have higher confidence (1.1x boost)
    if regular_indicator.confidence > 0:
        assert chained_indicator.confidence >= regular_indicator.confidence


def test_coordinated_fraud_pattern():
    """Test coordinated fraud with multiple actors."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    # Simulate coordinated fraud ring
    p1 = MuleAccountPattern(seed=42)
    p2 = ShippingFraudPattern(seed=42)

    coordination_metadata = {
        'shared_merchants': ['MRCH-RING-001', 'MRCH-RING-002'],
        'shared_locations': ['Suspicious_Area_A'],
        'time_clustering': True,
        'fraud_ring_id': 'RING-001'
    }

    combiner = FraudCombinationGenerator(seed=42)
    modified_txn, coord_indicator = combiner.apply_coordinated(
        txn.copy(), [p1, p2], customer, history, coordination_metadata
    )

    assert coord_indicator is not None
    assert 'combination_type' in coord_indicator.evidence
    assert coord_indicator.evidence['combination_type'] == 'coordinated'
    assert coord_indicator.evidence['coordination_actors'] == 2
    assert 'shared_merchants' in coord_indicator.evidence
    assert '[COORDINATED FRAUD]' in coord_indicator.reason
    assert modified_txn['Is_Fraud'] == 1


def test_coordinated_fraud_severity_elevation():
    """Test that coordinated fraud has elevated severity."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    p1 = CardTestingPattern(seed=42)
    p2 = TransactionReplayPattern(seed=42)

    combiner = FraudCombinationGenerator(seed=42)
    
    # Regular combination
    _, regular_indicator = combiner.combine_and_apply(txn.copy(), [p1, p2], customer, history)
    
    # Coordinated combination
    _, coord_indicator = combiner.apply_coordinated(txn.copy(), [p1, p2], customer, history)

    severity_order = ['none', 'low', 'medium', 'high', 'critical']
    regular_idx = severity_order.index(regular_indicator.severity)
    coord_idx = severity_order.index(coord_indicator.severity)
    
    # Coordinated should have equal or higher severity
    assert coord_idx >= regular_idx


def test_progressive_fraud_early_stage():
    """Test progressive fraud at early stage (low sophistication)."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    # Progressive patterns ordered by increasing sophistication
    patterns = [
        CardTestingPattern(seed=42),
        TransactionReplayPattern(seed=42),
        AccountTakeoverPattern(seed=42),
        MuleAccountPattern(seed=42)
    ]

    combiner = FraudCombinationGenerator(seed=42)
    modified_txn, prog_indicator = combiner.apply_progressive(
        txn.copy(), patterns, customer, history, sophistication_level=0.3
    )

    assert prog_indicator is not None
    assert 'combination_type' in prog_indicator.evidence
    assert prog_indicator.evidence['combination_type'] == 'progressive'
    assert prog_indicator.evidence['sophistication_level'] == 0.3
    assert prog_indicator.evidence['progression_stage'] == 'early'
    assert prog_indicator.evidence['patterns_applied'] == 1  # 30% of 4 = 1
    assert '[PROGRESSIVE FRAUD]' in prog_indicator.reason


def test_progressive_fraud_advanced_stage():
    """Test progressive fraud at advanced stage (high sophistication)."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    patterns = [
        CardTestingPattern(seed=42),
        TransactionReplayPattern(seed=42),
        AccountTakeoverPattern(seed=42),
        MuleAccountPattern(seed=42)
    ]

    combiner = FraudCombinationGenerator(seed=42)
    modified_txn, prog_indicator = combiner.apply_progressive(
        txn.copy(), patterns, customer, history, sophistication_level=0.8
    )

    assert prog_indicator is not None
    assert prog_indicator.evidence['progression_stage'] == 'advanced'
    assert prog_indicator.evidence['patterns_applied'] == 3  # 80% of 4 = 3
    assert prog_indicator.evidence['max_patterns'] == 4


def test_progressive_fraud_confidence_scaling():
    """Test that progressive fraud confidence scales with sophistication level."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    patterns = [
        CardTestingPattern(seed=42),
        TransactionReplayPattern(seed=42),
    ]

    combiner = FraudCombinationGenerator(seed=42)
    
    # Low sophistication
    _, low_prog = combiner.apply_progressive(txn.copy(), patterns, customer, history, sophistication_level=0.3)
    
    # High sophistication
    _, high_prog = combiner.apply_progressive(txn.copy(), patterns, customer, history, sophistication_level=0.9)

    # Higher sophistication should result in higher confidence
    # (confidence is scaled by 0.7 + 0.3 * sophistication_level)
    if low_prog.confidence > 0 and high_prog.confidence > 0:
        assert high_prog.confidence >= low_prog.confidence


def test_empty_pattern_list_handling():
    """Test that combination generator handles empty pattern lists gracefully."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    combiner = FraudCombinationGenerator(seed=42)
    
    # Empty pattern list
    modified_txn, indicator = combiner.combine_and_apply(txn.copy(), [], customer, history)
    
    assert indicator.fraud_type == FraudType.COMBINED
    assert indicator.confidence == 0.0
    assert indicator.severity == 'none'


def test_single_pattern_combination():
    """Test that combining a single pattern works correctly."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    p1 = CardTestingPattern(seed=42)

    combiner = FraudCombinationGenerator(seed=42)
    modified_txn, combined_indicator = combiner.combine_and_apply(txn.copy(), [p1], customer, history)

    assert combined_indicator is not None
    assert combined_indicator.fraud_type == FraudType.COMBINED
    assert modified_txn['Is_Fraud'] == 1


def test_combination_evidence_merging():
    """Test that evidence from multiple patterns is properly merged."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    p1 = CardTestingPattern(seed=42)
    p2 = TransactionReplayPattern(seed=42)

    combiner = FraudCombinationGenerator(seed=42)
    modified_txn, combined_indicator = combiner.combine_and_apply(txn.copy(), [p1, p2], customer, history)

    # Evidence should contain keys from both patterns
    assert combined_indicator.evidence is not None
    assert len(combined_indicator.evidence) > 0


def test_combination_severity_max():
    """Test that combined severity is the maximum of individual severities."""
    customer = DummyCustomer()
    txn = create_transaction()
    history = create_history()

    p1 = CardTestingPattern(seed=42)  # typically low severity
    p2 = AccountTakeoverPattern(seed=42)  # typically higher severity

    # Get individual severities
    _, ind1 = p1.apply_pattern(txn.copy(), customer, history)
    _, ind2 = p2.apply_pattern(txn.copy(), customer, history)

    combiner = FraudCombinationGenerator(seed=42)
    _, combined = combiner.combine_and_apply(txn.copy(), [p1, p2], customer, history)

    severity_order = ['none', 'low', 'medium', 'high', 'critical']
    max_individual_idx = max(severity_order.index(ind1.severity), severity_order.index(ind2.severity))
    combined_idx = severity_order.index(combined.severity)

    # Combined severity should be at least as high as the max individual
    assert combined_idx >= max_individual_idx
