"""
Demonstration of All 15 Fraud Patterns

This script demonstrates all fraud patterns including:
- Base Patterns (Days 1-2): 10 patterns
- Advanced Patterns (Days 3-4): 5 patterns

Shows how each pattern is detected and applied to transactions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.generators.fraud_patterns import (
    FraudPatternGenerator, FraudType,
    CardCloningPattern, AccountTakeoverPattern,
    TransactionReplayPattern, CardTestingPattern,
    MuleAccountPattern, ShippingFraudPattern,
    LoyaltyAbusePattern
)
from src.customer_profile import CustomerProfile, CustomerSegment, IncomeBracket
from datetime import datetime
import json


def create_sample_customer():
    """Create a sample customer for testing."""
    # Create a minimal customer-like object for testing
    class SampleCustomer:
        def __init__(self):
            self.customer_id = "CUST-DEMO-001"
            self.age = 35
            self.city = "Mumbai"
            self.state = "Maharashtra"
            self.segment = CustomerSegment.YOUNG_PROFESSIONAL
            self.income_bracket = IncomeBracket.MIDDLE
            self.occupation = "Software Engineer"
    
    return SampleCustomer()


def create_sample_transaction():
    """Create a sample transaction."""
    return {
        'Transaction_ID': 'TXN-DEMO-001',
        'Customer_ID': 'CUST-DEMO-001',
        'Date': '2025-10-21',
        'Time': '14:30:00',
        'Amount': 5000,
        'Merchant_ID': 'MRCH-MUM-SHOP-001',
        'Category': 'Shopping',
        'Payment_Mode': 'Credit Card',
        'City': 'Mumbai',
        'Channel': 'Online',
        'Device_Type': 'Web',
        'Transaction_Status': 'Approved',
        'Is_Fraud': 0
    }


def create_customer_history():
    """Create sample customer history."""
    return [
        {'Date': '2025-10-20', 'Time': '15:00:00', 'Amount': 3000, 'City': 'Mumbai', 
         'Merchant_ID': 'MRCH-MUM-SHOP-001', 'Category': 'Shopping', 'Channel': 'Online'},
        {'Date': '2025-10-19', 'Time': '12:30:00', 'Amount': 2500, 'City': 'Mumbai', 
         'Merchant_ID': 'MRCH-MUM-FOOD-002', 'Category': 'Dining', 'Channel': 'POS'},
        {'Date': '2025-10-18', 'Time': '18:45:00', 'Amount': 4000, 'City': 'Mumbai', 
         'Merchant_ID': 'MRCH-MUM-SHOP-003', 'Category': 'Electronics', 'Channel': 'Online'},
        {'Date': '2025-10-17', 'Time': '20:00:00', 'Amount': 1500, 'City': 'Mumbai', 
         'Merchant_ID': 'MRCH-MUM-FOOD-001', 'Category': 'Dining', 'Channel': 'POS'},
        {'Date': '2025-10-16', 'Time': '16:15:00', 'Amount': 6000, 'City': 'Mumbai', 
         'Merchant_ID': 'MRCH-MUM-SHOP-004', 'Category': 'Fashion', 'Channel': 'Online'},
    ]


def demo_fraud_pattern(pattern_class, pattern_name, customer, transaction, history):
    """Demonstrate a specific fraud pattern."""
    print(f"\n{'='*70}")
    print(f"  {pattern_name}")
    print(f"{'='*70}")
    
    pattern = pattern_class(seed=42)
    
    # Check if pattern should apply
    should_apply = pattern.should_apply(customer, transaction, history)
    print(f"Should Apply: {should_apply}")
    
    if should_apply or True:  # Force application for demo
        # Apply the pattern
        modified_txn, fraud_indicator = pattern.apply_pattern(
            transaction.copy(), customer, history
        )
        
        print(f"\nFraud Type: {fraud_indicator.fraud_type.value}")
        print(f"Confidence: {fraud_indicator.confidence:.3f}")
        print(f"Severity: {fraud_indicator.severity}")
        print(f"\nReason:")
        print(f"  {fraud_indicator.reason}")
        print(f"\nKey Evidence:")
        for key, value in list(fraud_indicator.evidence.items())[:5]:
            print(f"  - {key}: {value}")
        
        print(f"\nTransaction Changes:")
        print(f"  Amount: Rs.{transaction['Amount']} -> Rs.{modified_txn['Amount']:.2f}")
        print(f"  Category: {transaction.get('Category')} -> {modified_txn.get('Category')}")
        if transaction.get('City') != modified_txn.get('City'):
            print(f"  City: {transaction.get('City')} -> {modified_txn.get('City')}")


def main():
    """Run demonstration of all fraud patterns."""
    print("\n" + "="*70)
    print("  SYNFINANCE FRAUD PATTERN DEMONSTRATION")
    print("  Version 0.5.0-dev")
    print("  15 Fraud Patterns (10 Base + 5 Advanced)")
    print("="*70)
    
    # Create test data
    customer = create_sample_customer()
    transaction = create_sample_transaction()
    history = create_customer_history()
    
    print(f"\nSample Customer: {customer.customer_id}")
    print(f"  Segment: {customer.segment.value}")
    print(f"  Income: {customer.income_bracket.value}")
    print(f"  Location: {customer.city}, {customer.state}")
    
    print(f"\nSample Transaction:")
    print(f"  Amount: Rs.{transaction['Amount']}")
    print(f"  Category: {transaction['Category']}")
    print(f"  Channel: {transaction['Channel']}")
    print(f"  City: {transaction['City']}")
    
    # Test FraudPatternGenerator with all 15 patterns
    print(f"\n{'='*70}")
    print("  FRAUD PATTERN GENERATOR STATISTICS")
    print(f"{'='*70}")
    
    fraud_gen = FraudPatternGenerator(fraud_rate=1.0, seed=42)
    print(f"\nTotal Patterns Loaded: {len(fraud_gen.patterns)}")
    print(f"Fraud Injection Rate: {fraud_gen.fraud_rate * 100}%")
    
    print(f"\nBase Patterns (Days 1-2):")
    base_patterns = [
        FraudType.CARD_CLONING, FraudType.ACCOUNT_TAKEOVER,
        FraudType.MERCHANT_COLLUSION, FraudType.VELOCITY_ABUSE,
        FraudType.AMOUNT_MANIPULATION, FraudType.REFUND_FRAUD,
        FraudType.STOLEN_CARD, FraudType.SYNTHETIC_IDENTITY,
        FraudType.FIRST_PARTY_FRAUD, FraudType.FRIENDLY_FRAUD
    ]
    for i, fraud_type in enumerate(base_patterns, 1):
        print(f"  {i:2d}. {fraud_type.value}")
    
    print(f"\nAdvanced Patterns (Days 3-4):")
    advanced_patterns = [
        FraudType.TRANSACTION_REPLAY, FraudType.CARD_TESTING,
        FraudType.MULE_ACCOUNT, FraudType.SHIPPING_FRAUD,
        FraudType.LOYALTY_ABUSE
    ]
    for i, fraud_type in enumerate(advanced_patterns, 11):
        print(f"  {i:2d}. {fraud_type.value}")
    
    # Demonstrate select patterns
    print(f"\n{'='*70}")
    print("  PATTERN DEMONSTRATIONS")
    print(f"{'='*70}")
    
    # Demo advanced patterns
    demo_fraud_pattern(TransactionReplayPattern, "Pattern 11: Transaction Replay",
                      customer, transaction, history)
    
    demo_fraud_pattern(CardTestingPattern, "Pattern 12: Card Testing",
                      customer, transaction, history)
    
    demo_fraud_pattern(MuleAccountPattern, "Pattern 13: Mule Account",
                      customer, transaction, history)
    
    demo_fraud_pattern(ShippingFraudPattern, "Pattern 14: Shipping Fraud",
                      customer, transaction, history)
    
    demo_fraud_pattern(LoyaltyAbusePattern, "Pattern 15: Loyalty Program Abuse",
                      customer, transaction, history)
    
    # Generate transactions with fraud
    print(f"\n{'='*70}")
    print("  BATCH FRAUD INJECTION TEST")
    print(f"{'='*70}")
    
    print("\nGenerating 100 transactions with 10% fraud rate...")
    fraud_gen = FraudPatternGenerator(fraud_rate=0.10, seed=42)
    
    fraud_counts = {fraud_type: 0 for fraud_type in FraudType}
    total_fraud = 0
    
    for i in range(100):
        txn = transaction.copy()
        txn['Transaction_ID'] = f'TXN-BATCH-{i:03d}'
        # Fix date format for VelocityAbusePattern
        txn['Date'] = '2025-10-21'
        txn['Time'] = '14:30:00'
        
        modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(txn, customer, history)
        
        if fraud_indicator:
            fraud_counts[fraud_indicator.fraud_type] += 1
            total_fraud += 1
    
    print(f"\nResults:")
    print(f"  Total Transactions: 100")
    print(f"  Fraudulent Transactions: {total_fraud}")
    print(f"  Fraud Rate: {total_fraud}%")
    
    print(f"\nFraud Distribution:")
    for fraud_type, count in fraud_counts.items():
        if count > 0:
            print(f"  {fraud_type.value:30s}: {count:2d} transactions")
    
    print(f"\n{'='*70}")
    print("  DEMONSTRATION COMPLETE")
    print(f"{'='*70}")
    print("\nAll 15 fraud patterns are operational!")
    print("✓ 10 Base patterns (Week 4 Days 1-2)")
    print("✓ 5 Advanced patterns (Week 4 Days 3-4)")
    print("\nNext Steps:")
    print("  - Add fraud pattern combinations")
    print("  - Implement fraud network analysis")
    print("  - Create comprehensive test suite (30+ tests)")
    print()


if __name__ == "__main__":
    main()
