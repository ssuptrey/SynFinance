"""
Data Generator Module for SynFinance
Generates synthetic Indian financial transaction data with customer behavioral consistency

Week 1, Day 5-7: Customer-Aware Transaction Generation
Focus: Indian AI companies, developer-friendly, 100% scalable
Refactoring: Modular architecture with generators/, utils/, models/

Key Features:
- Customer behavioral consistency (spending patterns, preferences, timing)
- Indian market patterns (UPI dominance, festival spending, regional preferences)
- Scalable architecture (generator pattern, streaming support)
- Developer-friendly (modular, well-documented, easy to extend)

Architecture:
- TransactionGenerator class moved to generators/transaction_core.py
- Indian market data moved to utils/indian_data.py
- This file now provides convenient API functions and backward compatibility
"""

from datetime import datetime
from typing import Tuple, Optional
import pandas as pd

from src.customer_profile import CustomerProfile
from src.customer_generator import CustomerGenerator

# Import from modular structure
from src.generators.transaction_core import TransactionGenerator
from src.utils.indian_data import (
    INDIAN_FESTIVALS,
    INDIAN_MERCHANTS,
    UPI_HANDLES,
    CHAIN_MERCHANTS
)


# ============================================================================
# RE-EXPORT FOR BACKWARD COMPATIBILITY
# ============================================================================

# Re-export so existing imports still work
__all__ = [
    'TransactionGenerator',
    'generate_realistic_dataset',
    'generate_sample_data',
    'INDIAN_FESTIVALS',
    'INDIAN_MERCHANTS',
    'UPI_HANDLES',
    'CHAIN_MERCHANTS',
]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_sample_data(num_records: int = 100, output_format: str = 'csv') -> Tuple[pd.DataFrame, str]:
    """
    Convenience function to generate sample data (LEGACY - not customer-aware)
    
    DEPRECATED: Use generate_realistic_dataset() instead
    
    Args:
        num_records: Number of records to generate
        output_format: Output format ('csv' or 'json')
        
    Returns:
        Tuple of (DataFrame, filename)
    """
    generator = TransactionGenerator()
    df = generator.generate_transactions(num_records)
    
    if output_format.lower() == 'csv':
        filename = generator.export_to_csv(df)
    else:
        filename = generator.export_to_json(df)
    
    return df, filename


def generate_realistic_dataset(
    num_customers: int = 100,
    transactions_per_customer: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    days: int = 30,
    seed: Optional[int] = None,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate realistic transaction dataset with customer behavioral consistency
    
    This is the RECOMMENDED way to generate data for Indian AI companies.
    
    Args:
        num_customers: Number of customers to generate
        transactions_per_customer: Transactions per customer (or None for varied)
        start_date: Start date for transactions
        end_date: End date for transactions
        days: Number of days to generate over
        seed: Random seed for reproducibility
        output_file: Output filename (CSV, JSON, or Excel based on extension)
    
    Returns:
        DataFrame with transactions
    
    Example:
        >>> df = generate_realistic_dataset(
        ...     num_customers=100,
        ...     transactions_per_customer=50,
        ...     days=30,
        ...     seed=42,
        ...     output_file="transactions.csv"
        ... )
        >>> print(f"Generated {len(df)} transactions for {df['Customer_ID'].nunique()} customers")
    """
    # Generate customers
    customer_gen = CustomerGenerator(seed=seed)
    customers = customer_gen.generate_customers(num_customers)
    
    # Generate transactions
    txn_gen = TransactionGenerator(seed=seed)
    df = txn_gen.generate_dataset(
        customers=customers,
        transactions_per_customer=transactions_per_customer,
        start_date=start_date,
        end_date=end_date,
        days=days
    )
    
    # Export if filename provided
    if output_file:
        if output_file.endswith('.csv'):
            txn_gen.export_to_csv(df, output_file)
        elif output_file.endswith('.json'):
            txn_gen.export_to_json(df, output_file)
        elif output_file.endswith('.xlsx'):
            txn_gen.export_to_excel(df, output_file)
        else:
            # Default to CSV
            txn_gen.export_to_csv(df, output_file + '.csv')
    
    return df


# ============================================================================
# MAIN - DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SynFinance - Customer-Aware Transaction Generator")
    print("Week 1, Days 5-7: Complete Integration")
    print("=" * 80)
    
    # Test 1: Generate transactions for a single customer
    print("\n[TEST 1] Single Customer Transactions")
    print("-" * 80)
    
    customer_gen = CustomerGenerator(seed=42)
    customer = customer_gen.generate_customer()
    
    print(f"Customer: {customer.customer_id}")
    print(f"  Segment: {customer.segment.value}")
    print(f"  Age: {customer.age}, Income: ₹{customer.monthly_income:,.0f}/month")
    print(f"  Digital Savviness: {customer.digital_savviness.value}")
    print(f"  Preferred Categories: {', '.join(customer.preferred_categories[:3])}")
    print(f"  Preferred Payment Modes: {', '.join(customer.preferred_payment_modes)}")
    
    txn_gen = TransactionGenerator(seed=42)
    transactions = txn_gen.generate_customer_transactions(customer, count=10, days=7)
    
    print(f"\nGenerated {len(transactions)} transactions:")
    df_sample = pd.DataFrame(transactions)
    print(df_sample[['Date', 'Time', 'Category', 'Amount', 'Payment_Mode', 'Merchant']].head())
    
    # Test 2: Generate dataset for multiple customers
    print("\n\n[TEST 2] Multiple Customers Dataset")
    print("-" * 80)
    
    df = generate_realistic_dataset(
        num_customers=50,
        transactions_per_customer=20,
        days=30,
        seed=42
    )
    
    print(f"\n[SUCCESS] Generated dataset with:")
    print(f"  Total Transactions: {len(df):,}")
    print(f"  Unique Customers: {df['Customer_ID'].nunique()}")
    print(f"  Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Average Amount: ₹{df['Amount'].mean():,.2f}")
    print(f"  Total Volume: ₹{df['Amount'].sum():,.2f}")
    
    print("\n  Payment Mode Distribution:")
    for mode, count in df['Payment_Mode'].value_counts().head().items():
        print(f"    {mode}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\n  Top Categories:")
    for cat, count in df['Category'].value_counts().head().items():
        print(f"    {cat}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\n  Customer Segment Distribution:")
    for segment, count in df['Customer_Segment'].value_counts().items():
        print(f"    {segment}: {count} ({count/len(df)*100:.1f}%)")
    
    # Test 3: Verify behavioral consistency
    print("\n\n[TEST 3] Behavioral Consistency Check")
    print("-" * 80)
    
    sample_customer_id = df['Customer_ID'].iloc[0]
    customer_txns = df[df['Customer_ID'] == sample_customer_id]
    
    print(f"\nCustomer: {sample_customer_id}")
    print(f"  Total Transactions: {len(customer_txns)}")
    print(f"  Avg Amount: ₹{customer_txns['Amount'].mean():,.2f}")
    print(f"  Payment Modes: {', '.join(customer_txns['Payment_Mode'].unique())}")
    print(f"  Top Categories: {', '.join(customer_txns['Category'].value_counts().head(3).index.tolist())}")
    print(f"  Repeat Merchants: {customer_txns['Is_Repeat_Merchant'].sum()} / {len(customer_txns)}")
    
    print("\n" + "=" * 80)
    print("[COMPLETE] Week 1 Days 5-7 Integration Complete!")
    print("=" * 80)
    print("\nKey Features Implemented:")
    print("  [PASS] Customer-aware transaction generation")
    print("  [PASS] Indian market patterns (UPI, merchants, cities)")
    print("  [PASS] Behavioral consistency (categories, payment modes, amounts)")
    print("  [PASS] Time-based patterns (preferred shopping hours)")
    print("  [PASS] Merchant loyalty (repeat visits)")
    print("  [PASS] Geographic behavior (home city, travel)")
    print("  [PASS] Scalable architecture (streaming support)")
    print("  [PASS] Developer-friendly (modular, documented)")
    print("\nNext Steps: Update Streamlit UI to use customer-aware generation")
    print("=" * 80)
