"""
Generate 10K transaction dataset for Week 3 Days 2-3 correlation analysis.

This script:
1. Generates 10,000 transactions with all 43 fields
2. Uses diverse customers across all segments
3. Saves to CSV for pandas analysis
4. Validates data quality

Output: output/week3_analysis_dataset.csv
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generators.transaction_core import TransactionGenerator
from customer_generator import CustomerGenerator

def main():
    """Generate 10K transaction dataset."""
    print("=" * 70)
    print("Week 3 Days 2-3: Generating 10K Transaction Dataset")
    print("=" * 70)
    
    # Initialize generators
    print("\n[1/5] Initializing generators...")
    customer_gen = CustomerGenerator(seed=42)
    txn_gen = TransactionGenerator(seed=42)
    
    # Generate customers (100 customers, 100 transactions each = 10K total)
    print("\n[2/5] Generating 100 diverse customers...")
    customers = customer_gen.generate_customers(100)
    print(f"  Created {len(customers)} customers across all segments")
    
    # Generate 10K transactions
    print("\n[3/5] Generating 10,000 transactions (100 per customer)...")
    print("  This may take 1-2 minutes...")
    df = txn_gen.generate_dataset(
        customers=customers,
        transactions_per_customer=100,
        days=90  # 3 months of data
    )
    
    print(f"  Total transactions generated: {len(df)}")
    
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Rows: {len(df)}")
    
    # Validate data quality
    print("\n[4/5] Validating data quality...")
    print(f"  Total columns: {len(df.columns)}")
    
    # Count missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"  Missing values detected in {len(missing_cols)} columns:")
        for col in missing_cols.index[:10]:  # Show first 10
            print(f"    {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")
    else:
        print(f"  No missing values detected!")
    
    # Show field categories
    print(f"\n  Field categories:")
    print(f"    Transaction fields: {len([c for c in df.columns if 'Transaction' in c or 'Amount' in c or 'Date' in c])}")
    print(f"    Customer fields: {len([c for c in df.columns if 'Customer' in c])}")
    print(f"    Merchant fields: {len([c for c in df.columns if 'Merchant' in c])}")
    print(f"    Location fields: {len([c for c in df.columns if 'City' in c or 'State' in c or 'Region' in c])}")
    print(f"    Device fields: {len([c for c in df.columns if 'Device' in c or 'OS' in c or 'App' in c or 'Browser' in c])}")
    print(f"    Risk fields: {len([c for c in df.columns if 'Distance' in c or 'Risk' in c or 'Daily' in c or 'Time_Since' in c or 'Is_First' in c])}")
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / "output" / "week3_analysis_dataset.csv"
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"\nSaving to: {output_path}")
    df.to_csv(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    
    print(f"\nTransaction Statistics:")
    print(f"  Total transactions: {len(df):,}")
    print(f"  Unique customers: {df['Customer_ID'].nunique():,}")
    print(f"  Unique merchants: {df['Merchant_ID'].nunique():,}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    print(f"\nAmount Statistics:")
    print(f"  Min: ₹{df['Amount'].min():,.2f}")
    print(f"  Mean: ₹{df['Amount'].mean():,.2f}")
    print(f"  Median: ₹{df['Amount'].median():,.2f}")
    print(f"  Max: ₹{df['Amount'].max():,.2f}")
    
    print(f"\nPayment Mode Distribution:")
    for mode, count in df['Payment_Mode'].value_counts().head(5).items():
        pct = count / len(df) * 100
        print(f"  {mode}: {count:,} ({pct:.1f}%)")
    
    print(f"\nTransaction Status Distribution:")
    for status, count in df['Transaction_Status'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")
    
    print(f"\nTransaction Channel Distribution:")
    for channel, count in df['Transaction_Channel'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {channel}: {count:,} ({pct:.1f}%)")
    
    print(f"\nRegion Distribution:")
    for region, count in df['Region'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {region}: {count:,} ({pct:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print(f"Next steps: Run correlation analysis on {output_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
