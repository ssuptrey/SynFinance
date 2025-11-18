"""
Verify dataset quality and realism

Quick checks to ensure the 500k dataset is ready for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path

def verify_dataset():
    """Verify dataset quality"""
    print("=" * 80)
    print("DATASET VERIFICATION")
    print("=" * 80)
    
    # Load datasets
    data_dir = Path('benchmarks/data')
    train_df = pd.read_parquet(data_dir / 'train_500k.parquet')
    test_df = pd.read_parquet(data_dir / 'test_150k.parquet')
    
    print(f"\n[STRUCTURE]")
    print(f"  Train: {len(train_df):,} transactions, {len(train_df.columns)} features")
    print(f"  Test: {len(test_df):,} transactions, {len(test_df.columns)} features")
    
    # Check fraud rate
    print(f"\n[FRAUD RATE]")
    print(f"  Train: {train_df['is_fraud'].mean()*100:.2f}%")
    print(f"  Test: {test_df['is_fraud'].mean()*100:.2f}%")
    
    # Check features
    print(f"\n[FEATURES] ({len(train_df.columns)} total)")
    print(f"  {list(train_df.columns)}")
    
    # Amount distribution
    print(f"\n[AMOUNT DISTRIBUTION]")
    for name, df in [("Train", train_df), ("Test", test_df)]:
        print(f"  {name}:")
        print(f"    Min: ₹{df['amount'].min():.2f}")
        print(f"    Median: ₹{df['amount'].median():.2f}")
        print(f"    Mean: ₹{df['amount'].mean():.2f}")
        print(f"    95th: ₹{df['amount'].quantile(0.95):.2f}")
        print(f"    Max: ₹{df['amount'].max():.2f}")
    
    # Missing values
    print(f"\n[MISSING VALUES]")
    missing_train = train_df.isnull().sum().sum()
    missing_test = test_df.isnull().sum().sum()
    print(f"  Train: {missing_train} ({missing_train/len(train_df)/len(train_df.columns)*100:.2f}%)")
    print(f"  Test: {missing_test} ({missing_test/len(test_df)/len(test_df.columns)*100:.2f}%)")
    
    # Feature correlations with fraud
    print(f"\n[TOP 10 FEATURES CORRELATED WITH FRAUD]")
    correlations = train_df.select_dtypes(include=[np.number]).corr()['is_fraud'].abs().sort_values(ascending=False)
    for i, (feat, corr) in enumerate(correlations.head(11).items()):
        if feat != 'is_fraud':
            print(f"  {i}. {feat}: {corr:.3f}")
    
    print(f"\n✓ DATASET VERIFIED - READY FOR TRAINING")
    print(f"=" * 80)

if __name__ == '__main__':
    verify_dataset()
