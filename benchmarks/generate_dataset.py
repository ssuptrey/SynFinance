"""
Generate 500k Realistic UPI Transactions for Benchmark Validation

Creates a realistic fraud detection dataset with:
- 500,000 transactions (350k train, 150k test)
- 5% fraud rate (25,000 fraudulent transactions)
- 30+ features (transaction, temporal, geographic, behavioral, network)
- Realistic distributions matching Indian UPI patterns

Week 11 Day 2: Benchmark Validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import existing SynFinance generators
from src.customer_generator import CustomerGenerator
from src.generators.merchant_generator import MerchantGenerator
from src.generators.transaction_core import TransactionGenerator
from src.generators.geographic_generator import GeographicPatternGenerator
from src.generators.temporal_generator import TemporalPatternGenerator
from src.generators.fraud_patterns import FraudPatternGenerator
from src.generators.ml_features import MLFeatureEngineer


class BenchmarkDatasetGenerator:
    """
    Generate realistic 500k UPI transaction dataset for model validation
    """
    
    def __init__(self, n_transactions=500_000, fraud_rate=0.05, random_state=42):
        """
        Initialize dataset generator
        
        Args:
            n_transactions: Total transactions to generate
            fraud_rate: Percentage of fraudulent transactions (0-1)
            random_state: Random seed for reproducibility
        """
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Calculate splits
        self.n_fraud = int(n_transactions * fraud_rate)
        self.n_legitimate = n_transactions - self.n_fraud
        
        print(f"Dataset Configuration:")
        print(f"  Total transactions: {n_transactions:,}")
        print(f"  Legitimate: {self.n_legitimate:,} ({(1-fraud_rate)*100:.1f}%)")
        print(f"  Fraudulent: {self.n_fraud:,} ({fraud_rate*100:.1f}%)")
        
        # Initialize generators
        self.customer_gen = CustomerGenerator(seed=random_state)
        self.merchant_gen = MerchantGenerator(seed=random_state)
        # Note: Will generate customers/merchants in generate_base_transactions()
        self.n_customers = 50_000
        self.n_merchants = 10_000
    
    def generate_base_transactions(self):
        """Generate base transaction data (simplified for now)"""
        print("\n[1/5] Generating base transactions...")
        
        # Generate realistic transaction data
        transactions = []
        
        for i in range(self.n_transactions):
            # Basic transaction structure
            is_fraud = 1 if i < self.n_fraud else 0
            
            # Amount distribution (realistic Indian UPI)
            if is_fraud:
                # Fraud transactions tend to be higher
                amount = np.random.lognormal(mean=7.5, sigma=1.2)  # Higher amounts
            else:
                # Legitimate UPI transactions
                amount = np.random.lognormal(mean=6.2, sigma=1.0)  # Median ~₹500
            
            amount = max(10, min(amount, 50000))  # Clip to reasonable range
            
            txn = {
                'transaction_id': f'TXN{i+1:08d}',
                'customer_id': f'CUST{np.random.randint(1, self.n_customers+1):07d}',
                'merchant_id': f'MERCH{np.random.randint(1, self.n_merchants+1):06d}',
                'amount': round(amount, 2),
                'is_fraud': is_fraud
            }
            
            transactions.append(txn)
            
            if (i+1) % 100_000 == 0:
                print(f"  Generated {i+1:,} transactions...")
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        print(f"  ✓ Generated {len(df):,} total transactions")
        print(f"    Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
        print(f"    Legitimate: {(~df['is_fraud'].astype(bool)).sum():,}")
        
        return df, None, None
    
    def add_temporal_features(self, df):
        """Add temporal features"""
        print("\n[2/5] Adding temporal features...")
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            # Generate timestamps over 30 days
            start_date = datetime(2024, 10, 1)
            df['timestamp'] = [
                start_date + timedelta(
                    days=np.random.randint(0, 30),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                for _ in range(len(df))
            ]
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
        
        # Account age (days since account creation)
        df['days_since_account_creation'] = np.random.randint(1, 365, len(df))
        
        # Transaction recency
        df['days_since_last_transaction'] = np.random.exponential(3, len(df)).astype(int).clip(0, 30)
        
        # Transaction counts (velocity)
        df['transactions_today'] = np.random.poisson(2, len(df))
        df['transactions_this_week'] = np.random.poisson(10, len(df))
        
        # Rolling statistics (simulated)
        df['avg_amount_last_30d'] = df['amount'] * np.random.uniform(0.8, 1.2, len(df))
        df['std_amount_last_30d'] = df['avg_amount_last_30d'] * np.random.uniform(0.1, 0.3, len(df))
        
        print(f"  ✓ Added 10 temporal features")
        
        return df
    
    def add_geographic_features(self, df):
        """Add geographic features"""
        print("\n[3/5] Adding geographic features...")
        
        # Indian cities (realistic distribution)
        indian_cities = [
            ('Mumbai', 0.20), ('Delhi', 0.15), ('Bangalore', 0.12),
            ('Hyderabad', 0.10), ('Chennai', 0.08), ('Kolkata', 0.07),
            ('Pune', 0.06), ('Ahmedabad', 0.05), ('Jaipur', 0.04),
            ('Lucknow', 0.03), ('Other', 0.10)
        ]
        
        cities, weights = zip(*indian_cities)
        df['city'] = np.random.choice(cities, size=len(df), p=weights)
        
        # IP country (mostly India)
        df['ip_country'] = np.random.choice(
            ['IN', 'US', 'UK', 'SG', 'AE'],
            size=len(df),
            p=[0.95, 0.02, 0.01, 0.01, 0.01]
        )
        
        # Distance from home (km)
        df['distance_from_home'] = np.random.exponential(20, len(df)).clip(0, 1000)
        
        # Distance from last transaction (geographic velocity)
        df['distance_from_last_txn'] = np.random.exponential(10, len(df)).clip(0, 500)
        
        # New location flag
        df['new_location'] = (df['distance_from_home'] > 100).astype(int)
        
        print(f"  ✓ Added 5 geographic features")
        
        return df
    
    def add_behavioral_features(self, df):
        """Add behavioral features"""
        print("\n[4/5] Adding behavioral features...")
        
        # Amount deviation (Z-score from personal average)
        df['amount_deviation'] = (
            (df['amount'] - df['avg_amount_last_30d']) / 
            (df['std_amount_last_30d'] + 1e-6)
        )
        
        # New merchant flag (higher for fraud but not perfect)
        if 'is_fraud' in df.columns:
            fraud_mask = df['is_fraud'] == 1
            df['new_merchant'] = 0
            # 60% of fraud uses new merchant (not 100%)
            df.loc[fraud_mask, 'new_merchant'] = np.random.choice([0, 1], fraud_mask.sum(), p=[0.4, 0.6])
            # 25% of legitimate also uses new merchant
            df.loc[~fraud_mask, 'new_merchant'] = np.random.choice([0, 1], (~fraud_mask).sum(), p=[0.75, 0.25])
        else:
            df['new_merchant'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        
        # New device flag (higher for fraud but not perfect)
        if 'is_fraud' in df.columns:
            fraud_mask = df['is_fraud'] == 1
            df['new_device'] = 0
            # 40% of fraud uses new device
            df.loc[fraud_mask, 'new_device'] = np.random.choice([0, 1], fraud_mask.sum(), p=[0.6, 0.4])
            # 10% of legitimate also uses new device
            df.loc[~fraud_mask, 'new_device'] = np.random.choice([0, 1], (~fraud_mask).sum(), p=[0.9, 0.1])
        else:
            df['new_device'] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
        
        # Failed PIN attempts (slightly higher for fraud)
        if 'is_fraud' in df.columns:
            fraud_mask = df['is_fraud'] == 1
            df['failed_pin_attempts'] = 0
            # Fraud: higher PIN failures
            df.loc[fraud_mask, 'failed_pin_attempts'] = np.random.choice([0, 1, 2, 3], fraud_mask.sum(), p=[0.7, 0.15, 0.10, 0.05])
            # Legitimate: lower PIN failures
            df.loc[~fraud_mask, 'failed_pin_attempts'] = np.random.choice([0, 1, 2, 3], (~fraud_mask).sum(), p=[0.90, 0.08, 0.015, 0.005])
        else:
            df['failed_pin_attempts'] = np.random.choice([0, 1, 2, 3], size=len(df), p=[0.85, 0.10, 0.03, 0.02])
        
        print(f"  ✓ Added 4 behavioral features (realistic, not perfect predictors)")
        
        return df
    
    def add_network_features(self, df):
        """Add network/graph features (realistic, not perfect)"""
        print("\n[5/5] Adding network features...")
        
        # Connected to known fraudster (graph analysis)
        # Only slight correlation with fraud (many false positives in real life)
        if 'is_fraud' in df.columns:
            fraud_mask = df['is_fraud'] == 1
            df['connected_to_fraudster'] = 0
            # 15% of fraud connected to known fraudsters
            df.loc[fraud_mask, 'connected_to_fraudster'] = np.random.choice([0, 1], fraud_mask.sum(), p=[0.85, 0.15])
            # 3% of legitimate also connected (false positives)
            df.loc[~fraud_mask, 'connected_to_fraudster'] = np.random.choice([0, 1], (~fraud_mask).sum(), p=[0.97, 0.03])
        else:
            df['connected_to_fraudster'] = np.random.choice([0, 1], size=len(df), p=[0.98, 0.02])
        
        print(f"  ✓ Added 1 network feature (realistic correlation)")
        
        return df
    
    def add_upi_specific_features(self, df):
        """Add UPI-specific features (realistic correlations)"""
        print("\n[UPI] Adding UPI-specific features...")
        
        # UPI payment mode
        df['payment_mode'] = np.random.choice(
            ['QR', 'P2P', 'P2M', 'Intent'],
            size=len(df),
            p=[0.35, 0.30, 0.25, 0.10]
        )
        
        # UPI VPA (Virtual Payment Address)
        df['has_vpa'] = (df['payment_mode'].isin(['P2P', 'P2M'])).astype(int)
        
        # Device fingerprint change (realistic - not perfect indicator)
        if 'is_fraud' in df.columns:
            fraud_mask = df['is_fraud'] == 1
            df['device_fingerprint_change'] = 0
            # 30% of fraud has device change
            df.loc[fraud_mask, 'device_fingerprint_change'] = np.random.choice([0, 1], fraud_mask.sum(), p=[0.7, 0.3])
            # 5% of legitimate also has device change
            df.loc[~fraud_mask, 'device_fingerprint_change'] = np.random.choice([0, 1], (~fraud_mask).sum(), p=[0.95, 0.05])
        else:
            df['device_fingerprint_change'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
        
        # SIM serial change (SIM swap detection - rare but suspicious)
        if 'is_fraud' in df.columns:
            fraud_mask = df['is_fraud'] == 1
            df['sim_serial_change'] = 0
            # 20% of fraud involves SIM swap
            df.loc[fraud_mask, 'sim_serial_change'] = np.random.choice([0, 1], fraud_mask.sum(), p=[0.8, 0.2])
            # 0.5% of legitimate (people change phones)
            df.loc[~fraud_mask, 'sim_serial_change'] = np.random.choice([0, 1], (~fraud_mask).sum(), p=[0.995, 0.005])
        else:
            df['sim_serial_change'] = np.random.choice([0, 1], size=len(df), p=[0.99, 0.01])
        
        # UPI app version
        df['app_version'] = np.random.choice(['v10.2', 'v10.3', 'v11.0'], size=len(df), p=[0.2, 0.3, 0.5])
        
        print(f"  ✓ Added 5 UPI-specific features (realistic correlations)")
        
        return df
    
    def validate_realism(self, df):
        """Validate dataset realism"""
        print("\n[VALIDATION] Checking dataset realism...")
        
        stats = {}
        
        # Amount distribution
        stats['amount_median'] = df['amount'].median()
        stats['amount_95th'] = df['amount'].quantile(0.95)
        stats['amount_mean'] = df['amount'].mean()
        
        print(f"  Amount distribution:")
        print(f"    Median: ₹{stats['amount_median']:.2f}")
        print(f"    95th percentile: ₹{stats['amount_95th']:.2f}")
        print(f"    Mean: ₹{stats['amount_mean']:.2f}")
        
        # Temporal patterns
        peak_hours = df.groupby('hour').size().nlargest(3).index.tolist()
        stats['peak_hours'] = peak_hours
        print(f"  Peak hours: {peak_hours}")
        
        # Fraud rate
        if 'is_fraud' in df.columns:
            fraud_rate = df['is_fraud'].mean()
            stats['fraud_rate'] = fraud_rate
            print(f"  Fraud rate: {fraud_rate*100:.2f}%")
        
        # Save validation stats
        output_dir = Path('benchmarks/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'dataset_validation_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✓ Validation stats saved")
        
        return stats
    
    def generate_dataset(self):
        """Generate complete dataset with all features"""
        print("=" * 80)
        print("BENCHMARK DATASET GENERATION")
        print("=" * 80)
        
        # Generate base transactions
        df, customers, merchants = self.generate_base_transactions()
        
        # Add all feature groups
        df = self.add_temporal_features(df)
        df = self.add_geographic_features(df)
        df = self.add_behavioral_features(df)
        df = self.add_network_features(df)
        df = self.add_upi_specific_features(df)
        
        # Validate realism
        self.validate_realism(df)
        
        print("\n" + "=" * 80)
        print(f"DATASET COMPLETE: {len(df):,} transactions with {len(df.columns)} features")
        print("=" * 80)
        
        return df
    
    def save_train_test_split(self, df, test_size=0.3):
        """Save train/test splits"""
        print(f"\n[SPLIT] Creating {100*(1-test_size):.0f}/{100*test_size:.0f} train/test split...")
        
        # Shuffle dataset
        df_shuffled = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Split
        split_idx = int(len(df_shuffled) * (1 - test_size))
        train_df = df_shuffled.iloc[:split_idx]
        test_df = df_shuffled.iloc[split_idx:]
        
        print(f"  Train: {len(train_df):,} transactions")
        print(f"  Test: {len(test_df):,} transactions")
        
        # Save to parquet (efficient format)
        output_dir = Path('benchmarks/data')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / 'train_500k.parquet'
        test_path = output_dir / 'test_150k.parquet'
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        print(f"  ✓ Saved: {train_path}")
        print(f"  ✓ Saved: {test_path}")
        
        # Also save full dataset
        full_path = output_dir / 'full_500k.parquet'
        df_shuffled.to_parquet(full_path, index=False)
        print(f"  ✓ Saved: {full_path}")
        
        return train_df, test_df


def main():
    """Generate benchmark dataset"""
    # Generate 500k transactions
    generator = BenchmarkDatasetGenerator(
        n_transactions=500_000,
        fraud_rate=0.05,
        random_state=42
    )
    
    # Generate dataset
    df = generator.generate_dataset()
    
    # Save train/test splits
    train_df, test_df = generator.save_train_test_split(df, test_size=0.3)
    
    print("\n✓ DATASET GENERATION COMPLETE")
    print(f"\nNext steps:")
    print(f"  1. Verify data: python benchmarks/verify_dataset.py")
    print(f"  2. Train models: python benchmarks/train_models.py")
    print(f"  3. Evaluate: python benchmarks/evaluate_models.py")


if __name__ == '__main__':
    main()
