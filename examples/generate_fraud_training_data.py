"""
Generate ML Training Dataset for Fraud Detection

This script generates a complete fraud detection training dataset with:
- Transaction generation
- Fraud pattern injection
- ML feature engineering
- Dataset preparation (balance, split, normalize)
- Multiple export formats

Usage:
    python examples/generate_fraud_training_data.py --num-transactions 10000 --fraud-rate 0.1
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

from src.data_generator import DataGenerator
from src.generators.fraud_patterns import FraudPatternGenerator
from src.generators.ml_features import MLFeatureEngineer
from src.generators.ml_dataset_generator import MLDatasetGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate ML training dataset for fraud detection'
    )
    parser.add_argument(
        '--num-transactions',
        type=int,
        default=10000,
        help='Number of transactions to generate (default: 10000)'
    )
    parser.add_argument(
        '--fraud-rate',
        type=float,
        default=0.1,
        help='Fraud injection rate 0.0-1.0 (default: 0.1)'
    )
    parser.add_argument(
        '--num-customers',
        type=int,
        default=200,
        help='Number of customers (default: 200)'
    )
    parser.add_argument(
        '--num-days',
        type=int,
        default=90,
        help='Number of days for transaction history (default: 90)'
    )
    parser.add_argument(
        '--balance-strategy',
        type=str,
        choices=['undersample', 'oversample'],
        default='undersample',
        help='Dataset balancing strategy (default: undersample)'
    )
    parser.add_argument(
        '--target-fraud-rate',
        type=float,
        default=0.5,
        help='Target fraud rate after balancing (default: 0.5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/ml_training_data',
        help='Output directory (default: output/ml_training_data)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--export-parquet',
        action='store_true',
        help='Export to Parquet format (requires pyarrow)'
    )
    parser.add_argument(
        '--export-numpy',
        action='store_true',
        help='Export to NumPy arrays (sklearn-ready)'
    )
    
    return parser.parse_args()


def build_transaction_history(transactions):
    """Build transaction history lookup for feature engineering."""
    transaction_history = {}
    
    for txn in sorted(transactions, key=lambda x: x.timestamp):
        customer_id = txn.customer_id
        if customer_id not in transaction_history:
            transaction_history[customer_id] = []
        transaction_history[customer_id].append(txn)
    
    return transaction_history


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("ML TRAINING DATASET GENERATOR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Transactions: {args.num_transactions:,}")
    print(f"  Customers: {args.num_customers}")
    print(f"  Days: {args.num_days}")
    print(f"  Fraud Rate: {args.fraud_rate:.1%}")
    print(f"  Balance Strategy: {args.balance_strategy}")
    print(f"  Target Fraud Rate: {args.target_fraud_rate:.1%}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Random Seed: {args.seed}\n")
    
    # Step 1: Generate base transactions
    print("Step 1: Generating base transactions...")
    generator = DataGenerator(
        num_customers=args.num_customers,
        num_days=args.num_days,
        seed=args.seed
    )
    customers = generator.generate_customers()
    transactions = generator.generate_transactions(num_transactions=args.num_transactions)
    print(f"  Generated {len(transactions):,} transactions")
    print(f"  Customers: {len(customers)}")
    
    # Step 2: Inject fraud patterns
    print("\nStep 2: Injecting fraud patterns...")
    fraud_gen = FraudPatternGenerator(seed=args.seed)
    transactions = fraud_gen.inject_fraud_patterns(
        transactions,
        customers,
        fraud_rate=args.fraud_rate
    )
    
    fraud_stats = fraud_gen.get_fraud_statistics()
    print(f"  Total fraud: {fraud_stats['total_fraud']} ({fraud_stats['fraud_rate']:.2%})")
    print(f"  Fraud patterns:")
    for pattern, count in fraud_stats['fraud_by_type'].items():
        print(f"    {pattern}: {count}")
    
    # Step 3: Engineer ML features
    print("\nStep 3: Engineering ML features...")
    engineer = MLFeatureEngineer()
    
    # Build transaction history
    transaction_history = build_transaction_history(transactions)
    
    # Engineer features for all transactions
    features_list = []
    batch_size = 1000
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i+batch_size]
        batch_features = engineer.engineer_features(batch, transaction_history)
        features_list.extend(batch_features)
        
        if (i + batch_size) % 5000 == 0:
            print(f"  Processed {min(i+batch_size, len(transactions)):,}/{len(transactions):,} transactions")
    
    print(f"  Engineered {len(features_list):,} samples with 32 features each")
    
    # Step 4: Create ML-ready dataset
    print("\nStep 4: Creating ML-ready dataset...")
    dataset_gen = MLDatasetGenerator(seed=args.seed)
    
    split, metadata = dataset_gen.create_ml_ready_dataset(
        features_list,
        balance_strategy=args.balance_strategy,
        target_fraud_rate=args.target_fraud_rate,
        normalize=True,
        encode_categorical=True,
        train_ratio=0.70,
        val_ratio=0.15
    )
    
    print(f"  Train: {len(split.train):,} samples")
    print(f"  Validation: {len(split.validation):,} samples")
    print(f"  Test: {len(split.test):,} samples")
    
    # Validate quality
    train_quality = dataset_gen.validate_dataset_quality(split.train)
    print(f"\n  Dataset Quality:")
    print(f"    Fraud rate: {train_quality['fraud_rate']:.2%}")
    print(f"    Class balance: {train_quality['quality_checks']['class_balance']}")
    print(f"    Sufficient samples: {train_quality['quality_checks']['sufficient_samples']}")
    print(f"    No missing labels: {train_quality['quality_checks']['no_missing_labels']}")
    
    # Step 5: Export datasets
    print("\nStep 5: Exporting datasets...")
    
    # Always export CSV
    csv_files = {
        'train': os.path.join(args.output_dir, 'train.csv'),
        'validation': os.path.join(args.output_dir, 'validation.csv'),
        'test': os.path.join(args.output_dir, 'test.csv')
    }
    
    dataset_gen.export_to_csv(split.train, csv_files['train'])
    dataset_gen.export_to_csv(split.validation, csv_files['validation'])
    dataset_gen.export_to_csv(split.test, csv_files['test'])
    print(f"  CSV: {csv_files['train']}")
    print(f"  CSV: {csv_files['validation']}")
    print(f"  CSV: {csv_files['test']}")
    
    # Export metadata
    metadata_file = os.path.join(args.output_dir, 'metadata.json')
    dataset_gen.export_metadata(metadata_file, metadata)
    print(f"  Metadata: {metadata_file}")
    
    # Optional: Parquet export
    if args.export_parquet:
        print("\n  Exporting to Parquet format...")
        parquet_files = {
            'train': os.path.join(args.output_dir, 'train.parquet'),
            'validation': os.path.join(args.output_dir, 'validation.parquet'),
            'test': os.path.join(args.output_dir, 'test.parquet')
        }
        
        dataset_gen.export_to_parquet(split.train, parquet_files['train'])
        dataset_gen.export_to_parquet(split.validation, parquet_files['validation'])
        dataset_gen.export_to_parquet(split.test, parquet_files['test'])
        print(f"  Parquet: {parquet_files['train']}")
        print(f"  Parquet: {parquet_files['validation']}")
        print(f"  Parquet: {parquet_files['test']}")
    
    # Optional: NumPy export
    if args.export_numpy:
        print("\n  Exporting to NumPy arrays...")
        numpy_prefix_train = os.path.join(args.output_dir, 'train')
        numpy_prefix_val = os.path.join(args.output_dir, 'validation')
        numpy_prefix_test = os.path.join(args.output_dir, 'test')
        
        X_train, y_train, feature_names = dataset_gen.export_to_numpy(
            split.train, numpy_prefix_train
        )
        X_val, y_val, _ = dataset_gen.export_to_numpy(
            split.validation, numpy_prefix_val
        )
        X_test, y_test, _ = dataset_gen.export_to_numpy(
            split.test, numpy_prefix_test
        )
        
        print(f"  NumPy: {numpy_prefix_train}_X.npy, {numpy_prefix_train}_y.npy")
        print(f"  NumPy: {numpy_prefix_val}_X.npy, {numpy_prefix_val}_y.npy")
        print(f"  NumPy: {numpy_prefix_test}_X.npy, {numpy_prefix_test}_y.npy")
        print(f"  Feature names: {numpy_prefix_train}_features.json")
    
    # Step 6: Generate summary report
    print("\nStep 6: Generating summary report...")
    
    summary = {
        'generation_date': datetime.now().isoformat(),
        'configuration': {
            'num_transactions': args.num_transactions,
            'num_customers': args.num_customers,
            'num_days': args.num_days,
            'fraud_rate': args.fraud_rate,
            'balance_strategy': args.balance_strategy,
            'target_fraud_rate': args.target_fraud_rate,
            'seed': args.seed
        },
        'fraud_statistics': fraud_stats,
        'dataset_split': {
            'train_size': len(split.train),
            'validation_size': len(split.validation),
            'test_size': len(split.test),
            'train_fraud_rate': train_quality['fraud_rate'],
            'total_features': 32
        },
        'quality_checks': train_quality['quality_checks'],
        'output_files': {
            'csv': list(csv_files.values()),
            'metadata': metadata_file
        }
    }
    
    if args.export_parquet:
        summary['output_files']['parquet'] = list(parquet_files.values())
    
    if args.export_numpy:
        summary['output_files']['numpy'] = [
            f"{numpy_prefix_train}_X.npy",
            f"{numpy_prefix_train}_y.npy",
            f"{numpy_prefix_val}_X.npy",
            f"{numpy_prefix_val}_y.npy",
            f"{numpy_prefix_test}_X.npy",
            f"{numpy_prefix_test}_y.npy",
            f"{numpy_prefix_train}_features.json"
        ]
    
    summary_file = os.path.join(args.output_dir, 'generation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Summary: {summary_file}")
    
    # Print completion message
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Total samples: {len(features_list):,}")
    print(f"Train/Val/Test: {len(split.train)}/{len(split.validation)}/{len(split.test)}")
    print(f"Features: 32 (6 categories)")
    print(f"Fraud rate (train): {train_quality['fraud_rate']:.2%}")
    print("\nNext steps:")
    print("  1. Review generation_summary.json for statistics")
    print("  2. Train ML model: python examples/train_fraud_detector.py")
    print("  3. Validate quality: python scripts/validate_data_quality.py output/ml_training_data/train.csv")
    print("  4. See ML_DATASET_GUIDE.md for complete documentation")
    print()


if __name__ == '__main__':
    main()
