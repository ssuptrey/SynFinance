"""
Generate Anomaly Detection Dataset

This script generates a synthetic transaction dataset with both fraud patterns
and anomalies for training anomaly detection and fraud detection models.

Usage:
    python examples/generate_anomaly_dataset.py --num-transactions 10000 --anomaly-rate 0.05

Features:
    - Generates realistic customer profiles and transactions
    - Injects fraud patterns (configurable rate)
    - Injects anomaly patterns (configurable rate)
    - Exports to CSV with complete ground truth labels
    - Generates analysis report with statistics
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generator import DataGenerator
from src.generators.fraud_patterns import FraudPatternGenerator, apply_fraud_labels
from src.generators.anomaly_patterns import AnomalyPatternGenerator, apply_anomaly_labels


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic transaction dataset with anomalies and fraud patterns'
    )
    
    parser.add_argument(
        '--num-transactions',
        type=int,
        default=10000,
        help='Number of transactions to generate (default: 10000)'
    )
    
    parser.add_argument(
        '--num-customers',
        type=int,
        default=200,
        help='Number of customers to generate (default: 200)'
    )
    
    parser.add_argument(
        '--num-days',
        type=int,
        default=90,
        help='Number of days of transaction history (default: 90)'
    )
    
    parser.add_argument(
        '--fraud-rate',
        type=float,
        default=0.02,
        help='Fraud injection rate 0.0-1.0 (default: 0.02 = 2%%)'
    )
    
    parser.add_argument(
        '--anomaly-rate',
        type=float,
        default=0.05,
        help='Anomaly injection rate 0.0-1.0 (default: 0.05 = 5%%)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/anomaly_dataset',
        help='Output directory for generated files (default: output/anomaly_dataset)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def generate_base_transactions(num_transactions: int, num_customers: int, 
                               num_days: int, seed: int) -> tuple:
    """
    Generate base transactions and customer profiles
    
    Args:
        num_transactions: Number of transactions to generate
        num_customers: Number of customers
        num_days: Number of days of history
        seed: Random seed
        
    Returns:
        Tuple of (transactions, customers)
    """
    print(f"\n{'='*60}")
    print("STEP 1: Generating Base Transactions")
    print(f"{'='*60}")
    
    generator = DataGenerator(
        num_customers=num_customers,
        num_days=num_days,
        seed=seed
    )
    
    print(f"Generating {num_customers} customers...")
    customers = generator.generate_customers()
    print(f"✓ Generated {len(customers)} customers")
    
    print(f"\nGenerating {num_transactions} transactions...")
    transactions = generator.generate_transactions(num_transactions=num_transactions)
    print(f"✓ Generated {len(transactions)} transactions")
    
    return transactions, customers


def inject_fraud_patterns(transactions: list, customers: list, 
                          fraud_rate: float, seed: int) -> list:
    """
    Inject fraud patterns into transactions
    
    Args:
        transactions: List of transactions
        customers: List of customers
        fraud_rate: Target fraud rate
        seed: Random seed
        
    Returns:
        Transactions with fraud patterns
    """
    print(f"\n{'='*60}")
    print("STEP 2: Injecting Fraud Patterns")
    print(f"{'='*60}")
    
    fraud_gen = FraudPatternGenerator(seed=seed)
    
    print(f"Target fraud rate: {fraud_rate:.1%}")
    transactions = fraud_gen.inject_fraud_patterns(
        transactions,
        customers,
        fraud_rate=fraud_rate
    )
    
    # Get statistics
    stats = fraud_gen.get_statistics()
    print(f"\n✓ Fraud injection complete:")
    print(f"  - Total transactions: {stats['total_transactions']}")
    print(f"  - Fraud count: {stats['fraud_count']}")
    print(f"  - Actual fraud rate: {stats['fraud_rate']:.2%}")
    
    print(f"\n  Fraud by type:")
    for fraud_type, count in stats['frauds_by_type'].items():
        if count > 0:
            print(f"    - {fraud_type}: {count} ({count/stats['total_transactions']:.2%})")
    
    # Apply labels to non-fraudulent transactions
    transactions = apply_fraud_labels(transactions)
    
    return transactions


def inject_anomaly_patterns(transactions: list, customers: list,
                            anomaly_rate: float, seed: int) -> list:
    """
    Inject anomaly patterns into transactions
    
    Args:
        transactions: List of transactions
        customers: List of customers
        anomaly_rate: Target anomaly rate
        seed: Random seed
        
    Returns:
        Transactions with anomaly patterns
    """
    print(f"\n{'='*60}")
    print("STEP 3: Injecting Anomaly Patterns")
    print(f"{'='*60}")
    
    anomaly_gen = AnomalyPatternGenerator(seed=seed)
    
    print(f"Target anomaly rate: {anomaly_rate:.1%}")
    transactions = anomaly_gen.inject_anomaly_patterns(
        transactions,
        customers,
        anomaly_rate=anomaly_rate
    )
    
    # Get statistics
    stats = anomaly_gen.get_statistics()
    print(f"\n✓ Anomaly injection complete:")
    print(f"  - Total transactions: {stats['total_transactions']}")
    print(f"  - Anomaly count: {stats['anomaly_count']}")
    print(f"  - Actual anomaly rate: {stats['anomaly_rate']:.2%}")
    
    print(f"\n  Anomalies by type:")
    for anomaly_type, count in stats['anomalies_by_type'].items():
        if count > 0:
            print(f"    - {anomaly_type}: {count} ({count/stats['total_transactions']:.2%})")
    
    # Apply labels to non-anomalous transactions
    transactions = apply_anomaly_labels(transactions)
    
    return transactions


def analyze_dataset(transactions: list) -> Dict[str, Any]:
    """
    Analyze the generated dataset
    
    Args:
        transactions: List of transactions
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print("STEP 4: Analyzing Dataset")
    print(f"{'='*60}")
    
    total = len(transactions)
    
    # Count fraud and anomalies
    fraud_count = sum(1 for txn in transactions if txn.get('Fraud_Type', 'None') != 'None')
    anomaly_count = sum(1 for txn in transactions if txn.get('Anomaly_Type', 'None') != 'None')
    
    # Count overlap (transactions that are both fraud and anomaly)
    both_count = sum(1 for txn in transactions 
                    if txn.get('Fraud_Type', 'None') != 'None' 
                    and txn.get('Anomaly_Type', 'None') != 'None')
    
    # Severity analysis
    anomaly_severities = [txn.get('Anomaly_Severity', 0.0) for txn in transactions 
                         if txn.get('Anomaly_Type', 'None') != 'None']
    
    avg_severity = sum(anomaly_severities) / len(anomaly_severities) if anomaly_severities else 0.0
    high_severity = sum(1 for s in anomaly_severities if s >= 0.7)
    
    # Confidence analysis
    anomaly_confidences = [txn.get('Anomaly_Confidence', 0.0) for txn in transactions
                          if txn.get('Anomaly_Type', 'None') != 'None']
    
    avg_confidence = sum(anomaly_confidences) / len(anomaly_confidences) if anomaly_confidences else 0.0
    high_confidence = sum(1 for c in anomaly_confidences if c >= 0.7)
    
    analysis = {
        'total_transactions': total,
        'fraud_count': fraud_count,
        'fraud_rate': fraud_count / total,
        'anomaly_count': anomaly_count,
        'anomaly_rate': anomaly_count / total,
        'overlap_count': both_count,
        'overlap_rate': both_count / fraud_count if fraud_count > 0 else 0,
        'avg_anomaly_severity': avg_severity,
        'high_severity_count': high_severity,
        'high_severity_rate': high_severity / anomaly_count if anomaly_count > 0 else 0,
        'avg_anomaly_confidence': avg_confidence,
        'high_confidence_count': high_confidence,
        'high_confidence_rate': high_confidence / anomaly_count if anomaly_count > 0 else 0
    }
    
    print(f"\n✓ Dataset Analysis:")
    print(f"  - Total transactions: {analysis['total_transactions']}")
    print(f"  - Fraud transactions: {analysis['fraud_count']} ({analysis['fraud_rate']:.2%})")
    print(f"  - Anomaly transactions: {analysis['anomaly_count']} ({analysis['anomaly_rate']:.2%})")
    print(f"  - Overlap (fraud + anomaly): {analysis['overlap_count']} ({analysis['overlap_rate']:.1%} of fraud)")
    print(f"\n  Anomaly Quality Metrics:")
    print(f"  - Average severity: {analysis['avg_anomaly_severity']:.3f}")
    print(f"  - High severity (≥0.7): {analysis['high_severity_count']} ({analysis['high_severity_rate']:.1%})")
    print(f"  - Average confidence: {analysis['avg_anomaly_confidence']:.3f}")
    print(f"  - High confidence (≥0.7): {analysis['high_confidence_count']} ({analysis['high_confidence_rate']:.1%})")
    
    return analysis


def export_dataset(transactions: list, output_dir: str, analysis: Dict[str, Any]) -> None:
    """
    Export dataset to CSV and generate report
    
    Args:
        transactions: List of transactions
        output_dir: Output directory
        analysis: Analysis results
    """
    print(f"\n{'='*60}")
    print("STEP 5: Exporting Dataset")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to CSV
    csv_path = os.path.join(output_dir, 'anomaly_dataset.csv')
    print(f"\nExporting to CSV: {csv_path}")
    
    if transactions:
        import csv
        
        # Get all unique field names
        fieldnames = set()
        for txn in transactions:
            fieldnames.update(txn.keys())
        fieldnames = sorted(fieldnames)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(transactions)
        
        print(f"✓ Exported {len(transactions)} transactions to CSV")
        file_size = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
    
    # Generate summary report
    report_path = os.path.join(output_dir, 'dataset_summary.json')
    print(f"\nGenerating summary report: {report_path}")
    
    summary = {
        'generation_date': datetime.now().isoformat(),
        'dataset_statistics': analysis,
        'output_files': {
            'dataset_csv': csv_path,
            'summary_report': report_path
        }
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary report generated")
    
    # Generate text report
    txt_report_path = os.path.join(output_dir, 'dataset_summary.txt')
    print(f"\nGenerating text report: {txt_report_path}")
    
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ANOMALY DETECTION DATASET SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Dataset Statistics:\n")
        f.write(f"  Total Transactions: {analysis['total_transactions']}\n")
        f.write(f"  Fraud Transactions: {analysis['fraud_count']} ({analysis['fraud_rate']:.2%})\n")
        f.write(f"  Anomaly Transactions: {analysis['anomaly_count']} ({analysis['anomaly_rate']:.2%})\n")
        f.write(f"  Overlap (fraud + anomaly): {analysis['overlap_count']} ({analysis['overlap_rate']:.1%})\n\n")
        
        f.write("Anomaly Quality Metrics:\n")
        f.write(f"  Average Severity: {analysis['avg_anomaly_severity']:.3f}\n")
        f.write(f"  High Severity Count: {analysis['high_severity_count']} ({analysis['high_severity_rate']:.1%})\n")
        f.write(f"  Average Confidence: {analysis['avg_anomaly_confidence']:.3f}\n")
        f.write(f"  High Confidence Count: {analysis['high_confidence_count']} ({analysis['high_confidence_rate']:.1%})\n\n")
        
        f.write("Output Files:\n")
        f.write(f"  Dataset CSV: {csv_path}\n")
        f.write(f"  Summary JSON: {report_path}\n")
        f.write(f"  Text Report: {txt_report_path}\n")
    
    print(f"✓ Text report generated")


def main():
    """Main execution function"""
    args = parse_args()
    
    print("\n" + "="*60)
    print("ANOMALY DETECTION DATASET GENERATOR")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Transactions: {args.num_transactions}")
    print(f"  Customers: {args.num_customers}")
    print(f"  Days: {args.num_days}")
    print(f"  Fraud Rate: {args.fraud_rate:.1%}")
    print(f"  Anomaly Rate: {args.anomaly_rate:.1%}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Random Seed: {args.seed}")
    
    # Step 1: Generate base transactions
    transactions, customers = generate_base_transactions(
        args.num_transactions,
        args.num_customers,
        args.num_days,
        args.seed
    )
    
    # Step 2: Inject fraud patterns
    transactions = inject_fraud_patterns(
        transactions,
        customers,
        args.fraud_rate,
        args.seed
    )
    
    # Step 3: Inject anomaly patterns
    transactions = inject_anomaly_patterns(
        transactions,
        customers,
        args.anomaly_rate,
        args.seed
    )
    
    # Step 4: Analyze dataset
    analysis = analyze_dataset(transactions)
    
    # Step 5: Export dataset
    export_dataset(transactions, args.output_dir, analysis)
    
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n✓ Dataset successfully generated!")
    print(f"\nNext steps:")
    print(f"  1. Review dataset: {os.path.join(args.output_dir, 'anomaly_dataset.csv')}")
    print(f"  2. Check summary: {os.path.join(args.output_dir, 'dataset_summary.txt')}")
    print(f"  3. Train ML models using the dataset")
    print(f"  4. Analyze anomaly patterns and fraud detection")
    print()


if __name__ == '__main__':
    main()
