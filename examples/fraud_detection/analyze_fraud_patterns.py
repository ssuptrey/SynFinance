"""
Analyze Fraud Patterns in Generated Dataset

This script analyzes fraud patterns in a generated transaction dataset:
- Fraud pattern distribution
- Fraud combinations analysis
- Network analysis statistics
- Cross-pattern co-occurrence
- Temporal distribution
- Severity and confidence statistics

Usage:
    python examples/analyze_fraud_patterns.py --input output/fraud_dataset.csv
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List

import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze fraud patterns in generated dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file with fraud data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/fraud_analysis',
        help='Output directory for analysis results (default: output/fraud_analysis)'
    )
    
    return parser.parse_args()


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} transactions")
    return df


def analyze_fraud_distribution(df: pd.DataFrame) -> Dict:
    """Analyze fraud type distribution."""
    print("\nAnalyzing fraud pattern distribution...")
    
    # Overall fraud rate
    total_transactions = len(df)
    fraud_transactions = df[df['is_fraud'] == 1]
    fraud_count = len(fraud_transactions)
    fraud_rate = fraud_count / total_transactions if total_transactions > 0 else 0
    
    print(f"  Total transactions: {total_transactions:,}")
    print(f"  Fraud transactions: {fraud_count:,} ({fraud_rate:.2%})")
    
    # Fraud type distribution
    fraud_types = fraud_transactions['fraud_type'].value_counts()
    print(f"\n  Fraud patterns detected:")
    for fraud_type, count in fraud_types.items():
        pct = count / fraud_count * 100
        print(f"    {fraud_type}: {count} ({pct:.1f}%)")
    
    return {
        'total_transactions': int(total_transactions),
        'fraud_count': int(fraud_count),
        'fraud_rate': float(fraud_rate),
        'fraud_by_type': {str(k): int(v) for k, v in fraud_types.to_dict().items()}
    }


def analyze_severity_distribution(df: pd.DataFrame) -> Dict:
    """Analyze fraud severity distribution."""
    print("\nAnalyzing fraud severity distribution...")
    
    fraud_transactions = df[df['is_fraud'] == 1]
    
    if len(fraud_transactions) == 0:
        print("  No fraud transactions found")
        return {}
    
    severity_counts = fraud_transactions['fraud_severity'].value_counts()
    print(f"  Severity distribution:")
    for severity, count in severity_counts.items():
        pct = count / len(fraud_transactions) * 100
        print(f"    {severity}: {count} ({pct:.1f}%)")
    
    return {str(k): int(v) for k, v in severity_counts.to_dict().items()}


def analyze_confidence_statistics(df: pd.DataFrame) -> Dict:
    """Analyze fraud confidence score statistics."""
    print("\nAnalyzing fraud confidence statistics...")
    
    fraud_transactions = df[df['is_fraud'] == 1]
    
    if len(fraud_transactions) == 0:
        print("  No fraud transactions found")
        return {}
    
    confidence_scores = fraud_transactions['fraud_confidence']
    
    stats = {
        'mean': float(confidence_scores.mean()),
        'median': float(confidence_scores.median()),
        'std': float(confidence_scores.std()),
        'min': float(confidence_scores.min()),
        'max': float(confidence_scores.max()),
        'q25': float(confidence_scores.quantile(0.25)),
        'q75': float(confidence_scores.quantile(0.75))
    }
    
    print(f"  Confidence scores:")
    print(f"    Mean: {stats['mean']:.3f}")
    print(f"    Median: {stats['median']:.3f}")
    print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"    Q25-Q75: [{stats['q25']:.3f}, {stats['q75']:.3f}]")
    
    # Confidence bins
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels = ['Very Low (0-0.3)', 'Low (0.3-0.5)', 'Medium (0.5-0.7)', 'High (0.7-0.9)', 'Very High (0.9-1.0)']
    confidence_bins = pd.cut(confidence_scores, bins=bins, labels=labels)
    bin_counts = confidence_bins.value_counts()
    
    print(f"\n  Confidence bins:")
    for label, count in bin_counts.items():
        pct = count / len(fraud_transactions) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")
    
    stats['bins'] = {str(k): int(v) for k, v in bin_counts.to_dict().items()}
    
    return stats


def analyze_temporal_patterns(df: pd.DataFrame) -> Dict:
    """Analyze temporal distribution of fraud."""
    print("\nAnalyzing temporal fraud patterns...")
    
    fraud_transactions = df[df['is_fraud'] == 1].copy()
    
    if len(fraud_transactions) == 0:
        print("  No fraud transactions found")
        return {}
    
    # Convert timestamp to datetime if not already
    if 'timestamp' in fraud_transactions.columns:
        fraud_transactions['datetime'] = pd.to_datetime(fraud_transactions['timestamp'])
        fraud_transactions['hour'] = fraud_transactions['datetime'].dt.hour
        fraud_transactions['day_of_week'] = fraud_transactions['datetime'].dt.dayofweek
        
        # Hour distribution
        hour_counts = fraud_transactions['hour'].value_counts().sort_index()
        print(f"\n  Fraud by hour of day (top 5):")
        for hour, count in hour_counts.head(5).items():
            pct = count / len(fraud_transactions) * 100
            print(f"    {hour:02d}:00: {count} ({pct:.1f}%)")
        
        # Day of week distribution
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = fraud_transactions['day_of_week'].value_counts().sort_index()
        print(f"\n  Fraud by day of week:")
        for day, count in dow_counts.items():
            pct = count / len(fraud_transactions) * 100
            print(f"    {day_names[day]}: {count} ({pct:.1f}%)")
        
        return {
            'by_hour': {int(k): int(v) for k, v in hour_counts.to_dict().items()},
            'by_day_of_week': {day_names[int(k)]: int(v) for k, v in dow_counts.to_dict().items()}
        }
    
    return {}


def analyze_amount_patterns(df: pd.DataFrame) -> Dict:
    """Analyze transaction amount patterns for fraud."""
    print("\nAnalyzing fraud amount patterns...")
    
    fraud_transactions = df[df['is_fraud'] == 1]
    normal_transactions = df[df['is_fraud'] == 0]
    
    if len(fraud_transactions) == 0:
        print("  No fraud transactions found")
        return {}
    
    fraud_amounts = fraud_transactions['amount']
    normal_amounts = normal_transactions['amount']
    
    fraud_stats = {
        'fraud_mean': float(fraud_amounts.mean()),
        'fraud_median': float(fraud_amounts.median()),
        'fraud_std': float(fraud_amounts.std()),
        'fraud_min': float(fraud_amounts.min()),
        'fraud_max': float(fraud_amounts.max())
    }
    
    normal_stats = {
        'normal_mean': float(normal_amounts.mean()),
        'normal_median': float(normal_amounts.median()),
        'normal_std': float(normal_amounts.std())
    }
    
    print(f"  Fraud amounts:")
    print(f"    Mean: Rs.{fraud_stats['fraud_mean']:,.2f}")
    print(f"    Median: Rs.{fraud_stats['fraud_median']:,.2f}")
    print(f"    Range: Rs.{fraud_stats['fraud_min']:,.2f} - Rs.{fraud_stats['fraud_max']:,.2f}")
    
    print(f"\n  Normal amounts:")
    print(f"    Mean: Rs.{normal_stats['normal_mean']:,.2f}")
    print(f"    Median: Rs.{normal_stats['normal_median']:,.2f}")
    
    return {**fraud_stats, **normal_stats}


def analyze_fraud_evidence(df: pd.DataFrame) -> Dict:
    """Analyze fraud evidence patterns."""
    print("\nAnalyzing fraud evidence patterns...")
    
    fraud_transactions = df[df['is_fraud'] == 1]
    
    if len(fraud_transactions) == 0:
        print("  No fraud transactions found")
        return {}
    
    # Count common evidence indicators
    evidence_keywords = defaultdict(int)
    
    for evidence in fraud_transactions['fraud_evidence']:
        if pd.notna(evidence):
            # Parse JSON evidence
            try:
                evidence_dict = json.loads(evidence.replace("'", '"'))
                for key in evidence_dict.keys():
                    evidence_keywords[key] += 1
            except:
                pass
    
    print(f"  Common evidence indicators:")
    for keyword, count in sorted(evidence_keywords.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = count / len(fraud_transactions) * 100
        print(f"    {keyword}: {count} ({pct:.1f}%)")
    
    return dict(evidence_keywords)


def generate_summary_report(results: Dict, output_dir: str):
    """Generate comprehensive summary report."""
    print("\nGenerating summary report...")
    
    report_file = os.path.join(output_dir, 'fraud_analysis_report.json')
    
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'fraud_distribution': results.get('fraud_distribution', {}),
        'severity_distribution': results.get('severity_distribution', {}),
        'confidence_statistics': results.get('confidence_statistics', {}),
        'temporal_patterns': results.get('temporal_patterns', {}),
        'amount_patterns': results.get('amount_patterns', {}),
        'evidence_patterns': results.get('evidence_patterns', {})
    }
    
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Report saved to: {report_file}")
    
    # Generate text summary
    text_report_file = os.path.join(output_dir, 'fraud_analysis_summary.txt')
    
    with open(text_report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FRAUD PATTERN ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis Date: {summary['analysis_date']}\n\n")
        
        # Fraud distribution
        if 'fraud_distribution' in summary:
            dist = summary['fraud_distribution']
            f.write("FRAUD DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Transactions: {dist.get('total_transactions', 0):,}\n")
            f.write(f"Fraud Transactions: {dist.get('fraud_count', 0):,} ({dist.get('fraud_rate', 0):.2%})\n\n")
            
            if 'fraud_by_type' in dist:
                f.write("Fraud Patterns:\n")
                for fraud_type, count in dist['fraud_by_type'].items():
                    pct = count / dist['fraud_count'] * 100 if dist['fraud_count'] > 0 else 0
                    f.write(f"  {fraud_type}: {count} ({pct:.1f}%)\n")
            f.write("\n")
        
        # Severity distribution
        if 'severity_distribution' in summary:
            f.write("SEVERITY DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            for severity, count in summary['severity_distribution'].items():
                f.write(f"  {severity}: {count}\n")
            f.write("\n")
        
        # Confidence statistics
        if 'confidence_statistics' in summary:
            conf = summary['confidence_statistics']
            f.write("CONFIDENCE STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean: {conf.get('mean', 0):.3f}\n")
            f.write(f"Median: {conf.get('median', 0):.3f}\n")
            f.write(f"Range: [{conf.get('min', 0):.3f}, {conf.get('max', 0):.3f}]\n\n")
    
    print(f"  Text summary saved to: {text_report_file}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("FRAUD PATTERN ANALYSIS")
    print("=" * 80)
    
    # Load dataset
    df = load_dataset(args.input)
    
    # Run analyses
    results = {}
    
    results['fraud_distribution'] = analyze_fraud_distribution(df)
    results['severity_distribution'] = analyze_severity_distribution(df)
    results['confidence_statistics'] = analyze_confidence_statistics(df)
    results['temporal_patterns'] = analyze_temporal_patterns(df)
    results['amount_patterns'] = analyze_amount_patterns(df)
    results['evidence_patterns'] = analyze_fraud_evidence(df)
    
    # Generate summary report
    generate_summary_report(results, args.output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  - fraud_analysis_report.json (detailed JSON)")
    print(f"  - fraud_analysis_summary.txt (text summary)")
    print()


if __name__ == '__main__':
    main()
