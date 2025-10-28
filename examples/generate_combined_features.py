"""
Generate Combined ML Features Example

This example demonstrates the complete pipeline for generating unified ML features
that combine fraud-based and anomaly-based signals with interaction features.

Pipeline:
1. Generate synthetic transaction data with fraud and anomaly patterns
2. Generate fraud-based ML features (32 features)
3. Generate anomaly-based ML features (26 features)
4. Generate combined features with interactions (68 features total)
5. Export features in multiple formats
6. Calculate feature statistics

Author: SynFinance Development Team
Version: 0.7.0
Date: October 28, 2025
"""

import sys
import os
from datetime import datetime, timedelta
from dataclasses import asdict
import json
import csv
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.customer_generator import CustomerGenerator
from src.generators.transaction_core import TransactionGenerator
from src.generators.fraud_patterns import FraudPatternGenerator
from src.generators.anomaly_patterns import AnomalyPatternGenerator
from src.generators.ml_features import MLFeatureEngineer
from src.generators.anomaly_ml_features import AnomalyMLFeatureGenerator
from src.generators.combined_ml_features import CombinedMLFeatureGenerator


def main():
    """Generate and export combined ML features"""
    
    print("=" * 80)
    print("COMBINED ML FEATURES GENERATION")
    print("=" * 80)
    print()
    
    # Step 1: Initialize generators
    print("Step 1: Initializing generators...")
    print("-" * 40)
    
    # Create output directory
    output_dir = Path('output/combined_features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data generators
    customer_gen = CustomerGenerator(seed=42)
    txn_gen = TransactionGenerator(seed=42)
    fraud_gen = FraudPatternGenerator(fraud_rate=0.10, seed=42)
    anomaly_gen = AnomalyPatternGenerator(seed=42)
    
    # Initialize feature generators
    fraud_feature_gen = MLFeatureEngineer()
    anomaly_feature_gen = AnomalyMLFeatureGenerator()
    combined_feature_gen = CombinedMLFeatureGenerator()
    
    print(f"  Customer generator initialized")
    print(f"  Transaction generator initialized")
    print(f"  Fraud pattern generator initialized (10% fraud rate)")
    print(f"  Anomaly pattern generator initialized (will inject 15% anomaly rate)")
    print(f"  Fraud feature engineer initialized")
    print(f"  Anomaly feature generator initialized")
    print(f"  Combined feature generator initialized")
    print()
    
    # Step 2: Generate transaction data
    print("Step 2: Generating transaction data...")
    print("-" * 40)
    
    # Generate customers
    customers = customer_gen.generate_customers(count=50)
    print(f"  Generated {len(customers)} customers")
    
    # Generate base transactions
    df = txn_gen.generate_dataset(customers=customers, transactions_per_customer=10)
    transactions = df.to_dict('records')
    print(f"  Generated {len(transactions)} base transactions")
    
    # Create customer map for fraud/anomaly injection
    customer_map = {}
    for c in customers:
        from types import SimpleNamespace
        customer_ns = SimpleNamespace(
            Customer_ID=c.customer_id,
            Age=c.age,
            Gender=c.gender,
            City=c.city,
            city=c.city,
            State=c.state,
            Region=c.region,
            Occupation=c.occupation.value,
            Income_Bracket=c.income_bracket.value,
            Segment=c.segment.value,
            Digital_Savviness=c.digital_savviness.value,
            digital_savviness=c.digital_savviness,
            Risk_Profile=c.risk_profile.value,
            preferred_categories=c.preferred_categories if hasattr(c, 'preferred_categories') else [],
            preferred_merchants=c.preferred_merchants if hasattr(c, 'preferred_merchants') else [],
            preferred_payment_modes=c.preferred_payment_modes if hasattr(c, 'preferred_payment_modes') else [],
        )
        customer_map[c.customer_id] = customer_ns
    
    # Inject fraud patterns
    fraud_transactions = []
    customer_history_map = {}
    
    for txn in transactions:
        customer_id = txn.get('Customer_ID')
        customer_dict = customer_map.get(customer_id)
        history = customer_history_map.get(customer_id, [])
        
        # Apply fraud
        modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(txn, customer_dict, history)
        
        # Add fraud labels
        if fraud_indicator:
            modified_txn['Is_Fraud'] = 1
            modified_txn['Fraud_Type'] = fraud_indicator.fraud_type.value
            modified_txn['Fraud_Confidence'] = fraud_indicator.confidence
        else:
            modified_txn['Is_Fraud'] = 0
            modified_txn['Fraud_Type'] = None
        
        fraud_transactions.append(modified_txn)
        
        if customer_id not in customer_history_map:
            customer_history_map[customer_id] = []
        customer_history_map[customer_id].append(modified_txn)
    
    # Inject anomaly patterns (batch injection)
    final_transactions = anomaly_gen.inject_anomaly_patterns(
        fraud_transactions,
        customers,
        anomaly_rate=0.15
    )
    
    transactions = final_transactions
    
    # Get statistics
    fraud_count = sum(1 for t in transactions if t.get('Is_Fraud') == 1)
    anomaly_count = sum(1 for t in transactions if t.get('Anomaly_Type', 'None') != 'None')
    both_count = sum(1 for t in transactions 
                     if t.get('Is_Fraud') == 1 and t.get('Anomaly_Type', 'None') != 'None')
    
    print(f"  Total transactions: {len(transactions)}")
    print(f"  Fraudulent: {fraud_count} ({fraud_count/len(transactions)*100:.1f}%)")
    print(f"  With anomalies: {anomaly_count} ({anomaly_count/len(transactions)*100:.1f}%)")
    print(f"  Both fraud & anomaly: {both_count} ({both_count/len(transactions)*100:.1f}%)")
    print()
    
    # Step 3: Generate fraud-based ML features
    print("Step 3: Generating fraud-based ML features...")
    print("-" * 40)
    
    fraud_features_list = []
    for i, transaction in enumerate(transactions):
        customer_id = transaction.get('Customer_ID')
        customer_ns = customer_map.get(customer_id)
        customer_dict = vars(customer_ns) if customer_ns else {}
        customer_history = [t for t in transactions[:i] 
                          if t.get('Customer_ID') == customer_id]
        
        fraud_features = fraud_feature_gen.engineer_features(
            transaction, customer_dict, customer_history
        )
        fraud_features_list.append(fraud_features)
    
    print(f"  Generated fraud features for {len(fraud_features_list)} transactions")
    print(f"  Features per transaction: 32")
    print()
    
    # Step 4: Generate anomaly-based ML features
    print("Step 4: Generating anomaly-based ML features...")
    print("-" * 40)
    
    anomaly_features_list = []
    for i, transaction in enumerate(transactions):
        customer_history = [t for t in transactions[:i] 
                          if t.get('Customer_ID') == transaction.get('Customer_ID')]
        
        anomaly_features = anomaly_feature_gen.generate_features(transaction, customer_history)
        anomaly_features_list.append(anomaly_features)
    
    print(f"  Generated anomaly features for {len(anomaly_features_list)} transactions")
    print(f"  Features per transaction: 26")
    print()
    
    # Step 5: Generate combined features with interactions
    print("Step 5: Generating combined ML features with interactions...")
    print("-" * 40)
    
    # Convert fraud and anomaly features to dictionaries
    fraud_dicts = [f.to_dict() for f in fraud_features_list]
    anomaly_dicts = [asdict(f) for f in anomaly_features_list]
    
    # Generate combined features
    combined_features = combined_feature_gen.generate_batch_features(
        transactions, fraud_dicts, anomaly_dicts
    )
    
    print(f"  Generated combined features for {len(combined_features)} transactions")
    print(f"  Total features per transaction: 68")
    print(f"    - Fraud features: 32")
    print(f"    - Anomaly features: 26")
    print(f"    - Interaction features: 10")
    print()
    
    # Step 6: Calculate feature statistics
    print("Step 6: Calculating feature statistics...")
    print("-" * 40)
    
    stats = combined_feature_gen.get_feature_statistics(combined_features)
    
    print(f"  Calculated statistics for {len(stats)} features")
    print()
    
    # Display sample statistics
    print("Sample Feature Statistics:")
    print()
    
    sample_features = [
        'daily_txn_count',
        'travel_velocity_kmh',
        'current_severity',
        'risk_amplification_score',
        'ensemble_fraud_probability'
    ]
    
    for feature_name in sample_features:
        if feature_name in stats:
            feature_stats = stats[feature_name]
            print(f"  {feature_name}:")
            print(f"    Mean:   {feature_stats['mean']:.4f}")
            print(f"    Std:    {feature_stats['std']:.4f}")
            print(f"    Min:    {feature_stats['min']:.4f}")
            print(f"    Max:    {feature_stats['max']:.4f}")
            print(f"    Median: {feature_stats['median']:.4f}")
            print()
    
    # Step 7: Export features
    print("Step 7: Exporting combined features...")
    print("-" * 40)
    
    # Export to CSV
    csv_file = output_dir / 'combined_ml_features.csv'
    features_dicts = combined_feature_gen.export_to_dict_list(combined_features)
    
    if features_dicts:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=features_dicts[0].keys())
            writer.writeheader()
            writer.writerows(features_dicts)
        
        print(f"  CSV exported: {csv_file}")
        print(f"  Rows: {len(features_dicts)}")
        print(f"  Columns: {len(features_dicts[0])}")
    
    # Export statistics to JSON
    stats_file = output_dir / 'feature_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Statistics exported: {stats_file}")
    
    # Export sample features for inspection
    sample_file = output_dir / 'sample_features.json'
    sample_features_export = {
        'transaction_count': len(combined_features),
        'feature_count': 68,
        'fraud_count': fraud_count,
        'anomaly_count': anomaly_count,
        'both_count': both_count,
        'sample_transactions': [
            features_dicts[i] for i in range(min(5, len(features_dicts)))
        ]
    }
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_features_export, f, indent=2)
    
    print(f"  Sample features exported: {sample_file}")
    print()
    
    # Step 8: Analyze high-risk transactions
    print("Step 8: Analyzing high-risk transactions...")
    print("-" * 40)
    
    # Find high-risk transactions
    high_risk_transactions = [
        f for f in combined_features
        if f.ensemble_fraud_probability > 0.7
    ]
    
    print(f"  High-risk transactions (>0.7 probability): {len(high_risk_transactions)}")
    
    if high_risk_transactions:
        print()
        print("  High-Risk Transaction Examples:")
        print()
        
        for i, txn in enumerate(high_risk_transactions[:3], 1):
            print(f"  Transaction {i}: {txn.transaction_id}")
            print(f"    Customer: {txn.customer_id}")
            print(f"    Is Fraud: {txn.is_fraud}")
            print(f"    Fraud Type: {txn.fraud_type or 'None'}")
            print(f"    Anomaly Type: {txn.anomaly_type or 'None'}")
            print(f"    Ensemble Probability: {txn.ensemble_fraud_probability:.3f}")
            print(f"    Risk Amplification: {txn.risk_amplification_score:.3f}")
            print(f"    High Risk Combination: {txn.high_risk_combination}")
            print(f"    Velocity (1h): {txn.txn_frequency_1h} transactions")
            print(f"    Distance from Home: {txn.distance_from_home:.1f} km")
            print(f"    Current Severity: {txn.current_severity:.3f}")
            print()
    
    # Step 9: Feature importance analysis
    print("Step 9: Feature importance analysis...")
    print("-" * 40)
    
    # Analyze which features have highest variance (potential importance)
    high_variance_features = sorted(
        [(name, stats_dict['std']) for name, stats_dict in stats.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    print("  Top 10 features by variance (potential importance):")
    print()
    for i, (feature_name, variance) in enumerate(high_variance_features, 1):
        print(f"    {i:2d}. {feature_name:40s} (std: {variance:.4f})")
    print()
    
    # Analyze interaction features specifically
    print("  Interaction Feature Statistics:")
    print()
    
    interaction_features = [
        'high_risk_combination',
        'risk_amplification_score',
        'compound_severity_score',
        'behavioral_consistency_score',
        'pattern_alignment_score',
        'conflict_indicator',
        'velocity_severity_product',
        'geographic_risk_score',
        'weighted_risk_score',
        'ensemble_fraud_probability'
    ]
    
    for feature_name in interaction_features:
        if feature_name in stats:
            feature_stats = stats[feature_name]
            print(f"    {feature_name:40s} mean: {feature_stats['mean']:.4f}")
    print()
    
    # Summary
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Generated: {len(combined_features)} transactions with 68 ML features each")
    print(f"Output directory: {output_dir}")
    print()
    print("Files created:")
    print(f"  1. {csv_file.name} - All features in CSV format")
    print(f"  2. {stats_file.name} - Feature statistics")
    print(f"  3. {sample_file.name} - Sample transactions")
    print()
    print("Feature breakdown:")
    print("  - 32 Fraud-based features (velocity, geographic, behavioral, etc.)")
    print("  - 26 Anomaly-based features (frequency, severity, persistence, etc.)")
    print("  - 10 Interaction features (risk amplification, pattern alignment, etc.)")
    print()
    print(f"High-risk detection: {len(high_risk_transactions)} transactions identified")
    print()
    print("Ready for model training!")
    print()


if __name__ == '__main__':
    main()
