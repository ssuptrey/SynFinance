"""
Generate ML Features from Anomaly Patterns

Demonstrates complete ML feature engineering workflow:
1. Generate transactions with anomaly patterns
2. Calculate 27 ML features per transaction
3. Train Isolation Forest for unsupervised detection
4. Generate final ML dataset with all features

For Indian financial institutions and fraud detection systems.
"""

from datetime import datetime, timedelta
import pandas as pd
import json

from src.generators.anomaly_patterns import AnomalyPatternGenerator
from src.generators.anomaly_ml_features import (
    AnomalyMLFeatureGenerator,
    IsolationForestAnomalyDetector
)
from src.customer_generator import CustomerGenerator


def generate_sample_transactions(num_customers=10, transactions_per_customer=20):
    """
    Generate sample transactions with anomaly patterns
    
    Args:
        num_customers: Number of customers to simulate
        transactions_per_customer: Transactions per customer
        
    Returns:
        List of transaction dictionaries
    """
    print(f"Generating {num_customers * transactions_per_customer} transactions...")
    
    customer_gen = CustomerGenerator(num_customers=num_customers)
    customers = customer_gen.generate_customers()
    
    anomaly_gen = AnomalyPatternGenerator()
    transactions = []
    
    for customer in customers:
        customer_id = customer['Customer_ID']
        
        for i in range(transactions_per_customer):
            date = datetime.now() - timedelta(days=transactions_per_customer - i)
            
            transaction = {
                'Transaction_ID': f'TXN_{customer_id}_{i+1:03d}',
                'Customer_ID': customer_id,
                'Date': date.strftime('%Y-%m-%d'),
                'Hour': (date.hour + i) % 24,
                'Amount': 1000.0 + (i * 500),
                'Merchant_Category': 'Retail',
                'City': customer['City'],
                'Distance_From_Last_Txn_km': i * 2.0,
                'Time_Since_Last_Txn_hours': 24.0,
                'Fraud_Type': 'None',
                'Anomaly_Type': 'None',
                'Anomaly_Severity': 0.0,
                'Anomaly_Confidence': 0.0,
                'Anomaly_Evidence': ''
            }
            
            # Add anomaly pattern to some transactions
            if i % 5 == 0:
                anomaly_result = anomaly_gen.detect_anomalies(
                    transaction,
                    [t for t in transactions if t['Customer_ID'] == customer_id]
                )
                transaction.update(anomaly_result)
            
            transactions.append(transaction)
    
    print(f"Generated {len(transactions)} transactions")
    anomaly_count = sum(1 for t in transactions if t['Anomaly_Type'] != 'None')
    print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(transactions)*100:.1f}%)")
    
    return transactions


def build_customer_histories(transactions):
    """
    Build per-customer transaction histories
    
    Args:
        transactions: List of all transactions
        
    Returns:
        Dictionary mapping customer_id to list of transactions
    """
    print("Building customer histories...")
    
    histories = {}
    for txn in transactions:
        customer_id = txn['Customer_ID']
        if customer_id not in histories:
            histories[customer_id] = []
        histories[customer_id].append(txn)
    
    # Sort each customer's history by date
    for customer_id in histories:
        histories[customer_id].sort(key=lambda t: (t['Date'], t['Hour']))
    
    print(f"Built histories for {len(histories)} customers")
    
    return histories


def generate_ml_features(transactions, customer_histories):
    """
    Generate ML features for all transactions
    
    Args:
        transactions: List of all transactions
        customer_histories: Dictionary of customer histories
        
    Returns:
        List of AnomalyMLFeatures objects
    """
    print("Generating ML features...")
    
    feature_gen = AnomalyMLFeatureGenerator()
    detector = IsolationForestAnomalyDetector(contamination=0.1, random_state=42)
    
    # Train Isolation Forest
    print("Training Isolation Forest...")
    isolation_scores = detector.fit_predict(transactions)
    print(f"Generated {len(isolation_scores)} isolation scores")
    
    # Generate features with isolation scores
    print("Calculating 27 ML features per transaction...")
    features_list = feature_gen.generate_features_batch(
        transactions=transactions,
        customer_histories=customer_histories,
        isolation_scores=isolation_scores
    )
    
    print(f"Generated {len(features_list)} feature vectors")
    
    return features_list


def analyze_features(features_list):
    """
    Analyze and summarize ML features
    
    Args:
        features_list: List of AnomalyMLFeatures objects
    """
    print("\n" + "="*80)
    print("ML FEATURES ANALYSIS")
    print("="*80)
    
    # Convert to DataFrame
    df = pd.DataFrame([vars(f) for f in features_list])
    
    # Basic statistics
    print(f"\nDataset Shape: {df.shape[0]} transactions × {df.shape[1]} features")
    print(f"\nFeature Categories:")
    print(f"  - Frequency Features: 5")
    print(f"  - Severity Aggregates: 5")
    print(f"  - Type Distribution: 5")
    print(f"  - Persistence Metrics: 3")
    print(f"  - Cross-Pattern Features: 2")
    print(f"  - Evidence Features: 4")
    print(f"  - Unsupervised Features: 3")
    print(f"  Total: 27 features")
    
    # Anomaly statistics
    anomaly_count = df['is_anomaly'].sum()
    print(f"\nAnomaly Detection:")
    print(f"  - Detected by Isolation Forest: {anomaly_count}")
    print(f"  - Anomaly Rate: {anomaly_count/len(df)*100:.1f}%")
    
    # Frequency statistics
    print(f"\nFrequency Features:")
    print(f"  - Average hourly anomaly count: {df['hourly_anomaly_count'].mean():.2f}")
    print(f"  - Average daily anomaly count: {df['daily_anomaly_count'].mean():.2f}")
    print(f"  - Average weekly anomaly count: {df['weekly_anomaly_count'].mean():.2f}")
    print(f"  - Average trend: {df['anomaly_frequency_trend'].mean():.3f}")
    
    # Severity statistics
    print(f"\nSeverity Aggregates:")
    print(f"  - Average mean severity: {df['mean_severity_last_10'].mean():.3f}")
    print(f"  - Average max severity: {df['max_severity_last_10'].mean():.3f}")
    print(f"  - High severity rate: {df['high_severity_rate_last_10'].mean():.3f}")
    
    # Type distribution
    print(f"\nType Distribution:")
    print(f"  - Behavioral rate: {df['behavioral_anomaly_rate'].mean():.3f}")
    print(f"  - Geographic rate: {df['geographic_anomaly_rate'].mean():.3f}")
    print(f"  - Temporal rate: {df['temporal_anomaly_rate'].mean():.3f}")
    print(f"  - Amount rate: {df['amount_anomaly_rate'].mean():.3f}")
    print(f"  - Average diversity: {df['anomaly_type_diversity'].mean():.3f}")
    
    # Persistence metrics
    print(f"\nPersistence Metrics:")
    print(f"  - Average consecutive count: {df['consecutive_anomaly_count'].mean():.2f}")
    print(f"  - Max streak length: {df['anomaly_streak_length'].max()}")
    print(f"  - Average days since first: {df['days_since_first_anomaly'].mean():.1f}")
    
    # Cross-pattern features
    fraud_and_anomaly = df['is_fraud_and_anomaly'].sum()
    print(f"\nCross-Pattern Features:")
    print(f"  - Fraud + Anomaly cases: {fraud_and_anomaly}")
    print(f"  - Average correlation: {df['fraud_anomaly_correlation_score'].mean():.3f}")
    
    # Evidence features
    print(f"\nEvidence Features:")
    print(f"  - Impossible travel: {df['has_impossible_travel'].sum()}")
    print(f"  - Unusual category: {df['has_unusual_category'].sum()}")
    print(f"  - Unusual hour: {df['has_unusual_hour'].sum()}")
    print(f"  - Spending spike: {df['has_spending_spike'].sum()}")
    
    # Isolation Forest
    print(f"\nIsolation Forest:")
    print(f"  - Average score: {df['isolation_forest_score'].mean():.3f}")
    print(f"  - Min score: {df['isolation_forest_score'].min():.3f}")
    print(f"  - Max score: {df['isolation_forest_score'].max():.3f}")
    print(f"  - Average probability: {df['anomaly_probability'].mean():.3f}")
    
    # Top anomalous transactions
    print(f"\nTop 5 Most Anomalous Transactions (by Isolation Forest):")
    top_anomalies = df.nsmallest(5, 'isolation_forest_score')
    for idx, row in top_anomalies.iterrows():
        print(f"  {row['transaction_id']}:")
        print(f"    Customer: {row['customer_id']}")
        print(f"    IF Score: {row['isolation_forest_score']:.3f}")
        print(f"    Probability: {row['anomaly_probability']:.3f}")
        print(f"    Severity: {row['current_severity']:.3f}")
        print(f"    Hourly count: {row['hourly_anomaly_count']}")
    
    return df


def save_ml_dataset(features_df, output_path='output/anomaly_ml_features.csv'):
    """
    Save ML features to CSV file
    
    Args:
        features_df: DataFrame of ML features
        output_path: Output file path
    """
    print(f"\nSaving ML dataset to {output_path}...")
    features_df.to_csv(output_path, index=False)
    print(f"Saved {len(features_df)} rows with {len(features_df.columns)} columns")


def main():
    """Main execution"""
    
    print("="*80)
    print("ANOMALY ML FEATURE GENERATION DEMO")
    print("="*80)
    print()
    
    # Step 1: Generate transactions with anomaly patterns
    transactions = generate_sample_transactions(
        num_customers=10,
        transactions_per_customer=20
    )
    
    # Step 2: Build customer histories
    customer_histories = build_customer_histories(transactions)
    
    # Step 3: Generate ML features
    features_list = generate_ml_features(transactions, customer_histories)
    
    # Step 4: Analyze features
    features_df = analyze_features(features_list)
    
    # Step 5: Save dataset
    save_ml_dataset(features_df)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Review output/anomaly_ml_features.csv")
    print("  2. Use features for supervised ML training")
    print("  3. Tune Isolation Forest contamination parameter")
    print("  4. Analyze feature importance")
    print("  5. Integrate with fraud detection pipeline")


if __name__ == '__main__':
    main()
