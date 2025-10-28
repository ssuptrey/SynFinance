"""
Comprehensive Anomaly Pattern Analysis

Demonstrates complete anomaly detection and analysis pipeline:
1. Generate transactions with anomaly patterns
2. Detect and label anomalies
3. Perform correlation analysis (anomaly-fraud)
4. Analyze severity distributions
5. Detect temporal clustering
6. Generate geographic heatmaps
7. Generate ML features
8. Train Isolation Forest
9. Generate comprehensive reports

For Indian financial institutions selling synthetic fraud detection datasets.
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List
import os

from src.generators.anomaly_patterns import AnomalyPatternGenerator
from src.generators.anomaly_analysis import (
    AnomalyFraudCorrelationAnalyzer,
    SeverityDistributionAnalyzer,
    TemporalClusteringAnalyzer,
    GeographicHeatmapAnalyzer
)
from src.generators.anomaly_ml_features import (
    AnomalyMLFeatureGenerator,
    IsolationForestAnomalyDetector
)
from src.data_generator import SyntheticDataGenerator


def generate_dataset_with_anomalies(num_customers=100, transactions_per_customer=50):
    """
    Generate synthetic dataset with fraud and anomaly patterns
    
    Args:
        num_customers: Number of customers to simulate
        transactions_per_customer: Transactions per customer
        
    Returns:
        DataFrame with all transactions including anomalies
    """
    print("="*80)
    print("STEP 1: DATASET GENERATION")
    print("="*80)
    
    total_transactions = num_customers * transactions_per_customer
    print(f"Generating {total_transactions} transactions...")
    print(f"  - Customers: {num_customers}")
    print(f"  - Transactions per customer: {transactions_per_customer}")
    
    # Generate base dataset with fraud patterns
    generator = SyntheticDataGenerator(num_customers=num_customers)
    df = generator.generate_dataset(num_transactions=total_transactions)
    
    print(f"Generated {len(df)} base transactions")
    fraud_count = df[df['Fraud_Type'] != 'None'].shape[0]
    print(f"  - Fraud transactions: {fraud_count} ({fraud_count/len(df)*100:.1f}%)")
    
    # Add anomaly patterns
    print("\nDetecting anomaly patterns...")
    anomaly_gen = AnomalyPatternGenerator(anomaly_rate=0.10)
    
    transactions = df.to_dict('records')
    customer_histories = {}
    
    for i, txn in enumerate(transactions):
        customer_id = txn['Customer_ID']
        
        if customer_id not in customer_histories:
            customer_histories[customer_id] = []
        
        history = customer_histories[customer_id]
        
        # Detect anomalies
        anomaly_result = anomaly_gen.detect_anomaly_patterns(txn, history)
        txn.update(anomaly_result)
        
        customer_histories[customer_id].append(txn)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(transactions)} transactions...")
    
    df = pd.DataFrame(transactions)
    
    anomaly_count = df[df['Anomaly_Type'] != 'None'].shape[0]
    print(f"\nDetected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.1f}%)")
    
    # Anomaly type breakdown
    print("\nAnomaly Type Distribution:")
    for anomaly_type in ['BEHAVIORAL', 'GEOGRAPHIC', 'TEMPORAL', 'AMOUNT']:
        count = df[df['Anomaly_Type'] == anomaly_type].shape[0]
        print(f"  - {anomaly_type}: {count} ({count/len(df)*100:.2f}%)")
    
    return df, customer_histories


def analyze_anomaly_fraud_correlation(df):
    """
    Analyze correlation between anomalies and fraud
    
    Args:
        df: DataFrame with transactions
    """
    print("\n" + "="*80)
    print("STEP 2: ANOMALY-FRAUD CORRELATION ANALYSIS")
    print("="*80)
    
    analyzer = AnomalyFraudCorrelationAnalyzer()
    transactions = df.to_dict('records')
    
    result = analyzer.analyze_correlation(transactions)
    
    print(f"\nCorrelation Analysis Results:")
    print(f"  - Phi Coefficient: {result.phi_coefficient:.4f}")
    print(f"  - Chi-Square Statistic: {result.chi_square_stat:.4f}")
    print(f"  - P-Value: {result.p_value:.6f}")
    print(f"  - Significant: {result.is_significant}")
    
    print(f"\nContingency Table:")
    print(f"  - Both Fraud & Anomaly: {result.both_count}")
    print(f"  - Fraud Only: {result.fraud_only_count}")
    print(f"  - Anomaly Only: {result.anomaly_only_count}")
    print(f"  - Neither: {result.neither_count}")
    
    print(f"\nInterpretation:")
    if result.phi_coefficient > 0.3:
        print(f"  Strong positive correlation between fraud and anomalies")
    elif result.phi_coefficient > 0.1:
        print(f"  Moderate positive correlation between fraud and anomalies")
    else:
        print(f"  Weak correlation between fraud and anomalies")
    
    if result.is_significant:
        print(f"  Correlation is statistically significant (p < 0.05)")
    else:
        print(f"  Correlation is not statistically significant")
    
    return result


def analyze_severity_distribution(df):
    """
    Analyze anomaly severity distribution
    
    Args:
        df: DataFrame with transactions
    """
    print("\n" + "="*80)
    print("STEP 3: SEVERITY DISTRIBUTION ANALYSIS")
    print("="*80)
    
    analyzer = SeverityDistributionAnalyzer()
    transactions = df.to_dict('records')
    
    result = analyzer.analyze_severity_distribution(transactions)
    
    print(f"\nSeverity Statistics:")
    print(f"  - Mean Severity: {result.mean_severity:.4f}")
    print(f"  - Median Severity: {result.median_severity:.4f}")
    print(f"  - Std Deviation: {result.std_severity:.4f}")
    print(f"  - Min Severity: {result.min_severity:.4f}")
    print(f"  - Max Severity: {result.max_severity:.4f}")
    
    print(f"\nSeverity Bins (10 bins):")
    for i, count in enumerate(result.severity_bins):
        bin_start = i * 0.1
        bin_end = (i + 1) * 0.1
        print(f"  {bin_start:.1f}-{bin_end:.1f}: {count} transactions")
    
    print(f"\nType-Specific Severity:")
    for anomaly_type, severity in result.type_severity_means.items():
        print(f"  - {anomaly_type}: {severity:.4f}")
    
    print(f"\nOutliers (IQR method):")
    print(f"  - High severity outliers: {len(result.high_severity_outliers)}")
    if result.high_severity_outliers:
        print(f"  - Outlier severities: {result.high_severity_outliers[:5]}")
    
    return result


def analyze_temporal_clustering(df):
    """
    Analyze temporal clustering of anomalies
    
    Args:
        df: DataFrame with transactions
    """
    print("\n" + "="*80)
    print("STEP 4: TEMPORAL CLUSTERING ANALYSIS")
    print("="*80)
    
    analyzer = TemporalClusteringAnalyzer()
    transactions = df.to_dict('records')
    
    result = analyzer.analyze_temporal_clustering(transactions)
    
    print(f"\nTemporal Clustering Results:")
    print(f"  - Total Clusters: {result.cluster_count}")
    print(f"  - Average Cluster Size: {result.average_cluster_size:.2f}")
    print(f"  - Max Cluster Size: {result.max_cluster_size}")
    
    print(f"\nHourly Distribution:")
    for hour, count in sorted(result.hourly_distribution.items()):
        print(f"  Hour {hour:02d}: {count} anomalies")
    
    print(f"\nBurst Detection:")
    print(f"  - Bursts Detected: {len(result.burst_periods)}")
    if result.burst_periods:
        print(f"  - Burst Periods:")
        for period in result.burst_periods[:5]:
            print(f"    {period}")
    
    print(f"\nClustering Metrics:")
    print(f"  - Average Time Between Anomalies: {result.average_time_between_anomalies:.2f} hours")
    
    return result


def analyze_geographic_patterns(df):
    """
    Analyze geographic distribution of anomalies
    
    Args:
        df: DataFrame with transactions
    """
    print("\n" + "="*80)
    print("STEP 5: GEOGRAPHIC HEATMAP ANALYSIS")
    print("="*80)
    
    analyzer = GeographicHeatmapAnalyzer()
    transactions = df.to_dict('records')
    
    result = analyzer.analyze_geographic_distribution(transactions)
    
    print(f"\nGeographic Analysis Results:")
    print(f"  - Total Cities: {len(result.city_anomaly_counts)}")
    
    print(f"\nTop 10 Cities by Anomaly Count:")
    sorted_cities = sorted(result.city_anomaly_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (city, count) in enumerate(sorted_cities[:10], 1):
        severity = result.city_severity_averages.get(city, 0.0)
        print(f"  {i}. {city}: {count} anomalies (avg severity: {severity:.3f})")
    
    print(f"\nHigh-Risk Routes (Top 10):")
    sorted_routes = sorted(result.high_risk_routes.items(), key=lambda x: x[1], reverse=True)
    for i, (route, count) in enumerate(sorted_routes[:10], 1):
        print(f"  {i}. {route}: {count} transitions")
    
    print(f"\nDistance-Severity Correlation:")
    print(f"  - Correlation Coefficient: {result.distance_severity_correlation:.4f}")
    if abs(result.distance_severity_correlation) > 0.3:
        print(f"  - Strong correlation between distance and severity")
    elif abs(result.distance_severity_correlation) > 0.1:
        print(f"  - Moderate correlation between distance and severity")
    else:
        print(f"  - Weak correlation between distance and severity")
    
    return result


def generate_ml_features(df, customer_histories):
    """
    Generate ML features for all transactions
    
    Args:
        df: DataFrame with transactions
        customer_histories: Customer transaction histories
    """
    print("\n" + "="*80)
    print("STEP 6: ML FEATURE GENERATION")
    print("="*80)
    
    print("Initializing feature generators...")
    feature_gen = AnomalyMLFeatureGenerator()
    detector = IsolationForestAnomalyDetector(contamination=0.10, random_state=42)
    
    transactions = df.to_dict('records')
    
    # Train Isolation Forest
    print("Training Isolation Forest model...")
    isolation_scores = detector.fit_predict(transactions)
    print(f"  - Generated {len(isolation_scores)} isolation scores")
    print(f"  - Score range: [{min(isolation_scores):.3f}, {max(isolation_scores):.3f}]")
    
    # Generate ML features
    print("\nGenerating 27 ML features per transaction...")
    features_list = feature_gen.generate_features_batch(
        transactions=transactions,
        customer_histories=customer_histories,
        isolation_scores=isolation_scores
    )
    
    print(f"Generated {len(features_list)} feature vectors")
    
    # Convert to DataFrame
    features_df = pd.DataFrame([vars(f) for f in features_list])
    
    print("\nML Feature Summary:")
    print(f"  - Total Features: {features_df.shape[1]}")
    print(f"  - Frequency Features: 5")
    print(f"  - Severity Aggregates: 5")
    print(f"  - Type Distribution: 5")
    print(f"  - Persistence Metrics: 3")
    print(f"  - Cross-Pattern Features: 2")
    print(f"  - Evidence Features: 4")
    print(f"  - Unsupervised Features: 3")
    
    # Anomaly detection results
    anomaly_predictions = features_df['is_anomaly'].sum()
    print(f"\nIsolation Forest Anomaly Detection:")
    print(f"  - Predicted Anomalies: {anomaly_predictions} ({anomaly_predictions/len(features_df)*100:.1f}%)")
    print(f"  - Average Anomaly Probability: {features_df['anomaly_probability'].mean():.3f}")
    
    # Top features by importance
    print("\nTop 5 Transactions by Anomaly Probability:")
    top_anomalies = features_df.nsmallest(5, 'isolation_forest_score')
    for idx, row in top_anomalies.iterrows():
        print(f"  {row['transaction_id']}:")
        print(f"    IF Score: {row['isolation_forest_score']:.3f}")
        print(f"    Probability: {row['anomaly_probability']:.3f}")
        print(f"    Hourly Anomalies: {row['hourly_anomaly_count']}")
    
    return features_df


def save_results(df, features_df, output_dir='output'):
    """
    Save analysis results to files
    
    Args:
        df: Main dataset DataFrame
        features_df: ML features DataFrame
        output_dir: Output directory path
    """
    print("\n" + "="*80)
    print("STEP 7: SAVING RESULTS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main dataset
    main_path = os.path.join(output_dir, 'anomaly_analysis_dataset.csv')
    df.to_csv(main_path, index=False)
    print(f"Saved main dataset to {main_path}")
    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
    
    # Save ML features
    features_path = os.path.join(output_dir, 'anomaly_ml_features_complete.csv')
    features_df.to_csv(features_path, index=False)
    print(f"\nSaved ML features to {features_path}")
    print(f"  - Rows: {len(features_df)}")
    print(f"  - Columns: {len(features_df.columns)}")
    
    # Save summary statistics
    summary = {
        'dataset_info': {
            'total_transactions': len(df),
            'total_customers': df['Customer_ID'].nunique(),
            'fraud_count': int(df[df['Fraud_Type'] != 'None'].shape[0]),
            'fraud_rate': float(df[df['Fraud_Type'] != 'None'].shape[0] / len(df)),
            'anomaly_count': int(df[df['Anomaly_Type'] != 'None'].shape[0]),
            'anomaly_rate': float(df[df['Anomaly_Type'] != 'None'].shape[0] / len(df))
        },
        'anomaly_types': {
            'BEHAVIORAL': int(df[df['Anomaly_Type'] == 'BEHAVIORAL'].shape[0]),
            'GEOGRAPHIC': int(df[df['Anomaly_Type'] == 'GEOGRAPHIC'].shape[0]),
            'TEMPORAL': int(df[df['Anomaly_Type'] == 'TEMPORAL'].shape[0]),
            'AMOUNT': int(df[df['Anomaly_Type'] == 'AMOUNT'].shape[0])
        },
        'ml_features': {
            'total_features': int(features_df.shape[1]),
            'anomalies_detected': int(features_df['is_anomaly'].sum()),
            'average_anomaly_probability': float(features_df['anomaly_probability'].mean()),
            'average_severity': float(features_df['current_severity'].mean())
        }
    }
    
    summary_path = os.path.join(output_dir, 'anomaly_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary statistics to {summary_path}")


def generate_final_report(df, features_df, correlation_result, severity_result, 
                         temporal_result, geographic_result):
    """
    Generate comprehensive analysis report
    
    Args:
        df: Main dataset
        features_df: ML features dataset
        correlation_result: Correlation analysis result
        severity_result: Severity analysis result
        temporal_result: Temporal clustering result
        geographic_result: Geographic analysis result
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE ANOMALY ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nGeneration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n1. DATASET OVERVIEW")
    print(f"   - Total Transactions: {len(df):,}")
    print(f"   - Total Customers: {df['Customer_ID'].nunique():,}")
    print(f"   - Date Range: {df['Date'].min()} to {df['Date'].max()}")
    
    fraud_count = df[df['Fraud_Type'] != 'None'].shape[0]
    anomaly_count = df[df['Anomaly_Type'] != 'None'].shape[0]
    both_count = df[(df['Fraud_Type'] != 'None') & (df['Anomaly_Type'] != 'None')].shape[0]
    
    print(f"\n2. FRAUD & ANOMALY STATISTICS")
    print(f"   - Fraud Transactions: {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
    print(f"   - Anomaly Transactions: {anomaly_count:,} ({anomaly_count/len(df)*100:.2f}%)")
    print(f"   - Both Fraud & Anomaly: {both_count:,} ({both_count/len(df)*100:.2f}%)")
    
    print(f"\n3. CORRELATION ANALYSIS")
    print(f"   - Phi Coefficient: {correlation_result.phi_coefficient:.4f}")
    print(f"   - Statistical Significance: {'Yes' if correlation_result.is_significant else 'No'}")
    print(f"   - P-Value: {correlation_result.p_value:.6f}")
    
    print(f"\n4. SEVERITY ANALYSIS")
    print(f"   - Average Severity: {severity_result.mean_severity:.4f}")
    print(f"   - Severity Range: [{severity_result.min_severity:.4f}, {severity_result.max_severity:.4f}]")
    print(f"   - High Severity Outliers: {len(severity_result.high_severity_outliers)}")
    
    print(f"\n5. TEMPORAL PATTERNS")
    print(f"   - Temporal Clusters: {temporal_result.cluster_count}")
    print(f"   - Burst Periods Detected: {len(temporal_result.burst_periods)}")
    print(f"   - Peak Hour: {max(temporal_result.hourly_distribution.items(), key=lambda x: x[1])[0]}")
    
    print(f"\n6. GEOGRAPHIC PATTERNS")
    print(f"   - Cities with Anomalies: {len(geographic_result.city_anomaly_counts)}")
    print(f"   - High-Risk Routes: {len(geographic_result.high_risk_routes)}")
    print(f"   - Distance-Severity Correlation: {geographic_result.distance_severity_correlation:.4f}")
    
    print(f"\n7. ML FEATURE ENGINEERING")
    print(f"   - Total ML Features: {features_df.shape[1]}")
    print(f"   - Isolation Forest Predictions: {features_df['is_anomaly'].sum():,}")
    print(f"   - Average Anomaly Probability: {features_df['anomaly_probability'].mean():.4f}")
    
    print(f"\n8. RECOMMENDATIONS")
    if correlation_result.phi_coefficient > 0.3:
        print(f"   - Strong fraud-anomaly correlation detected - use combined detection")
    if len(severity_result.high_severity_outliers) > 0:
        print(f"   - High severity outliers present - investigate manually")
    if len(temporal_result.burst_periods) > 0:
        print(f"   - Temporal bursts detected - possible coordinated activity")
    if abs(geographic_result.distance_severity_correlation) > 0.3:
        print(f"   - Distance correlates with severity - monitor long-distance transactions")
    
    print("\n" + "="*80)


def main():
    """Main execution"""
    
    print("="*80)
    print("COMPREHENSIVE ANOMALY PATTERN ANALYSIS")
    print("Production-Ready Analysis for Indian Financial Institutions")
    print("="*80)
    print()
    
    # Generate dataset
    df, customer_histories = generate_dataset_with_anomalies(
        num_customers=100,
        transactions_per_customer=50
    )
    
    # Run analyses
    correlation_result = analyze_anomaly_fraud_correlation(df)
    severity_result = analyze_severity_distribution(df)
    temporal_result = analyze_temporal_clustering(df)
    geographic_result = analyze_geographic_patterns(df)
    
    # Generate ML features
    features_df = generate_ml_features(df, customer_histories)
    
    # Save results
    save_results(df, features_df)
    
    # Generate final report
    generate_final_report(df, features_df, correlation_result, severity_result,
                         temporal_result, geographic_result)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput Files:")
    print("  - output/anomaly_analysis_dataset.csv")
    print("  - output/anomaly_ml_features_complete.csv")
    print("  - output/anomaly_analysis_summary.json")
    print("\nNext Steps:")
    print("  1. Review anomaly patterns in output files")
    print("  2. Train supervised ML models using ML features")
    print("  3. Integrate anomaly detection into fraud pipeline")
    print("  4. Tune Isolation Forest contamination parameter")
    print("  5. Deploy to production environment")


if __name__ == '__main__':
    main()
