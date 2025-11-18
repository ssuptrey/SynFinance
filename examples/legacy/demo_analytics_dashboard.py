"""
Fraud Detection Analytics Demo

Demonstrates the full analytics pipeline:
1. Generate synthetic data with fraud patterns
2. Train ML model
3. Analyze feature importance
4. Generate visualizations
5. Create HTML dashboard

Run this script to see the analytics capabilities in action.
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.customer_generator import CustomerGenerator
from src.generators.transaction_core import TransactionGenerator
from src.generators.fraud_patterns import FraudPatternGenerator
from src.generators.anomaly_patterns import AnomalyPatternGenerator
from src.generators.ml_features import MLFeatureEngineer
from src.generators.anomaly_ml_features import AnomalyMLFeatureGenerator
from src.generators.combined_ml_features import CombinedMLFeatureGenerator
from src.analytics.advanced_analytics import (
    CorrelationAnalyzer,
    FeatureImportanceAnalyzer,
    ModelPerformanceAnalyzer,
    StatisticalTestAnalyzer,
)
from src.analytics.visualization import VisualizationFramework
from src.analytics.dashboard import HTMLDashboardGenerator


def main():
    """Run the full analytics demo"""
    
    print("=" * 80)
    print("FRAUD DETECTION ANALYTICS DEMO")
    print("=" * 80)
    print()
    
    # =========================================================================
    # Step 1: Generate Synthetic Data
    # =========================================================================
    print("Step 1: Generating synthetic transaction data...")
    print("-" * 80)
    
    # Generate customers
    customer_gen = CustomerGenerator()
    customers = customer_gen.generate_customers(50)
    print(f"  ✓ Generated {len(customers)} customers")
    
    # Generate transactions as DataFrame, then convert to dict records
    txn_gen = TransactionGenerator()
    df = txn_gen.generate_dataset(customers=customers, transactions_per_customer=10)
    transactions = df.to_dict('records')
    print(f"  ✓ Generated {len(transactions)} base transactions")
    
    # Create customer map for fraud/anomaly injection
    from types import SimpleNamespace
    customer_map = {}
    for c in customers:
        customer_ns = SimpleNamespace(
            Customer_ID=c.customer_id,
            city=c.city,
            digital_savviness=c.digital_savviness,
        )
        customer_map[c.customer_id] = customer_ns
    
    # Inject fraud patterns
    fraud_gen = FraudPatternGenerator()
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
    
    transactions = fraud_transactions
    fraud_count = sum(1 for t in transactions if t.get('Is_Fraud') == 1)
    print(f"  ✓ Injected fraud: {fraud_count} fraudulent ({fraud_count/len(transactions)*100:.1f}%)")
    
    # Inject anomalies
    anomaly_gen = AnomalyPatternGenerator()
    transactions = anomaly_gen.inject_anomaly_patterns(transactions, customers, anomaly_rate=0.15)
    anomaly_count = sum(1 for t in transactions if t.get('Anomaly_Type', 'None') != 'None')
    print(f"  ✓ Injected anomalies: {anomaly_count} anomalous ({anomaly_count/len(transactions)*100:.1f}%)")
    
    print()
    
    # =========================================================================
    # Step 2: Generate ML Features
    # =========================================================================
    print("Step 2: Generating ML features...")
    print("-" * 80)
    
    # Initialize feature generators
    ml_engineer = MLFeatureEngineer()
    anomaly_feature_gen = AnomalyMLFeatureGenerator()
    combined_gen = CombinedMLFeatureGenerator()
    
    # Generate fraud features (per transaction with history)
    fraud_features_list = []
    for i, transaction in enumerate(transactions):
        customer_id = transaction.get('Customer_ID')
        customer_ns = customer_map.get(customer_id)
        customer_dict = vars(customer_ns) if customer_ns else {}
        customer_history = [t for t in transactions[:i] 
                          if t.get('Customer_ID') == customer_id]
        
        fraud_features = ml_engineer.engineer_features(
            transaction, customer_dict, customer_history
        )
        fraud_features_list.append(fraud_features)
    
    print(f"  ✓ Generated fraud features: 32 features per transaction")
    
    # Generate anomaly features (per transaction with history)
    anomaly_features_list = []
    for i, transaction in enumerate(transactions):
        customer_history = [t for t in transactions[:i] 
                          if t.get('Customer_ID') == transaction.get('Customer_ID')]
        
        anomaly_features = anomaly_feature_gen.generate_features(transaction, customer_history)
        anomaly_features_list.append(anomaly_features)
    
    print(f"  ✓ Generated anomaly features: 26 features per transaction")
    
    # Convert features to dictionaries for combined generation
    from dataclasses import asdict
    fraud_dicts = [f.to_dict() for f in fraud_features_list]
    anomaly_dicts = [asdict(f) for f in anomaly_features_list]
    
    # Generate combined features with interactions
    combined_features = combined_gen.generate_batch_features(
        transactions, fraud_dicts, anomaly_dicts
    )
    print(f"  ✓ Combined features: 68 total features (32 fraud + 26 anomaly + 10 interaction)")
    
    # Prepare data for ML
    X = np.array([f.get_feature_values() for f in combined_features])
    y = np.array([f.is_fraud for f in combined_features])
    feature_names = combined_features[0].get_feature_names()
    
    print(f"  ✓ Feature matrix shape: {X.shape}")
    print(f"  ✓ Class balance: {y.sum()} fraud / {len(y) - y.sum()} normal")
    print()
    
    # =========================================================================
    # Step 3: Train ML Model
    # =========================================================================
    print("Step 3: Training fraud detection model...")
    print("-" * 80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"  ✓ Train set: {len(X_train)} samples")
    print(f"  ✓ Test set: {len(X_test)} samples")
    
    # Train RandomForest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print(f"  ✓ Model trained: RandomForest (100 trees, max_depth=10)")
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    print()
    
    # =========================================================================
    # Step 4: Correlation Analysis
    # =========================================================================
    print("Step 4: Analyzing feature correlations...")
    print("-" * 80)
    
    corr_analyzer = CorrelationAnalyzer(threshold=0.7)
    corr_result = corr_analyzer.analyze(X_train, feature_names, method='pearson')
    high_corr = corr_result.get_highly_correlated_pairs()
    
    print(f"  ✓ Correlation matrix computed: {len(feature_names)}x{len(feature_names)}")
    print(f"  ✓ High correlations (>{corr_result.threshold}): {len(high_corr)} pairs")
    
    if high_corr:
        print(f"\n  Top 5 correlated pairs:")
        for i, (f1, f2, corr) in enumerate(high_corr[:5], 1):
            print(f"    {i}. {f1} ↔ {f2}: {corr:.3f}")
    print()
    
    # =========================================================================
    # Step 5: Feature Importance Analysis
    # =========================================================================
    print("Step 5: Analyzing feature importance...")
    print("-" * 80)
    
    importance_analyzer = FeatureImportanceAnalyzer(n_repeats=5)
    importance_results = importance_analyzer.analyze_all(
        clf, X_train, y_train, X_test, y_test, feature_names
    )
    
    print(f"  ✓ Feature importance methods: {len(importance_results)}")
    for result in importance_results:
        top_5 = result.get_top_features(5)
        print(f"\n  {result.method.upper()} - Top 5 features:")
        for i, (feat, imp) in enumerate(top_5, 1):
            print(f"    {i}. {feat}: {imp:.4f}")
    print()
    
    # =========================================================================
    # Step 6: Model Performance Metrics
    # =========================================================================
    print("Step 6: Evaluating model performance...")
    print("-" * 80)
    
    perf_analyzer = ModelPerformanceAnalyzer()
    metrics = perf_analyzer.analyze(y_test, y_pred, y_pred_proba)
    
    print(f"  ✓ Accuracy:  {metrics.accuracy:.4f}")
    print(f"  ✓ Precision: {metrics.precision:.4f}")
    print(f"  ✓ Recall:    {metrics.recall:.4f}")
    print(f"  ✓ F1-Score:  {metrics.f1:.4f}")
    print(f"  ✓ ROC-AUC:   {metrics.roc_auc:.4f}")
    print(f"  ✓ Avg Prec:  {metrics.average_precision:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    {metrics.confusion_matrix[0]}")
    print(f"    {metrics.confusion_matrix[1]}")
    print()
    
    # =========================================================================
    # Step 7: Statistical Tests
    # =========================================================================
    print("Step 7: Running statistical significance tests...")
    print("-" * 80)
    
    stat_analyzer = StatisticalTestAnalyzer()
    
    # Test if fraud transactions have different ensemble probability
    ensemble_probs = np.array([f.ensemble_fraud_probability for f in combined_features])
    test_result = stat_analyzer.test_fraud_vs_normal(ensemble_probs, y)
    
    print(f"  ✓ T-test (fraud vs. normal on ensemble probability):")
    print(f"    Statistic: {test_result.statistic:.4f}")
    print(f"    P-value: {test_result.p_value:.2e}")
    print(f"    Significant: {'YES' if test_result.is_significant else 'NO'} (α=0.05)")
    print()
    
    # =========================================================================
    # Step 8: Generate Visualizations
    # =========================================================================
    print("Step 8: Generating visualizations...")
    print("-" * 80)
    
    viz = VisualizationFramework()
    output_dir = Path("output/analytics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    charts = {}
    
    # Feature distribution
    fig = viz.plot_distribution(
        ensemble_probs,
        title="Ensemble Fraud Probability Distribution",
        xlabel="Probability"
    )
    charts['ensemble_dist'] = fig
    print(f"  ✓ Distribution plot created")
    
    # Correlation heatmap (top 20 features)
    top_20_idx = np.argsort(importance_results[0].importances)[::-1][:20]
    fig = viz.plot_correlation_heatmap(
        corr_result.correlation_matrix[np.ix_(top_20_idx, top_20_idx)],
        [feature_names[i] for i in top_20_idx],
        title="Top 20 Features Correlation Heatmap"
    )
    charts['correlation_heatmap'] = fig
    print(f"  ✓ Correlation heatmap created")
    
    # Feature importance
    fig = viz.plot_feature_importance(
        feature_names,
        importance_results[0].importances,
        title=f"{importance_results[0].method.title()} Feature Importance",
        top_n=15,
        std=importance_results[0].importances_std
    )
    charts['tree_based_importance'] = fig
    print(f"  ✓ Feature importance plot created")
    
    # Confusion matrix
    fig = viz.plot_confusion_matrix(
        metrics.confusion_matrix,
        labels=['Normal', 'Fraud'],
        title="Confusion Matrix"
    )
    charts['confusion_matrix'] = fig
    print(f"  ✓ Confusion matrix created")
    
    # ROC curve
    fig = viz.plot_roc_curve(
        metrics.fpr,
        metrics.tpr,
        metrics.roc_auc,
        title="ROC Curve"
    )
    charts['roc_curve'] = fig
    print(f"  ✓ ROC curve created")
    
    # Precision-Recall curve
    fig = viz.plot_precision_recall_curve(
        metrics.precision_curve,
        metrics.recall_curve,
        metrics.average_precision,
        title="Precision-Recall Curve"
    )
    charts['pr_curve'] = fig
    print(f"  ✓ Precision-Recall curve created")
    print()
    
    # =========================================================================
    # Step 9: Generate HTML Dashboard
    # =========================================================================
    print("Step 9: Generating HTML dashboard...")
    print("-" * 80)
    
    dashboard_gen = HTMLDashboardGenerator()
    
    # Prepare dataset info
    dataset_info = {
        'total_transactions': len(transactions),
        'fraud_transactions': fraud_count,
        'fraud_rate': fraud_count / len(transactions),
        'total_features': len(feature_names),
    }
    
    # Prepare anomaly stats
    anomaly_stats = {
        'total_anomalies': anomaly_count,
        'anomaly_rate': anomaly_count / len(transactions),
        'avg_severity': np.mean([
            t.anomaly_severity for t in transactions
            if hasattr(t, 'anomaly_severity') and t.anomaly_severity is not None
        ]) if anomaly_count > 0 else 0,
        'high_severity_count': sum(
            1 for t in transactions
            if hasattr(t, 'anomaly_severity') and t.anomaly_severity and t.anomaly_severity > 0.7
        ),
    }
    
    # Generate dashboard
    dashboard_path = output_dir / "fraud_detection_dashboard.html"
    dashboard_gen.generate_dashboard(
        output_path=str(dashboard_path),
        title="Fraud Detection Analytics Dashboard",
        subtitle=f"Analysis of {len(transactions)} Transactions",
        dataset_info=dataset_info,
        model_metrics=metrics,
        importance_results=importance_results,
        correlation_results=corr_result,
        anomaly_stats=anomaly_stats,
        charts=charts,
    )
    
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("ANALYTICS COMPLETE!")
    print("=" * 80)
    print()
    print(f"Dashboard saved to: {dashboard_path}")
    print(f"Open the dashboard in your browser to explore the full analysis.")
    print()
    print("Key Findings:")
    print(f"  • Model F1-Score: {metrics.f1:.2%}")
    print(f"  • ROC-AUC: {metrics.roc_auc:.4f}")
    print(f"  • Top Feature: {importance_results[0].get_top_features(1)[0][0]}")
    print(f"  • High Correlations: {len(high_corr)} pairs")
    print()


if __name__ == "__main__":
    main()
