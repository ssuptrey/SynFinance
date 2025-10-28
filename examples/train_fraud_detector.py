"""
Train Fraud Detection Models

This script trains fraud detection models (Random Forest and XGBoost) on SynFinance
generated data with comprehensive evaluation metrics and feature importance analysis.

Usage:
    python train_fraud_detector.py [--num-transactions 5000] [--fraud-rate 0.1]

Author: SynFinance Development Team
Version: 0.5.0
Date: October 26, 2025
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# SynFinance imports
sys.path.insert(0, os.path.abspath('..'))
from src.data_generator import DataGenerator
from src.generators.fraud_patterns import FraudPatternGenerator
from src.generators.ml_features import MLFeatureEngineer
from src.generators.ml_dataset_generator import MLDatasetGenerator

# Try XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠ XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--num-transactions', type=int, default=5000,
                       help='Number of transactions to generate (default: 5000)')
    parser.add_argument('--fraud-rate', type=float, default=0.1,
                       help='Target fraud rate (default: 0.1)')
    parser.add_argument('--output-dir', type=str, default='output/ml_training',
                       help='Output directory for results (default: output/ml_training)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    return parser.parse_args()


def generate_data(num_transactions, fraud_rate, seed=42):
    """Generate transaction data with fraud patterns."""
    print("=" * 60)
    print("STEP 1: Generating Transaction Data")
    print("=" * 60)
    
    # Calculate num customers and days
    num_customers = max(100, num_transactions // 50)
    num_days = max(30, num_transactions // (num_customers * 10))
    
    # Initialize generator
    generator = DataGenerator(
        num_customers=num_customers,
        start_date=datetime(2025, 1, 1),
        num_days=num_days
    )
    
    # Generate customers
    print(f"Generating {num_customers} customer profiles...")
    customers = generator.generate_customers()
    print(f"✓ Generated {len(customers)} customers")
    
    # Generate transactions
    print(f"\nGenerating {num_transactions} transactions...")
    transactions = generator.generate_transactions(num_transactions=num_transactions)
    print(f"✓ Generated {len(transactions)} transactions")
    
    # Inject fraud
    print(f"\nInjecting fraud patterns (target rate: {fraud_rate:.1%})...")
    fraud_gen = FraudPatternGenerator(seed=seed)
    transactions = fraud_gen.inject_fraud_patterns(
        transactions,
        customers,
        fraud_rate=fraud_rate,
        patterns=['card_cloning', 'velocity_abuse', 'geographic_impossible',
                 'merchant_collusion', 'account_takeover']
    )
    
    fraud_count = sum(1 for t in transactions if t.get('is_fraud', 0) == 1)
    actual_rate = fraud_count / len(transactions)
    
    print(f"✓ Fraud injection complete")
    print(f"  - Fraud transactions: {fraud_count}/{len(transactions)}")
    print(f"  - Actual fraud rate: {actual_rate:.2%}")
    
    return transactions, customers


def engineer_features(transactions, customers):
    """Engineer ML features from transaction data."""
    print("\n" + "=" * 60)
    print("STEP 2: Engineering ML Features")
    print("=" * 60)
    
    feature_engineer = MLFeatureEngineer()
    
    # Build transaction history
    print("Building transaction history...")
    history_lookup = {}
    for txn in transactions:
        customer_id = txn['customer_id']
        if customer_id not in history_lookup:
            history_lookup[customer_id] = []
        history_lookup[customer_id].append(txn)
    
    # Sort by timestamp
    for customer_id in history_lookup:
        history_lookup[customer_id].sort(key=lambda x: x['timestamp'])
    
    print(f"✓ Built history for {len(history_lookup)} customers")
    
    # Engineer features
    print(f"\nEngineering features for {len(transactions)} transactions...")
    features_list = []
    
    for i, txn in enumerate(transactions):
        customer = next(c for c in customers if c['customer_id'] == txn['customer_id'])
        customer_history = history_lookup[txn['customer_id']]
        txn_index = customer_history.index(txn)
        history = customer_history[:txn_index]
        
        ml_features = feature_engineer.engineer_features(
            transaction=txn,
            customer=customer,
            transaction_history=history
        )
        features_list.append(ml_features.to_dict())
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(transactions)} transactions")
    
    print(f"✓ Feature engineering complete")
    print(f"  - Features per transaction: {len(features_list[0]) - 3}")
    
    return features_list


def create_ml_dataset(features_list, seed=42):
    """Create ML-ready dataset with balancing and splitting."""
    print("\n" + "=" * 60)
    print("STEP 3: Creating ML-Ready Dataset")
    print("=" * 60)
    
    dataset_gen = MLDatasetGenerator(seed=seed)
    
    split, metadata = dataset_gen.create_ml_ready_dataset(
        features_list,
        balance_strategy='undersample',
        target_fraud_rate=0.5,
        normalize=True,
        encode_categorical=True
    )
    
    stats = split.get_stats()
    print(f"\n✓ Dataset creation complete")
    print(f"  - Train: {stats['train_size']} samples ({stats['train_fraud_rate']:.1%} fraud)")
    print(f"  - Validation: {stats['validation_size']} samples ({stats['validation_fraud_rate']:.1%} fraud)")
    print(f"  - Test: {stats['test_size']} samples ({stats['test_fraud_rate']:.1%} fraud)")
    
    return split, metadata


def prepare_arrays(dataset):
    """Convert dataset to numpy arrays."""
    exclude_cols = {'transaction_id', 'is_fraud', 'fraud_type', 'fraud_type_encoded'}
    
    if len(dataset) == 0:
        return np.array([]), np.array([]), []
    
    all_cols = set(dataset[0].keys())
    feature_cols = sorted(all_cols - exclude_cols)
    
    X = np.array([[sample.get(col, 0) for col in feature_cols] for sample in dataset])
    y = np.array([sample.get('is_fraud', 0) for sample in dataset])
    
    return X, y, feature_cols


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model."""
    print("\n" + "=" * 60)
    print("STEP 4: Training Random Forest")
    print("=" * 60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = f1_score(y_train, model.predict(X_train))
    val_score = f1_score(y_val, model.predict(X_val))
    
    print(f"✓ Training complete")
    print(f"  - Train F1-Score: {train_score:.3f}")
    print(f"  - Validation F1-Score: {val_score:.3f}")
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    if not XGBOOST_AVAILABLE:
        return None
    
    print("\n" + "=" * 60)
    print("STEP 5: Training XGBoost")
    print("=" * 60)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    train_score = f1_score(y_train, model.predict(X_train))
    val_score = f1_score(y_val, model.predict(X_val))
    
    print(f"✓ Training complete")
    print(f"  - Train F1-Score: {train_score:.3f}")
    print(f"  - Validation F1-Score: {val_score:.3f}")
    
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation."""
    print(f"\n{model_name} Test Set Evaluation:")
    print("-" * 40)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    print(f"F1-Score:          {f1:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Avg Precision:     {avg_precision:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
    
    return {
        'predictions': y_pred,
        'probabilities': y_proba,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm
    }


def plot_confusion_matrices(results, output_dir):
    """Plot confusion matrices for all models."""
    models = [k for k in results.keys() if k != 'feature_names']
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(models):
        cm = results[model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{model_name} Confusion Matrix')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_xticklabels(['Normal', 'Fraud'])
        axes[idx].set_yticklabels(['Normal', 'Fraud'])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrices to {output_dir}/confusion_matrices.png")


def plot_roc_curves(results, output_dir):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 6))
    
    for model_name in results.keys():
        if model_name == 'feature_names':
            continue
        
        y_test = results[model_name].get('y_test')
        y_proba = results[model_name]['probabilities']
        
        if y_test is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = results[model_name]['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Fraud Detection Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved ROC curves to {output_dir}/roc_curves.png")


def plot_feature_importance(model, feature_names, output_dir, top_n=15):
    """Plot feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved feature importance to {output_dir}/feature_importance.png")
    
    # Return top features
    top_features = [(feature_names[i], importances[i]) for i in indices]
    return top_features


def save_results(results, output_dir):
    """Save all results to JSON."""
    # Prepare serializable results
    save_dict = {}
    for model_name, metrics in results.items():
        if model_name == 'feature_names':
            save_dict[model_name] = metrics
            continue
        
        save_dict[model_name] = {
            'f1_score': float(metrics['f1_score']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'roc_auc': float(metrics['roc_auc']),
            'avg_precision': float(metrics['avg_precision']),
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
    
    with open(f'{output_dir}/evaluation_results.json', 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"✓ Saved results to {output_dir}/evaluation_results.json")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("SYNFINANCE FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Transactions: {args.num_transactions}")
    print(f"Fraud Rate: {args.fraud_rate:.1%}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Random Seed: {args.seed}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    np.random.seed(args.seed)
    
    # Generate data
    transactions, customers = generate_data(
        args.num_transactions,
        args.fraud_rate,
        args.seed
    )
    
    # Engineer features
    features_list = engineer_features(transactions, customers)
    
    # Create ML dataset
    split, metadata = create_ml_dataset(features_list, args.seed)
    
    # Prepare arrays
    print("\nPreparing data arrays...")
    X_train, y_train, feature_names = prepare_arrays(split.train)
    X_val, y_val, _ = prepare_arrays(split.validation)
    X_test, y_test, _ = prepare_arrays(split.test)
    print(f"✓ Data arrays ready: {X_train.shape[1]} features")
    
    # Train models
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val) if XGBOOST_AVAILABLE else None
    
    # Evaluate models
    print("\n" + "=" * 60)
    print("STEP 6: Model Evaluation")
    print("=" * 60)
    
    results = {'feature_names': feature_names}
    
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_results['y_test'] = y_test
    results['Random Forest'] = rf_results
    
    if xgb_model:
        xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        xgb_results['y_test'] = y_test
        results['XGBoost'] = xgb_results
    
    # Visualizations
    print("\n" + "=" * 60)
    print("STEP 7: Generating Visualizations")
    print("=" * 60)
    
    plot_confusion_matrices(results, args.output_dir)
    plot_roc_curves(results, args.output_dir)
    top_features = plot_feature_importance(rf_model, feature_names, args.output_dir)
    
    # Save results
    save_results(results, args.output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest Model: Random Forest")
    print(f"  - F1-Score: {rf_results['f1_score']:.4f}")
    print(f"  - ROC-AUC: {rf_results['roc_auc']:.4f}")
    
    print(f"\nTop 5 Most Important Features:")
    for feat, importance in top_features[:5]:
        print(f"  - {feat}: {importance:.4f}")
    
    print(f"\nAll results saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
