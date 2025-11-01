"""
Ensemble ML Models Demo

Demonstrates Random Forest, XGBoost, and Voting Ensemble for fraud detection.
Week 8 Day 3: Ensemble ML Models
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.ml.models.random_forest import RandomForestModel
from src.ml.models.xgboost_model import XGBoostModel
from src.ml.ensemble.voting import VotingEnsemble


def generate_fraud_dataset(n_samples: int = 5000):
    """Generate synthetic fraud detection dataset"""
    print(f"\n{'='*70}")
    print("GENERATING SYNTHETIC FRAUD DATASET")
    print(f"{'='*70}")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=2,
        weights=[0.95, 0.05],  # 95% legitimate, 5% fraud
        random_state=42,
        flip_y=0.01  # 1% label noise
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    feature_names = [f"feature_{i}" for i in range(30)]
    
    print(f"✓ Training samples: {len(X_train):,}")
    print(f"✓ Testing samples: {len(X_test):,}")
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Fraud rate (train): {y_train.mean():.1%}")
    print(f"✓ Fraud rate (test): {y_test.mean():.1%}")
    
    return X_train, X_test, y_train, y_test, feature_names


def demo_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Demonstrate Random Forest model"""
    print(f"\n{'='*70}")
    print("RANDOM FOREST MODEL")
    print(f"{'='*70}")
    
    # Initialize and train
    rf = RandomForestModel(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced"
    )
    
    print("\nTraining Random Forest...")
    metrics = rf.train(
        X_train, y_train,
        validation_data=(X_test, y_test),
        feature_names=feature_names
    )
    
    # Display metrics
    print("\n📊 Training Metrics:")
    print(f"  Accuracy:  {metrics['train_accuracy']:.4f}")
    print(f"  Precision: {metrics['train_precision']:.4f}")
    print(f"  Recall:    {metrics['train_recall']:.4f}")
    print(f"  F1 Score:  {metrics['train_f1']:.4f}")
    print(f"  ROC AUC:   {metrics['train_roc_auc']:.4f}")
    
    print("\n📊 Validation Metrics:")
    print(f"  Accuracy:  {metrics['val_accuracy']:.4f}")
    print(f"  Precision: {metrics['val_precision']:.4f}")
    print(f"  Recall:    {metrics['val_recall']:.4f}")
    print(f"  F1 Score:  {metrics['val_f1']:.4f}")
    print(f"  ROC AUC:   {metrics['val_roc_auc']:.4f}")
    
    # Feature importance
    importance = rf.get_feature_importance()
    print("\n🔍 Top 5 Most Important Features:")
    for i, (feature, score) in enumerate(list(importance.items())[:5], 1):
        print(f"  {i}. {feature}: {score:.4f}")
    
    return rf


def demo_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """Demonstrate XGBoost model"""
    print(f"\n{'='*70}")
    print("XGBOOST MODEL")
    print(f"{'='*70}")
    
    # Initialize and train
    xgb_model = XGBoostModel(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    print("\nTraining XGBoost...")
    metrics = xgb_model.train(
        X_train, y_train,
        validation_data=(X_test, y_test),
        feature_names=feature_names,
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Display metrics
    print("\n📊 Training Metrics:")
    print(f"  Accuracy:  {metrics['train_accuracy']:.4f}")
    print(f"  Precision: {metrics['train_precision']:.4f}")
    print(f"  Recall:    {metrics['train_recall']:.4f}")
    print(f"  F1 Score:  {metrics['train_f1']:.4f}")
    print(f"  ROC AUC:   {metrics['train_roc_auc']:.4f}")
    
    print("\n📊 Validation Metrics:")
    print(f"  Accuracy:  {metrics['val_accuracy']:.4f}")
    print(f"  Precision: {metrics['val_precision']:.4f}")
    print(f"  Recall:    {metrics['val_recall']:.4f}")
    print(f"  F1 Score:  {metrics['val_f1']:.4f}")
    print(f"  ROC AUC:   {metrics['val_roc_auc']:.4f}")
    
    if "best_iteration" in metrics:
        print(f"\n⚡ Early stopping at iteration: {metrics['best_iteration']}")
    
    # Feature importance
    importance = xgb_model.get_feature_importance()
    print("\n🔍 Top 5 Most Important Features:")
    for i, (feature, score) in enumerate(list(importance.items())[:5], 1):
        print(f"  {i}. {feature}: {score:.4f}")
    
    return xgb_model


def demo_voting_ensemble(rf, xgb_model, X_test, y_test):
    """Demonstrate Voting Ensemble"""
    print(f"\n{'='*70}")
    print("VOTING ENSEMBLE")
    print(f"{'='*70}")
    
    # Create soft voting ensemble
    print("\nCreating Soft Voting Ensemble...")
    soft_ensemble = VotingEnsemble(
        models=[rf, xgb_model],
        voting="soft",
        weights=[0.5, 0.5]
    )
    
    print("✓ Ensemble created with 2 models (equal weights)")
    
    # Evaluate
    metrics = soft_ensemble.evaluate(X_test, y_test)
    
    print("\n📊 Soft Voting Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Hard voting ensemble
    print("\nCreating Hard Voting Ensemble...")
    hard_ensemble = VotingEnsemble(
        models=[rf, xgb_model],
        voting="hard",
        weights=[0.5, 0.5]
    )
    
    metrics = hard_ensemble.evaluate(X_test, y_test)
    
    print("\n📊 Hard Voting Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Get individual predictions
    individual = soft_ensemble.get_model_predictions(X_test[:5])
    
    print("\n🔍 Individual Model Predictions (first 5 samples):")
    print("\n  Sample | RF Prob | XGB Prob | Ensemble Prob")
    print("  " + "-" * 48)
    
    rf_probas = individual["RandomForest_0"]["probas"][:5]
    xgb_probas = individual["XGBoost_1"]["probas"][:5]
    ensemble_probas = soft_ensemble.predict_proba(X_test[:5])
    
    for i in range(5):
        print(f"  {i+1:6} | {rf_probas[i]:7.4f} | {xgb_probas[i]:8.4f} | {ensemble_probas[i]:13.4f}")
    
    return soft_ensemble


def demo_model_persistence(rf, output_dir="output/ml_models"):
    """Demonstrate model saving and loading"""
    print(f"\n{'='*70}")
    print("MODEL PERSISTENCE")
    print(f"{'='*70}")
    
    # Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{output_dir}/random_forest_fraud_detector.pkl"
    
    print(f"\nSaving model to: {model_path}")
    rf.save(model_path)
    print("✓ Model saved successfully")
    
    # Load model
    print("\nLoading model...")
    loaded_rf = RandomForestModel.load(model_path)
    print("✓ Model loaded successfully")
    
    # Verify
    print("\n📋 Loaded Model Info:")
    info = loaded_rf.get_info()
    print(f"  Name:         {info['model_name']}")
    print(f"  Version:      {info['model_version']}")
    print(f"  Trained:      {info['is_trained']}")
    print(f"  Features:     {info['feature_count']}")
    print(f"  Training Date: {info['training_date']}")


def compare_models(rf, xgb_model, ensemble, X_test, y_test):
    """Compare all models"""
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    
    rf_metrics = rf.evaluate(X_test, y_test)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    ens_metrics = ensemble.evaluate(X_test, y_test)
    
    print("\n📊 Performance Comparison:")
    print("\n  Metric      | Random Forest | XGBoost | Ensemble")
    print("  " + "-" * 58)
    
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for metric in metrics:
        rf_val = rf_metrics[metric]
        xgb_val = xgb_metrics[metric]
        ens_val = ens_metrics[metric]
        
        # Mark best with ⭐
        best_val = max(rf_val, xgb_val, ens_val)
        rf_mark = "⭐" if rf_val == best_val else "  "
        xgb_mark = "⭐" if xgb_val == best_val else "  "
        ens_mark = "⭐" if ens_val == best_val else "  "
        
        print(f"  {metric:11} | {rf_mark}{rf_val:.4f}      | {xgb_mark}{xgb_val:.4f}  | {ens_mark}{ens_val:.4f}")


def main():
    """Run ensemble ML models demo"""
    print(f"\n{'='*70}")
    print(" 🚀 ENSEMBLE ML MODELS DEMO - Week 8 Day 3")
    print(f"{'='*70}")
    
    # Generate data
    X_train, X_test, y_train, y_test, feature_names = generate_fraud_dataset()
    
    # Train individual models
    rf = demo_random_forest(X_train, X_test, y_train, y_test, feature_names)
    xgb_model = demo_xgboost(X_train, X_test, y_train, y_test, feature_names)
    
    # Create ensemble
    ensemble = demo_voting_ensemble(rf, xgb_model, X_test, y_test)
    
    # Model persistence
    demo_model_persistence(rf)
    
    # Compare models
    compare_models(rf, xgb_model, ensemble, X_test, y_test)
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
