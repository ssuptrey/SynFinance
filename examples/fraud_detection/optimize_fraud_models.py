"""
Comprehensive Model Optimization Demo for SynFinance Fraud Detection

This script demonstrates the complete model optimization workflow:
1. Generate fraud dataset with combined ML features
2. Hyperparameter optimization (grid search + random search)
3. Build multiple ensemble models (voting, stacking, bagging)
4. Feature selection (RFE, LASSO, correlation)
5. Model comparison with business recommendations
6. Register best models to ModelRegistry
7. Export comparison reports and registry summary

Designed for production deployment in Indian financial markets.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from generators.combined_ml_features import CombinedFeatureGenerator
from ml.model_optimization import (
    HyperparameterOptimizer,
    EnsembleModelBuilder,
    FeatureSelector
)
from ml.model_registry import (
    ModelRegistry,
    ModelComparison,
    ModelMetadata
)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_subsection(title):
    """Print formatted subsection header"""
    print(f"\n--- {title} ---\n")


def generate_fraud_dataset(n_samples=5000, fraud_rate=0.03):
    """
    Generate comprehensive fraud dataset with all ML features
    
    Args:
        n_samples: Number of transactions to generate
        fraud_rate: Proportion of fraudulent transactions
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    print_subsection("Generating Fraud Dataset")
    print(f"Total samples: {n_samples:,}")
    print(f"Fraud rate: {fraud_rate*100:.1f}%")
    print(f"Expected fraud cases: {int(n_samples * fraud_rate):,}")
    
    generator = CombinedFeatureGenerator()
    
    # Generate data
    print("\nGenerating transactions with combined ML features...")
    transactions = generator.generate_dataset(
        n_samples=n_samples,
        fraud_rate=fraud_rate
    )
    
    print(f"Generated {len(transactions):,} transactions")
    print(f"Actual fraud cases: {sum(transactions['is_fraud']):,}")
    print(f"Actual fraud rate: {sum(transactions['is_fraud'])/len(transactions)*100:.2f}%")
    
    # Prepare features and target
    feature_columns = [col for col in transactions.columns 
                      if col not in ['transaction_id', 'is_fraud', 'customer_id', 
                                    'merchant_id', 'timestamp']]
    
    X = transactions[feature_columns].values
    y = transactions['is_fraud'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Train-test split (70-30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"  Fraud cases: {sum(y_train):,} ({sum(y_train)/len(y_train)*100:.2f}%)")
    print(f"Test set: {len(X_test):,} samples")
    print(f"  Fraud cases: {sum(y_test):,} ({sum(y_test)/len(y_test)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test, feature_columns


def optimize_hyperparameters(X_train, y_train):
    """
    Perform hyperparameter optimization using grid and random search
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Dict of optimized models
    """
    print_subsection("Hyperparameter Optimization")
    
    optimized_models = {}
    optimizer = HyperparameterOptimizer(
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    
    # 1. Optimize Random Forest with Grid Search
    print("1. Random Forest - Grid Search")
    print("   Parameter grid: n_estimators, max_depth, min_samples_split")
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_result = optimizer.grid_search(
        model=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        X_train=X_train,
        y_train=y_train
    )
    
    print(f"   Best F1 score: {rf_result.best_score:.4f}")
    print(f"   Best parameters: {rf_result.best_params}")
    print(f"   Total combinations tested: {rf_result.n_iterations}")
    
    optimized_models['random_forest'] = {
        'model': rf_result.best_estimator,
        'score': rf_result.best_score,
        'params': rf_result.best_params,
        'method': 'grid_search'
    }
    
    # 2. Optimize Gradient Boosting with Random Search
    print("\n2. Gradient Boosting - Random Search")
    print("   Parameter distributions: n_estimators, learning_rate, max_depth")
    
    gb_param_distributions = {
        'n_estimators': [100, 150, 200, 250, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    gb_result = optimizer.random_search(
        model=GradientBoostingClassifier(random_state=42),
        param_distributions=gb_param_distributions,
        X_train=X_train,
        y_train=y_train,
        n_iter=30
    )
    
    print(f"   Best F1 score: {gb_result.best_score:.4f}")
    print(f"   Best parameters: {gb_result.best_params}")
    print(f"   Iterations tested: {gb_result.n_iterations}")
    
    optimized_models['gradient_boosting'] = {
        'model': gb_result.best_estimator,
        'score': gb_result.best_score,
        'params': gb_result.best_params,
        'method': 'random_search'
    }
    
    # 3. Optimize Logistic Regression with Grid Search
    print("\n3. Logistic Regression - Grid Search")
    print("   Parameter grid: C, penalty, solver")
    
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000]
    }
    
    lr_result = optimizer.grid_search(
        model=LogisticRegression(random_state=42),
        param_grid=lr_param_grid,
        X_train=X_train,
        y_train=y_train
    )
    
    print(f"   Best F1 score: {lr_result.best_score:.4f}")
    print(f"   Best parameters: {lr_result.best_params}")
    print(f"   Total combinations tested: {lr_result.n_iterations}")
    
    optimized_models['logistic_regression'] = {
        'model': lr_result.best_estimator,
        'score': lr_result.best_score,
        'params': lr_result.best_params,
        'method': 'grid_search'
    }
    
    print("\n" + "-"*80)
    print("Hyperparameter Optimization Summary:")
    for name, info in optimized_models.items():
        print(f"  {name:20s}: F1={info['score']:.4f} ({info['method']})")
    
    return optimized_models


def build_ensembles(optimized_models, X_train, y_train, X_test, y_test):
    """
    Build various ensemble models
    
    Args:
        optimized_models: Dict of optimized base models
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Dict of ensemble results
    """
    print_subsection("Ensemble Model Building")
    
    ensemble_builder = EnsembleModelBuilder()
    ensemble_results = {}
    
    # Prepare base models
    base_models = [
        ('rf', optimized_models['random_forest']['model']),
        ('gb', optimized_models['gradient_boosting']['model']),
        ('lr', optimized_models['logistic_regression']['model'])
    ]
    
    # 1. Soft Voting Ensemble
    print("1. Soft Voting Ensemble")
    print("   Combines probability predictions from all base models")
    
    voting_soft_result = ensemble_builder.create_voting_ensemble(
        base_models=base_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        voting='soft'
    )
    
    print(f"   Ensemble F1 score: {voting_soft_result.ensemble_score:.4f}")
    print(f"   Base model scores: {[f'{s:.4f}' for s in voting_soft_result.individual_scores]}")
    print(f"   Improvement: {voting_soft_result.improvement:.4f}")
    
    ensemble_results['voting_soft'] = voting_soft_result
    
    # 2. Hard Voting Ensemble
    print("\n2. Hard Voting Ensemble")
    print("   Uses majority vote from predictions")
    
    voting_hard_result = ensemble_builder.create_voting_ensemble(
        base_models=base_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        voting='hard'
    )
    
    print(f"   Ensemble F1 score: {voting_hard_result.ensemble_score:.4f}")
    print(f"   Improvement: {voting_hard_result.improvement:.4f}")
    
    ensemble_results['voting_hard'] = voting_hard_result
    
    # 3. Stacking Ensemble
    print("\n3. Stacking Ensemble")
    print("   Uses meta-learner (Logistic Regression) on base model predictions")
    
    stacking_result = ensemble_builder.create_stacking_ensemble(
        base_models=base_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta_learner=LogisticRegression(random_state=42, max_iter=1000)
    )
    
    print(f"   Ensemble F1 score: {stacking_result.ensemble_score:.4f}")
    print(f"   Improvement: {stacking_result.improvement:.4f}")
    
    ensemble_results['stacking'] = stacking_result
    
    # 4. Bagging Ensemble
    print("\n4. Bagging Ensemble (Random Forest)")
    print("   Bootstrap aggregating with 100 estimators")
    
    bagging_result = ensemble_builder.create_bagging_ensemble(
        base_model=DecisionTreeClassifier(max_depth=20, random_state=42),
        n_estimators=100,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    print(f"   Ensemble F1 score: {bagging_result.ensemble_score:.4f}")
    print(f"   Base model score: {bagging_result.individual_scores[0]:.4f}")
    print(f"   Improvement: {bagging_result.improvement:.4f}")
    
    ensemble_results['bagging'] = bagging_result
    
    print("\n" + "-"*80)
    print("Ensemble Model Summary:")
    for name, result in ensemble_results.items():
        print(f"  {name:20s}: F1={result.ensemble_score:.4f} (improvement={result.improvement:+.4f})")
    
    return ensemble_results


def perform_feature_selection(X_train, y_train, feature_names):
    """
    Perform feature selection using multiple methods
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
        
    Returns:
        Dict of feature selection results
    """
    print_subsection("Feature Selection")
    
    selector = FeatureSelector()
    selection_results = {}
    
    # Base model for RFE
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 1. Recursive Feature Elimination (RFE)
    print("1. Recursive Feature Elimination (RFE)")
    print(f"   Selecting top 20 features from {len(feature_names)}")
    
    rfe_result = selector.rfe_selection(
        model=base_model,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        n_features_to_select=20,
        step=1
    )
    
    print(f"   Selected {len(rfe_result.selected_features)} features")
    print(f"   Top 10 features by rank:")
    for i, (feat, rank) in enumerate(zip(rfe_result.selected_features[:10], 
                                          rfe_result.feature_rankings[:10]), 1):
        print(f"     {i:2d}. {feat:30s} (rank={rank})")
    
    selection_results['rfe'] = rfe_result
    
    # 2. LASSO Feature Selection
    print("\n2. LASSO (L1 Regularization) Feature Selection")
    print("   Alpha=0.001 (regularization strength)")
    
    lasso_result = selector.lasso_selection(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        alpha=0.001
    )
    
    print(f"   Selected {len(lasso_result.selected_features)} features")
    print(f"   Top 10 features by importance:")
    for i, (feat, imp) in enumerate(zip(lasso_result.selected_features[:10], 
                                         lasso_result.feature_importances[:10]), 1):
        print(f"     {i:2d}. {feat:30s} (importance={imp:.4f})")
    
    selection_results['lasso'] = lasso_result
    
    # 3. Correlation-Based Selection
    print("\n3. Correlation-Based Feature Selection")
    print("   Removing highly correlated features (threshold=0.9)")
    
    corr_result = selector.correlation_selection(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        threshold=0.9
    )
    
    print(f"   Selected {len(corr_result.selected_features)} features")
    print(f"   Removed {len(feature_names) - len(corr_result.selected_features)} correlated features")
    
    selection_results['correlation'] = corr_result
    
    # 4. Combined Selection (intersection of all methods)
    print("\n4. Combined Feature Selection")
    print("   Features selected by ALL methods")
    
    rfe_set = set(rfe_result.selected_features)
    lasso_set = set(lasso_result.selected_features)
    corr_set = set(corr_result.selected_features)
    
    combined_features = list(rfe_set & lasso_set & corr_set)
    
    print(f"   RFE selected: {len(rfe_set)} features")
    print(f"   LASSO selected: {len(lasso_set)} features")
    print(f"   Correlation selected: {len(corr_set)} features")
    print(f"   Combined (intersection): {len(combined_features)} features")
    
    if combined_features:
        print(f"\n   Combined feature set:")
        for i, feat in enumerate(sorted(combined_features), 1):
            print(f"     {i:2d}. {feat}")
    
    selection_results['combined'] = combined_features
    
    return selection_results


def compare_models(optimized_models, ensemble_results, X_test, y_test, feature_names):
    """
    Compare all models and generate business recommendations
    
    Args:
        optimized_models: Dict of optimized base models
        ensemble_results: Dict of ensemble results
        X_test, y_test: Test data
        feature_names: List of feature names
        
    Returns:
        ModelComparison instance with results
    """
    print_subsection("Model Comparison & Business Recommendations")
    
    comparison = ModelComparison()
    
    # Add base models
    print("Adding models to comparison:")
    
    for name, info in optimized_models.items():
        metadata = ModelMetadata(
            model_id=f"{name}_optimized",
            model_name=name.replace('_', ' ').title(),
            model_type=name,
            version='1.0',
            created_at=datetime.now(),
            hyperparameters=info['params'],
            feature_names=feature_names,
            training_samples=len(X_test) * 7 // 3,  # Approximate from 70-30 split
            training_duration_seconds=0.0,
            metrics={},
            tags=['optimized', info['method']],
            description=f"Optimized using {info['method']}",
            author="SynFinance Model Optimizer"
        )
        
        comparison.add_model(
            model_name=name,
            model=info['model'],
            X_test=X_test,
            y_test=y_test,
            metadata=metadata
        )
        print(f"  Added: {name}")
    
    # Add ensemble models
    for name, result in ensemble_results.items():
        metadata = ModelMetadata(
            model_id=f"ensemble_{name}",
            model_name=f"Ensemble {name.replace('_', ' ').title()}",
            model_type='ensemble',
            version='1.0',
            created_at=datetime.now(),
            hyperparameters={},
            feature_names=feature_names,
            training_samples=len(X_test) * 7 // 3,
            training_duration_seconds=0.0,
            metrics={},
            tags=['ensemble', result.ensemble_type],
            description=f"{result.ensemble_type} ensemble model",
            author="SynFinance Model Optimizer"
        )
        
        comparison.add_model(
            model_name=f"ensemble_{name}",
            model=result.ensemble_model,
            X_test=X_test,
            y_test=y_test,
            metadata=metadata
        )
        print(f"  Added: ensemble_{name}")
    
    # Run comparison
    print(f"\nComparing {len(comparison.models)} models...")
    
    result = comparison.compare(
        primary_metric='f1',
        business_priority='balanced',
        min_recall=0.70,
        max_fpr=0.10
    )
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    print(f"\nBest model: {result.best_model}")
    print(f"\nBest model metrics:")
    for metric, value in result.best_metrics.items():
        print(f"  {metric:15s}: {value:.4f}")
    
    print(f"\n\nFull metrics comparison:")
    print(result.metrics_table)
    
    print(f"\n\nModel rankings (by F1 score):")
    for i, (model, score) in enumerate(result.rankings, 1):
        print(f"  {i}. {model:30s}: {score:.4f}")
    
    print(f"\n\nBUSINESS RECOMMENDATIONS:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    
    return comparison, result


def register_models(comparison, registry_dir='models'):
    """
    Register top models to ModelRegistry
    
    Args:
        comparison: ModelComparison instance
        registry_dir: Directory for model registry
        
    Returns:
        ModelRegistry instance
    """
    print_subsection(f"Model Registration (Registry: {registry_dir}/)")
    
    registry = ModelRegistry(base_dir=registry_dir)
    
    # Register top 3 models
    result = comparison.compare(primary_metric='f1')
    top_models = result.rankings[:3]
    
    print(f"Registering top 3 models:")
    
    for rank, (model_name, score) in enumerate(top_models, 1):
        model_info = comparison.models[model_name]
        
        # Register model
        path = registry.register_model(
            model=model_info['model'],
            model_name=model_name,
            metadata=model_info['metadata'],
            overwrite=True
        )
        
        print(f"\n{rank}. {model_name}")
        print(f"   F1 Score: {score:.4f}")
        print(f"   Saved to: {path}")
        print(f"   Model type: {model_info['metadata'].model_type}")
        print(f"   Tags: {', '.join(model_info['metadata'].tags)}")
    
    # List all registered models
    print(f"\n\nAll registered models:")
    all_models = registry.list_models()
    for i, name in enumerate(all_models, 1):
        metadata = registry.get_metadata(name)
        print(f"  {i}. {name:30s} (v{metadata.version}, {metadata.model_type})")
    
    return registry


def export_reports(comparison, registry, output_dir='output/optimization'):
    """
    Export all comparison and registry reports
    
    Args:
        comparison: ModelComparison instance
        registry: ModelRegistry instance
        output_dir: Output directory for reports
    """
    print_subsection(f"Exporting Reports (Directory: {output_dir}/)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export comparison report
    comparison_path = os.path.join(output_dir, 'model_comparison_report.txt')
    comparison.export_comparison_report(comparison_path)
    print(f"1. Model Comparison Report: {comparison_path}")
    print(f"   Size: {os.path.getsize(comparison_path):,} bytes")
    
    # Export registry report
    registry_path = os.path.join(output_dir, 'model_registry_report.txt')
    registry.export_registry_report(registry_path)
    print(f"\n2. Model Registry Report: {registry_path}")
    print(f"   Size: {os.path.getsize(registry_path):,} bytes")
    
    print(f"\nAll reports exported to: {output_dir}/")


def main():
    """
    Main execution function
    """
    print("="*80)
    print("  SynFinance Fraud Detection - Model Optimization Pipeline")
    print("  Production-Ready System for Indian Financial Markets")
    print("="*80)
    print(f"\nExecution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Generate Dataset
    print_section("STEP 1: Dataset Generation")
    X_train, X_test, y_train, y_test, feature_names = generate_fraud_dataset(
        n_samples=5000,
        fraud_rate=0.03
    )
    
    # Step 2: Hyperparameter Optimization
    print_section("STEP 2: Hyperparameter Optimization")
    optimized_models = optimize_hyperparameters(X_train, y_train)
    
    # Step 3: Ensemble Building
    print_section("STEP 3: Ensemble Model Building")
    ensemble_results = build_ensembles(
        optimized_models, X_train, y_train, X_test, y_test
    )
    
    # Step 4: Feature Selection
    print_section("STEP 4: Feature Selection")
    selection_results = perform_feature_selection(X_train, y_train, feature_names)
    
    # Step 5: Model Comparison
    print_section("STEP 5: Model Comparison & Business Recommendations")
    comparison, result = compare_models(
        optimized_models, ensemble_results, X_test, y_test, feature_names
    )
    
    # Step 6: Model Registration
    print_section("STEP 6: Model Registration")
    registry = register_models(comparison, registry_dir='models')
    
    # Step 7: Export Reports
    print_section("STEP 7: Export Reports")
    export_reports(comparison, registry, output_dir='output/optimization')
    
    # Final Summary
    print_section("PIPELINE COMPLETE")
    print("Summary:")
    print(f"  Dataset: {len(X_train) + len(X_test):,} transactions")
    print(f"  Features: {len(feature_names)}")
    print(f"  Optimized models: {len(optimized_models)}")
    print(f"  Ensemble models: {len(ensemble_results)}")
    print(f"  Best model: {result.best_model}")
    print(f"  Best F1 score: {result.best_metrics['f1']:.4f}")
    print(f"  Registered models: {len(registry.list_models())}")
    print(f"\nExecution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
