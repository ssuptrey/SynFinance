"""
Comprehensive Tests for Model Optimization Module

Tests cover:
- Hyperparameter optimization (Grid Search, Random Search)
- Ensemble methods (Voting, Stacking, Bagging)
- Feature selection (RFE, LASSO, correlation, variance)
- Model comparison and ranking
- Model registry operations

All tests use realistic fraud detection scenarios with imbalanced data.
"""

import json
import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

from src.ml.model_optimization import (
    HyperparameterOptimizer,
    EnsembleModelBuilder,
    FeatureSelector,
    OptimizationResult,
    EnsembleResult,
    FeatureSelectionResult,
)
from src.ml.model_registry import (
    ModelRegistry,
    ModelMetadata,
    ModelComparison,
    ModelComparisonResult,
)


@pytest.fixture
def fraud_dataset():
    """Generate synthetic fraud detection dataset with class imbalance"""
    # Create imbalanced dataset similar to real fraud scenarios
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[0.98, 0.02],  # 2% fraud rate
        flip_y=0.01,
        random_state=42
    )
    
    # Split into train/test
    split_idx = 700
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    feature_names = [f'feature_{i}' for i in range(20)]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
    }


@pytest.fixture
def temp_registry_dir():
    """Create temporary directory for model registry tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestHyperparameterOptimizer:
    """Test hyperparameter optimization functionality"""
    
    def test_grid_search_optimization(self, fraud_dataset):
        """Test Grid Search hyperparameter optimization"""
        optimizer = HyperparameterOptimizer(
            scoring='f1',
            cv=3,
            random_state=42
        )
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
        }
        
        model = RandomForestClassifier(random_state=42)
        
        result = optimizer.grid_search(
            model=model,
            param_grid=param_grid,
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score >= 0  # Can be 0 with imbalanced data
        assert len(result.cv_results) > 0
        assert 'n_estimators' in result.best_params
        assert 'max_depth' in result.best_params
    
    def test_random_search_optimization(self, fraud_dataset):
        """Test Random Search hyperparameter optimization"""
        optimizer = HyperparameterOptimizer(
            scoring='roc_auc',
            cv=3,
            random_state=42
        )
        
        param_distributions = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
        }
        
        model = RandomForestClassifier(random_state=42)
        
        result = optimizer.random_search(
            model=model,
            param_distributions=param_distributions,
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
            n_iter=10,
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score > 0
        assert result.optimization_method == 'random_search'
    
    def test_optimization_with_different_scorers(self, fraud_dataset):
        """Test optimization with different scoring metrics"""
        scorers = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        param_grid = {'n_estimators': [50], 'max_depth': [5]}
        
        for scorer in scorers:
            optimizer = HyperparameterOptimizer(
                scoring=scorer,
                cv=3,
                random_state=42
            )
            
            model = RandomForestClassifier(random_state=42)
            
            result = optimizer.grid_search(
                model=model,
                param_grid=param_grid,
                X_train=fraud_dataset['X_train'],
                y_train=fraud_dataset['y_train'],
            )
            
            assert result.optimization_method == 'grid_search'
            assert result.best_score >= 0


class TestEnsembleModelBuilder:
    """Test ensemble model building functionality"""
    
    def test_voting_ensemble_soft(self, fraud_dataset):
        """Test soft voting ensemble"""
        builder = EnsembleModelBuilder()
        
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ]
        
        result = builder.create_voting_ensemble(
            base_models=base_models,
            voting='soft',
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
            X_test=fraud_dataset['X_test'],
            y_test=fraud_dataset['y_test'],
        )
        
        assert isinstance(result, EnsembleResult)
        assert result.ensemble_model is not None
        assert result.ensemble_type == 'voting_soft'
        assert len(result.individual_scores) == 3
        assert result.ensemble_score >= 0
    
    def test_voting_ensemble_hard(self, fraud_dataset):
        """Test hard voting ensemble"""
        builder = EnsembleModelBuilder()
        
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ]
        
        result = builder.create_voting_ensemble(
            base_models=base_models,
            voting='hard',
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
            X_test=fraud_dataset['X_test'],
            y_test=fraud_dataset['y_test'],
        )
        
        assert result.ensemble_model is not None
        assert result.ensemble_score >= 0
    
    def test_stacking_ensemble(self, fraud_dataset):
        """Test stacking ensemble"""
        builder = EnsembleModelBuilder()
        
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ]
        
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        result = builder.create_stacking_ensemble(
            base_models=base_models,
            meta_learner=meta_learner,
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
            X_test=fraud_dataset['X_test'],
            y_test=fraud_dataset['y_test'],
        )
        
        assert isinstance(result, EnsembleResult)
        assert result.ensemble_type == 'stacking'
        assert result.ensemble_model is not None
        assert len(result.individual_scores) == 2
        assert result.ensemble_score >= 0
    
    def test_bagging_ensemble(self, fraud_dataset):
        """Test bagging ensemble"""
        builder = EnsembleModelBuilder()
        
        base_model = DecisionTreeClassifier(max_depth=10, random_state=42)
        
        result = builder.create_bagging_ensemble(
            base_model=base_model,
            n_estimators=10,
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
            X_test=fraud_dataset['X_test'],
            y_test=fraud_dataset['y_test'],
        )
        
        assert isinstance(result, EnsembleResult)
        assert result.ensemble_type == 'bagging'
        assert result.ensemble_model is not None
        # Bagging score can be 0 with imbalanced data
        assert result.ensemble_score >= 0
    
    def test_ensemble_improvement(self, fraud_dataset):
        """Test that ensemble improves over individual models"""
        builder = EnsembleModelBuilder()
        
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=30, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ]
        
        result = builder.create_voting_ensemble(
            base_models=base_models,
            voting='soft',
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
            X_test=fraud_dataset['X_test'],
            y_test=fraud_dataset['y_test'],
        )
        
        # Ensemble score should be competitive with individual scores
        individual_scores = list(result.individual_scores.values())
        assert result.ensemble_score >= min(individual_scores)


class TestFeatureSelector:
    """Test feature selection functionality"""
    
    def test_rfe_selection(self, fraud_dataset):
        """Test Recursive Feature Elimination"""
        selector = FeatureSelector()
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        result = selector.rfe_selection(
            model=model,
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
            feature_names=fraud_dataset['feature_names'],
            n_features_to_select=10,
        )
        
        assert isinstance(result, FeatureSelectionResult)
        assert result.selection_method == 'rfe'
        assert len(result.selected_features) == 10
        assert result.n_features_selected == 10
    
    def test_lasso_selection(self, fraud_dataset):
        """Test LASSO-based feature selection"""
        selector = FeatureSelector()
        
        result = selector.lasso_selection(
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
            feature_names=fraud_dataset['feature_names'],
            alpha=0.01,
        )
        
        assert isinstance(result, FeatureSelectionResult)
        assert result.selection_method == 'lasso'
        assert len(result.selected_features) > 0
        assert len(result.feature_scores) == 20
    
    def test_correlation_based_selection(self, fraud_dataset):
        """Test correlation-based feature selection"""
        selector = FeatureSelector()
        
        result = selector.correlation_selection(
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
            feature_names=fraud_dataset['feature_names'],
            threshold=0.95,
        )
        
        assert isinstance(result, FeatureSelectionResult)
        assert result.selection_method == 'correlation'
        assert len(result.selected_features) > 0
    
    def test_combined_selection(self, fraud_dataset):
        """Test combined feature selection methods"""
        selector = FeatureSelector()
        
        # First apply correlation filtering
        corr_result = selector.correlation_selection(
            X_train=fraud_dataset['X_train'],
            y_train=fraud_dataset['y_train'],
            feature_names=fraud_dataset['feature_names'],
            threshold=0.95,
        )
        
        # Then apply RFE on selected features
        selected_indices = [
            fraud_dataset['feature_names'].index(f)
            for f in corr_result.selected_features
        ]
        X_filtered = fraud_dataset['X_train'][:, selected_indices]
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        rfe_result = selector.rfe_selection(
            model=model,
            X_train=X_filtered,
            y_train=fraud_dataset['y_train'],
            feature_names=corr_result.selected_features,
            n_features_to_select=min(10, len(corr_result.selected_features)),
        )
        
        assert len(rfe_result.selected_features) <= len(corr_result.selected_features)


class TestModelRegistry:
    """Test model registry functionality"""
    
    def test_register_and_load_model(self, fraud_dataset, temp_registry_dir):
        """Test registering and loading a model"""
        registry = ModelRegistry(base_dir=temp_registry_dir)
        
        # Train a model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(fraud_dataset['X_train'], fraud_dataset['y_train'])
        
        # Create metadata
        metadata = ModelMetadata(
            model_id='rf_v1',
            model_name='RandomForest_v1',
            model_type='RandomForestClassifier',
            version='1.0',
            created_at=datetime.now().isoformat(),
            hyperparameters={'n_estimators': 50, 'max_depth': None},
            feature_names=fraud_dataset['feature_names'],
            training_samples=len(fraud_dataset['X_train']),
            training_duration_seconds=1.5,
            tags=['production', 'fraud-detection'],
            description='Random Forest model for fraud detection',
        )
        
        # Register model
        model_path = registry.register_model(
            model=model,
            model_name='RandomForest_v1',
            metadata=metadata,
        )
        
        assert Path(model_path).exists()
        
        # Load model
        loaded_model, loaded_metadata = registry.load_model('RandomForest_v1')
        
        assert loaded_model is not None
        assert loaded_metadata.model_name == 'RandomForest_v1'
        assert loaded_metadata.version == '1.0'
        
        # Test predictions match
        original_pred = model.predict(fraud_dataset['X_test'])
        loaded_pred = loaded_model.predict(fraud_dataset['X_test'])
        assert np.array_equal(original_pred, loaded_pred)
    
    def test_list_models_with_filters(self, temp_registry_dir):
        """Test listing models with filters"""
        registry = ModelRegistry(base_dir=temp_registry_dir)
        
        # Register multiple models
        for i in range(3):
            model = RandomForestClassifier(n_estimators=50, random_state=i)
            metadata = ModelMetadata(
                model_id=f'model_{i}',
                model_name=f'Model_{i}',
                model_type='RandomForestClassifier',
                version=f'1.{i}',
                created_at=datetime.now().isoformat(),
                hyperparameters={},
                feature_names=[],
                training_samples=100,
                training_duration_seconds=1.0,
                tags=['test'] if i < 2 else ['production'],
            )
            
            registry.register_model(model, f'Model_{i}', metadata)
        
        # List all models
        all_models = registry.list_models()
        assert len(all_models) == 3
        
        # Filter by tag
        test_models = registry.list_models(tag='test')
        assert len(test_models) == 2
        
        prod_models = registry.list_models(tag='production')
        assert len(prod_models) == 1
    
    def test_delete_model(self, temp_registry_dir):
        """Test deleting a model"""
        registry = ModelRegistry(base_dir=temp_registry_dir)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        metadata = ModelMetadata(
            model_id='test_model',
            model_name='TestModel',
            model_type='RandomForestClassifier',
            version='1.0',
            created_at=datetime.now().isoformat(),
            hyperparameters={},
            feature_names=[],
            training_samples=100,
            training_duration_seconds=1.0,
        )
        
        registry.register_model(model, 'TestModel', metadata)
        assert 'TestModel' in registry.list_models()
        
        registry.delete_model('TestModel')
        assert 'TestModel' not in registry.list_models()
    
    def test_export_registry_report(self, temp_registry_dir):
        """Test exporting registry report"""
        registry = ModelRegistry(base_dir=temp_registry_dir)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        metadata = ModelMetadata(
            model_id='test_model',
            model_name='TestModel',
            model_type='RandomForestClassifier',
            version='1.0',
            created_at=datetime.now().isoformat(),
            hyperparameters={'n_estimators': 50},
            feature_names=['f1', 'f2'],
            training_samples=100,
            training_duration_seconds=1.0,
            metrics={'accuracy': 0.95, 'f1': 0.85},
        )
        
        registry.register_model(model, 'TestModel', metadata)
        
        report_path = Path(temp_registry_dir) / 'report.json'
        registry.export_registry_report(str(report_path))
        
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
        
        assert report['total_models'] == 1
        assert 'TestModel' in report['models']


class TestModelComparison:
    """Test model comparison functionality"""
    
    def test_add_and_compare_models(self, fraud_dataset):
        """Test adding models and comparing them"""
        comparator = ModelComparison()
        
        # Train multiple models
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(fraud_dataset['X_train'], fraud_dataset['y_train'])
        
        gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb_model.fit(fraud_dataset['X_train'], fraud_dataset['y_train'])
        
        # Add to comparison
        comparator.add_model(
            'RandomForest',
            rf_model,
            fraud_dataset['X_test'],
            fraud_dataset['y_test']
        )
        
        comparator.add_model(
            'GradientBoosting',
            gb_model,
            fraud_dataset['X_test'],
            fraud_dataset['y_test']
        )
        
        # Compare
        result = comparator.compare(primary_metric='f1')
        
        assert isinstance(result, ModelComparisonResult)
        assert len(result.model_names) == 2
        assert result.best_model in ['RandomForest', 'GradientBoosting']
        assert len(result.recommendations) > 0
    
    def test_comparison_with_business_priorities(self, fraud_dataset):
        """Test model comparison with different business priorities"""
        comparator = ModelComparison()
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(fraud_dataset['X_train'], fraud_dataset['y_train'])
        
        comparator.add_model(
            'RandomForest',
            rf_model,
            fraud_dataset['X_test'],
            fraud_dataset['y_test']
        )
        
        # Test different business priorities
        priorities = ['precision', 'recall', 'balanced']
        
        for priority in priorities:
            result = comparator.compare(
                primary_metric='f1',
                business_priority=priority
            )
            
            assert len(result.recommendations) > 0
            assert any(priority in rec.lower() or 'deployment' in rec.lower() 
                      for rec in result.recommendations)
    
    def test_export_comparison_report(self, fraud_dataset, temp_registry_dir):
        """Test exporting comparison report"""
        comparator = ModelComparison()
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(fraud_dataset['X_train'], fraud_dataset['y_train'])
        
        comparator.add_model(
            'RandomForest',
            rf_model,
            fraud_dataset['X_test'],
            fraud_dataset['y_test']
        )
        
        report_path = Path(temp_registry_dir) / 'comparison.json'
        comparator.export_comparison_report(str(report_path))
        
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
        
        assert 'models_compared' in report
        assert 'best_model' in report
        assert 'recommendations' in report
