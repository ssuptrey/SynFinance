"""
Machine Learning Module

Provides comprehensive ML capabilities:
- Hyperparameter optimization (Grid, Random, Bayesian)
- Ensemble methods (Voting, Stacking, Bagging)
- Feature selection (RFE, LASSO, correlation, variance)
- Model comparison and ranking
- Model registry and persistence

This module is production-ready for enterprise fraud detection systems.
"""

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

__all__ = [
    # Optimization
    'HyperparameterOptimizer',
    'EnsembleModelBuilder',
    'FeatureSelector',
    'OptimizationResult',
    'EnsembleResult',
    'FeatureSelectionResult',
    
    # Registry and Comparison
    'ModelRegistry',
    'ModelMetadata',
    'ModelComparison',
    'ModelComparisonResult',
]
