"""
Machine Learning Module

Provides comprehensive ML capabilities:
- Base model interface for all fraud detectors
- Individual models (Random Forest, XGBoost, Neural Network)
- Ensemble methods (Voting, Stacking)
- Hyperparameter optimization (Grid, Random, Bayesian)
- Feature selection (RFE, LASSO, correlation, variance)
- Model comparison and ranking
- Model registry and persistence

This module is production-ready for enterprise fraud detection systems.
Week 8 Day 3: Ensemble ML Models
"""

# Base model
from src.ml.base_model import BaseModel

# Individual models
from src.ml.models import RandomForestModel, XGBoostModel

# Ensemble models
from src.ml.ensemble import VotingEnsemble

# Optimization and registry
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
    # Base model
    'BaseModel',
    
    # Individual models
    'RandomForestModel',
    'XGBoostModel',
    
    # Ensemble
    'VotingEnsemble',
    
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
