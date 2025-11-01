"""
XGBoost Fraud Detection Model

Implements XGBoost gradient boosting classifier for fraud detection.
Week 8 Day 3: Ensemble ML Models
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from src.ml.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost fraud detection model
    
    Uses gradient boosting with advanced regularization and handling
    of imbalanced data for fraud detection.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            min_child_weight: Minimum sum of instance weight in child
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            scale_pos_weight: Balance of positive/negative weights
            random_state: Random seed
        """
        super().__init__(
            model_name="XGBoost",
            model_version="1.0.0"
        )
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            tree_method="hist",
            eval_metric="logloss"
        )
        
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state
        }
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 20,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            validation_data: Optional (X_val, y_val) tuple
            feature_names: Optional list of feature names
            early_stopping_rounds: Stop if no improvement
            verbose: Print training progress
        
        Returns:
            Dictionary containing training metrics
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Set up early stopping if validation data provided
        fit_params = {}
        if validation_data is not None:
            X_val, y_val = validation_data
            # New XGBoost API uses eval_set without early_stopping_rounds parameter
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = verbose
            
            # Set early stopping attribute on model before training
            if early_stopping_rounds:
                self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        
        # Train model
        self.model.fit(X, y, **fit_params)
        self.is_trained = True
        self.training_date = datetime.utcnow()
        
        # Evaluate on training data
        train_metrics = self.evaluate(X, y)
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        
        # Add best iteration info
        if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None:
            train_metrics["best_iteration"] = int(self.model.best_iteration)
        
        # Evaluate on validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self.evaluate(X_val, y_val)
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
            train_metrics.update(val_metrics)
        
        # Store metrics
        self.metrics = train_metrics
        
        return train_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud labels
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Fraud probabilities (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Return probability of fraud class (class 1)
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Features (n_samples, n_features)
            y: True labels (n_samples,)
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
            "pr_auc": average_precision_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)
            
            # Specificity
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.feature_names is None:
            return None
        
        importances = self.model.feature_importances_
        
        # Sort by importance
        feature_importance = dict(zip(self.feature_names, importances))
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return feature_importance
