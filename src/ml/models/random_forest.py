"""
Random Forest Fraud Detection Model

Implements Random Forest classifier for fraud detection.
Week 8 Day 3: Ensemble ML Models
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
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


class RandomForestModel(BaseModel):
    """
    Random Forest fraud detection model
    
    Uses ensemble of decision trees with bagging for robust fraud detection.
    Good for handling non-linear relationships and feature interactions.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest model
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf node
            max_features: Number of features to consider for split
            class_weight: Weight balancing strategy
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        super().__init__(
            model_name="RandomForest",
            model_version="1.0.0"
        )
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
            "random_state": random_state
        }
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train Random Forest model
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            validation_data: Optional (X_val, y_val) tuple
            feature_names: Optional list of feature names
        
        Returns:
            Dictionary containing training metrics
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        self.training_date = datetime.utcnow()
        
        # Evaluate on training data
        train_metrics = self.evaluate(X, y)
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        
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
