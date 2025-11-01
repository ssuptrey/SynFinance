"""
Voting Ensemble Classifier

Combines multiple fraud detection models using voting strategy.
Week 8 Day 3: Ensemble ML Models
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Literal
from datetime import datetime
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


class VotingEnsemble(BaseModel):
    """
    Voting ensemble for fraud detection
    
    Combines predictions from multiple models using:
    - Hard voting: Majority vote of predicted labels
    - Soft voting: Weighted average of predicted probabilities
    """
    
    def __init__(
        self,
        models: List[BaseModel],
        voting: Literal["hard", "soft"] = "soft",
        weights: Optional[List[float]] = None,
        model_version: str = "1.0.0"
    ):
        """
        Initialize voting ensemble
        
        Args:
            models: List of trained base models
            voting: Voting strategy ("hard" or "soft")
            weights: Optional model weights (must sum to 1)
            model_version: Version identifier
        """
        super().__init__(
            model_name="VotingEnsemble",
            model_version=model_version
        )
        
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")
        
        self.models = models
        self.voting = voting
        
        # Set up weights
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            self.weights = np.array(weights)
        else:
            # Equal weights
            self.weights = np.ones(len(models)) / len(models)
        
        self.hyperparameters = {
            "voting": voting,
            "n_models": len(models),
            "model_names": [m.model_name for m in models],
            "weights": self.weights.tolist()
        }
        
        # Ensemble is "trained" if all base models are trained
        self.is_trained = all(m.is_trained for m in models)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train all base models in ensemble
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            validation_data: Optional (X_val, y_val) tuple
            feature_names: Optional list of feature names
        
        Returns:
            Dictionary containing training metrics for each model
        """
        all_metrics = {}
        
        # Train each model
        for i, model in enumerate(self.models):
            print(f"Training {model.model_name}...")
            
            model_metrics = model.train(
                X, y,
                validation_data=validation_data,
                feature_names=feature_names
            )
            
            # Store metrics with model prefix
            prefix = f"{model.model_name}_{i}"
            all_metrics.update({f"{prefix}_{k}": v for k, v in model_metrics.items()})
        
        self.is_trained = True
        self.training_date = datetime.utcnow()
        self.feature_names = feature_names
        
        # Evaluate ensemble performance
        ensemble_train = self.evaluate(X, y)
        all_metrics.update({f"ensemble_train_{k}": v for k, v in ensemble_train.items()})
        
        if validation_data is not None:
            X_val, y_val = validation_data
            ensemble_val = self.evaluate(X_val, y_val)
            all_metrics.update({f"ensemble_val_{k}": v for k, v in ensemble_val.items()})
        
        self.metrics = all_metrics
        
        return all_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud labels using voting
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        if self.voting == "hard":
            # Hard voting: majority vote
            predictions = np.array([model.predict(X) for model in self.models])
            
            # Weighted voting
            weighted_votes = predictions.T @ self.weights
            return (weighted_votes >= 0.5).astype(int)
        
        else:
            # Soft voting: average probabilities
            probas = self.predict_proba(X)
            return (probas >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities using weighted average
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Fraud probabilities (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        # Get probabilities from each model
        probas = np.array([model.predict_proba(X) for model in self.models])
        
        # Weighted average
        weighted_proba = probas.T @ self.weights
        
        return weighted_proba
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance
        
        Args:
            X: Features (n_samples, n_features)
            y: True labels (n_samples,)
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
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
        Get average feature importance across models
        
        Returns:
            Dictionary mapping feature names to average importance
        """
        if not self.is_trained:
            return None
        
        # Collect importances from all models
        all_importances = []
        for model in self.models:
            importance = model.get_feature_importance()
            if importance is not None:
                all_importances.append(importance)
        
        if not all_importances:
            return None
        
        # Average importances
        feature_names = list(all_importances[0].keys())
        avg_importance = {}
        
        for feature in feature_names:
            scores = [imp.get(feature, 0.0) for imp in all_importances]
            avg_importance[feature] = np.mean(scores)
        
        # Sort by importance
        avg_importance = dict(
            sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return avg_importance
    
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual predictions from each model
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Dictionary mapping model names to predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        predictions = {}
        for i, model in enumerate(self.models):
            key = f"{model.model_name}_{i}"
            predictions[key] = {
                "labels": model.predict(X),
                "probas": model.predict_proba(X)
            }
        
        return predictions
