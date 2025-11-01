"""
Base Model Interface

Defines the abstract base class for all fraud detection models.
Week 8 Day 3: Ensemble ML Models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract base class for fraud detection models
    
    All fraud detection models must implement this interface to ensure
    consistency across different model types and enable ensemble methods.
    """
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model
            model_version: Version string
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.is_trained = False
        self.training_date = None
        self.feature_names = None
        self.metrics = {}
    
    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Train the model on provided data
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional validation data tuple (X_val, y_val)
        
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud labels
        
        Args:
            X: Features to predict on
        
        Returns:
            Array of predicted labels (0 or 1)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities
        
        Args:
            X: Features to predict on
        
        Returns:
            Array of fraud probabilities
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Features to evaluate on
            y: True labels
        
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save model to disk
        
        Args:
            path: Path to save model file
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_date": self.training_date,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "is_trained": self.is_trained
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Load model from disk
        
        Args:
            path: Path to model file
        
        Returns:
            Loaded model instance
        """
        model_data = joblib.load(path)
        
        # Create new instance
        instance = cls.__new__(cls)
        instance.model_name = model_data.get("model_name", "Unknown")
        instance.model_version = model_data.get("model_version", "0.0.0")
        instance.model = model_data["model"]
        instance.hyperparameters = model_data.get("hyperparameters", {})
        instance.training_date = model_data.get("training_date")
        instance.feature_names = model_data.get("feature_names")
        instance.metrics = model_data.get("metrics", {})
        instance.is_trained = model_data.get("is_trained", False)
        
        return instance
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available
        """
        if not self.is_trained or self.feature_names is None:
            return None
        
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "training_date": self.training_date.isoformat() if self.training_date else None,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "metrics": self.metrics
        }
    
    def __repr__(self) -> str:
        """String representation of model"""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_name} v{self.model_version} ({status})"
