"""
Model Registry and Comparison Module

Provides comprehensive model management capabilities including:
- Model persistence (save/load with metadata)
- Model versioning and tracking
- Multi-model performance comparison
- Recommendation generation based on business metrics
- Production deployment readiness checks

This module is production-ready for enterprise fraud detection systems.
"""

import json
import pickle
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


@dataclass
class ModelMetadata:
    """Metadata for a registered model"""
    
    model_id: str
    model_name: str
    model_type: str
    version: str
    created_at: str
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    training_samples: int
    training_duration_seconds: float
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    author: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary"""
        return cls(**data)


@dataclass
class ModelComparisonResult:
    """Result of comparing multiple models"""
    
    model_names: List[str]
    metrics_table: pd.DataFrame
    best_model: str
    best_metrics: Dict[str, float]
    rankings: Dict[str, List[str]]  # metric -> list of model names ranked
    recommendations: List[str]
    comparison_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_names': self.model_names,
            'metrics_table': self.metrics_table.to_dict(),
            'best_model': self.best_model,
            'best_metrics': self.best_metrics,
            'rankings': self.rankings,
            'recommendations': self.recommendations,
            'comparison_timestamp': self.comparison_timestamp,
        }
    
    def get_ranking_by_metric(self, metric: str) -> List[str]:
        """Get model ranking for specific metric"""
        if metric not in self.rankings:
            raise ValueError(f"Metric '{metric}' not found in rankings")
        return self.rankings[metric]
    
    def get_model_rank(self, model_name: str, metric: str) -> int:
        """Get rank of model for specific metric (1-indexed)"""
        ranking = self.get_ranking_by_metric(metric)
        if model_name not in ranking:
            raise ValueError(f"Model '{model_name}' not found in ranking")
        return ranking.index(model_name) + 1


class ModelRegistry:
    """
    Model registry for saving, loading, and managing trained models
    
    Provides production-ready model management with:
    - Persistent storage with metadata
    - Version control
    - Model tagging and search
    - Automatic metadata tracking
    - Safe serialization/deserialization
    
    Example:
        >>> registry = ModelRegistry(base_dir="models/production")
        >>> registry.register_model(
        ...     model=trained_rf,
        ...     model_name="RandomForest_v1",
        ...     metadata=metadata
        ... )
        >>> loaded_model, metadata = registry.load_model("RandomForest_v1")
    """
    
    def __init__(self, base_dir: str = "models"):
        """
        Initialize model registry
        
        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.base_dir / "models"
        self.metadata_dir = self.base_dir / "metadata"
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Registry index
        self.index_path = self.base_dir / "registry_index.json"
        self._load_index()
    
    def _load_index(self) -> None:
        """Load registry index from disk"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                'models': {},
                'tags': {},
                'versions': {},
            }
    
    def _save_index(self) -> None:
        """Save registry index to disk"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register_model(
        self,
        model: BaseEstimator,
        model_name: str,
        metadata: ModelMetadata,
        overwrite: bool = False,
    ) -> str:
        """
        Register a trained model with metadata
        
        Args:
            model: Trained scikit-learn model
            model_name: Unique name for the model
            metadata: Model metadata
            overwrite: Whether to overwrite existing model
            
        Returns:
            Path to saved model
            
        Raises:
            ValueError: If model already exists and overwrite=False
        """
        # Check if model exists
        if model_name in self.index['models'] and not overwrite:
            raise ValueError(
                f"Model '{model_name}' already exists. "
                f"Set overwrite=True to replace."
            )
        
        # Generate paths
        model_path = self.models_dir / f"{model_name}.pkl"
        metadata_path = self.metadata_dir / f"{model_name}.json"
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update index
        self.index['models'][model_name] = {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'registered_at': datetime.now().isoformat(),
            'version': metadata.version,
            'model_type': metadata.model_type,
        }
        
        # Update tags index
        for tag in metadata.tags:
            if tag not in self.index['tags']:
                self.index['tags'][tag] = []
            if model_name not in self.index['tags'][tag]:
                self.index['tags'][tag].append(model_name)
        
        # Update version index
        if metadata.version not in self.index['versions']:
            self.index['versions'][metadata.version] = []
        if model_name not in self.index['versions'][metadata.version]:
            self.index['versions'][metadata.version].append(model_name)
        
        self._save_index()
        
        return str(model_path)
    
    def load_model(
        self,
        model_name: str,
    ) -> Tuple[BaseEstimator, ModelMetadata]:
        """
        Load a registered model with its metadata
        
        Args:
            model_name: Name of model to load
            
        Returns:
            Tuple of (model, metadata)
            
        Raises:
            ValueError: If model not found
        """
        if model_name not in self.index['models']:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_info = self.index['models'][model_name]
        
        # Load model
        with open(model_info['model_path'], 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        with open(model_info['metadata_path'], 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        return model, metadata
    
    def list_models(
        self,
        tag: Optional[str] = None,
        version: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> List[str]:
        """
        List registered models with optional filtering
        
        Args:
            tag: Filter by tag
            version: Filter by version
            model_type: Filter by model type
            
        Returns:
            List of model names
        """
        models = set(self.index['models'].keys())
        
        # Filter by tag
        if tag is not None:
            if tag in self.index['tags']:
                models &= set(self.index['tags'][tag])
            else:
                return []
        
        # Filter by version
        if version is not None:
            if version in self.index['versions']:
                models &= set(self.index['versions'][version])
            else:
                return []
        
        # Filter by model type
        if model_type is not None:
            filtered = [
                name for name in models
                if self.index['models'][name]['model_type'] == model_type
            ]
            models = set(filtered)
        
        return sorted(list(models))
    
    def get_metadata(self, model_name: str) -> ModelMetadata:
        """
        Get metadata for a registered model
        
        Args:
            model_name: Name of model
            
        Returns:
            Model metadata
            
        Raises:
            ValueError: If model not found
        """
        if model_name not in self.index['models']:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        metadata_path = self.index['models'][model_name]['metadata_path']
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        return ModelMetadata.from_dict(metadata_dict)
    
    def delete_model(self, model_name: str) -> None:
        """
        Delete a registered model
        
        Args:
            model_name: Name of model to delete
            
        Raises:
            ValueError: If model not found
        """
        if model_name not in self.index['models']:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_info = self.index['models'][model_name]
        
        # Delete files
        Path(model_info['model_path']).unlink(missing_ok=True)
        Path(model_info['metadata_path']).unlink(missing_ok=True)
        
        # Update index
        version = model_info['version']
        model_type = model_info['model_type']
        
        del self.index['models'][model_name]
        
        # Clean up tags
        for tag in list(self.index['tags'].keys()):
            if model_name in self.index['tags'][tag]:
                self.index['tags'][tag].remove(model_name)
            if not self.index['tags'][tag]:
                del self.index['tags'][tag]
        
        # Clean up versions
        if version in self.index['versions']:
            if model_name in self.index['versions'][version]:
                self.index['versions'][version].remove(model_name)
            if not self.index['versions'][version]:
                del self.index['versions'][version]
        
        self._save_index()
    
    def export_registry_report(self, output_path: str) -> None:
        """
        Export registry summary report
        
        Args:
            output_path: Path to save report
        """
        report = {
            'total_models': len(self.index['models']),
            'models': {},
            'tags': self.index['tags'],
            'versions': self.index['versions'],
            'generated_at': datetime.now().isoformat(),
        }
        
        for model_name in self.index['models']:
            metadata = self.get_metadata(model_name)
            report['models'][model_name] = {
                'version': metadata.version,
                'model_type': metadata.model_type,
                'created_at': metadata.created_at,
                'metrics': metadata.metrics,
                'tags': metadata.tags,
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)


class ModelComparison:
    """
    Compare multiple models on same dataset
    
    Provides comprehensive model comparison with:
    - Side-by-side metric comparison
    - Statistical ranking
    - Business-focused recommendations
    - Visualization support
    - Production deployment guidance
    
    Example:
        >>> comparator = ModelComparison()
        >>> comparator.add_model("RandomForest", rf_model, X_test, y_test)
        >>> comparator.add_model("XGBoost", xgb_model, X_test, y_test)
        >>> result = comparator.compare(
        ...     primary_metric='roc_auc',
        ...     business_priority='recall'
        ... )
        >>> print(result.best_model)
    """
    
    def __init__(self):
        """Initialize model comparison"""
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.metadata = {}
    
    def add_model(
        self,
        model_name: str,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a model to comparison
        
        Args:
            model_name: Unique name for model
            model: Trained model
            X_test: Test features
            y_test: Test labels
            metadata: Optional model metadata
        """
        if model_name in self.models:
            warnings.warn(f"Model '{model_name}' already exists, overwriting")
        
        self.models[model_name] = model
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None
        
        self.predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self.metrics[model_name] = metrics
        
        # Store metadata
        self.metadata[model_name] = metadata or {}
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = np.nan
            metrics['avg_precision'] = np.nan
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = float(tn)
            metrics['false_positives'] = float(fp)
            metrics['false_negatives'] = float(fn)
            metrics['true_positives'] = float(tp)
            
            # Specificity
            if tn + fp > 0:
                metrics['specificity'] = tn / (tn + fp)
            else:
                metrics['specificity'] = np.nan
        
        return metrics
    
    def compare(
        self,
        primary_metric: str = 'f1',
        business_priority: str = 'balanced',
        min_recall: float = 0.7,
        max_fpr: float = 0.1,
    ) -> ModelComparisonResult:
        """
        Compare all added models
        
        Args:
            primary_metric: Primary metric for ranking ('f1', 'roc_auc', 'precision', 'recall')
            business_priority: Business priority ('precision', 'recall', 'balanced')
            min_recall: Minimum acceptable recall (for fraud detection)
            max_fpr: Maximum acceptable false positive rate
            
        Returns:
            Comparison result with rankings and recommendations
        """
        if not self.models:
            raise ValueError("No models added for comparison")
        
        # Create metrics table
        metrics_data = []
        for model_name in sorted(self.models.keys()):
            row = {'Model': model_name}
            row.update(self.metrics[model_name])
            metrics_data.append(row)
        
        metrics_table = pd.DataFrame(metrics_data)
        metrics_table = metrics_table.set_index('Model')
        
        # Calculate rankings for each metric
        rankings = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision']:
            if metric in metrics_table.columns:
                # Sort descending (higher is better)
                sorted_models = metrics_table[metric].sort_values(ascending=False)
                rankings[metric] = sorted_models.index.tolist()
        
        # Determine best model based on primary metric
        if primary_metric not in metrics_table.columns:
            raise ValueError(f"Primary metric '{primary_metric}' not available")
        
        best_model = metrics_table[primary_metric].idxmax()
        best_metrics = metrics_table.loc[best_model].to_dict()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics_table,
            business_priority,
            min_recall,
            max_fpr,
            primary_metric,
        )
        
        return ModelComparisonResult(
            model_names=list(self.models.keys()),
            metrics_table=metrics_table,
            best_model=best_model,
            best_metrics=best_metrics,
            rankings=rankings,
            recommendations=recommendations,
            comparison_timestamp=datetime.now().isoformat(),
        )
    
    def _generate_recommendations(
        self,
        metrics_table: pd.DataFrame,
        business_priority: str,
        min_recall: float,
        max_fpr: float,
        primary_metric: str,
    ) -> List[str]:
        """Generate business-focused recommendations"""
        recommendations = []
        
        # Best overall model
        best_overall = metrics_table[primary_metric].idxmax()
        recommendations.append(
            f"Best model by {primary_metric}: {best_overall} "
            f"({metrics_table.loc[best_overall, primary_metric]:.4f})"
        )
        
        # Check recall requirement for fraud detection
        high_recall_models = metrics_table[metrics_table['recall'] >= min_recall]
        if not high_recall_models.empty:
            best_high_recall = high_recall_models['f1'].idxmax()
            recommendations.append(
                f"For fraud detection (recall >= {min_recall}): {best_high_recall} "
                f"(recall={metrics_table.loc[best_high_recall, 'recall']:.4f}, "
                f"f1={metrics_table.loc[best_high_recall, 'f1']:.4f})"
            )
        else:
            recommendations.append(
                f"WARNING: No models meet minimum recall requirement ({min_recall}). "
                f"Consider ensemble methods or threshold tuning."
            )
        
        # Check false positive rate
        if 'false_positives' in metrics_table.columns and 'true_negatives' in metrics_table.columns:
            metrics_table_copy = metrics_table.copy()
            metrics_table_copy['fpr'] = (
                metrics_table_copy['false_positives'] / 
                (metrics_table_copy['false_positives'] + metrics_table_copy['true_negatives'])
            )
            low_fpr_models = metrics_table_copy[metrics_table_copy['fpr'] <= max_fpr]
            
            if not low_fpr_models.empty:
                best_low_fpr = low_fpr_models['f1'].idxmax()
                recommendations.append(
                    f"Low false positive rate (FPR <= {max_fpr}): {best_low_fpr} "
                    f"(fpr={metrics_table_copy.loc[best_low_fpr, 'fpr']:.4f})"
                )
        
        # Business priority recommendation
        if business_priority == 'precision':
            best_precision = metrics_table['precision'].idxmax()
            recommendations.append(
                f"For precision-focused deployment: {best_precision} "
                f"(precision={metrics_table.loc[best_precision, 'precision']:.4f})"
            )
        elif business_priority == 'recall':
            best_recall = metrics_table['recall'].idxmax()
            recommendations.append(
                f"For recall-focused deployment: {best_recall} "
                f"(recall={metrics_table.loc[best_recall, 'recall']:.4f})"
            )
        else:  # balanced
            best_f1 = metrics_table['f1'].idxmax()
            recommendations.append(
                f"For balanced deployment: {best_f1} "
                f"(f1={metrics_table.loc[best_f1, 'f1']:.4f})"
            )
        
        # ROC-AUC recommendation (ranking ability)
        if 'roc_auc' in metrics_table.columns:
            best_roc = metrics_table['roc_auc'].idxmax()
            recommendations.append(
                f"Best ranking ability (ROC-AUC): {best_roc} "
                f"(roc_auc={metrics_table.loc[best_roc, 'roc_auc']:.4f})"
            )
        
        # Production deployment recommendation
        if 'roc_auc' in metrics_table.columns and 'recall' in metrics_table.columns:
            # Find model with good balance of ROC-AUC and recall
            metrics_table_copy = metrics_table.copy()
            metrics_table_copy['deployment_score'] = (
                0.4 * metrics_table_copy['roc_auc'] +
                0.4 * metrics_table_copy['recall'] +
                0.2 * metrics_table_copy['precision']
            )
            best_deployment = metrics_table_copy['deployment_score'].idxmax()
            recommendations.append(
                f"Recommended for production deployment: {best_deployment} "
                f"(balanced score={metrics_table_copy.loc[best_deployment, 'deployment_score']:.4f})"
            )
        
        return recommendations
    
    def export_comparison_report(self, output_path: str) -> None:
        """
        Export detailed comparison report
        
        Args:
            output_path: Path to save report
        """
        result = self.compare()
        
        report = {
            'comparison_timestamp': result.comparison_timestamp,
            'models_compared': result.model_names,
            'best_model': result.best_model,
            'best_metrics': result.best_metrics,
            'metrics_table': result.metrics_table.to_dict(),
            'rankings': result.rankings,
            'recommendations': result.recommendations,
            'metadata': self.metadata,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
