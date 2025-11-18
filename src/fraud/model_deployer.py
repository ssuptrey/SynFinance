"""
Model Deployer

ML model deployment, versioning, and A/B testing infrastructure.
Supports multiple deployment strategies with performance monitoring.

Week 10 Day 4: Advanced Fraud Detection
"""

import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from scipy import stats

from src.fraud import DeployedModel, ABTestConfig, DeploymentStrategy


class ModelDeployer:
    """
    ML model deployment and A/B testing framework
    
    Features:
    - Multiple deployment strategies (blue-green, canary, A/B, shadow)
    - Real-time performance monitoring (accuracy, latency, drift)
    - Automated rollback on degradation
    - Traffic splitting and gradual rollout
    - Model drift detection (KS test)
    - Champion/Challenger comparison
    
    Deployment strategies:
    1. Blue-Green: Instant switch between versions
    2. Canary: Gradual traffic shift (10% → 50% → 100%)
    3. A/B Testing: Champion vs Challenger comparison
    4. Shadow: Challenger runs without affecting decisions
    """
    
    def __init__(self):
        """Initialize model deployer"""
        # Deployed models registry
        self.models: Dict[str, DeployedModel] = {}
        
        # Active model (receives traffic)
        self.active_model_id: Optional[str] = None
        
        # A/B testing configurations
        self.ab_tests: Dict[str, ABTestConfig] = {}
        
        # Performance tracking
        self.prediction_history = defaultdict(lambda: deque(maxlen=10000))
        self.latency_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Rollback history
        self.rollback_history = []
    
    def deploy_model(
        self,
        model: Any,
        model_id: str,
        model_name: str = "",
        version: str = "1.0",
        deployment_strategy: str = 'blue_green',
        traffic_percentage: float = 100.0
    ) -> DeployedModel:
        """
        Deploy model to production
        
        Args:
            model: Model instance (must have predict() and predict_proba() methods)
            model_id: Unique model identifier
            model_name: Human-readable model name
            version: Model version
            deployment_strategy: Deployment strategy (blue_green, canary, ab_test, shadow)
            traffic_percentage: Initial traffic percentage (0-100)
            
        Returns:
            DeployedModel metadata
        """
        # Validate model has required methods
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            raise ValueError("Model must have predict() and predict_proba() methods")
        
        # Create deployment
        deployed_model = DeployedModel(
            model_id=model_id,
            model_name=model_name or model_id,
            version=version,
            deployment_strategy=DeploymentStrategy(deployment_strategy),
            deployed_at=datetime.now(),
            traffic_percentage=traffic_percentage
        )
        
        # Store model
        self.models[model_id] = deployed_model
        
        # Store model object (in production, this would be in model registry)
        setattr(deployed_model, '_model', model)
        
        # Handle deployment strategy
        if deployment_strategy == 'blue_green':
            self._deploy_blue_green(model_id)
        elif deployment_strategy == 'canary':
            self._deploy_canary(model_id)
        elif deployment_strategy == 'shadow':
            deployed_model.traffic_percentage = 0.0
        
        return deployed_model
    
    def _deploy_blue_green(self, new_model_id: str):
        """Blue-green deployment: instant switch"""
        # Switch active model
        old_model_id = self.active_model_id
        self.active_model_id = new_model_id
        
        # Set traffic
        self.models[new_model_id].traffic_percentage = 100.0
        
        if old_model_id and old_model_id in self.models:
            self.models[old_model_id].traffic_percentage = 0.0
    
    def _deploy_canary(self, new_model_id: str, initial_traffic: float = 10.0):
        """Canary deployment: gradual rollout"""
        # Start with small traffic percentage
        self.models[new_model_id].traffic_percentage = initial_traffic
        
        if self.active_model_id and self.active_model_id in self.models:
            self.models[self.active_model_id].traffic_percentage = 100.0 - initial_traffic
    
    def start_ab_test(
        self,
        champion_id: str,
        challenger_id: str,
        traffic_split: float = 0.1,
        duration_hours: int = 168,  # 1 week
        min_improvement: float = 0.02,
        min_samples: int = 1000
    ) -> ABTestConfig:
        """
        Start A/B test with champion vs challenger
        
        Args:
            champion_id: Champion model ID (current production)
            challenger_id: Challenger model ID (new model)
            traffic_split: Traffic to challenger (0-1, default: 10%)
            duration_hours: Test duration in hours
            min_improvement: Minimum improvement to promote (default: 2%)
            min_samples: Minimum samples needed
            
        Returns:
            ABTestConfig
        """
        if champion_id not in self.models or challenger_id not in self.models:
            raise ValueError("Both champion and challenger must be deployed")
        
        test_id = f"ab_test_{int(datetime.now().timestamp())}"
        
        config = ABTestConfig(
            test_id=test_id,
            champion_model_id=champion_id,
            challenger_model_id=challenger_id,
            traffic_split=traffic_split,
            started_at=datetime.now(),
            duration_hours=duration_hours,
            min_improvement=min_improvement,
            min_samples=min_samples
        )
        
        self.ab_tests[test_id] = config
        
        # Set traffic percentages
        self.models[champion_id].traffic_percentage = (1.0 - traffic_split) * 100
        self.models[challenger_id].traffic_percentage = traffic_split * 100
        
        return config
    
    def predict(
        self,
        X: np.ndarray,
        return_model_id: bool = False
    ) -> Any:
        """
        Make prediction using deployed models
        
        Routes traffic according to deployment strategy
        
        Args:
            X: Input features
            return_model_id: Whether to return model ID used
            
        Returns:
            Predictions (and optionally model ID)
        """
        # Select model based on traffic routing
        model_id = self._route_traffic()
        
        if not model_id or model_id not in self.models:
            raise ValueError("No active model available")
        
        deployed_model = self.models[model_id]
        model = getattr(deployed_model, '_model')
        
        # Measure latency
        start_time = time.time()
        predictions = model.predict(X)
        latency_ms = (time.time() - start_time) * 1000
        
        # Track latency
        self.latency_history[model_id].append(latency_ms)
        
        # Track predictions for drift detection
        for pred in predictions:
            self.prediction_history[model_id].append(float(pred))
        
        if return_model_id:
            return predictions, model_id
        
        return predictions
    
    def predict_proba(
        self,
        X: np.ndarray,
        return_model_id: bool = False
    ) -> Any:
        """Make probability predictions"""
        model_id = self._route_traffic()
        
        if not model_id or model_id not in self.models:
            raise ValueError("No active model available")
        
        deployed_model = self.models[model_id]
        model = getattr(deployed_model, '_model')
        
        # Measure latency
        start_time = time.time()
        probabilities = model.predict_proba(X)
        latency_ms = (time.time() - start_time) * 1000
        
        # Track latency
        self.latency_history[model_id].append(latency_ms)
        
        if return_model_id:
            return probabilities, model_id
        
        return probabilities
    
    def _route_traffic(self) -> Optional[str]:
        """Route traffic to models based on current configuration"""
        if not self.models:
            return None
        
        # Get models with non-zero traffic
        active_models = [
            (model_id, model.traffic_percentage)
            for model_id, model in self.models.items()
            if model.traffic_percentage > 0
        ]
        
        if not active_models:
            return self.active_model_id
        
        # Weighted random selection
        total_weight = sum(weight for _, weight in active_models)
        rand = random.uniform(0, total_weight)
        
        cumulative = 0.0
        for model_id, weight in active_models:
            cumulative += weight
            if rand <= cumulative:
                return model_id
        
        return active_models[-1][0]
    
    def monitor_model_performance(
        self,
        model_id: str,
        time_window: str = '1h',
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Get real-time model performance metrics
        
        Args:
            model_id: Model ID to monitor
            time_window: Time window (1h, 24h, 7d)
            y_true: Optional ground truth labels
            y_pred: Optional predictions
            
        Returns:
            Dictionary with performance metrics
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        deployed_model = self.models[model_id]
        metrics = {}
        
        # Latency metrics
        if model_id in self.latency_history and self.latency_history[model_id]:
            latencies = list(self.latency_history[model_id])
            metrics['latency_p50'] = float(np.percentile(latencies, 50))
            metrics['latency_p95'] = float(np.percentile(latencies, 95))
            metrics['latency_p99'] = float(np.percentile(latencies, 99))
            metrics['latency_avg'] = float(np.mean(latencies))
        
        # Accuracy metrics (if labels provided)
        if y_true is not None and y_pred is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
            
            # Update deployed model metrics
            deployed_model.accuracy = metrics['accuracy']
            deployed_model.precision = metrics['precision']
            deployed_model.recall = metrics['recall']
            deployed_model.f1_score = metrics['f1_score']
        
        # Drift detection
        drift_score = self.detect_drift(model_id)
        if drift_score is not None:
            metrics['drift_score'] = drift_score
            deployed_model.prediction_drift_score = drift_score
        
        # Traffic percentage
        metrics['traffic_percentage'] = deployed_model.traffic_percentage
        
        # Deployment info
        metrics['deployed_at'] = deployed_model.deployed_at.isoformat()
        metrics['version'] = deployed_model.version
        metrics['deployment_strategy'] = deployed_model.deployment_strategy.value
        
        return metrics
    
    def detect_drift(
        self,
        model_id: str,
        reference_predictions: Optional[np.ndarray] = None,
        threshold: float = 0.05
    ) -> Optional[float]:
        """
        Detect model drift using Kolmogorov-Smirnov test
        
        Args:
            model_id: Model ID
            reference_predictions: Reference predictions (baseline)
            threshold: P-value threshold for drift detection
            
        Returns:
            KS statistic (0-1), or None if insufficient data
        """
        if model_id not in self.prediction_history:
            return None
        
        current_predictions = list(self.prediction_history[model_id])
        
        if len(current_predictions) < 100:
            return None  # Insufficient data
        
        if reference_predictions is None:
            # Use first half as reference, second half as current
            split = len(current_predictions) // 2
            reference_predictions = current_predictions[:split]
            current_predictions = current_predictions[split:]
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(reference_predictions, current_predictions)
        
        return float(ks_statistic)
    
    def rollback_deployment(
        self,
        from_model_id: str,
        to_model_id: str,
        reason: str = ""
    ):
        """
        Rollback deployment from one model to another
        
        Args:
            from_model_id: Current model ID
            to_model_id: Previous model ID to rollback to
            reason: Reason for rollback
        """
        if from_model_id not in self.models or to_model_id not in self.models:
            raise ValueError("Both models must be deployed")
        
        # Switch traffic
        self.models[from_model_id].traffic_percentage = 0.0
        self.models[to_model_id].traffic_percentage = 100.0
        self.active_model_id = to_model_id
        
        # Record rollback
        self.rollback_history.append({
            'timestamp': datetime.now(),
            'from_model': from_model_id,
            'to_model': to_model_id,
            'reason': reason
        })
    
    def auto_rollback_if_degraded(
        self,
        model_id: str,
        previous_model_id: str,
        accuracy_threshold: float = 0.85,
        latency_threshold_ms: float = 150.0
    ) -> bool:
        """
        Automatically rollback if model performance degrades
        
        Args:
            model_id: Current model ID
            previous_model_id: Previous model to rollback to
            accuracy_threshold: Minimum acceptable accuracy
            latency_threshold_ms: Maximum acceptable latency (p95)
            
        Returns:
            True if rollback occurred
        """
        metrics = self.monitor_model_performance(model_id)
        
        # Check accuracy
        if 'accuracy' in metrics and metrics['accuracy'] < accuracy_threshold:
            self.rollback_deployment(
                model_id,
                previous_model_id,
                f"Accuracy dropped below {accuracy_threshold} ({metrics['accuracy']:.3f})"
            )
            return True
        
        # Check latency
        if 'latency_p95' in metrics and metrics['latency_p95'] > latency_threshold_ms:
            self.rollback_deployment(
                model_id,
                previous_model_id,
                f"Latency exceeded {latency_threshold_ms}ms (p95: {metrics['latency_p95']:.1f}ms)"
            )
            return True
        
        return False
    
    def promote_to_champion(
        self,
        challenger_id: str,
        stop_ab_test: bool = True
    ):
        """
        Promote challenger to champion (100% traffic)
        
        Args:
            challenger_id: Challenger model ID
            stop_ab_test: Whether to stop active A/B test
        """
        if challenger_id not in self.models:
            raise ValueError(f"Model {challenger_id} not found")
        
        # Set as active model with full traffic
        self.active_model_id = challenger_id
        self.models[challenger_id].traffic_percentage = 100.0
        
        # Reduce traffic to other models
        for model_id, model in self.models.items():
            if model_id != challenger_id:
                model.traffic_percentage = 0.0
        
        # Stop A/B tests involving this model
        if stop_ab_test:
            for test_id, config in list(self.ab_tests.items()):
                if config.challenger_model_id == challenger_id:
                    del self.ab_tests[test_id]
    
    def evaluate_ab_test(
        self,
        test_id: str,
        y_true_champion: np.ndarray,
        y_pred_champion: np.ndarray,
        y_true_challenger: np.ndarray,
        y_pred_challenger: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate A/B test results
        
        Args:
            test_id: A/B test ID
            y_true_champion: True labels for champion predictions
            y_pred_champion: Champion predictions
            y_true_challenger: True labels for challenger predictions
            y_pred_challenger: Challenger predictions
            
        Returns:
            Evaluation results with recommendation
        """
        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        config = self.ab_tests[test_id]
        
        from sklearn.metrics import accuracy_score, f1_score
        
        # Calculate metrics
        champion_accuracy = accuracy_score(y_true_champion, y_pred_champion)
        challenger_accuracy = accuracy_score(y_true_challenger, y_pred_challenger)
        
        champion_f1 = f1_score(y_true_champion, y_pred_champion, average='binary', zero_division=0)
        challenger_f1 = f1_score(y_true_challenger, y_pred_challenger, average='binary', zero_division=0)
        
        improvement = challenger_f1 - champion_f1
        
        # Check promotion criteria
        should_promote = (
            improvement >= config.min_improvement and
            len(y_pred_challenger) >= config.min_samples
        )
        
        result = {
            'test_id': test_id,
            'champion_id': config.champion_model_id,
            'challenger_id': config.challenger_model_id,
            'champion_accuracy': champion_accuracy,
            'challenger_accuracy': challenger_accuracy,
            'champion_f1': champion_f1,
            'challenger_f1': challenger_f1,
            'improvement': improvement,
            'should_promote': should_promote,
            'samples_champion': len(y_pred_champion),
            'samples_challenger': len(y_pred_challenger),
            'recommendation': "Promote challenger" if should_promote else "Keep champion"
        }
        
        return result
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get overall deployment status"""
        return {
            'total_models': len(self.models),
            'active_model': self.active_model_id,
            'active_ab_tests': len(self.ab_tests),
            'models': {
                model_id: {
                    'version': model.version,
                    'traffic_percentage': model.traffic_percentage,
                    'deployed_at': model.deployed_at.isoformat(),
                    'strategy': model.deployment_strategy.value
                }
                for model_id, model in self.models.items()
            },
            'total_rollbacks': len(self.rollback_history)
        }
