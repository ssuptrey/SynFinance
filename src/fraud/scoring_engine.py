"""
Fraud Scoring Engine

Real-time fraud risk scoring with multi-factor analysis.
Combines multiple scoring components with configurable weights.

Week 10 Day 4: Advanced Fraud Detection
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.fraud import RiskScore, RiskFactor


class ScoringComponent(ABC):
    """Base class for scoring components"""
    
    def __init__(self, name: str, weight: float = 1.0, enabled: bool = True):
        """
        Initialize scoring component
        
        Args:
            name: Component name
            weight: Component weight (0-1)
            enabled: Whether component is active
        """
        self.name = name
        self.weight = weight
        self.enabled = enabled
    
    @abstractmethod
    def calculate_score(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RiskFactor:
        """
        Calculate risk score for this component
        
        Args:
            transaction: Transaction data
            context: Additional context (customer profile, historical data, etc.)
            
        Returns:
            RiskFactor with score (0-100) and explanation
        """
        pass
    
    def is_applicable(self, transaction: Dict[str, Any]) -> bool:
        """Check if component applies to transaction"""
        return self.enabled


class AmountAnomalyComponent(ScoringComponent):
    """Score based on transaction amount anomaly"""
    
    def __init__(self, weight: float = 0.2):
        super().__init__("Amount Anomaly", weight)
    
    def calculate_score(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RiskFactor:
        amount = transaction.get('amount', 0.0)
        
        # Get customer's average transaction amount
        avg_amount = context.get('avg_amount', 100.0)
        std_amount = context.get('std_amount', 50.0)
        
        # Calculate Z-score
        if std_amount > 0:
            z_score = abs((amount - avg_amount) / std_amount)
        else:
            z_score = 0.0
        
        # Convert Z-score to risk score (0-100)
        # Z > 3 = high risk, Z > 2 = medium, Z > 1 = low
        if z_score > 3:
            score = min(100, 70 + (z_score - 3) * 10)
            severity = "critical"
            explanation = f"Amount ${amount:.2f} is {z_score:.1f} std deviations above average ${avg_amount:.2f}"
        elif z_score > 2:
            score = 50 + (z_score - 2) * 20
            severity = "high"
            explanation = f"Amount ${amount:.2f} is {z_score:.1f} std deviations above average"
        elif z_score > 1:
            score = 20 + (z_score - 1) * 30
            severity = "medium"
            explanation = f"Amount ${amount:.2f} is slightly above average ${avg_amount:.2f}"
        else:
            score = z_score * 20
            severity = "low"
            explanation = f"Amount ${amount:.2f} is within normal range"
        
        return RiskFactor(
            name=self.name,
            score=score,
            weight=self.weight,
            explanation=explanation,
            severity=severity
        )


class VelocityAnomalyComponent(ScoringComponent):
    """Score based on transaction velocity violations"""
    
    def __init__(self, weight: float = 0.25):
        super().__init__("Velocity Anomaly", weight)
    
    def calculate_score(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RiskFactor:
        velocity_result = context.get('velocity_result')
        
        if not velocity_result or not velocity_result.violated_rules:
            return RiskFactor(
                name=self.name,
                score=0.0,
                weight=self.weight,
                explanation="No velocity violations detected",
                severity="low"
            )
        
        # Get velocity risk score
        score = velocity_result.get_risk_score()
        severity = velocity_result.max_severity
        
        violation_count = len(velocity_result.violated_rules)
        rule_names = [r.name for r in velocity_result.violated_rules[:3]]
        
        explanation = f"{violation_count} velocity violations: {', '.join(rule_names)}"
        
        return RiskFactor(
            name=self.name,
            score=score,
            weight=self.weight,
            explanation=explanation,
            severity=severity
        )


class BehavioralAnomalyComponent(ScoringComponent):
    """Score based on behavioral deviations"""
    
    def __init__(self, weight: float = 0.2):
        super().__init__("Behavioral Anomaly", weight)
    
    def calculate_score(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RiskFactor:
        behavioral_anomalies = context.get('behavioral_anomalies', [])
        
        if not behavioral_anomalies:
            return RiskFactor(
                name=self.name,
                score=0.0,
                weight=self.weight,
                explanation="No behavioral anomalies detected",
                severity="low"
            )
        
        # Score based on number and significance of anomalies
        significant_anomalies = [a for a in behavioral_anomalies if a.is_significant]
        
        if len(significant_anomalies) >= 3:
            score = 85.0
            severity = "critical"
        elif len(significant_anomalies) == 2:
            score = 65.0
            severity = "high"
        elif len(significant_anomalies) == 1:
            score = 40.0
            severity = "medium"
        else:
            score = 20.0
            severity = "low"
        
        anomaly_types = [a.anomaly_type.value for a in significant_anomalies[:3]]
        explanation = f"{len(significant_anomalies)} significant behavioral anomalies: {', '.join(anomaly_types)}"
        
        return RiskFactor(
            name=self.name,
            score=score,
            weight=self.weight,
            explanation=explanation,
            severity=severity
        )


class PatternMatchComponent(ScoringComponent):
    """Score based on fraud pattern matches"""
    
    def __init__(self, weight: float = 0.3):
        super().__init__("Pattern Match", weight)
    
    def calculate_score(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RiskFactor:
        pattern_matches = context.get('pattern_matches', [])
        
        if not pattern_matches:
            return RiskFactor(
                name=self.name,
                score=0.0,
                weight=self.weight,
                explanation="No fraud patterns detected",
                severity="low"
            )
        
        # Get highest risk pattern
        max_score = max(pm.get_risk_score() for pm in pattern_matches)
        top_pattern = max(pattern_matches, key=lambda pm: pm.get_risk_score())
        
        severity_map = {
            'critical': "critical",
            'high': "high",
            'medium': "medium",
            'low': "low"
        }
        severity = severity_map.get(top_pattern.pattern.severity, "medium")
        
        explanation = f"Matched pattern '{top_pattern.pattern.name}' with {top_pattern.confidence:.0%} confidence"
        
        return RiskFactor(
            name=self.name,
            score=max_score,
            weight=self.weight,
            explanation=explanation,
            severity=severity
        )


class MLModelComponent(ScoringComponent):
    """Score based on ML model predictions"""
    
    def __init__(self, weight: float = 0.25):
        super().__init__("ML Model", weight)
    
    def calculate_score(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RiskFactor:
        ml_prediction = context.get('ml_prediction')
        ml_probability = context.get('ml_probability', 0.0)
        
        if ml_prediction is None:
            return RiskFactor(
                name=self.name,
                score=50.0,  # Neutral if model unavailable
                weight=self.weight,
                explanation="ML model prediction unavailable",
                severity="medium"
            )
        
        # Convert probability to score (0-100)
        score = ml_probability * 100
        
        if score > 80:
            severity = "critical"
            explanation = f"ML model predicts {ml_probability:.1%} fraud probability (HIGH RISK)"
        elif score > 60:
            severity = "high"
            explanation = f"ML model predicts {ml_probability:.1%} fraud probability"
        elif score > 40:
            severity = "medium"
            explanation = f"ML model predicts {ml_probability:.1%} fraud probability"
        else:
            severity = "low"
            explanation = f"ML model predicts {ml_probability:.1%} fraud probability (LOW RISK)"
        
        return RiskFactor(
            name=self.name,
            score=score,
            weight=self.weight,
            explanation=explanation,
            severity=severity
        )


class FraudScoringEngine:
    """
    Real-time fraud scoring engine with multi-factor analysis
    
    Combines multiple scoring components with configurable weights:
    - Amount anomaly detection
    - Velocity violations
    - Behavioral deviations
    - Fraud pattern matches
    - ML model predictions
    
    Features:
    - Sub-100ms latency target
    - Pluggable scoring components
    - Automatic score normalization
    - Confidence interval calculation
    - Factor weighting and calibration
    """
    
    def __init__(self):
        """Initialize fraud scoring engine"""
        self.components: List[ScoringComponent] = []
        self.calibration_data: Optional[pd.DataFrame] = None
        self.performance_stats = {
            'total_scores': 0,
            'total_time_ms': 0.0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0
        }
        
        # Add default components
        self._add_default_components()
    
    def _add_default_components(self):
        """Add default scoring components"""
        self.components = [
            AmountAnomalyComponent(weight=0.2),
            VelocityAnomalyComponent(weight=0.25),
            BehavioralAnomalyComponent(weight=0.2),
            PatternMatchComponent(weight=0.3),
            MLModelComponent(weight=0.25)
        ]
    
    def add_component(self, component: ScoringComponent):
        """
        Add custom scoring component
        
        Args:
            component: ScoringComponent instance
        """
        self.components.append(component)
    
    def remove_component(self, name: str):
        """
        Remove scoring component by name
        
        Args:
            name: Component name
        """
        self.components = [c for c in self.components if c.name != name]
    
    def set_component_weight(self, name: str, weight: float):
        """
        Update component weight
        
        Args:
            name: Component name
            weight: New weight (0-1)
        """
        for component in self.components:
            if component.name == name:
                component.weight = weight
                break
    
    def calculate_risk_score(
        self,
        transaction: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RiskScore:
        """
        Calculate comprehensive fraud risk score
        
        Args:
            transaction: Transaction data (amount, merchant_id, timestamp, etc.)
            context: Optional context (customer_profile, velocity_result, etc.)
            
        Returns:
            RiskScore with final score, confidence, and factor breakdown
        """
        start_time = time.time()
        
        if context is None:
            context = {}
        
        # Calculate all factor scores
        factors: List[RiskFactor] = []
        total_weight = 0.0
        
        for component in self.components:
            if component.is_applicable(transaction):
                factor = component.calculate_score(transaction, context)
                factors.append(factor)
                total_weight += component.weight
        
        # Normalize weights if they don't sum to 1.0
        if total_weight > 0:
            for factor in factors:
                factor.weight = factor.weight / total_weight
        
        # Calculate weighted average score
        final_score = sum(f.score * f.weight for f in factors)
        
        # Calculate confidence based on factor agreement
        confidence = self._calculate_confidence(factors)
        
        # Track performance
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_performance_stats(elapsed_ms)
        
        return RiskScore(
            transaction_id=transaction.get('transaction_id', 'UNKNOWN'),
            customer_id=transaction.get('customer_id', 'UNKNOWN'),
            score=final_score,
            confidence=confidence,
            factors=factors,
            timestamp=datetime.now(),
            model_version="1.0"
        )
    
    def _calculate_confidence(self, factors: List[RiskFactor]) -> float:
        """
        Calculate confidence based on factor score agreement
        
        Higher confidence when factors agree (all high or all low)
        Lower confidence when factors conflict
        
        Args:
            factors: List of risk factors
            
        Returns:
            Confidence score (0-1)
        """
        if not factors:
            return 0.5
        
        # Calculate normalized scores
        scores = np.array([f.score for f in factors])
        
        # Confidence based on inverse of standard deviation
        # Low std = high agreement = high confidence
        if len(scores) > 1:
            std = np.std(scores)
            # Normalize std to 0-1 range (max std is ~50 for scores 0-100)
            normalized_std = min(std / 50, 1.0)
            confidence = 1.0 - normalized_std
        else:
            confidence = 0.8  # Moderate confidence for single factor
        
        return confidence
    
    def _update_performance_stats(self, elapsed_ms: float):
        """Update performance statistics"""
        self.performance_stats['total_scores'] += 1
        self.performance_stats['total_time_ms'] += elapsed_ms
        self.performance_stats['avg_latency_ms'] = (
            self.performance_stats['total_time_ms'] / 
            self.performance_stats['total_scores']
        )
        self.performance_stats['max_latency_ms'] = max(
            self.performance_stats['max_latency_ms'],
            elapsed_ms
        )
    
    def calibrate_thresholds(
        self,
        historical_data: pd.DataFrame,
        fraud_label_column: str = 'is_fraud'
    ):
        """
        Auto-calibrate decision thresholds based on historical data
        
        Args:
            historical_data: Historical transactions with fraud labels
            fraud_label_column: Column name for fraud labels (0/1)
        """
        self.calibration_data = historical_data
        
        # Calculate scores for historical data
        scores = []
        for _, row in historical_data.iterrows():
            transaction = row.to_dict()
            risk_score = self.calculate_risk_score(transaction)
            scores.append(risk_score.score)
        
        historical_data['risk_score'] = scores
        
        # Calculate optimal thresholds based on precision-recall tradeoff
        fraud_scores = historical_data[historical_data[fraud_label_column] == 1]['risk_score']
        legitimate_scores = historical_data[historical_data[fraud_label_column] == 0]['risk_score']
        
        # Store calibration statistics
        self.calibration_stats = {
            'fraud_score_mean': fraud_scores.mean(),
            'fraud_score_std': fraud_scores.std(),
            'legitimate_score_mean': legitimate_scores.mean(),
            'legitimate_score_std': legitimate_scores.std(),
            'total_samples': len(historical_data),
            'fraud_samples': len(fraud_scores),
            'legitimate_samples': len(legitimate_scores)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_scores': 0,
            'total_time_ms': 0.0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0
        }
    
    def explain_score(self, risk_score: RiskScore) -> str:
        """
        Generate human-readable explanation of risk score
        
        Args:
            risk_score: RiskScore instance
            
        Returns:
            Formatted explanation string
        """
        lines = []
        lines.append(f"Risk Score: {risk_score.score:.1f}/100 ({risk_score.get_risk_level().upper()})")
        lines.append(f"Confidence: {risk_score.confidence:.1%}")
        lines.append("")
        lines.append("Top Risk Factors:")
        
        for i, factor in enumerate(risk_score.get_top_factors(5), 1):
            weighted_score = factor.score * factor.weight
            lines.append(f"  {i}. {factor.name}: {factor.score:.1f} (weight: {factor.weight:.2f}, contribution: {weighted_score:.1f})")
            lines.append(f"     {factor.explanation}")
            lines.append(f"     Severity: {factor.severity.upper()}")
        
        return "\n".join(lines)
