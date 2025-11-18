"""
Fraud Detection Module

Comprehensive fraud detection system with:
- Real-time fraud scoring engine
- Velocity checking and anomaly detection
- Behavioral analysis and profiling
- Pattern detection (card testing, ATO, bust-out, etc.)
- Automated decision engine
- ML model deployment and A/B testing

Week 10 Day 4: Advanced Fraud Detection & Real-Time Analytics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


# Enums for decision types
class DecisionType(str, Enum):
    """Fraud decision categories"""
    APPROVE = "APPROVE"
    REVIEW_LOW = "REVIEW_LOW"
    REVIEW_HIGH = "REVIEW_HIGH"
    REVIEW_URGENT = "REVIEW_URGENT"
    DECLINE = "DECLINE"


class AnomalyType(str, Enum):
    """Types of behavioral anomalies"""
    AMOUNT_ANOMALY = "AMOUNT_ANOMALY"
    MERCHANT_ANOMALY = "MERCHANT_ANOMALY"
    TIME_ANOMALY = "TIME_ANOMALY"
    LOCATION_ANOMALY = "LOCATION_ANOMALY"
    FREQUENCY_ANOMALY = "FREQUENCY_ANOMALY"


class DeploymentStrategy(str, Enum):
    """Model deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    AB_TEST = "ab_test"
    SHADOW = "shadow"


# Core dataclasses
@dataclass
class RiskFactor:
    """Individual risk factor contribution"""
    name: str
    score: float  # 0-100
    weight: float  # 0-1
    explanation: str
    severity: str  # low, medium, high, critical


@dataclass
class RiskScore:
    """Comprehensive risk score result"""
    transaction_id: str
    customer_id: str
    score: float  # Final weighted score 0-100
    confidence: float  # 0-1
    factors: List[RiskFactor]
    timestamp: datetime
    model_version: str = "1.0"
    
    def get_top_factors(self, n: int = 3) -> List[RiskFactor]:
        """Get top N risk factors by weighted score"""
        return sorted(
            self.factors,
            key=lambda f: f.score * f.weight,
            reverse=True
        )[:n]
    
    def get_risk_level(self) -> str:
        """Get categorical risk level"""
        if self.score < 30:
            return "low"
        elif self.score < 50:
            return "medium"
        elif self.score < 70:
            return "high"
        else:
            return "critical"


@dataclass
class VelocityRule:
    """Velocity check rule definition"""
    name: str
    time_window: str  # 1min, 5min, 15min, 1hr, 24hr
    metric: str  # count, amount, unique_merchants, distance
    threshold: float
    severity: str  # low, medium, high, critical
    
    def get_window_seconds(self) -> int:
        """Convert time window to seconds"""
        mapping = {
            '1min': 60,
            '5min': 300,
            '15min': 900,
            '1hr': 3600,
            '24hr': 86400
        }
        return mapping.get(self.time_window, 3600)


@dataclass
class VelocityResult:
    """Result of velocity check"""
    customer_id: str
    timestamp: datetime
    violated_rules: List[VelocityRule]
    velocity_stats: Dict[str, Any]  # Actual values per time window
    is_violation: bool
    max_severity: str  # Highest severity from violated rules
    
    def get_risk_score(self) -> float:
        """Convert velocity violations to risk score (0-100)"""
        if not self.violated_rules:
            return 0.0
        
        severity_scores = {
            'low': 20,
            'medium': 50,
            'high': 75,
            'critical': 95
        }
        
        return severity_scores.get(self.max_severity, 0.0)


@dataclass
class CustomerProfile:
    """Baseline customer behavioral profile"""
    customer_id: str
    created_at: datetime
    last_updated: datetime
    
    # Transaction amount statistics
    avg_amount: float
    std_amount: float
    min_amount: float
    max_amount: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    
    # Merchant preferences
    top_merchants: List[str]
    top_categories: List[str]
    unique_merchants: int
    
    # Temporal patterns
    hour_of_day_distribution: Dict[int, float]  # Hour -> frequency
    day_of_week_distribution: Dict[str, float]  # Day name -> frequency
    
    # Geographic patterns
    home_location: Optional[Dict[str, float]] = None  # lat, lon
    frequent_cities: List[str] = field(default_factory=list)
    
    # Activity patterns
    daily_transaction_count: float = 0.0
    weekly_transaction_count: float = 0.0
    
    # Total transactions used to build profile
    transaction_count: int = 0


@dataclass
class BehavioralAnomaly:
    """Detected behavioral anomaly"""
    anomaly_type: AnomalyType
    field_name: str
    expected_value: Any
    actual_value: Any
    deviation_score: float  # Z-score or chi-square value
    p_value: float  # Statistical significance
    is_significant: bool  # p_value < 0.05
    explanation: str


@dataclass
class FraudPattern:
    """Known fraud pattern definition"""
    pattern_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]  # Pattern matching rules
    severity: str  # low, medium, high, critical
    min_transactions: int  # Minimum transactions needed to match
    time_window: str  # Time window for pattern (e.g., "24hr", "7d")
    
    def calculate_confidence(self, matched_transactions: List[Dict]) -> float:
        """Calculate match confidence (0-1)"""
        if len(matched_transactions) < self.min_transactions:
            return 0.0
        
        # Simple confidence based on completeness
        confidence = min(len(matched_transactions) / (self.min_transactions * 2), 1.0)
        return confidence


@dataclass
class PatternMatch:
    """Detected fraud pattern match"""
    pattern: FraudPattern
    customer_id: str
    matched_transactions: List[Dict[str, Any]]
    confidence: float  # 0-1
    timestamp: datetime
    explanation: str
    
    def get_risk_score(self) -> float:
        """Convert pattern match to risk score (0-100)"""
        severity_base = {
            'low': 30,
            'medium': 60,
            'high': 85,
            'critical': 98
        }
        
        base_score = severity_base.get(self.pattern.severity, 50)
        return base_score * self.confidence


@dataclass
class FraudDecision:
    """Automated fraud decision"""
    transaction_id: str
    customer_id: str
    decision: DecisionType
    risk_score: float  # 0-100
    confidence: float  # 0-1
    reasoning: List[str]  # Human-readable explanations
    recommended_action: str
    manual_review_priority: Optional[int] = None  # 1-10 for review decisions
    timestamp: datetime = field(default_factory=datetime.now)
    
    def should_decline(self) -> bool:
        """Check if transaction should be declined"""
        return self.decision == DecisionType.DECLINE
    
    def needs_review(self) -> bool:
        """Check if transaction needs manual review"""
        return self.decision in [
            DecisionType.REVIEW_LOW,
            DecisionType.REVIEW_HIGH,
            DecisionType.REVIEW_URGENT
        ]


@dataclass
class DeployedModel:
    """Production model deployment metadata"""
    model_id: str
    model_name: str
    version: str
    deployment_strategy: DeploymentStrategy
    deployed_at: datetime
    traffic_percentage: float = 100.0  # % of traffic using this model
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    latency_p50: Optional[float] = None  # milliseconds
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    
    # Drift detection
    prediction_drift_score: Optional[float] = None  # KS statistic
    feature_drift_score: Optional[float] = None


@dataclass
class ABTestConfig:
    """A/B testing configuration"""
    test_id: str
    champion_model_id: str
    challenger_model_id: str
    traffic_split: float  # % traffic to challenger (0-1)
    started_at: datetime
    duration_hours: int = 168  # 1 week default
    
    # Promotion criteria
    min_improvement: float = 0.02  # 2% improvement required
    min_samples: int = 1000  # Minimum transactions
    max_latency_ms: float = 100.0  # Maximum acceptable latency


# Module exports
from src.fraud.scoring_engine import FraudScoringEngine, ScoringComponent
from src.fraud.velocity_checker import VelocityChecker
from src.fraud.behavioral_analyzer import BehavioralAnalyzer
from src.fraud.pattern_detector import PatternDetector
from src.fraud.decision_engine import DecisionEngine
from src.fraud.model_deployer import ModelDeployer


__all__ = [
    # Enums
    'DecisionType',
    'AnomalyType',
    'DeploymentStrategy',
    
    # Dataclasses
    'RiskFactor',
    'RiskScore',
    'VelocityRule',
    'VelocityResult',
    'CustomerProfile',
    'BehavioralAnomaly',
    'FraudPattern',
    'PatternMatch',
    'FraudDecision',
    'DeployedModel',
    'ABTestConfig',
    
    # Main classes
    'FraudScoringEngine',
    'ScoringComponent',
    'VelocityChecker',
    'BehavioralAnalyzer',
    'PatternDetector',
    'DecisionEngine',
    'ModelDeployer',
]
