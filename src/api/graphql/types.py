"""
GraphQL Type Definitions

Defines all GraphQL types for transactions, customers, merchants, etc.
Uses Strawberry for type-safe schema definition.
"""

from typing import Optional, List
from datetime import datetime
from decimal import Decimal
import strawberry


@strawberry.type
class TransactionType:
    """GraphQL type for Transaction"""
    
    transaction_id: str
    customer_id: str
    merchant_id: str
    amount: float
    category: str
    payment_mode: str
    timestamp: datetime
    
    # Location fields
    city: Optional[str] = None
    state: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Fraud detection fields
    is_fraud: bool = False
    fraud_type: Optional[str] = None
    fraud_confidence: Optional[float] = None
    
    # Anomaly detection fields
    is_anomaly: bool = False
    anomaly_score: Optional[float] = None
    anomaly_type: Optional[str] = None
    
    # Risk scores
    risk_score: Optional[float] = None
    velocity_score: Optional[float] = None


@strawberry.type
class CustomerType:
    """GraphQL type for Customer"""
    
    customer_id: str
    name: str
    age: int
    gender: str
    occupation: str
    income_bracket: str
    customer_segment: str
    
    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    
    # Behavioral features
    digital_savviness: Optional[float] = None
    loyalty_score: Optional[float] = None
    risk_level: Optional[str] = None
    
    # Stats
    total_transactions: Optional[int] = None
    total_spent: Optional[float] = None
    avg_transaction_amount: Optional[float] = None


@strawberry.type
class MerchantType:
    """GraphQL type for Merchant"""
    
    merchant_id: str
    name: str
    category: str
    subcategory: Optional[str] = None
    
    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    
    # Reputation and risk
    reputation_score: Optional[float] = None
    is_chain: bool = False
    risk_level: Optional[str] = None
    
    # Stats
    total_transactions: Optional[int] = None
    total_revenue: Optional[float] = None


@strawberry.type
class MLFeaturesType:
    """GraphQL type for ML Features"""
    
    transaction_id: str
    
    # Aggregate features
    daily_transaction_count: int = 0
    weekly_transaction_count: int = 0
    daily_transaction_amount: float = 0.0
    average_daily_amount: float = 0.0
    
    # Velocity features
    transaction_frequency_1h: int = 0
    transaction_frequency_24h: int = 0
    amount_velocity_1h: float = 0.0
    
    # Geographic features
    distance_from_home: float = 0.0
    unique_cities_7d: int = 0
    
    # Temporal features
    is_unusual_hour: bool = False
    is_weekend: bool = False
    hour_of_day: int = 0
    day_of_week: int = 0
    
    # Behavioral features
    category_diversity_7d: int = 0
    merchant_loyalty_score: float = 0.0
    is_new_merchant: bool = False
    
    # Anomaly features
    amount_anomaly_score: float = 0.0
    velocity_anomaly_score: float = 0.0
    geographic_anomaly_score: float = 0.0
    
    # Interaction features
    risk_amplification_score: float = 0.0
    ensemble_fraud_probability: float = 0.0


@strawberry.type
class FraudPatternType:
    """GraphQL type for detected fraud patterns"""
    
    pattern_id: str
    transaction_id: str
    pattern_type: str
    confidence: float
    severity: str
    
    # Evidence
    evidence: List[str]
    detected_at: datetime
    
    # Additional context
    related_transactions: Optional[List[str]] = None
    recommended_action: Optional[str] = None


@strawberry.type
class SystemHealthType:
    """GraphQL type for system health status"""
    
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    
    # Component health
    database: str
    cache: str
    api: str
    ml_model: str
    
    # Metrics
    uptime_seconds: int
    cpu_usage_percent: float
    memory_usage_percent: float
    active_connections: int
    
    # Errors
    recent_errors: Optional[List[str]] = None


@strawberry.type
class GenerationStatsType:
    """GraphQL type for generation statistics"""
    
    total_transactions_generated: int
    total_customers: int
    total_merchants: int
    
    # Fraud stats
    fraud_count: int
    fraud_rate: float
    
    # Anomaly stats
    anomaly_count: int
    anomaly_rate: float
    
    # Performance stats
    generation_rate_per_second: float
    average_processing_time_ms: float
    
    # Time window
    period_start: datetime
    period_end: datetime


@strawberry.type
class GenerationResultType:
    """GraphQL type for generation operation result"""
    
    success: bool
    message: str
    transactions_generated: int
    fraud_injected: int
    execution_time_seconds: float
    output_file: Optional[str] = None


@strawberry.type
class ModelTrainingResultType:
    """GraphQL type for model training result"""
    
    success: bool
    message: str
    model_id: str
    algorithm: str
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Training details
    training_time_seconds: float
    samples_trained: int
    features_used: int


@strawberry.type
class ValidationResultType:
    """GraphQL type for data validation result"""
    
    success: bool
    quality_score: float
    
    # Validation details
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    
    # Issues
    critical_issues: List[str]
    recommendations: List[str]


@strawberry.input
class TransactionFilterInput:
    """Input type for filtering transactions"""
    
    customer_id: Optional[str] = None
    merchant_id: Optional[str] = None
    category: Optional[str] = None
    payment_mode: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    is_fraud: Optional[bool] = None
    is_anomaly: Optional[bool] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@strawberry.input
class CustomerFilterInput:
    """Input type for filtering customers"""
    
    occupation: Optional[str] = None
    income_bracket: Optional[str] = None
    customer_segment: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None


@strawberry.input
class MerchantFilterInput:
    """Input type for filtering merchants"""
    
    category: Optional[str] = None
    city: Optional[str] = None
    is_chain: Optional[bool] = None
    min_reputation: Optional[float] = None


@strawberry.input
class GenerateTransactionsInput:
    """Input type for generating transactions"""
    
    count: int
    fraud_rate: Optional[float] = 0.05
    seed: Optional[int] = None
    output_format: Optional[str] = "csv"
    include_ml_features: bool = True


@strawberry.input
class TrainModelInput:
    """Input type for training ML model"""
    
    algorithm: str  # random_forest, xgboost, logistic_regression
    features: Optional[List[str]] = None
    train_split_ratio: float = 0.8
    cross_validation_folds: int = 5
    hyperparameter_tuning: bool = False
