"""
Database Models for SynFinance

SQLAlchemy ORM models for storing transactions, customers, merchants,
fraud patterns, and ML predictions.

Week 7 Day 5: Database Integration
"""

from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, CheckConstraint, Numeric, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

Base = declarative_base()


class Transaction(Base):
    """
    Transaction model
    
    Stores all transaction data including fraud detection results.
    """
    __tablename__ = 'transactions'
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    
    # Foreign keys
    customer_id: Mapped[str] = mapped_column(String(50), ForeignKey('customers.customer_id'), nullable=False, index=True)
    merchant_id: Mapped[str] = mapped_column(String(50), ForeignKey('merchants.merchant_id'), nullable=False, index=True)
    
    # Transaction details
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default='USD')
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    
    # Location
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(50))
    country: Mapped[str] = mapped_column(String(50), default='USA')
    zip_code: Mapped[Optional[str]] = mapped_column(String(10))
    
    # Transaction type
    transaction_type: Mapped[str] = mapped_column(String(50), nullable=False)  # purchase, withdrawal, transfer
    category: Mapped[Optional[str]] = mapped_column(String(100))
    merchant_category: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    
    # Payment method
    payment_method: Mapped[str] = mapped_column(String(50))  # credit_card, debit_card, bank_transfer
    card_type: Mapped[Optional[str]] = mapped_column(String(20))  # visa, mastercard, amex
    
    # Channel
    channel: Mapped[str] = mapped_column(String(50))  # online, pos, atm, mobile
    is_online: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    
    # Temporal features
    hour_of_day: Mapped[int] = mapped_column(Integer)
    day_of_week: Mapped[int] = mapped_column(Integer)
    day_of_month: Mapped[int] = mapped_column(Integer)
    month: Mapped[int] = mapped_column(Integer)
    is_weekend: Mapped[bool] = mapped_column(Boolean, default=False)
    is_holiday: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Fraud detection
    is_fraud: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    fraud_type: Mapped[Optional[str]] = mapped_column(String(100))
    fraud_score: Mapped[Optional[float]] = mapped_column(Float)
    fraud_reason: Mapped[Optional[str]] = mapped_column(Text)
    
    # Anomaly detection
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    anomaly_type: Mapped[Optional[str]] = mapped_column(String(100))
    anomaly_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Velocity features
    transactions_last_hour: Mapped[Optional[int]] = mapped_column(Integer)
    transactions_last_day: Mapped[Optional[int]] = mapped_column(Integer)
    amount_last_hour: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    amount_last_day: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    
    # Distance features
    distance_from_home: Mapped[Optional[float]] = mapped_column(Float)
    distance_from_last: Mapped[Optional[float]] = mapped_column(Float)
    
    # Time features
    time_since_last_transaction: Mapped[Optional[float]] = mapped_column(Float)
    time_of_day_category: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Risk features
    merchant_risk_score: Mapped[Optional[float]] = mapped_column(Float)
    customer_risk_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    customer = relationship("Customer", back_populates="transactions")
    merchant = relationship("Merchant", back_populates="transactions")
    features = relationship("MLFeatures", back_populates="transaction", uselist=False)
    predictions = relationship("ModelPrediction", back_populates="transaction")
    
    # Indexes
    __table_args__ = (
        Index('idx_transaction_timestamp', 'timestamp'),
        Index('idx_transaction_customer_timestamp', 'customer_id', 'timestamp'),
        Index('idx_transaction_merchant_timestamp', 'merchant_id', 'timestamp'),
        Index('idx_transaction_fraud', 'is_fraud', 'timestamp'),
        Index('idx_transaction_amount', 'amount'),
        CheckConstraint('amount >= 0', name='check_amount_positive'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'customer_id': self.customer_id,
            'merchant_id': self.merchant_id,
            'amount': float(self.amount),
            'currency': self.currency,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'city': self.city,
            'state': self.state,
            'country': self.country,
            'transaction_type': self.transaction_type,
            'category': self.category,
            'is_fraud': self.is_fraud,
            'fraud_type': self.fraud_type,
            'fraud_score': self.fraud_score,
            'is_anomaly': self.is_anomaly
        }


class Customer(Base):
    """
    Customer model
    
    Stores customer profile and historical data.
    """
    __tablename__ = 'customers'
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    
    # Personal information
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Demographics
    date_of_birth: Mapped[Optional[datetime]] = mapped_column(DateTime)
    age: Mapped[Optional[int]] = mapped_column(Integer)
    gender: Mapped[Optional[str]] = mapped_column(String(10))
    occupation: Mapped[Optional[str]] = mapped_column(String(100))
    income_level: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Address
    address_line1: Mapped[Optional[str]] = mapped_column(String(255))
    address_line2: Mapped[Optional[str]] = mapped_column(String(255))
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(50))
    country: Mapped[str] = mapped_column(String(50), default='USA')
    zip_code: Mapped[Optional[str]] = mapped_column(String(10))
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    
    # Account information
    account_created: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    account_status: Mapped[str] = mapped_column(String(20), default='active')
    credit_score: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Behavioral features
    avg_transaction_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    transaction_count: Mapped[int] = mapped_column(Integer, default=0)
    total_spent: Mapped[Decimal] = mapped_column(Numeric(15, 2), default=0)
    fraud_history_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Risk assessment
    risk_score: Mapped[Optional[float]] = mapped_column(Float)
    risk_category: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    transactions = relationship("Transaction", back_populates="customer")
    
    __table_args__ = (
        Index('idx_customer_email', 'email'),
        Index('idx_customer_risk', 'risk_score'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'age': self.age,
            'city': self.city,
            'state': self.state,
            'account_status': self.account_status,
            'transaction_count': self.transaction_count,
            'risk_score': self.risk_score
        }


class Merchant(Base):
    """
    Merchant model
    
    Stores merchant information and risk profiles.
    """
    __tablename__ = 'merchants'
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    merchant_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    
    # Merchant information
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    mcc_code: Mapped[Optional[str]] = mapped_column(String(4))  # Merchant Category Code
    
    # Location
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(50))
    country: Mapped[str] = mapped_column(String(50), default='USA')
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    
    # Risk assessment
    risk_score: Mapped[Optional[float]] = mapped_column(Float)
    risk_category: Mapped[Optional[str]] = mapped_column(String(20))
    fraud_report_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Business metrics
    total_transactions: Mapped[int] = mapped_column(Integer, default=0)
    total_revenue: Mapped[Decimal] = mapped_column(Numeric(15, 2), default=0)
    avg_transaction_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    transactions = relationship("Transaction", back_populates="merchant")
    
    __table_args__ = (
        Index('idx_merchant_category', 'category'),
        Index('idx_merchant_risk', 'risk_score'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'merchant_id': self.merchant_id,
            'name': self.name,
            'category': self.category,
            'city': self.city,
            'state': self.state,
            'risk_score': self.risk_score,
            'total_transactions': self.total_transactions
        }


class MLFeatures(Base):
    """
    ML Features model
    
    Stores engineered features for machine learning models.
    """
    __tablename__ = 'ml_features'
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(50), ForeignKey('transactions.transaction_id'), unique=True, nullable=False, index=True)
    
    # Amount-based features
    amount_normalized: Mapped[Optional[float]] = mapped_column(Float)
    amount_log: Mapped[Optional[float]] = mapped_column(Float)
    amount_zscore: Mapped[Optional[float]] = mapped_column(Float)
    amount_deviation_from_avg: Mapped[Optional[float]] = mapped_column(Float)
    
    # Temporal features
    hour_sin: Mapped[Optional[float]] = mapped_column(Float)
    hour_cos: Mapped[Optional[float]] = mapped_column(Float)
    day_of_week_sin: Mapped[Optional[float]] = mapped_column(Float)
    day_of_week_cos: Mapped[Optional[float]] = mapped_column(Float)
    month_sin: Mapped[Optional[float]] = mapped_column(Float)
    month_cos: Mapped[Optional[float]] = mapped_column(Float)
    
    # Velocity features
    tx_velocity_1h: Mapped[Optional[int]] = mapped_column(Integer)
    tx_velocity_6h: Mapped[Optional[int]] = mapped_column(Integer)
    tx_velocity_24h: Mapped[Optional[int]] = mapped_column(Integer)
    amount_velocity_1h: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    amount_velocity_6h: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    amount_velocity_24h: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    
    # Pattern features
    unusual_time: Mapped[Optional[bool]] = mapped_column(Boolean)
    unusual_location: Mapped[Optional[bool]] = mapped_column(Boolean)
    unusual_amount: Mapped[Optional[bool]] = mapped_column(Boolean)
    unusual_merchant: Mapped[Optional[bool]] = mapped_column(Boolean)
    
    # Behavioral features
    customer_lifetime_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    customer_transaction_count: Mapped[Optional[int]] = mapped_column(Integer)
    customer_avg_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    customer_std_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    
    # Merchant features
    merchant_fraud_rate: Mapped[Optional[float]] = mapped_column(Float)
    merchant_avg_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    merchant_transaction_count: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Distance features
    distance_from_home_km: Mapped[Optional[float]] = mapped_column(Float)
    distance_from_last_km: Mapped[Optional[float]] = mapped_column(Float)
    
    # Time since features
    time_since_last_tx_seconds: Mapped[Optional[float]] = mapped_column(Float)
    time_since_account_created_days: Mapped[Optional[float]] = mapped_column(Float)
    
    # Aggregated features (JSON for flexibility)
    additional_features: Mapped[Optional[Dict]] = mapped_column(JSON)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    transaction = relationship("Transaction", back_populates="features")
    
    __table_args__ = (
        Index('idx_features_transaction', 'transaction_id'),
    )


class ModelPrediction(Base):
    """
    Model Prediction model
    
    Stores ML model predictions and metadata.
    """
    __tablename__ = 'model_predictions'
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(50), ForeignKey('transactions.transaction_id'), nullable=False, index=True)
    
    # Model information
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Prediction
    prediction: Mapped[bool] = mapped_column(Boolean, nullable=False)  # fraud or not
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Feature importance (top 10)
    feature_importance: Mapped[Optional[Dict]] = mapped_column(JSON)
    
    # Prediction details
    prediction_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    features_used: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Model performance (if ground truth available)
    ground_truth: Mapped[Optional[bool]] = mapped_column(Boolean)
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    transaction = relationship("Transaction", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_transaction', 'transaction_id'),
        Index('idx_prediction_model', 'model_name', 'model_version'),
        Index('idx_prediction_confidence', 'confidence_score'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'prediction': self.prediction,
            'confidence_score': self.confidence_score,
            'fraud_probability': self.fraud_probability,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class FraudPattern(Base):
    """
    Fraud Pattern model
    
    Stores detected fraud patterns for analysis.
    """
    __tablename__ = 'fraud_patterns'
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # Pattern information
    pattern_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    pattern_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Pattern characteristics
    characteristics: Mapped[Dict] = mapped_column(JSON, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # low, medium, high, critical
    
    # Detection statistics
    detection_count: Mapped[int] = mapped_column(Integer, default=0)
    last_detected: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Confidence
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_pattern_type', 'pattern_type'),
        Index('idx_pattern_severity', 'severity'),
    )


class AnomalyPattern(Base):
    """
    Anomaly Pattern model
    
    Stores detected anomaly patterns for analysis.
    """
    __tablename__ = 'anomaly_patterns'
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # Pattern information
    anomaly_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    anomaly_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Anomaly characteristics
    characteristics: Mapped[Dict] = mapped_column(JSON, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Detection statistics
    detection_count: Mapped[int] = mapped_column(Integer, default=0)
    last_detected: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Score
    anomaly_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_anomaly_type', 'anomaly_type'),
        Index('idx_anomaly_severity', 'severity'),
    )
