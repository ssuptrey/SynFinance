"""
WebSocket Event Types and Models

Defines event types and data models for WebSocket communication.
Week 8 Day 2: WebSocket Support
"""

from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict


class EventType(str, Enum):
    """WebSocket event types"""
    TRANSACTION_CREATED = "transaction_created"
    FRAUD_DETECTED = "fraud_detected"
    MODEL_TRAINING_STARTED = "model_training_started"
    MODEL_TRAINING_PROGRESS = "model_training_progress"
    MODEL_TRAINING_COMPLETE = "model_training_complete"
    SYSTEM_ALERT = "system_alert"
    CONNECTION_ESTABLISHED = "connection_established"
    PING = "ping"
    PONG = "pong"


@dataclass
class Event:
    """Base event class for WebSocket messages"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization"""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "tenant_id": self.tenant_id
        }
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Alias for to_dict for clarity"""
        return self.to_dict()


@dataclass
class TransactionCreatedEvent(Event):
    """Event fired when a new transaction is created"""
    
    def __init__(
        self,
        transaction_id: str,
        customer_id: str,
        merchant_id: str,
        amount: float,
        timestamp: datetime,
        is_fraud: bool = False,
        fraud_score: Optional[float] = None,
        tenant_id: Optional[str] = None,
        event_id: Optional[str] = None
    ):
        super().__init__(
            event_type=EventType.TRANSACTION_CREATED,
            data={
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "merchant_id": merchant_id,
                "amount": amount,
                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                "is_fraud": is_fraud,
                "fraud_score": fraud_score
            },
            tenant_id=tenant_id,
            event_id=event_id
        )


@dataclass
class FraudDetectedEvent(Event):
    """Event fired when fraud is detected"""
    
    def __init__(
        self,
        transaction_id: str,
        fraud_score: float,
        fraud_type: str,
        amount: float,
        customer_id: str,
        merchant_id: str,
        risk_level: str = "medium",
        details: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        event_id: Optional[str] = None
    ):
        super().__init__(
            event_type=EventType.FRAUD_DETECTED,
            data={
                "transaction_id": transaction_id,
                "fraud_score": fraud_score,
                "fraud_type": fraud_type,
                "amount": amount,
                "customer_id": customer_id,
                "merchant_id": merchant_id,
                "risk_level": risk_level,
                "details": details or {}
            },
            tenant_id=tenant_id,
            event_id=event_id
        )


@dataclass
class ModelTrainingProgressEvent(Event):
    """Event fired during model training to report progress"""
    
    def __init__(
        self,
        model_id: str,
        model_type: str,
        progress: float,
        stage: str,
        metrics: Optional[Dict[str, float]] = None,
        status: str = "training",
        tenant_id: Optional[str] = None,
        event_id: Optional[str] = None
    ):
        super().__init__(
            event_type=EventType.MODEL_TRAINING_PROGRESS,
            data={
                "model_id": model_id,
                "model_type": model_type,
                "progress": progress,
                "stage": stage,
                "metrics": metrics or {},
                "status": status
            },
            tenant_id=tenant_id,
            event_id=event_id
        )


@dataclass
class SystemAlertEvent(Event):
    """Event fired for system-level alerts"""
    
    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        event_id: Optional[str] = None
    ):
        super().__init__(
            event_type=EventType.SYSTEM_ALERT,
            data={
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "details": details or {}
            },
            tenant_id=tenant_id,
            event_id=event_id
        )
