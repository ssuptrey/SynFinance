"""
WebSocket Module

Real-time WebSocket communication for fraud detection events.
Week 8 Day 2: WebSocket Support
"""

from src.api.websocket.events import (
    Event,
    EventType,
    TransactionCreatedEvent,
    FraudDetectedEvent,
    ModelTrainingProgressEvent,
    SystemAlertEvent
)

from src.api.websocket.connection_manager import (
    ConnectionManager,
    get_connection_manager
)

__all__ = [
    "Event",
    "EventType",
    "TransactionCreatedEvent",
    "FraudDetectedEvent",
    "ModelTrainingProgressEvent",
    "SystemAlertEvent",
    "ConnectionManager",
    "get_connection_manager"
]
