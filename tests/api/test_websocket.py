"""
WebSocket Tests

Tests for WebSocket connection management, event broadcasting, and message handling.
Week 8 Day 2: WebSocket Support
"""

import pytest
import json
import asyncio
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from datetime import datetime

from src.api.api_server import app
from src.api.websocket import (
    get_connection_manager,
    Event,
    EventType,
    TransactionCreatedEvent,
    FraudDetectedEvent,
    ModelTrainingProgressEvent
)
from src.api.websocket.handlers import WebSocketHandler


class TestWebSocketEvents:
    """Test WebSocket event models"""
    
    def test_event_creation(self):
        """Test basic event creation"""
        event = Event(
            event_type=EventType.SYSTEM_ALERT,
            data={"message": "test"}
        )
        
        assert event.event_type == EventType.SYSTEM_ALERT
        assert event.data == {"message": "test"}
        assert event.timestamp is not None
    
    def test_event_to_dict(self):
        """Test event serialization"""
        event = Event(
            event_type=EventType.PING,
            data={"test": "data"},
            event_id="test_id",
            tenant_id="tenant_1"
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["event_type"] == "ping"
        assert event_dict["data"] == {"test": "data"}
        assert event_dict["event_id"] == "test_id"
        assert event_dict["tenant_id"] == "tenant_1"
        assert "timestamp" in event_dict
    
    def test_transaction_created_event(self):
        """Test transaction created event"""
        event = TransactionCreatedEvent(
            transaction_id="TXN001",
            customer_id="CUST001",
            merchant_id="MERCH001",
            amount=1000.0,
            timestamp=datetime(2025, 11, 1, 10, 30),
            is_fraud=False,
            fraud_score=0.1
        )
        
        assert event.event_type == EventType.TRANSACTION_CREATED
        assert event.data["transaction_id"] == "TXN001"
        assert event.data["amount"] == 1000.0
        assert event.data["is_fraud"] is False
    
    def test_fraud_detected_event(self):
        """Test fraud detected event"""
        event = FraudDetectedEvent(
            transaction_id="TXN002",
            fraud_score=0.95,
            fraud_type="velocity",
            amount=5000.0,
            customer_id="CUST002",
            merchant_id="MERCH002",
            risk_level="high"
        )
        
        assert event.event_type == EventType.FRAUD_DETECTED
        assert event.data["fraud_score"] == 0.95
        assert event.data["fraud_type"] == "velocity"
        assert event.data["risk_level"] == "high"
    
    def test_model_training_progress_event(self):
        """Test model training progress event"""
        event = ModelTrainingProgressEvent(
            model_id="model_123",
            model_type="random_forest",
            progress=0.5,
            stage="training",
            metrics={"accuracy": 0.92},
            status="in_progress"
        )
        
        assert event.event_type == EventType.MODEL_TRAINING_PROGRESS
        assert event.data["progress"] == 0.5
        assert event.data["metrics"]["accuracy"] == 0.92


class TestConnectionManager:
    """Test WebSocket connection manager"""
    
    @pytest.fixture
    def connection_manager(self):
        """Get fresh connection manager for each test"""
        from src.api.websocket.connection_manager import ConnectionManager
        return ConnectionManager()
    
    @pytest.mark.asyncio
    async def test_connect_client(self, connection_manager):
        """Test client connection"""
        # Create mock WebSocket
        class MockWebSocket:
            def __init__(self):
                self.accepted = False
                self.messages = []
            
            async def accept(self):
                self.accepted = True
            
            async def send_json(self, data):
                self.messages.append(data)
        
        mock_ws = MockWebSocket()
        
        await connection_manager.connect(mock_ws, "client_1")
        
        assert mock_ws.accepted
        assert connection_manager.get_connection_count() == 1
        assert "client_1" in connection_manager.active_connections
    
    def test_disconnect_client(self, connection_manager):
        """Test client disconnection"""
        # Manually add a connection
        class MockWebSocket:
            pass
        
        connection_manager.active_connections["client_1"] = MockWebSocket()
        connection_manager.client_metadata["client_1"] = {}
        
        assert connection_manager.get_connection_count() == 1
        
        connection_manager.disconnect("client_1")
        
        assert connection_manager.get_connection_count() == 0
        assert "client_1" not in connection_manager.active_connections
    
    def test_subscribe_to_topic(self, connection_manager):
        """Test topic subscription"""
        connection_manager.subscribe("client_1", "fraud_alerts")
        
        assert "client_1" in connection_manager.subscriptions["fraud_alerts"]
        assert connection_manager.get_topic_subscribers("fraud_alerts") == 1
    
    def test_unsubscribe_from_topic(self, connection_manager):
        """Test topic unsubscription"""
        connection_manager.subscribe("client_1", "fraud_alerts")
        assert connection_manager.get_topic_subscribers("fraud_alerts") == 1
        
        connection_manager.unsubscribe("client_1", "fraud_alerts")
        assert connection_manager.get_topic_subscribers("fraud_alerts") == 0
    
    @pytest.mark.asyncio
    async def test_send_personal_message(self, connection_manager):
        """Test sending message to specific client"""
        class MockWebSocket:
            def __init__(self):
                self.messages = []
            
            async def send_json(self, data):
                self.messages.append(data)
        
        mock_ws = MockWebSocket()
        connection_manager.active_connections["client_1"] = mock_ws
        
        event = Event(
            event_type=EventType.PING,
            data={"test": "message"}
        )
        
        result = await connection_manager.send_personal_message(event, "client_1")
        
        assert result is True
        assert len(mock_ws.messages) == 1
        assert mock_ws.messages[0]["event_type"] == "ping"
    
    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, connection_manager):
        """Test broadcasting to all clients"""
        class MockWebSocket:
            def __init__(self):
                self.messages = []
            
            async def send_json(self, data):
                self.messages.append(data)
        
        mock_ws1 = MockWebSocket()
        mock_ws2 = MockWebSocket()
        
        connection_manager.active_connections["client_1"] = mock_ws1
        connection_manager.active_connections["client_2"] = mock_ws2
        
        event = Event(
            event_type=EventType.SYSTEM_ALERT,
            data={"message": "broadcast test"}
        )
        
        sent_count = await connection_manager.broadcast(event)
        
        assert sent_count == 2
        assert len(mock_ws1.messages) == 1
        assert len(mock_ws2.messages) == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_to_topic_subscribers(self, connection_manager):
        """Test broadcasting to topic subscribers only"""
        class MockWebSocket:
            def __init__(self):
                self.messages = []
            
            async def send_json(self, data):
                self.messages.append(data)
        
        mock_ws1 = MockWebSocket()
        mock_ws2 = MockWebSocket()
        mock_ws3 = MockWebSocket()
        
        connection_manager.active_connections["client_1"] = mock_ws1
        connection_manager.active_connections["client_2"] = mock_ws2
        connection_manager.active_connections["client_3"] = mock_ws3
        
        # Subscribe clients 1 and 2 to topic
        connection_manager.subscribe("client_1", "fraud_alerts")
        connection_manager.subscribe("client_2", "fraud_alerts")
        
        event = FraudDetectedEvent(
            transaction_id="TXN123",
            fraud_score=0.9,
            fraud_type="velocity",
            amount=1000.0,
            customer_id="CUST001",
            merchant_id="MERCH001"
        )
        
        sent_count = await connection_manager.broadcast(event, topic="fraud_alerts")
        
        assert sent_count == 2
        assert len(mock_ws1.messages) == 1
        assert len(mock_ws2.messages) == 1
        assert len(mock_ws3.messages) == 0  # Not subscribed
    
    @pytest.mark.asyncio
    async def test_broadcast_with_tenant_filter(self, connection_manager):
        """Test broadcasting with tenant filtering"""
        class MockWebSocket:
            def __init__(self):
                self.messages = []
            
            async def send_json(self, data):
                self.messages.append(data)
        
        mock_ws1 = MockWebSocket()
        mock_ws2 = MockWebSocket()
        
        connection_manager.active_connections["client_1"] = mock_ws1
        connection_manager.active_connections["client_2"] = mock_ws2
        connection_manager.client_tenants["client_1"] = "tenant_a"
        connection_manager.client_tenants["client_2"] = "tenant_b"
        
        event = Event(
            event_type=EventType.SYSTEM_ALERT,
            data={"message": "tenant specific"},
            tenant_id="tenant_a"
        )
        
        sent_count = await connection_manager.broadcast(event, tenant_id="tenant_a")
        
        assert sent_count == 1
        assert len(mock_ws1.messages) == 1  # tenant_a
        assert len(mock_ws2.messages) == 0  # tenant_b
    
    def test_get_client_info(self, connection_manager):
        """Test retrieving client information"""
        class MockWebSocket:
            pass
        
        connection_manager.active_connections["client_1"] = MockWebSocket()
        connection_manager.client_metadata["client_1"] = {"user": "test_user"}
        connection_manager.client_tenants["client_1"] = "tenant_1"
        connection_manager.subscribe("client_1", "fraud_alerts")
        connection_manager.subscribe("client_1", "transactions")
        
        client_info = connection_manager.get_client_info("client_1")
        
        assert client_info is not None
        assert client_info["client_id"] == "client_1"
        assert client_info["tenant_id"] == "tenant_1"
        assert client_info["metadata"]["user"] == "test_user"
        assert "fraud_alerts" in client_info["subscriptions"]
        assert "transactions" in client_info["subscriptions"]


class TestWebSocketHandler:
    """Test WebSocket message handler"""
    
    @pytest.fixture
    def setup(self):
        """Setup handler and connection manager"""
        from src.api.websocket.connection_manager import ConnectionManager
        connection_manager = ConnectionManager()
        handler = WebSocketHandler(connection_manager)
        return handler, connection_manager
    
    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, setup):
        """Test handling subscribe message"""
        handler, connection_manager = setup
        
        class MockWebSocket:
            pass
        
        connection_manager.active_connections["client_1"] = MockWebSocket()
        
        message = {
            "type": "subscribe",
            "topic": "fraud_alerts"
        }
        
        await handler.handle_message(MockWebSocket(), "client_1", message)
        
        assert "client_1" in connection_manager.subscriptions["fraud_alerts"]
    
    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(self, setup):
        """Test handling unsubscribe message"""
        handler, connection_manager = setup
        
        class MockWebSocket:
            pass
        
        connection_manager.active_connections["client_1"] = MockWebSocket()
        connection_manager.subscribe("client_1", "fraud_alerts")
        
        message = {
            "type": "unsubscribe",
            "topic": "fraud_alerts"
        }
        
        await handler.handle_message(MockWebSocket(), "client_1", message)
        
        assert "client_1" not in connection_manager.subscriptions["fraud_alerts"]
    
    @pytest.mark.asyncio
    async def test_handle_ping_message(self, setup):
        """Test handling ping message"""
        handler, connection_manager = setup
        
        class MockWebSocket:
            def __init__(self):
                self.messages = []
            
            async def send_json(self, data):
                self.messages.append(data)
        
        mock_ws = MockWebSocket()
        connection_manager.active_connections["client_1"] = mock_ws
        
        message = {"type": "ping"}
        
        await handler.handle_message(mock_ws, "client_1", message)
        
        # Should receive pong
        assert len(mock_ws.messages) == 1
        assert mock_ws.messages[0]["event_type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_handle_invalid_message_type(self, setup):
        """Test handling invalid message type"""
        handler, connection_manager = setup
        
        class MockWebSocket:
            def __init__(self):
                self.messages = []
            
            async def send_json(self, data):
                self.messages.append(data)
        
        mock_ws = MockWebSocket()
        
        message = {"type": "invalid_type"}
        
        await handler.handle_message(mock_ws, "client_1", message)
        
        # Should receive error
        assert len(mock_ws.messages) == 1
        assert mock_ws.messages[0]["event_type"] == "error"
    
    @pytest.mark.asyncio
    async def test_handle_missing_message_type(self, setup):
        """Test handling message with missing type"""
        handler, connection_manager = setup
        
        class MockWebSocket:
            def __init__(self):
                self.messages = []
            
            async def send_json(self, data):
                self.messages.append(data)
        
        mock_ws = MockWebSocket()
        
        message = {"data": "some data"}  # No type field
        
        await handler.handle_message(mock_ws, "client_1", message)
        
        # Should receive error
        assert len(mock_ws.messages) == 1
        assert mock_ws.messages[0]["event_type"] == "error"


class TestWebSocketEndpoint:
    """Test WebSocket API endpoint"""
    
    def test_websocket_stats_endpoint(self):
        """Test WebSocket statistics endpoint"""
        client = TestClient(app)
        response = client.get("/ws/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_connections" in data
        assert "clients" in data
        assert "timestamp" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
