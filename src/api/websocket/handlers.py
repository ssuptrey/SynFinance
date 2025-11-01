"""
WebSocket Message Handlers

Handles incoming WebSocket messages and routes them appropriately.
Week 8 Day 2: WebSocket Support
"""

import json
from typing import Dict, Any, Optional
from fastapi import WebSocket

from src.api.websocket.connection_manager import ConnectionManager
from src.api.websocket.events import Event, EventType
from src.observability import get_logger, LogCategory

logger = get_logger(__name__)


class WebSocketHandler:
    """Handles WebSocket message routing and processing"""
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize WebSocket handler
        
        Args:
            connection_manager: Connection manager instance
        """
        self.connection_manager = connection_manager
    
    async def handle_message(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any]
    ) -> None:
        """
        Handle incoming WebSocket message
        
        Args:
            websocket: WebSocket connection
            client_id: Client identifier
            message: Parsed message data
        """
        message_type = message.get("type")
        
        if not message_type:
            await self._send_error(websocket, "Missing message type")
            return
        
        try:
            if message_type == "subscribe":
                await self._handle_subscribe(client_id, message)
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe(client_id, message)
            elif message_type == "ping":
                await self._handle_ping(websocket, client_id)
            elif message_type == "get_status":
                await self._handle_status(websocket, client_id)
            else:
                await self._send_error(
                    websocket,
                    f"Unknown message type: {message_type}"
                )
        
        except Exception as e:
            logger.error(
                f"Error handling WebSocket message",
                category=LogCategory.SYSTEM,
                extra={"error": str(e), "client_id": client_id, "message_type": message_type}
            )
            await self._send_error(websocket, f"Internal error: {str(e)}")
    
    async def _handle_subscribe(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle subscription request"""
        topic = message.get("topic")
        
        if not topic:
            return
        
        self.connection_manager.subscribe(client_id, topic)
        
        logger.info(
            f"Client {client_id} subscribed to {topic}",
            category=LogCategory.SYSTEM
        )
    
    async def _handle_unsubscribe(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle unsubscribe request"""
        topic = message.get("topic")
        
        if not topic:
            return
        
        self.connection_manager.unsubscribe(client_id, topic)
        
        logger.info(
            f"Client {client_id} unsubscribed from {topic}",
            category=LogCategory.SYSTEM
        )
    
    async def _handle_ping(self, websocket: WebSocket, client_id: str) -> None:
        """Handle ping request"""
        event = Event(
            event_type=EventType.PONG,
            data={"message": "pong"}
        )
        await self.connection_manager.send_personal_message(event, client_id)
    
    async def _handle_status(self, websocket: WebSocket, client_id: str) -> None:
        """Handle status request"""
        client_info = self.connection_manager.get_client_info(client_id)
        
        if client_info:
            event = Event(
                event_type=EventType.SYSTEM_ALERT,
                data={
                    "type": "status_response",
                    "client_info": client_info,
                    "total_connections": self.connection_manager.get_connection_count()
                }
            )
            await self.connection_manager.send_personal_message(event, client_id)
    
    async def _send_error(self, websocket: WebSocket, error_message: str) -> None:
        """Send error message to client"""
        await websocket.send_json({
            "event_type": "error",
            "data": {"message": error_message}
        })
