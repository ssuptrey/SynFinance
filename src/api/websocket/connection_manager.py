"""
WebSocket Connection Manager

Manages WebSocket connections, client tracking, and event broadcasting.
Week 8 Day 2: WebSocket Support
"""

import asyncio
import json
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict

from src.api.websocket.events import Event, EventType
from src.observability import get_logger, LogCategory

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections and event broadcasting
    
    Handles:
    - Connection lifecycle (connect, disconnect)
    - Client metadata tracking
    - Event broadcasting to subscribed clients
    - Topic-based subscriptions
    """
    
    def __init__(self):
        """Initialize connection manager"""
        # Active connections: client_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Client metadata: client_id -> metadata dict
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Subscriptions: topic -> set of client_ids
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Tenant associations: client_id -> tenant_id
        self.client_tenants: Dict[str, str] = {}
        
        # Connection timestamps
        self.connection_times: Dict[str, datetime] = {}
        
        logger.info(
            "WebSocket Connection Manager initialized",
            category=LogCategory.SYSTEM
        )
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Accept and register a new WebSocket connection
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
            tenant_id: Optional tenant ID for multi-tenancy
            metadata: Optional client metadata
        """
        await websocket.accept()
        
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = metadata or {}
        self.connection_times[client_id] = datetime.utcnow()
        
        if tenant_id:
            self.client_tenants[client_id] = tenant_id
        
        logger.info(
            f"WebSocket client connected: {client_id}",
            category=LogCategory.SYSTEM,
            extra={
                "client_id": client_id,
                "tenant_id": tenant_id,
                "total_connections": len(self.active_connections)
            }
        )
        
        # Send connection established event
        await self.send_personal_message(
            Event(
                event_type=EventType.CONNECTION_ESTABLISHED,
                data={
                    "client_id": client_id,
                    "server_time": datetime.utcnow().isoformat(),
                    "message": "WebSocket connection established"
                }
            ),
            client_id
        )
    
    def disconnect(self, client_id: str) -> None:
        """
        Disconnect and cleanup a client
        
        Args:
            client_id: Client identifier to disconnect
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if client_id in self.client_metadata:
            del self.client_metadata[client_id]
        
        if client_id in self.client_tenants:
            del self.client_tenants[client_id]
        
        if client_id in self.connection_times:
            del self.connection_times[client_id]
        
        # Remove from all subscriptions
        for topic_subscribers in self.subscriptions.values():
            topic_subscribers.discard(client_id)
        
        logger.info(
            f"WebSocket client disconnected: {client_id}",
            category=LogCategory.SYSTEM,
            extra={
                "client_id": client_id,
                "remaining_connections": len(self.active_connections)
            }
        )
    
    def subscribe(self, client_id: str, topic: str) -> None:
        """
        Subscribe client to a topic
        
        Args:
            client_id: Client identifier
            topic: Topic to subscribe to
        """
        self.subscriptions[topic].add(client_id)
        
        logger.debug(
            f"Client {client_id} subscribed to topic: {topic}",
            category=LogCategory.SYSTEM,
            extra={
                "client_id": client_id,
                "topic": topic,
                "topic_subscribers": len(self.subscriptions[topic])
            }
        )
    
    def unsubscribe(self, client_id: str, topic: str) -> None:
        """
        Unsubscribe client from a topic
        
        Args:
            client_id: Client identifier
            topic: Topic to unsubscribe from
        """
        self.subscriptions[topic].discard(client_id)
        
        logger.debug(
            f"Client {client_id} unsubscribed from topic: {topic}",
            category=LogCategory.SYSTEM
        )
    
    async def send_personal_message(self, event: Event, client_id: str) -> bool:
        """
        Send message to a specific client
        
        Args:
            event: Event to send
            client_id: Target client ID
        
        Returns:
            True if sent successfully, False otherwise
        """
        if client_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[client_id]
            await websocket.send_json(event.to_dict())
            return True
        except Exception as e:
            logger.error(
                f"Failed to send message to client {client_id}",
                category=LogCategory.SYSTEM,
                extra={"error": str(e), "client_id": client_id}
            )
            return False
    
    async def broadcast(
        self,
        event: Event,
        topic: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> int:
        """
        Broadcast event to multiple clients
        
        Args:
            event: Event to broadcast
            topic: If specified, only send to subscribers of this topic
            tenant_id: If specified, only send to clients in this tenant
        
        Returns:
            Number of clients that received the message
        """
        # Determine target clients
        if topic:
            target_clients = self.subscriptions.get(topic, set())
        else:
            target_clients = set(self.active_connections.keys())
        
        # Filter by tenant if specified
        if tenant_id:
            target_clients = {
                client_id for client_id in target_clients
                if self.client_tenants.get(client_id) == tenant_id
            }
        
        # Send to all target clients
        sent_count = 0
        failed_clients = []
        
        for client_id in target_clients:
            try:
                websocket = self.active_connections.get(client_id)
                if websocket:
                    await websocket.send_json(event.to_dict())
                    sent_count += 1
            except Exception as e:
                logger.warning(
                    f"Failed to send broadcast to client {client_id}",
                    category=LogCategory.SYSTEM,
                    extra={"error": str(e), "client_id": client_id}
                )
                failed_clients.append(client_id)
        
        # Cleanup failed connections
        for client_id in failed_clients:
            self.disconnect(client_id)
        
        logger.debug(
            f"Broadcast event {event.event_type} to {sent_count} clients",
            category=LogCategory.SYSTEM,
            extra={
                "event_type": event.event_type,
                "sent_count": sent_count,
                "topic": topic,
                "tenant_id": tenant_id
            }
        )
        
        return sent_count
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_topic_subscribers(self, topic: str) -> int:
        """Get number of subscribers for a topic"""
        return len(self.subscriptions.get(topic, set()))
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get client information"""
        if client_id not in self.active_connections:
            return None
        
        return {
            "client_id": client_id,
            "tenant_id": self.client_tenants.get(client_id),
            "metadata": self.client_metadata.get(client_id, {}),
            "connected_at": self.connection_times.get(client_id).isoformat()
            if client_id in self.connection_times else None,
            "subscriptions": [
                topic for topic, subs in self.subscriptions.items()
                if client_id in subs
            ]
        }
    
    def get_all_clients(self) -> List[Dict[str, Any]]:
        """Get information about all connected clients"""
        return [
            self.get_client_info(client_id)
            for client_id in self.active_connections.keys()
        ]


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get global connection manager instance"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager
