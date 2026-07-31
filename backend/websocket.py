"""
WebSocket manager for real-time alert broadcasting.

Allows dashboard clients to receive instant notifications when
critical defects are detected, without polling.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts messages.

    Usage:
        manager = WebSocketManager()
        await manager.connect(websocket, device_id="CAM-001")
        await manager.broadcast_alert({"type": "broken_yarn", "severity": "critical"})
    """

    def __init__(self):
        self._connections: List[Dict[str, Any]] = []

    async def connect(self, websocket: WebSocket, client_id: str = ""):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        conn = {
            "websocket": websocket,
            "client_id": client_id or f"client-{len(self._connections)}",
            "connected_at": None,  # will be set after first message
        }
        self._connections.append(conn)
        logger.info(
            f"WebSocket connected: {conn['client_id']} (total: {len(self._connections)})"
        )

        # Send welcome message
        await self._send(
            websocket,
            {
                "type": "connected",
                "client_id": conn["client_id"],
                "message": "Connected to Fabric Defect Detection alerts",
            },
        )

    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected client."""
        self._connections = [
            c for c in self._connections if c["websocket"] != websocket
        ]
        logger.info(f"WebSocket disconnected (remaining: {len(self._connections)})")

    async def broadcast(self, message: dict):
        """Send a message to all connected clients."""
        dead = []
        for conn in self._connections:
            try:
                await self._send(conn["websocket"], message)
            except Exception:
                dead.append(conn)
        for d in dead:
            self._connections.remove(d)

    async def broadcast_alert(self, alert: dict):
        """
        Broadcast a defect alert to all connected clients.

        Args:
            alert: Dict with type, severity, device_id, confidence, action.
        """
        await self.broadcast(
            {
                "type": "alert",
                "timestamp": None,  # Will be set by client
                "data": alert,
            }
        )

    async def broadcast_detection(self, detection: dict):
        """Broadcast a new detection to all clients."""
        await self.broadcast(
            {
                "type": "detection",
                "data": detection,
            }
        )

    async def broadcast_stats(self, stats: dict):
        """Broadcast updated system stats."""
        await self.broadcast(
            {
                "type": "stats",
                "data": stats,
            }
        )

    async def _send(self, websocket: WebSocket, message: dict):
        """Send a JSON message to a single client."""
        from datetime import datetime, timezone

        message["timestamp"] = datetime.now(timezone.utc).isoformat()
        await websocket.send_text(json.dumps(message, default=str))

    @property
    def active_connections(self) -> int:
        return len(self._connections)

    def get_client_ids(self) -> List[str]:
        return [c["client_id"] for c in self._connections]


# Singleton
_manager: WebSocketManager | None = None


def get_ws_manager() -> WebSocketManager:
    global _manager
    if _manager is None:
        _manager = WebSocketManager()
    return _manager
