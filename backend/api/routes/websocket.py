"""
BridgeGuardian AI — Real-Time WebSocket Telemetry Gateway
Streams live tile processing progress, stream status, and real-time defect alerts to field engineers.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections for live campaign monitoring."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, campaign_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if campaign_id not in self.active_connections:
            self.active_connections[campaign_id] = []
        self.active_connections[campaign_id].append(websocket)

    def disconnect(self, campaign_id: str, websocket: WebSocket) -> None:
        if campaign_id in self.active_connections:
            if websocket in self.active_connections[campaign_id]:
                self.active_connections[campaign_id].remove(websocket)

    async def broadcast(self, campaign_id: str, message: Dict[str, Any]) -> None:
        if campaign_id in self.active_connections:
            for connection in self.active_connections[campaign_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@router.websocket("/ws/campaigns/{campaign_id}")
async def websocket_campaign_endpoint(websocket: WebSocket, campaign_id: str):
    """
    WebSocket endpoint broadcasting real-time campaign tile processing updates and defect alerts.
    """
    await manager.connect(campaign_id, websocket)
    try:
        # Send initial connected handshake
        await websocket.send_json({
            "type": "CONNECTED",
            "campaign_id": campaign_id,
            "message": "Connected to BridgeGuardian AI Real-Time Telemetry Gateway",
        })

        while True:
            # Keep connection alive and listen for client pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "PONG"})
    except WebSocketDisconnect:
        manager.disconnect(campaign_id, websocket)
