"""
BridgeGuardian AI — Real-Time WebSocket Telemetry Gateway
Streams live multi-stage inspection updates, status changes, and defect alerts to connected clients.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

INSPECTION_STAGES = [
    {"stage": "Queued", "progress": 10, "message": "Inspection request queued in task queue"},
    {"stage": "Processing", "progress": 30, "message": "Preprocessing drone image data"},
    {"stage": "Feature Extraction", "progress": 55, "message": "Extracting OpenCV geometric crack and rust features"},
    {"stage": "Prediction", "progress": 75, "message": "Calculating Structural Health Index and RUL estimations"},
    {"stage": "Generating Report", "progress": 90, "message": "Compiling structured PDF report archive"},
    {"stage": "Completed", "progress": 100, "message": "Inspection evaluation and report generation completed"},
]


class ConnectionManager:
    """Manages active WebSocket connections per campaign / inspection channel."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, channel_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if channel_id not in self.active_connections:
            self.active_connections[channel_id] = []
        self.active_connections[channel_id].append(websocket)

    def disconnect(self, channel_id: str, websocket: WebSocket) -> None:
        if channel_id in self.active_connections:
            if websocket in self.active_connections[channel_id]:
                self.active_connections[channel_id].remove(websocket)

    async def broadcast(self, channel_id: str, message: Dict[str, Any]) -> None:
        if channel_id in self.active_connections:
            for connection in list(self.active_connections[channel_id]):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.disconnect(channel_id, connection)


manager = ConnectionManager()


@router.websocket("/ws/campaigns/{campaign_id}")
async def websocket_campaign_endpoint(websocket: WebSocket, campaign_id: str):
    """
    WebSocket endpoint broadcasting real-time inspection stage progress updates.
    Simulates or streams real-time stage progression:
    Queued -> Processing -> Feature Extraction -> Prediction -> Generating Report -> Completed
    """
    await manager.connect(campaign_id, websocket)
    try:
        await websocket.send_json({
            "type": "CONNECTED",
            "campaign_id": campaign_id,
            "message": "Connected to BridgeGuardian AI Real-Time Gateway",
        })

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "PONG"})
            elif data == "start_simulation":
                for step in INSPECTION_STAGES:
                    await websocket.send_json({
                        "type": "STATUS_UPDATE",
                        "campaign_id": campaign_id,
                        "stage": step["stage"],
                        "progress": step["progress"],
                        "message": step["message"],
                    })
                    await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        manager.disconnect(campaign_id, websocket)
