"""WebSocket endpoint for real-time signal push."""

import json
import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()

# Connected WebSocket clients
_connected: set[WebSocket] = set()


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """WebSocket endpoint for real-time signal streaming.

    Clients connect here to receive signals as they are generated
    by the background scanner.
    """
    await websocket.accept()
    _connected.add(websocket)
    logger.info(f"WebSocket client connected (total: {len(_connected)})")

    try:
        # Send initial connection confirmation
        await websocket.send_json({"type": "connected", "message": "Connected to live signal feed"})

        # Keep connection alive, listening for client messages
        while True:
            data = await websocket.receive_text()
            # Clients can send pings or filter preferences
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        _connected.discard(websocket)
        logger.info(f"WebSocket client removed (total: {len(_connected)})")


async def broadcast_signals(signals: list):
    """Broadcast new signals to all connected WebSocket clients.

    Called by the scheduler when new signals are generated.
    """
    if not _connected or not signals:
        return

    payload = {
        "type": "signals",
        "count": len(signals),
        "signals": [
            {
                "ticker": s.ticker,
                "direction": s.direction.value,
                "confidence": s.confidence,
                "strategy": s.strategy,
                "reasoning": s.reasoning,
                "price": s.price_at_signal,
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
            }
            for s in signals
        ],
    }

    # Broadcast to all clients concurrently
    disconnected = set()
    for ws in _connected:
        try:
            await ws.send_json(payload)
        except Exception:
            disconnected.add(ws)

    # Clean up disconnected
    for ws in disconnected:
        _connected.discard(ws)
