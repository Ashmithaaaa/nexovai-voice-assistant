"""
server.py - FastAPI Server & WebSocket Gateway

Serves the Clinic Management Dashboard and handles real-time voice sessions
connecting the browser frontend to the Pipecat AI agent.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Load environment variables
load_dotenv()

from backend.bot import run_bot
from backend.data import (
    get_doctors_data,
    get_appointments_data,
    register_listener,
    DOCTORS,
    APPOINTMENTS,
)

app = FastAPI(title="Aura Health Clinic - Voice AI Receptionist")

# Enable CORS for local testing & integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active connected client WebSockets
connected_clients: Set[WebSocket] = set()


# Register data listener for live dashboard synchronization
async def on_data_changed(event_type: str, data: Dict[str, Any]):
    """Broadcast state changes to all connected browser dashboards."""
    logger.info(f"[Server] Data changed ({event_type}). Broadcasting to {len(connected_clients)} clients.")
    message = json.dumps({
        "type": "clinic_update",
        "event": event_type,
        "doctors": get_doctors_data(),
        "appointments": get_appointments_data(),
        "event_data": data,
    })

    disconnected = set()
    for ws in list(connected_clients):
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)

    connected_clients.difference_update(disconnected)

# Wire data listener
register_listener(on_data_changed)


@app.get("/favicon.ico")
async def favicon():
    """Return empty 204 for favicon to prevent 404 noise."""
    from fastapi import Response
    return Response(status_code=204)



# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Health check & configuration status."""
    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip() and not os.getenv("GEMINI_API_KEY", "").startswith("your_"))
    has_eleven = bool(os.getenv("ELEVENLABS_API_KEY", "").strip() and not os.getenv("ELEVENLABS_API_KEY", "").startswith("your_"))

    return {
        "status": "healthy",
        "llm": "Google Gemini (gemini-2.0-flash)",
        "stt": "ElevenLabs STT",
        "tts": "ElevenLabs TTS",
        "gemini_configured": has_gemini,
        "elevenlabs_configured": has_eleven,
        "active_connections": len(connected_clients),
    }


@app.get("/api/doctors")
async def get_doctors():
    """Return all clinic doctors and real-time slot statuses."""
    return {"doctors": get_doctors_data()}


@app.get("/api/appointments")
async def get_appointments():
    """Return all appointments with status."""
    return {"appointments": get_appointments_data()}


@app.post("/api/reset")
async def reset_data():
    """Reset data store to clean initial state for demonstration."""
    from backend.data import DOCTORS as D_INIT, APPOINTMENTS as A_INIT
    # Reset slots
    for doc in DOCTORS:
        for s in doc["available_slots"]:
            s["status"] = "available"
    # Reset appointments
    APPOINTMENTS.clear()
    APPOINTMENTS.extend([
        {
            "id": "apt-101",
            "doctor_name": "Dr. Arvind Rao",
            "doctor_specialty": "Cardiology",
            "patient_name": "Siddharth Verma",
            "patient_phone": "9876543210",
            "date": "Tomorrow",
            "time_slot": "11:30 AM",
            "status": "Confirmed",
            "created_at": "Today, 10:15 AM",
        },
        {
            "id": "apt-102",
            "doctor_name": "Dr. Priya Sharma",
            "doctor_specialty": "Dermatology",
            "patient_name": "Kavita Nair",
            "patient_phone": "9123456789",
            "date": "Tomorrow",
            "time_slot": "01:30 PM",
            "status": "Confirmed",
            "created_at": "Today, 11:45 AM",
        },
    ])

    await on_data_changed("reset", {})
    return {"status": "success", "message": "Clinic database reset to initial seed data."}


# ---------------------------------------------------------------------------
# WebSocket Endpoint for Voice AI & Dashboard Real-time Events
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bi-directional WebSocket connection for:
      - Raw 16kHz PCM audio streaming (microphone in, synthesized speech out)
      - Live transcript and conversational events
      - Real-time dashboard synchronization
    """
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"[WebSocket] Client connected. Active clients: {len(connected_clients)}")

    # Send initial clinic snapshot
    try:
        await websocket.send_text(json.dumps({
            "type": "clinic_update",
            "event": "initial_load",
            "doctors": get_doctors_data(),
            "appointments": get_appointments_data(),
        }))
    except Exception as e:
        logger.error(f"[WebSocket] Failed to send initial snapshot: {e}")

    # Event callback delivering Pipecat & bot events to this client
    async def dispatch_event(event_data: Dict[str, Any]):
        try:
            await websocket.send_text(json.dumps(event_data))
        except Exception:
            pass

    try:
        # Run voice bot pipeline
        await run_bot(websocket, dispatch_event)
    except WebSocketDisconnect:
        logger.info("[WebSocket] Client disconnected cleanly.")
    except Exception as e:
        logger.warning(f"[WebSocket] Session terminated: {e}")
    finally:
        connected_clients.discard(websocket)
        logger.info(f"[WebSocket] Active clients remaining: {len(connected_clients)}")


# ---------------------------------------------------------------------------
# Static Frontend Serving
# ---------------------------------------------------------------------------
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(frontend_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server.app", host="0.0.0.0", port=8000, reload=True)
