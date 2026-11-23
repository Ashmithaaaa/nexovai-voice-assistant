"""
Root entry point: python server.py
Starts the FastAPI server on http://localhost:8000
"""

import os
import sys
import uvicorn

# Ensure the root directory is on the python module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.server import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"\n=======================================================")
    print(f"  Aura Health Clinic - Voice AI Receptionist Server")
    print(f"  Serving on: http://localhost:{port}")
    print(f"  WebSocket URL: ws://localhost:{port}/ws")
    print(f"=======================================================\n")
    uvicorn.run("backend.server:app", host=host, port=port, reload=True)
