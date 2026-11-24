"""
Root entry point: python server.py
Starts the FastAPI server on http://localhost:8000
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("\n" + "=" * 62)
print("  [+] Aura Healthcare -- Clinical Voice Receptionist")
print("  [*] Initializing server on http://localhost:8000 ...")
print("=" * 62 + "\n", flush=True)

# Ensure the root directory is on the python module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.server import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"  [+] Web Dashboard:  http://localhost:{port}")
    print(f"  [+] Live WebSocket: ws://localhost:{port}/ws")
    print("=" * 62 + "\n", flush=True)
    uvicorn.run(app, host=host, port=port)
