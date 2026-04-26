"""
run_all.py — Start the full AuraGuard stack with one command.

Services started (in order):
  1. mediamtx          — RTMP server (receives stream from SpecBridge iOS app)
  2. AI Brain          — FastAPI on :8000 (processes events, speaks via ElevenLabs)
  3. Vision Engine     — Reads RTMP stream, runs face + health detection
  4. Dashboard         — Streamlit caregiver portal on :8501

Usage:
  python run_all.py

Prerequisites:
  - mediamtx binary must be in PATH
  - .env file must be present with required keys
  - SpecBridge iOS app streams to rtmp://<this-machine-IP>:1935/live/stream
"""

import os
import shutil
import signal
import socket
import subprocess
import sys
import time

SERVICES = [
    {
        "name": "mediamtx",
        "cmd": ["mediamtx"],
        "delay": 2.0,
        "note": "RTMP server ready on :1935",
    },
    {
        "name": "AI Brain",
        "cmd": [
            sys.executable, "-m", "uvicorn",
            "services.brain.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "warning",
        ],
        "delay": 3.0,
        "note": "Brain API ready on http://localhost:8000",
    },
    {
        "name": "Vision Engine",
        "cmd": [sys.executable, "-m", "services.vision.face_recognition_engine"],
        "delay": 1.0,
        "note": "Vision Engine reading RTMP stream",
    },
    {
        "name": "Dashboard",
        "cmd": [
            sys.executable, "-m", "streamlit", "run",
            "services/dashboard/app.py",
            "--server.port", "8501",
            "--server.headless", "true",
        ],
        "delay": 0,
        "note": "Dashboard ready at http://localhost:8501",
    },
    {
        "name": "Event Audio",
        "cmd": [
            sys.executable, "-m", "uvicorn",
            "services.webapp.app:app",
            "--host", "0.0.0.0",
            "--port", "8502",
            "--log-level", "warning",
        ],
        "delay": 0,
        "note": "Event audio webapp ready at http://localhost:8502",
    },
]

_procs: list[subprocess.Popen] = []


def _shutdown(sig=None, frame=None):
    print("\n[AuraGuard] Shutting down all services...")
    for p in reversed(_procs):
        if p.poll() is None:
            p.terminate()
    for p in _procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print("[AuraGuard] All services stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if something is listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _kill_existing_mediamtx():
    """Kill any mediamtx process already holding port 1935."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", "tcp:1935"], capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid], capture_output=True)
                print(f"[mediamtx] Killed stale process on port 1935 (PID {pid})")
        if pids:
            time.sleep(0.5)
    except Exception:
        pass


def _check_mediamtx():
    if shutil.which("mediamtx") is None:
        print("[AuraGuard] ERROR: 'mediamtx' not found in PATH.")
        print("  Download from https://github.com/bluenviron/mediamtx/releases")
        print("  and place it in /usr/local/bin or add its directory to PATH.")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  AuraGuard AI — Starting all services")
    print("=" * 60)

    _check_mediamtx()
    _kill_existing_mediamtx()

    for svc in SERVICES:
        name = svc["name"]
        cmd = svc["cmd"]
        print(f"\n[{name}] Starting...")
        try:
            proc = subprocess.Popen(cmd)
            _procs.append(proc)
        except FileNotFoundError as e:
            print(f"[{name}] FAILED to start: {e}")
            _shutdown()

        if svc["delay"] > 0:
            time.sleep(svc["delay"])

        if proc.poll() is not None:
            print(f"[{name}] CRASHED immediately (exit code {proc.returncode})")
            _shutdown()

        # For mediamtx, confirm it's actually listening on :1935
        if name == "mediamtx":
            if _port_open("localhost", 1935):
                print(f"[{name}] Port 1935 confirmed open ✓")
            else:
                print(f"[{name}] WARNING: port 1935 is NOT open after startup — mediamtx may have crashed")
                print(f"[{name}] Check the output above for error messages")

        print(f"[{name}] {svc['note']}")

    print("\n" + "=" * 60)
    print("  All services running. Press Ctrl+C to stop.")
    print("  Dashboard:  http://localhost:8501")
    print("  Brain API:  http://localhost:8000")
    print("  Event Audio: http://localhost:8502")
    print("  RTMP:       rtmp://localhost:1935/live/stream")
    print("=" * 60 + "\n")

    # Watch for unexpected exits
    while True:
        for i, proc in enumerate(_procs):
            if proc.poll() is not None:
                name = SERVICES[i]["name"]
                print(f"[AuraGuard] WARNING: {name} exited unexpectedly (code {proc.returncode})")
        time.sleep(5)


if __name__ == "__main__":
    main()
