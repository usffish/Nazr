"""
services/webapp/app.py — Event audio webapp.

Watches tempfiles/ for new event JSON files and speaks them aloud via ElevenLabs.
Falls back to browser Web Speech API if ElevenLabs is unavailable.
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse

load_dotenv()

TEMPFILES = Path("/Users/mtb/Programming/Hackabull-2026/tempfiles")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

app = FastAPI()
# Pre-seed _seen with all files that exist at startup so they are never replayed
_seen: set[str] = {p.name for p in TEMPFILES.glob("event_*.json")} if TEMPFILES.exists() else set()
_queue: deque[dict] = deque()   # events pushed via /ingest


@app.post("/ingest")
async def ingest(request: Request):
    """Receive an event directly from the vision engine."""
    event = await request.json()
    _queue.append(event)
    return {"status": "ok"}


@app.get("/poll")
def poll():
    """Return all pending events (pushed via /ingest + any new JSON files)."""
    new = []

    while _queue:
        new.append(_queue.popleft())

    for path in sorted(TEMPFILES.glob("event_*.json")):
        if path.name not in _seen:
            _seen.add(path.name)
            try:
                ev = json.loads(path.read_text())
                if not any(e.get("event_id") == ev.get("event_id") for e in new):
                    new.append(ev)
            except Exception:
                pass

    return new


@app.post("/speak")
async def speak(request: Request):
    """Convert text to speech via ElevenLabs and return MP3 audio."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "no text"}, status_code=400)

    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        return JSONResponse({"error": "elevenlabs not configured"}, status_code=503)

    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio_iter = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_flash_v2_5",
        )
        audio_bytes = b"".join(audio_iter)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>AuraGuard — Live Events</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e0e0e0;
      padding: 32px;
    }
    h1 { font-size: 1.5rem; margin-bottom: 8px; color: #fff; }
    #status { font-size: 0.85rem; color: #888; margin-bottom: 28px; }
    #status.speaking { color: #4ade80; }
    #feed { display: flex; flex-direction: column; gap: 14px; }
    .card {
      background: #1a1d27;
      border: 1px solid #2a2d3a;
      border-radius: 10px;
      padding: 18px 22px;
      animation: slideIn 0.3s ease;
    }
    .card.identity { border-left: 4px solid #60a5fa; }
    .card.health   { border-left: 4px solid #4ade80; }
    @keyframes slideIn {
      from { opacity: 0; transform: translateY(-8px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .badge {
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 3px 9px;
      border-radius: 20px;
    }
    .badge.identity { background: #1e3a5f; color: #60a5fa; }
    .badge.health   { background: #14532d; color: #4ade80; }
    .ts { font-size: 0.75rem; color: #555; }
    .voice-script {
      font-size: 1rem;
      line-height: 1.5;
      color: #d1d5db;
      margin-bottom: 8px;
    }
    .meta { font-size: 0.78rem; color: #6b7280; }
    pre {
      background: #111318;
      border-radius: 6px;
      padding: 12px;
      font-size: 0.75rem;
      color: #9ca3af;
      overflow-x: auto;
      margin-top: 10px;
    }
    #mute-btn {
      position: fixed;
      top: 24px; right: 32px;
      background: #1a1d27;
      border: 1px solid #2a2d3a;
      color: #e0e0e0;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.85rem;
    }
    #mute-btn:hover { background: #2a2d3a; }
  </style>
</head>
<body>
  <h1>AuraGuard — Live Events</h1>
  <div id="status">Listening for events…</div>
  <button id="mute-btn" onclick="toggleMute()">🔊 Mute</button>
  <div id="feed"></div>

  <script>
    let muted = false;
    let currentAudio = null;

    function toggleMute() {
      muted = !muted;
      document.getElementById('mute-btn').textContent = muted ? '🔇 Unmute' : '🔊 Mute';
      if (muted && currentAudio) {
        currentAudio.pause();
        currentAudio = null;
      }
    }

    function voiceText(event) {
      if (event.voice_script) return event.voice_script;
      if (event.type === 'health') {
        const item = event.metadata?.detected_item || event.subtype;
        return 'Health alert: ' + item + ' detected.';
      }
      const name = event.metadata?.person_profile?.name || 'someone';
      return name + ' has been detected.';
    }

    function speakFallback(text) {
      const synth = window.speechSynthesis;
      synth.cancel();
      const utt = new SpeechSynthesisUtterance(text);
      utt.rate = 0.95;
      utt.pitch = 1;
      const statusEl = document.getElementById('status');
      utt.onstart  = () => { statusEl.textContent = '🔊 Speaking…'; statusEl.className = 'speaking'; };
      utt.onend    = () => { statusEl.textContent = 'Listening for events…'; statusEl.className = ''; };
      utt.onerror  = () => { statusEl.textContent = 'Listening for events…'; statusEl.className = ''; };
      synth.speak(utt);
    }

    async function speak(text) {
      if (muted || !text) return;
      const statusEl = document.getElementById('status');

      // Stop any currently playing audio
      if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
      }

      try {
        const res = await fetch('/speak', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });

        if (!res.ok) {
          speakFallback(text);
          return;
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        currentAudio = audio;

        statusEl.textContent = '🔊 Speaking…';
        statusEl.className = 'speaking';

        audio.onended = () => {
          statusEl.textContent = 'Listening for events…';
          statusEl.className = '';
          URL.revokeObjectURL(url);
          if (currentAudio === audio) currentAudio = null;
        };
        audio.onerror = () => {
          statusEl.textContent = 'Listening for events…';
          statusEl.className = '';
          URL.revokeObjectURL(url);
          if (currentAudio === audio) currentAudio = null;
          speakFallback(text);
        };

        await audio.play();
      } catch (_) {
        speakFallback(text);
      }
    }

    function formatTs(ts) {
      try { return new Date(ts).toLocaleTimeString(); } catch { return ts; }
    }

    function addCard(event) {
      const feed = document.getElementById('feed');
      const card = document.createElement('div');
      card.className = 'card ' + (event.type || '');

      const subtype = (event.subtype || '').replace(/_/g, ' ');
      const ts = formatTs(event.timestamp || '');

      let bodyHtml = '';
      if (event.voice_script) {
        bodyHtml += `<div class="voice-script">${event.voice_script}</div>`;
      }
      if (event.type === 'health') {
        const item = event.metadata?.detected_item || '—';
        bodyHtml += `<div class="meta">Detected: <strong>${item}</strong></div>`;
      }
      if (event.type === 'identity') {
        const p = event.metadata?.person_profile || {};
        if (p.name) bodyHtml += `<div class="meta">${p.relationship || ''} · ${p.name}</div>`;
      }
      bodyHtml += `<pre>${JSON.stringify(event, null, 2)}</pre>`;

      card.innerHTML = `
        <div class="card-header">
          <span class="badge ${event.type}">${subtype || event.type}</span>
          <span class="ts">${ts}</span>
        </div>
        ${bodyHtml}
      `;

      feed.insertBefore(card, feed.firstChild);
    }

    async function poll() {
      try {
        const res = await fetch('/poll');
        const events = await res.json();
        for (const ev of events) {
          addCard(ev);
          speak(voiceText(ev));
        }
      } catch (_) {}
    }

    setInterval(poll, 1000);
    poll();
  </script>
</body>
</html>"""
