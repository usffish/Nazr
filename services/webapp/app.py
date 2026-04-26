"""
services/webapp/app.py — Event audio webapp.

Watches tempfiles/ for new event JSON files and speaks them aloud via ElevenLabs.
Audio is routed through a canvas-backed video element so that Picture-in-Picture
mode keeps audio playing while SpecBridge is in the foreground on iOS.
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
_seen: set[str] = {p.name for p in TEMPFILES.glob("event_*.json")} if TEMPFILES.exists() else set()
_queue: deque[dict] = deque()


@app.post("/ingest")
async def ingest(request: Request):
    event = await request.json()
    _queue.append(event)
    return {"status": "ok"}


@app.get("/poll")
def poll():
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


async def _tts_elevenlabs(text: str) -> bytes:
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio_iter = client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id="eleven_flash_v2_5",
    )
    return b"".join(audio_iter)


async def _tts_edge(text: str) -> bytes:
    import edge_tts
    chunks: list[bytes] = []
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


@app.post("/speak")
async def speak(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "no text"}, status_code=400)

    # Try ElevenLabs first, fall back to edge-tts (free) on any failure
    if ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
        try:
            audio_bytes = await _tts_elevenlabs(text)
            return Response(content=audio_bytes, media_type="audio/mpeg")
        except Exception:
            pass  # fall through to edge-tts

    try:
        audio_bytes = await _tts_edge(text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AuraGuard — Live Events</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e0e0e0;
      padding: 32px 24px;
    }
    h1 { font-size: 1.5rem; margin-bottom: 6px; color: #fff; }
    #status { font-size: 0.85rem; color: #888; margin-bottom: 20px; }
    #status.speaking { color: #4ade80; }

    #controls {
      display: flex;
      gap: 10px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }
    .btn {
      background: #1a1d27;
      border: 1px solid #2a2d3a;
      color: #e0e0e0;
      padding: 10px 18px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.9rem;
    }
    .btn:hover { background: #2a2d3a; }
    .btn.pip-active { border-color: #4ade80; color: #4ade80; }

    #pip-hint {
      font-size: 0.78rem;
      color: #555;
      margin-bottom: 20px;
    }

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
      display: flex; justify-content: space-between;
      align-items: center; margin-bottom: 10px;
    }
    .badge {
      font-size: 0.7rem; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.08em;
      padding: 3px 9px; border-radius: 20px;
    }
    .badge.identity { background: #1e3a5f; color: #60a5fa; }
    .badge.health   { background: #14532d; color: #4ade80; }
    .ts { font-size: 0.75rem; color: #555; }
    .voice-script { font-size: 1rem; line-height: 1.5; color: #d1d5db; margin-bottom: 8px; }
    .meta { font-size: 0.78rem; color: #6b7280; }
    pre {
      background: #111318; border-radius: 6px; padding: 12px;
      font-size: 0.75rem; color: #9ca3af; overflow-x: auto; margin-top: 10px;
    }

    /* Hidden PiP video — not shown in page, only in PiP overlay */
    #pip-video { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
    #pip-canvas { display: none; }
  </style>
</head>
<body>
  <h1>AuraGuard — Live Events</h1>
  <div id="status">Tap "Start" to activate audio</div>

  <div id="controls">
    <button class="btn" id="start-btn" onclick="startAudio()">▶ Start</button>
    <button class="btn" id="pip-btn"   onclick="togglePiP()" disabled>⧉ Enter PiP</button>
    <button class="btn" id="mute-btn"  onclick="toggleMute()" disabled>🔊 Mute</button>
  </div>

  <div id="pip-hint">
    Tap <strong>Enter PiP</strong> then switch to SpecBridge — AuraGuard floats on top and audio plays through your glasses.
  </div>

  <canvas id="pip-canvas" width="320" height="180"></canvas>
  <video  id="pip-video" playsinline></video>

  <div id="feed"></div>

  <script>
    let muted = false;
    let audioCtx = null;
    let audioDest = null;
    let pipVideo = null;
    let canvasCtx = null;
    let pipActive = false;
    let statusText = 'Listening for events…';
    let lastEvent = null;

    // ── Canvas drawing (what appears in the PiP window) ────────────────────────
    function drawPiP() {
      if (!canvasCtx) return;
      const c = document.getElementById('pip-canvas');
      const w = c.width, h = c.height;

      canvasCtx.fillStyle = '#0f1117';
      canvasCtx.fillRect(0, 0, w, h);

      // Header bar
      canvasCtx.fillStyle = '#1a1d27';
      canvasCtx.fillRect(0, 0, w, 36);
      canvasCtx.fillStyle = '#4ade80';
      canvasCtx.font = 'bold 15px -apple-system, sans-serif';
      canvasCtx.textAlign = 'left';
      canvasCtx.fillText('AuraGuard', 12, 24);

      if (lastEvent) {
        const ev = lastEvent;
        const isIdentity = ev.type === 'identity';
        canvasCtx.fillStyle = isIdentity ? '#60a5fa' : '#4ade80';
        canvasCtx.font = 'bold 13px sans-serif';
        canvasCtx.textAlign = 'left';
        const badge = (ev.subtype || ev.type || '').replace(/_/g,' ').toUpperCase();
        canvasCtx.fillText(badge, 12, 60);

        const script = ev.voice_script || '';
        canvasCtx.fillStyle = '#d1d5db';
        canvasCtx.font = '12px sans-serif';
        wrapText(canvasCtx, script, 12, 82, w - 24, 18);
      } else {
        canvasCtx.fillStyle = '#555';
        canvasCtx.font = '13px sans-serif';
        canvasCtx.textAlign = 'center';
        canvasCtx.fillText(statusText, w / 2, h / 2);
      }

      requestAnimationFrame(drawPiP);
    }

    function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
      const words = text.split(' ');
      let line = '';
      for (const word of words) {
        const test = line ? line + ' ' + word : word;
        if (ctx.measureText(test).width > maxWidth && line) {
          ctx.fillText(line, x, y);
          line = word;
          y += lineHeight;
          if (y > 170) { ctx.fillText(line + '…', x, y); return; }
        } else {
          line = test;
        }
      }
      if (line) ctx.fillText(line, x, y);
    }

    // ── Audio setup (must happen after a user gesture on iOS) ──────────────────
    function startAudio() {
      if (audioCtx) return;

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioDest = audioCtx.createMediaStreamDestination();

      const canvas = document.getElementById('pip-canvas');
      canvasCtx = canvas.getContext('2d');
      drawPiP();

      const videoStream = canvas.captureStream(10);
      const audioTrack = audioDest.stream.getAudioTracks()[0];
      if (audioTrack) videoStream.addTrack(audioTrack);

      pipVideo = document.getElementById('pip-video');
      pipVideo.srcObject = videoStream;
      pipVideo.muted = false;
      pipVideo.play().catch(() => {});

      document.getElementById('start-btn').textContent = '✓ Audio Ready';
      document.getElementById('start-btn').disabled = true;
      document.getElementById('pip-btn').disabled = false;
      document.getElementById('mute-btn').disabled = false;
      document.getElementById('status').textContent = 'Listening for events…';

      // Listen for PiP being dismissed externally (e.g. user closes it)
      pipVideo.addEventListener('leavepictureinpicture', () => {
        pipActive = false;
        document.getElementById('pip-btn').textContent = '⧉ Enter PiP';
        document.getElementById('pip-btn').classList.remove('pip-active');
      });
    }

    // ── PiP toggle ─────────────────────────────────────────────────────────────
    async function togglePiP() {
      if (!audioCtx) { startAudio(); return; }
      if (audioCtx.state === 'suspended') await audioCtx.resume();

      try {
        if (!document.pictureInPictureElement) {
          await pipVideo.requestPictureInPicture();
          pipActive = true;
          document.getElementById('pip-btn').textContent = '⧉ Exit PiP';
          document.getElementById('pip-btn').classList.add('pip-active');
          document.getElementById('pip-hint').textContent =
            'PiP active — switch to SpecBridge. Audio plays through your glasses.';
        } else {
          await document.exitPictureInPicture();
        }
      } catch (e) {
        alert('PiP error: ' + e.message);
      }
    }

    // ── Mute ───────────────────────────────────────────────────────────────────
    function toggleMute() {
      muted = !muted;
      document.getElementById('mute-btn').textContent = muted ? '🔇 Unmute' : '🔊 Mute';
    }

    // ── Speak via ElevenLabs → AudioContext → PiP video ───────────────────────
    function speakFallback(text) {
      const synth = window.speechSynthesis;
      synth.cancel();
      const utt = new SpeechSynthesisUtterance(text);
      utt.rate = 0.95;
      synth.speak(utt);
    }

    async function speak(text) {
      if (muted || !text) return;
      if (!audioCtx) { speakFallback(text); return; }
      if (audioCtx.state === 'suspended') await audioCtx.resume();

      const statusEl = document.getElementById('status');
      try {
        const res = await fetch('/speak', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });

        if (!res.ok) { speakFallback(text); return; }

        const arrayBuffer = await res.arrayBuffer();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const source = audioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioDest);  // routes into PiP video stream
        source.start();

        statusText = '🔊 Speaking…';
        statusEl.textContent = '🔊 Speaking…';
        statusEl.className = 'speaking';

        source.onended = () => {
          statusText = 'Listening for events…';
          statusEl.textContent = 'Listening for events…';
          statusEl.className = '';
        };
      } catch (_) {
        speakFallback(text);
      }
    }

    // ── Cards ──────────────────────────────────────────────────────────────────
    function voiceText(ev) {
      if (ev.voice_script) return ev.voice_script;
      if (ev.type === 'health') return 'Health alert: ' + (ev.metadata?.detected_item || ev.subtype) + ' detected.';
      return (ev.metadata?.person_profile?.name || 'someone') + ' has been detected.';
    }

    function formatTs(ts) {
      try { return new Date(ts).toLocaleTimeString(); } catch { return ts; }
    }

    function addCard(ev) {
      lastEvent = ev;
      const feed = document.getElementById('feed');
      const card = document.createElement('div');
      card.className = 'card ' + (ev.type || '');
      const subtype = (ev.subtype || '').replace(/_/g, ' ');
      const ts = formatTs(ev.timestamp || '');
      let body = '';
      if (ev.voice_script) body += `<div class="voice-script">${ev.voice_script}</div>`;
      if (ev.type === 'health') body += `<div class="meta">Detected: <strong>${ev.metadata?.detected_item || '—'}</strong></div>`;
      if (ev.type === 'identity') {
        const p = ev.metadata?.person_profile || {};
        if (p.name) body += `<div class="meta">${p.relationship || ''} · ${p.name}</div>`;
      }
      body += `<pre>${JSON.stringify(ev, null, 2)}</pre>`;
      card.innerHTML = `
        <div class="card-header">
          <span class="badge ${ev.type}">${subtype || ev.type}</span>
          <span class="ts">${ts}</span>
        </div>${body}`;
      feed.insertBefore(card, feed.firstChild);
    }

    // ── Poll ───────────────────────────────────────────────────────────────────
    async function poll() {
      try {
        const res = await fetch('/poll');
        const events = await res.json();
        for (const ev of events) {
          addCard(ev);
          await speak(voiceText(ev));
        }
      } catch (_) {}
    }

    setInterval(poll, 1000);
    poll();
  </script>
</body>
</html>"""
