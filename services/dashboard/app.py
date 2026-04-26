"""
AuraGuard Caregiver Dashboard — Admin + Monitoring
Tabs: Admin | Live Events | Food & Water Log | Health Trends
"""

import os
import uuid
from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi

load_dotenv()

st.set_page_config(page_title="AuraGuard Dashboard", layout="wide", page_icon="🧠")

BRAIN_URL = "http://localhost:8000"
PATIENT_NAME = os.getenv("PATIENT_NAME", "Patient")
PATIENT_ID = os.getenv("PATIENT_ID", "patient_001")
REFRESH_SECONDS = 10


@st.cache_resource
def get_collection():
    uri = os.getenv("MONGODB_URI")
    db = os.getenv("MONGODB_DB", "auraguard")
    col = os.getenv("MONGODB_COLLECTION", "events")
    client = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
    return client[db][col]


def brain_get(path: str, timeout: int = 4):
    try:
        return requests.get(f"{BRAIN_URL}{path}", timeout=timeout)
    except Exception:
        return None


def brain_post(path: str, json: dict, timeout: int = 20):
    try:
        return requests.post(f"{BRAIN_URL}{path}", json=json, timeout=timeout)
    except Exception:
        return None


def check_rtsp() -> bool:
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-i",
             os.getenv("RTSP_STREAM_URL", "rtsp://localhost:8554/live/stream"),
             "-t", "1"],
            capture_output=True, timeout=4,
        )
        return r.returncode == 0
    except Exception:
        return False


# ── Page header ───────────────────────────────────────────────────────────────

st.title(f"🧠 AuraGuard — {PATIENT_NAME}'s Dashboard")

tab_admin, tab_events, tab_food, tab_trends = st.tabs([
    "Admin", "Live Events", "Food & Water Log", "Health Trends"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ADMIN
# ═══════════════════════════════════════════════════════════════════════════════

with tab_admin:

    # ── System Status ──────────────────────────────────────────────────────────
    st.subheader("System Status")

    status_resp = brain_get("/status")
    brain_ok = status_resp is not None and status_resp.status_code == 200
    status_data = status_resp.json() if brain_ok else {}

    stream_ok = check_rtsp()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brain API", "✓ Online" if brain_ok else "✗ Offline",
              delta=None, delta_color="off")
    c2.metric("MongoDB",
              "✓ Connected" if status_data.get("mongodb") == "connected" else "✗ Error",
              delta=None, delta_color="off")
    c3.metric("RTSP Stream", "✓ Live" if stream_ok else "○ No stream",
              delta=None, delta_color="off")
    c4.metric("Events logged", status_data.get("event_count", "—"),
              delta=None, delta_color="off")

    st.divider()

    # ── Audio Test ─────────────────────────────────────────────────────────────
    st.subheader("Test Audio (TTS)")
    st.caption("Plays a test message through the glasses speaker via ElevenLabs + afplay.")

    if st.button("🔊 Play Test Voice", use_container_width=False):
        with st.spinner("Synthesizing and playing..."):
            r = brain_post("/test/voice", {})
        if r and r.status_code == 200:
            st.success(f"Played: \"{r.json().get('script')}\"")
        else:
            st.error(f"Failed — Brain returned {r.status_code if r else 'no response'}")

    st.divider()

    # ── Identity Event Tests ───────────────────────────────────────────────────
    st.subheader("Test Face Recognition")
    st.caption("Injects a mock identity event into the Brain — triggers TTS and MongoDB log.")

    known_faces = [
        ("Ismail", "son"),
        ("Mohammed", "brother"),
        ("Custom", ""),
    ]

    cols = st.columns(3)
    for i, (name, rel) in enumerate(known_faces):
        with cols[i]:
            if name == "Custom":
                c_name = st.text_input("Name", "Visitor", key="custom_name")
                c_rel = st.text_input("Relationship", "friend", key="custom_rel")
            else:
                c_name, c_rel = name, rel

            if st.button(f"👤 Recognize {c_name}", key=f"face_{i}", use_container_width=True):
                payload = {
                    "type": "identity",
                    "subtype": "face_recognized",
                    "name": c_name,
                    "relationship": c_rel,
                }
                with st.spinner("Processing..."):
                    r = brain_post("/test/event", payload)
                if r and r.status_code == 200:
                    st.success("Event processed ✓")
                else:
                    st.error(f"Failed — {r.status_code if r else 'no response'}")

    st.divider()

    # ── Health Event Tests ─────────────────────────────────────────────────────
    st.subheader("Test Health Detection")
    st.caption("Injects a mock health event — triggers Gemini verification, TTS, and MongoDB log.")

    h_cols = st.columns(3)
    health_events = [
        ("eating", "🍽️ Eating"),
        ("drinking", "💧 Drinking"),
        ("medicine_taken", "💊 Medicine"),
    ]

    for i, (subtype, label) in enumerate(health_events):
        with h_cols[i]:
            if st.button(label, key=f"health_{subtype}", use_container_width=True):
                payload = {"type": "health", "subtype": subtype}
                with st.spinner("Processing..."):
                    r = brain_post("/test/event", payload)
                if r and r.status_code == 200:
                    st.success("Event processed ✓")
                else:
                    st.error(f"Failed — {r.status_code if r else 'no response'}")

    st.divider()

    # ── Raw API tester ─────────────────────────────────────────────────────────
    st.subheader("Raw API Test")
    endpoint = st.selectbox("Endpoint", ["/health", "/status", "/test/voice"])
    if st.button("Call endpoint"):
        if endpoint == "/test/voice":
            r = brain_post(endpoint, {})
        else:
            r = brain_get(endpoint)
        if r:
            st.json(r.json())
        else:
            st.error("No response from Brain")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_events:
    st.subheader("Recent Events")
    st.caption("Last 20 events from MongoDB, newest first.")

    try:
        col = get_collection()
        docs = list(col.find({}, {"_id": 0, "image_b64": 0}).sort("timestamp", -1).limit(20))
        if docs:
            for doc in docs:
                ts = doc.get("timestamp", "")[:19].replace("T", " ")
                etype = doc.get("type", "")
                subtype = doc.get("subtype", "")
                status = doc.get("processing_status", "")
                verified = doc.get("verified", None)
                script = doc.get("voice_script", "")
                meta = doc.get("metadata", {})

                color = "#2a9d8f" if etype == "identity" else "#f4a261"
                icon = "👤" if etype == "identity" else ("🍽️" if subtype == "eating" else "💧" if subtype == "drinking" else "💊")

                with st.expander(f"{icon} {ts}  —  {subtype}  ({status})", expanded=False):
                    c1, c2 = st.columns(2)
                    c1.write(f"**Type:** {etype} / {subtype}")
                    c1.write(f"**Confidence:** {doc.get('confidence', 0):.0%}")
                    c1.write(f"**Verified:** {verified}")
                    c2.write(f"**Status:** {status}")
                    if script:
                        c2.write(f"**Voice:** _{script}_")
                    if meta:
                        st.json(meta)
        else:
            st.info("No events logged yet.")
    except Exception as e:
        st.error(f"MongoDB error: {e}")

    st.caption(f"Auto-refreshes every {REFRESH_SECONDS}s")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FOOD & WATER LOG
# ═══════════════════════════════════════════════════════════════════════════════

with tab_food:
    st.subheader(f"Today — {datetime.now().strftime('%A, %B %d')}")

    try:
        col = get_collection()
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        docs = list(col.find(
            {"type": "health", "subtype": {"$in": ["eating", "drinking"]},
             "timestamp": {"$gte": start.isoformat()}},
            {"_id": 0, "timestamp": 1, "subtype": 1, "confidence": 1, "verified": 1},
        ).sort("timestamp", -1))

        if docs:
            df = pd.DataFrame(docs)
            df["time"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
            df["activity"] = df["subtype"].str.capitalize()
            df["hour"] = df["time"].dt.hour

            hourly = df.groupby(["hour", "activity"]).size().reset_index(name="count")
            fig = px.bar(hourly, x="hour", y="count", color="activity",
                         color_discrete_map={"Eating": "#f4a261", "Drinking": "#457b9d"},
                         title="Activity per Hour", barmode="group",
                         labels={"hour": "Hour", "count": "Events"})
            fig.update_xaxes(tickmode="linear", tick0=0, dtick=1)
            st.plotly_chart(fig, use_container_width=True)

            disp = df[["time", "activity", "confidence", "verified"]].copy()
            disp["time"] = disp["time"].dt.strftime("%I:%M %p")
            disp["confidence"] = disp["confidence"].map("{:.0%}".format)
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("No eating or drinking events today.")
    except Exception as e:
        st.error(f"MongoDB error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HEALTH TRENDS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_trends:
    st.subheader("Last 30 Days")

    try:
        col = get_collection()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        pipeline = [
            {"$match": {"type": "health", "timestamp": {"$gte": cutoff}}},
            {"$addFields": {"date": {"$substr": ["$timestamp", 0, 10]}}},
            {"$group": {"_id": {"date": "$date", "subtype": "$subtype"}, "count": {"$sum": 1}}},
            {"$sort": {"_id.date": 1}},
        ]
        docs = list(col.aggregate(pipeline))
        if docs:
            rows = [{"date": d["_id"]["date"], "activity": d["_id"]["subtype"].capitalize(),
                     "count": d["count"]} for d in docs]
            df = pd.DataFrame(rows)
            fig = px.line(df, x="date", y="count", color="activity", markers=True,
                          color_discrete_map={"Eating": "#f4a261", "Drinking": "#457b9d",
                                              "Medicine_taken": "#2a9d8f"},
                          title="Daily Health Activity (30 days)",
                          labels={"date": "Date", "count": "Events"})
            st.plotly_chart(fig, use_container_width=True)

            totals = df.groupby("activity")["count"].sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Eating (30d)", int(totals.get("Eating", 0)))
            c2.metric("Drinking (30d)", int(totals.get("Drinking", 0)))
            c3.metric("Medicine (30d)", int(totals.get("Medicine_taken", 0)))
        else:
            st.info("No health events in the last 30 days.")
    except Exception as e:
        st.error(f"MongoDB error: {e}")


# Auto-refresh
st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">', unsafe_allow_html=True)
