import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so `services` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(page_title="AuraGuard Caregiver Portal", layout="wide")

# ── Settings ──────────────────────────────────────────────────────────────────
from dashboard.settings import get_settings

try:
    settings = get_settings()
except SystemExit:
    st.error("Missing required environment variables. Check your .env file.")
    sys.exit(1)

# ── Data helpers ──────────────────────────────────────────────────────────────
from dashboard.data.mongodb_reader import fetch_latest_events
from dashboard.data.snowflake_reader import fetch_health_trends

# ── Components ────────────────────────────────────────────────────────────────
from dashboard.components.event_feed import render_event_feed
from dashboard.components.health_charts import render_health_chart

# ── Paths ─────────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = Path(__file__).parent.parent / "vision" / "known_faces"

# ── Session state defaults ────────────────────────────────────────────────────
if "events" not in st.session_state:
    st.session_state.events = []
if "mongo_error" not in st.session_state:
    st.session_state.mongo_error = False
if "health_df" not in st.session_state:
    st.session_state.health_df = None
if "snowflake_error" not in st.session_state:
    st.session_state.snowflake_error = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

# ── Fetch data ────────────────────────────────────────────────────────────────
events, mongo_error = fetch_latest_events()
st.session_state.events = events
st.session_state.mongo_error = mongo_error

health_df, snowflake_error = fetch_health_trends()
st.session_state.health_df = health_df
st.session_state.snowflake_error = snowflake_error

st.session_state.last_refresh = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ── Sidebar — Family Sync ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Family Sync")
    uploaded_photo = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
    name_input = st.text_input("Name")
    relationship_input = st.text_input("Relationship")
    background_input = st.text_area("Background")
    last_convo_input = st.text_area("Last Conversation")

    if st.button("Save"):
        missing = []
        if not uploaded_photo:
            missing.append("photo")
        if not name_input.strip():
            missing.append("Name")
        if not relationship_input.strip():
            missing.append("Relationship")
        if not background_input.strip():
            missing.append("Background")
        if not last_convo_input.strip():
            missing.append("Last Conversation")

        if missing:
            st.error(f"Please fill in: {', '.join(missing)}")
        else:
            slug = name_input.strip().lower().replace(" ", "_")
            KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

            img_path = KNOWN_FACES_DIR / f"{slug}.jpg"
            img_path.write_bytes(uploaded_photo.read())

            profile = {
                "name": name_input.strip(),
                "relationship": relationship_input.strip(),
                "background": background_input.strip(),
                "last_conversation": last_convo_input.strip(),
            }
            json_path = KNOWN_FACES_DIR / f"{slug}.json"
            json_path.write_text(json.dumps(profile, indent=2))

            st.success(f"Saved! Vision Engine will detect {name_input.strip()} on next startup.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("AuraGuard AI — Caregiver Portal")
st.caption(f"Patient: **{settings.PATIENT_NAME}** · Last refresh: {st.session_state.last_refresh}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_feed, tab_health = st.tabs(["Live Feed", "Health Trends"])

with tab_feed:
    render_event_feed(st.session_state.events, st.session_state.mongo_error)

with tab_health:
    render_health_chart(st.session_state.health_df, st.session_state.snowflake_error)

# ── Auto-refresh every 5 seconds ──────────────────────────────────────────────
time.sleep(5)
st.rerun()
