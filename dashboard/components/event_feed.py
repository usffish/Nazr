import pandas as pd
import streamlit as st

DISPLAY_COLUMNS = [
    "timestamp", "type", "subtype", "confidence",
    "verified", "voice_script", "processing_status",
]

# High-contrast color pairs (background, text)
_TYPE_COLORS = {
    "health":   {"bg": "#1a3a5c", "fg": "#e8f4fd"},   # deep blue
    "identity": {"bg": "#1a4731", "fg": "#d4f5e2"},   # deep green
}
_DEFAULT_COLORS = {"bg": "#2e2e2e", "fg": "#f0f0f0"}


def _row_color(row):
    colors = _TYPE_COLORS.get(row.get("type"), _DEFAULT_COLORS)
    return [
        f"background-color: {colors['bg']}; color: {colors['fg']}; font-weight: 500"
        for _ in row
    ]


def _format_confidence(val):
    try:
        return f"{float(val):.0%}"
    except (TypeError, ValueError):
        return val


def render_event_feed(events: list[dict], mongo_error: bool):
    if mongo_error:
        st.warning("MongoDB unavailable — showing cached data")

    if not events:
        st.info("No events yet.")
        return

    df = pd.DataFrame(events)

    for col in DISPLAY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[DISPLAY_COLUMNS].copy()

    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].apply(_format_confidence)

    styled = (
        df.style
        .apply(_row_color, axis=1)
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#0d1117"),
                ("color", "#c9d1d9"),
                ("font-weight", "600"),
                ("border-bottom", "2px solid #30363d"),
                ("padding", "8px 12px"),
            ]},
            {"selector": "td", "props": [
                ("padding", "6px 12px"),
                ("border-bottom", "1px solid #30363d"),
            ]},
        ])
    )

    st.dataframe(styled, use_container_width=True, height=500)
