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
    event_type = row["type"] if "type" in row.index else None
    colors = _TYPE_COLORS.get(event_type, _DEFAULT_COLORS)
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
        st.info("No events yet. Make sure the Brain service is running and processing events.")
        return

    df = pd.DataFrame(events)

    for col in DISPLAY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[DISPLAY_COLUMNS].copy()

    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].apply(_format_confidence)

    # Render each row as a colored card
    for _, row in df.iterrows():
        event_type = row.get("type") or ""
        colors = _TYPE_COLORS.get(event_type, _DEFAULT_COLORS)
        bg, fg = colors["bg"], colors["fg"]

        label = f"**{row.get('subtype', '')}** · {row.get('timestamp', '')}"
        confidence = row.get("confidence", "")
        script = row.get("voice_script", "") or ""
        status = row.get("processing_status", "")
        verified = row.get("verified", "")

        st.markdown(
            f"""
            <div style="background:{bg}; color:{fg}; padding:10px 14px; border-radius:6px;
                        margin-bottom:6px; font-family:monospace; font-size:13px;">
                <span style="font-weight:700; font-size:14px;">{row.get('type','').upper()} — {row.get('subtype','')}</span>
                &nbsp;&nbsp;<span style="opacity:0.7">{row.get('timestamp','')}</span>
                &nbsp;&nbsp;<span>confidence: {confidence}</span>
                &nbsp;&nbsp;<span>status: {status}</span>
                &nbsp;&nbsp;<span>verified: {verified}</span>
                <div style="margin-top:6px; opacity:0.85; font-style:italic;">{script[:120]}{'…' if len(script) > 120 else ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
