import pandas as pd
import plotly.express as px
import streamlit as st


def render_health_chart(df: pd.DataFrame | None, snowflake_error: bool):
    if snowflake_error:
        st.warning("Health trends unavailable — MongoDB connection failed")
        return

    if df is None or df.empty:
        st.info("No health events in the last 24 hours.")
        return

    fig = px.line(
        df,
        x="hour",
        y="count",
        color="subtype",
        markers=True,
        title="Health Events — Last 24 Hours",
        labels={"hour": "Time", "count": "Events", "subtype": "Activity"},
        color_discrete_map={
            "eating": "#4fc3f7",
            "drinking": "#81c784",
            "medicine_taken": "#ffb74d",
        },
    )
    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#c9d1d9",
        xaxis=dict(gridcolor="#30363d"),
        yaxis=dict(gridcolor="#30363d", dtick=1),
        legend_title_text="Activity",
    )
    st.plotly_chart(fig, use_container_width=True)
