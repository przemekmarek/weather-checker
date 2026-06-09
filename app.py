"""
Multi-model point forecast viewer
---------------------------------
Click anywhere on the map and get an hourly forecast chart comparing
UKMO (Met Office), UKV 2 km, ECMWF IFS, GFS and ICON for that point:
wind speed + direction arrows, with cloud cover and precipitation overlaid.

Data source: Open-Meteo (https://open-meteo.com), which serves the native
model output of all five models (the same models Windy.com displays) via a
free, key-less API. Windy's own Point Forecast API requires a paid
Professional key for ECMWF/UKV, so Open-Meteo is used by default.

Run with:  streamlit run app.py
"""

import datetime as dt

import folium
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Multi-model wind forecast", layout="wide")

# model id (Open-Meteo) -> (label, colour)
MODELS = {
    "ukmo_uk_deterministic_2km": ("UKV 2 km (Met Office)", "#d62728"),
    "ukmo_global_deterministic_10km": ("UKMO Global 10 km", "#ff7f0e"),
    "ecmwf_ifs025": ("ECMWF IFS 0.25°", "#1f77b4"),
    "gfs_seamless": ("GFS (NOAA)", "#2ca02c"),
    "icon_seamless": ("ICON (DWD)", "#9467bd"),
}

HOURLY_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation",
]

WIND_UNITS = {"kn": "kt", "ms": "m/s", "kmh": "km/h", "mph": "mph"}

DEFAULT_LAT, DEFAULT_LON = 55.86, -4.25  # Glasgow


# ----------------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lat: float, lon: float, model_ids: tuple[str, ...],
                   days: int, wind_unit: str) -> dict:
    """One Open-Meteo call covering all requested models for the point."""
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": ",".join(HOURLY_VARS),
        "models": ",".join(model_ids),
        "forecast_days": days,
        "windspeed_unit": wind_unit,
        "timezone": "auto",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast",
                     params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def to_frames(payload: dict, model_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Split the multi-model hourly block into one tidy frame per model."""
    hourly = payload.get("hourly", {})
    time_index = pd.to_datetime(hourly.get("time", []))
    frames: dict[str, pd.DataFrame] = {}

    for mid in model_ids:
        cols = {}
        for var in HOURLY_VARS:
            # multi-model responses suffix each variable with the model id;
            # single-model responses don't
            key = f"{var}_{mid}" if f"{var}_{mid}" in hourly else var
            if key in hourly:
                cols[var] = hourly[key]
        if not cols:
            continue
        df = pd.DataFrame(cols, index=time_index)
        # UKV is UK-only: outside its domain everything comes back null
        if df["wind_speed_10m"].notna().sum() == 0:
            continue
        frames[mid] = df
    return frames


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def build_figure(frames: dict[str, pd.DataFrame], wind_unit_label: str,
                 show_gusts: bool, arrow_every: int = 3) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.58, 0.42],
        specs=[[{}], [{"secondary_y": True}]],
        subplot_titles=(
            f"Wind speed ({wind_unit_label}) — arrows show direction (blowing towards)",
            "Cloud cover (%) and precipitation (mm/h)",
        ),
    )

    for mid, df in frames.items():
        label, colour = MODELS[mid]

        # --- wind speed line -------------------------------------------------
        fig.add_trace(go.Scatter(
            x=df.index, y=df["wind_speed_10m"],
            mode="lines", name=label, legendgroup=mid,
            line=dict(color=colour, width=2),
            hovertemplate="%{y:.1f} " + wind_unit_label + "<extra>" + label + "</extra>",
        ), row=1, col=1)

        if show_gusts and df["wind_gusts_10m"].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df["wind_gusts_10m"],
                mode="lines", name=f"{label} gusts", legendgroup=mid,
                showlegend=False,
                line=dict(color=colour, width=1, dash="dot"),
                hovertemplate="gust %{y:.1f} " + wind_unit_label +
                              "<extra>" + label + "</extra>",
            ), row=1, col=1)

        # --- direction arrows along the speed line ---------------------------
        sub = df.iloc[::arrow_every]
        # met. direction = where wind comes FROM; arrow points where it goes
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub["wind_speed_10m"],
            mode="markers", legendgroup=mid, showlegend=False,
            marker=dict(
                symbol="arrow", size=11, color=colour,
                angle=(sub["wind_direction_10m"] + 180) % 360,
                line=dict(width=0.5, color="white"),
            ),
            customdata=sub["wind_direction_10m"],
            hovertemplate="from %{customdata:.0f}°<extra>" + label + "</extra>",
        ), row=1, col=1)

        # --- cloud cover (left axis) -----------------------------------------
        fig.add_trace(go.Scatter(
            x=df.index, y=df["cloud_cover"],
            mode="lines", name=f"{label} cloud", legendgroup=mid,
            showlegend=False,
            line=dict(color=colour, width=1.5),
            opacity=0.7,
            hovertemplate="cloud %{y:.0f}%<extra>" + label + "</extra>",
        ), row=2, col=1, secondary_y=False)

        # --- precipitation (right axis, bars) ---------------------------------
        if df["precipitation"].notna().any() and df["precipitation"].sum() > 0:
            fig.add_trace(go.Bar(
                x=df.index, y=df["precipitation"],
                name=f"{label} precip", legendgroup=mid, showlegend=False,
                marker_color=colour, opacity=0.35,
                hovertemplate="precip %{y:.1f} mm<extra>" + label + "</extra>",
            ), row=2, col=1, secondary_y=True)

    # day separators
    if frames:
        any_df = next(iter(frames.values()))
        for day in pd.date_range(any_df.index[0].normalize(),
                                 any_df.index[-1].normalize(), freq="D")[1:]:
            fig.add_vline(x=day, line_width=1, line_color="rgba(0,0,0,0.15)")

    fig.update_layout(
        height=720, barmode="overlay", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
        margin=dict(l=50, r=50, t=80, b=40),
    )
    fig.update_yaxes(title_text=wind_unit_label, rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Cloud cover %", range=[0, 100],
                     row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="mm/h", rangemode="tozero",
                     row=2, col=1, secondary_y=True)
    return fig


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("Multi-model point forecast")
st.caption("Click anywhere on the map to compare UKV, UKMO, ECMWF, GFS and "
           "ICON forecasts for that point (native model data via Open-Meteo).")

with st.sidebar:
    st.header("Settings")
    selected = st.multiselect(
        "Models",
        options=list(MODELS),
        default=list(MODELS),
        format_func=lambda m: MODELS[m][0],
    )
    days = st.slider("Forecast days", 1, 7, 4)
    unit = st.selectbox("Wind unit", options=list(WIND_UNITS),
                        format_func=lambda u: WIND_UNITS[u], index=0)
    show_gusts = st.checkbox("Show gusts (dotted)", value=True)
    arrow_every = st.slider("Direction arrow every N hours", 1, 6, 3)

if "point" not in st.session_state:
    st.session_state.point = (DEFAULT_LAT, DEFAULT_LON)

col_map, col_chart = st.columns([1, 1.6], gap="medium")

with col_map:
    lat0, lon0 = st.session_state.point
    m = folium.Map(location=[lat0, lon0], zoom_start=8, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Map").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(m)
    folium.LayerControl().add_to(m)
    folium.Marker([lat0, lon0], tooltip="Forecast point").add_to(m)

    map_state = st_folium(m, height=560, use_container_width=True,
                          returned_objects=["last_clicked"])

    if map_state and map_state.get("last_clicked"):
        clicked = (map_state["last_clicked"]["lat"],
                   map_state["last_clicked"]["lng"])
        if clicked != st.session_state.point:
            st.session_state.point = clicked
            st.rerun()

with col_chart:
    lat, lon = st.session_state.point
    st.markdown(f"**Point:** {lat:.4f}°, {lon:.4f}°")

    if not selected:
        st.info("Select at least one model in the sidebar.")
    else:
        try:
            with st.spinner("Fetching forecasts…"):
                payload = fetch_forecast(lat, lon, tuple(selected),
                                         days, unit)
            frames = to_frames(payload, selected)
        except requests.RequestException as exc:
            st.error(f"Forecast request failed: {exc}")
            frames = {}

        if frames:
            missing = [MODELS[m][0] for m in selected if m not in frames]
            if missing:
                st.warning("No data here for: " + ", ".join(missing) +
                           " (point outside model domain).")
            fig = build_figure(frames, WIND_UNITS[unit], show_gusts,
                               arrow_every)
            st.plotly_chart(fig, use_container_width=True)

            elev = payload.get("elevation")
            updated = dt.datetime.now().strftime("%H:%M")
            st.caption(f"Model grid elevation ≈ {elev} m · 10 m wind · "
                       f"retrieved {updated} · cached 30 min per point")
        elif selected:
            st.info("No forecast data returned for this point.")
