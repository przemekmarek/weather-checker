"""
Multi-model point forecast viewer
---------------------------------
Click anywhere on the map and get an hourly forecast chart comparing
UK-relevant models for that point. Wind is shown at a selectable target
altitude (~600 / ~1000 / ~1500 m, or 10 m surface). For each model the app
automatically picks the closest vertical coordinate that model actually
publishes -- a pressure level (e.g. 950 hPa) or a height level
(e.g. 180 m AGL, as for UKV) -- and says which one in the legend.
Models with no upper-level data at all (e.g. ECMWF if only surface is
published) are dropped from upper-level views with a warning.

Data source: Open-Meteo (https://open-meteo.com).

Run with:  streamlit run app.py
"""

import datetime as dt
import re

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
    "icon_seamless": ("ICON Global (DWD)", "#9467bd"),
    "icon_eu": ("ICON-EU 7 km (DWD)", "#17becf"),
    "dmi_harmonie_arome_europe": ("DMI HARMONIE 2 km", "#8c564b"),
    "knmi_harmonie_arome_europe": ("KNMI HARMONIE 5.5 km", "#e377c2"),
    "meteofrance_arpege_europe": ("ARPEGE Europe 11 km", "#bcbd22"),
}

# pressure level -> approx altitude (m AMSL, ICAO standard atmosphere)
PRESSURE_ALT = {1000: 110, 975: 320, 950: 600, 925: 800,
                900: 1000, 850: 1500, 800: 1900}
# above-ground height levels commonly published via Open-Meteo
HEIGHT_LEVELS = [80, 100, 120, 180]

# target key -> (label, target altitude in m; None = 10 m surface)
TARGETS = {
    "600": ("~600 m (≈950 hPa)", 600),
    "1000": ("~1000 m (≈900 hPa)", 1000),
    "1500": ("~1500 m (≈850 hPa)", 1500),
    "sfc": ("10 m (surface)", None),
}

WIND_UNITS = {"kn": "kt", "ms": "m/s", "kmh": "km/h", "mph": "mph"}

BASEMAPS = {
    "Map": dict(tiles="OpenStreetMap", attr=None),
    "Terrain": dict(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="© OpenTopoMap (CC-BY-SA), © OpenStreetMap contributors",
    ),
    "Hillshade": dict(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Hillshade",
    ),
    "Satellite": dict(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    ),
}

DEFAULT_LAT, DEFAULT_LON = 55.86, -4.25  # Glasgow


# ----------------------------------------------------------------------------
# Level candidates
# ----------------------------------------------------------------------------

def candidate_levels(target: int | None) -> list[tuple[str, float, str, str]]:
    """Vertical coordinates worth requesting for a target altitude.

    Returns (variable suffix, nominal altitude m, display label, kind),
    sorted by closeness to the target. Pressure wins ties over height.
    """
    if target is None:
        return [("10m", 10.0, "10 m", "height")]
    cands = []
    for p, alt in PRESSURE_ALT.items():
        if abs(alt - target) <= 700:
            cands.append((f"{p}hPa", float(alt), f"{p} hPa", "pressure"))
    for h in HEIGHT_LEVELS:
        cands.append((f"{h}m", float(h), f"{h} m AGL", "height"))
    cands.sort(key=lambda c: (abs(c[1] - target), 0 if c[3] == "pressure" else 1))
    return cands


# ----------------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lat: float, lon: float, model_ids: tuple[str, ...],
                   days: int, wind_unit: str, target_key: str) -> dict:
    """One Open-Meteo call covering all models and all candidate levels.

    If the API rejects a variable name some endpoint doesn't know
    (HTTP 400 with the offending name in `reason`), that level is removed
    and the request retried, so an over-optimistic candidate list degrades
    gracefully rather than failing.
    """
    target = TARGETS[target_key][1]
    suffixes = [c[0] for c in candidate_levels(target)]
    hourly = ([f"wind_{w}_{s}" for s in suffixes
               for w in ("speed", "direction")]
              + ["cloud_cover", "precipitation"])

    for _ in range(len(suffixes) + 1):
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "hourly": ",".join(hourly),
            "models": ",".join(model_ids),
            "forecast_days": days,
            "wind_speed_unit": wind_unit,
            "timezone": "auto",
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params=params, timeout=30)
        if r.ok:
            return r.json()
        try:
            reason = r.json().get("reason", "")
        except ValueError:
            reason = r.text
        bad = [v for v in hourly if re.search(rf"\b{re.escape(v)}\b", reason)]
        if r.status_code == 400 and bad:
            # drop the whole offending level (speed + direction)
            bad_sfx = {v.split("_", 2)[2] for v in bad}
            hourly = [v for v in hourly
                      if v in ("cloud_cover", "precipitation")
                      or v.split("_", 2)[2] not in bad_sfx]
            continue
        r.raise_for_status()
    raise requests.RequestException("No valid wind variables accepted by API")


def to_frames(payload: dict, model_ids: list[str], target_key: str
              ) -> dict[str, tuple[pd.DataFrame, str]]:
    """Per model: pick the closest level with data; return (frame, level label)."""
    hourly = payload.get("hourly", {})
    time_index = pd.to_datetime(hourly.get("time", []))
    target = TARGETS[target_key][1]
    cands = candidate_levels(target)
    frames: dict[str, tuple[pd.DataFrame, str]] = {}

    def col(var: str, mid: str):
        # multi-model responses suffix each variable with the model id;
        # single-model responses don't
        for key in (f"{var}_{mid}", var):
            vals = hourly.get(key)
            if vals is not None and any(v is not None for v in vals):
                return vals
        return None

    for mid in model_ids:
        chosen = None
        for sfx, _alt, lvl_label, _kind in cands:  # already closest-first
            speed = col(f"wind_speed_{sfx}", mid)
            direction = col(f"wind_direction_{sfx}", mid)
            if speed is not None and direction is not None:
                chosen = (speed, direction, lvl_label)
                break
        if chosen is None:
            continue
        df = pd.DataFrame({
            "wind_speed": chosen[0],
            "wind_direction": chosen[1],
            "cloud_cover": col("cloud_cover", mid),
            "precipitation": col("precipitation", mid),
        }, index=time_index)
        frames[mid] = (df, chosen[2])
    return frames


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def build_figure(frames: dict[str, tuple[pd.DataFrame, str]],
                 wind_unit_label: str, target_label: str,
                 arrow_every: int = 3) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.58, 0.42],
        specs=[[{}], [{"secondary_y": True}]],
        subplot_titles=(
            f"Wind near {target_label} ({wind_unit_label}) — level used per "
            f"model shown in legend; arrows show direction (blowing towards)",
            "Cloud cover (%) and precipitation (mm/h)",
        ),
    )

    for mid, (df, lvl_label) in frames.items():
        label, colour = MODELS[mid]
        name = f"{label} · {lvl_label}"

        # --- wind speed line -------------------------------------------------
        fig.add_trace(go.Scatter(
            x=df.index, y=df["wind_speed"],
            mode="lines", name=name, legendgroup=mid,
            line=dict(color=colour, width=2),
            hovertemplate="%{y:.1f} " + wind_unit_label +
                          "<extra>" + name + "</extra>",
        ), row=1, col=1)

        # --- direction arrows along the speed line ---------------------------
        sub = df.iloc[::arrow_every]
        # met. direction = where wind comes FROM; arrow points where it goes
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub["wind_speed"],
            mode="markers", legendgroup=mid, showlegend=False,
            marker=dict(
                symbol="arrow", size=11, color=colour,
                angle=(sub["wind_direction"] + 180) % 360,
                line=dict(width=0.5, color="white"),
            ),
            customdata=sub["wind_direction"],
            hovertemplate="from %{customdata:.0f}°<extra>" + name + "</extra>",
        ), row=1, col=1)

        # --- cloud cover (left axis) -----------------------------------------
        if df["cloud_cover"].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df["cloud_cover"],
                mode="lines", name=f"{label} cloud", legendgroup=mid,
                showlegend=False,
                line=dict(color=colour, width=1.5),
                opacity=0.7,
                hovertemplate="cloud %{y:.0f}%<extra>" + label + "</extra>",
            ), row=2, col=1, secondary_y=False)

        # --- precipitation (right axis, bars) ---------------------------------
        if df["precipitation"].fillna(0).sum() > 0:
            fig.add_trace(go.Bar(
                x=df.index, y=df["precipitation"],
                name=f"{label} precip", legendgroup=mid, showlegend=False,
                marker_color=colour, opacity=0.35,
                hovertemplate="precip %{y:.1f} mm<extra>" + label + "</extra>",
            ), row=2, col=1, secondary_y=True)

    # day separators and day-name labels
    if frames:
        idx = next(iter(frames.values()))[0].index
        days_seq = pd.date_range(idx[0].normalize(), idx[-1].normalize(),
                                 freq="D")
        for day in days_seq[1:]:
            fig.add_vline(x=day, line_width=1, line_color="rgba(0,0,0,0.15)")
        for day in days_seq:
            day_hours = idx[(idx >= day) & (idx < day + pd.Timedelta("1D"))]
            if len(day_hours) < 6:  # skip stub partial days
                continue
            mid_t = day_hours[0] + (day_hours[-1] - day_hours[0]) / 2
            fig.add_annotation(
                x=mid_t, y=0.99, xref="x", yref="y domain",
                text=day.strftime("%A"), showarrow=False,
                yanchor="top", font=dict(size=13, color="rgba(0,0,0,0.5)"),
            )

    fig.update_layout(
        height=740, barmode="overlay", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
        margin=dict(l=50, r=50, t=90, b=40),
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
st.caption("Click anywhere on the map to compare model forecasts for that "
           "point (native model data via Open-Meteo). Each model uses the "
           "pressure or height level closest to the chosen altitude.")

with st.sidebar:
    st.header("Settings")
    selected = st.multiselect(
        "Models",
        options=list(MODELS),
        default=list(MODELS),
        format_func=lambda m: MODELS[m][0],
    )
    target_key = st.selectbox(
        "Wind altitude", options=list(TARGETS),
        format_func=lambda k: TARGETS[k][0], index=0,
    )
    days = st.slider("Forecast days", 1, 7, 4)
    unit = st.selectbox("Wind unit", options=list(WIND_UNITS),
                        format_func=lambda u: WIND_UNITS[u], index=0)
    arrow_every = st.slider("Direction arrow every N hours", 1, 6, 3)

if "point" not in st.session_state:
    st.session_state.point = (DEFAULT_LAT, DEFAULT_LON)
if "view" not in st.session_state:
    st.session_state.view = {"center": list(st.session_state.point), "zoom": 8}

col_map, col_chart = st.columns([1, 1.6], gap="medium")

with col_map:
    # basemap choice lives in Streamlit state, so it survives reruns
    basemap = st.radio("Basemap", options=list(BASEMAPS), horizontal=True,
                       key="basemap", label_visibility="collapsed")

    lat0, lon0 = st.session_state.point
    view = st.session_state.view
    bm = BASEMAPS[basemap]
    m = folium.Map(location=view["center"], zoom_start=view["zoom"],
                   tiles=bm["tiles"], attr=bm["attr"])
    folium.Marker([lat0, lon0], tooltip="Forecast point").add_to(m)

    map_state = st_folium(
        m, height=560, use_container_width=True,
        returned_objects=["last_clicked", "center", "zoom"],
        key="map",
    )

    # remember where the user has panned/zoomed to
    if map_state:
        c, z = map_state.get("center"), map_state.get("zoom")
        if c and z:
            st.session_state.view = {"center": [c["lat"], c["lng"]],
                                     "zoom": z}

    if map_state and map_state.get("last_clicked"):
        clicked = (map_state["last_clicked"]["lat"],
                   map_state["last_clicked"]["lng"])
        if clicked != st.session_state.point:
            st.session_state.point = clicked
            st.rerun()

with col_chart:
    lat, lon = st.session_state.point
    target_label = TARGETS[target_key][0]
    st.markdown(f"**Point:** {lat:.4f}°, {lon:.4f}° · "
                f"**Altitude:** {target_label}")

    if not selected:
        st.info("Select at least one model in the sidebar.")
    else:
        try:
            with st.spinner("Fetching forecasts…"):
                payload = fetch_forecast(lat, lon, tuple(selected),
                                         days, unit, target_key)
            frames = to_frames(payload, selected, target_key)
        except requests.RequestException as exc:
            st.error(f"Forecast request failed: {exc}")
            frames = {}

        if frames:
            missing = [MODELS[m][0] for m in selected if m not in frames]
            if missing:
                st.warning("Dropped (no wind data at this point near "
                           f"{target_label}): " + ", ".join(missing))
            fig = build_figure(frames, WIND_UNITS[unit], target_label,
                               arrow_every)
            st.plotly_chart(fig, use_container_width=True)

            elev = payload.get("elevation")
            updated = dt.datetime.now().strftime("%H:%M")
            st.caption(f"Model grid elevation ≈ {elev} m · pressure-level "
                       f"altitudes are AMSL (std. atmosphere); height levels "
                       f"are AGL · retrieved {updated} · cached 30 min")
        elif selected:
            st.info("No forecast data returned for this point near "
                    f"{target_label}.")
