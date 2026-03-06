import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk

from preprocessing import preprocess_site, SITE_NAME
from des import holt_forecast, tune_holt
from aqi import categorize_aqi 

# ---------- Calculation Helpers ----------

def calculate_heat_index(temp_c, humidity):
    """NOAA/NWS Heat Index Equation (Celsius)"""
    T = (temp_c * 9/5) + 32
    RH = humidity
    # Simple formula for low temps
    hi = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    
    # Rothfusz Regression for higher temps
    if hi > 80:
        hi = -42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH \
             - 6.83783e-3*T*T - 5.481717e-2*RH*RH + 1.22874e-3*T*T*RH \
             + 8.5282e-4*T*RH*RH - 1.99e-6*T*T*RH*RH
        
        # Adjustments
        if RH < 13 and 80 <= T <= 112:
            hi -= ((13-RH)/4) * np.sqrt((17-np.abs(T-95))/17)
        elif RH > 85 and 80 <= T <= 87:
            hi += ((RH-85)/10) * ((87-T)/5)
            
    return (hi - 32) * 5/9

def get_pagasa_hi_category(hi_c):
    """PAGASA Heat Index Categories"""
    if hi_c < 27: return "Not Hazardous", "#00e400"
    if hi_c <= 32: return "Caution", "#ffff00"
    if hi_c <= 41: return "Extreme Caution", "#ff7e00"
    if hi_c <= 51: return "Danger", "#ff0000"
    return "Extreme Danger", "#7e0023"

def categorize_pm25_denr(pm25_value):
    v = float(pm25_value)
    if v <= 25.0: return "Good", "#00e400"
    if v <= 35.0: return "Fair", "#ffff00"
    if v <= 45.0: return "USG", "#ff7e00"
    if v <= 55.0: return "Very Unhealthy", "#ff0000"
    if v <= 90.0: return "Acutely Unhealthy", "#8f3f97"
    return "Emergency", "#7e0023"

def hex_to_rgb_tuple(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

# ---------- Page Config & Signals ----------
st.set_page_config(page_title="Barangay Microclimate Forecast", layout="wide")

signals = {
    "tempC":      {"label": "Temperature", "unit": "°C",      "color": "#ff6b6b", "clip": (None, None)},
    "humidity":   {"label": "Humidity",    "unit": "%",       "color": "#4ecdc4", "clip": (0, 100)},
    "heat_index": {"label": "Heat Index",  "unit": "°C",      "color": "#ff8c00", "clip": (None, None)},
    "aqi":        {"label": "AQI",         "unit": "",        "color": "#6bcb77", "clip": (0, None)},
}

# ---------- Core Logic ----------

@st.cache_data(show_spinner=False)
def process_site_data(df, site_code, steps, alpha, beta, auto_tune):
    site_df = df[df["Location"] == site_code].copy()
    if site_df.empty: return None

    proc = preprocess_site(site_df)
    
    # Inject Heat Index calculation into preprocessed data
    if "tempC" in proc.columns and "humidity" in proc.columns:
        proc["heat_index"] = proc.apply(lambda r: calculate_heat_index(r["tempC"], r["humidity"]), axis=1)

    last_ts = proc.index[-1]
    future_idx = pd.date_range(last_ts + pd.Timedelta("5min"), periods=steps, freq="5min")
    results, chosen = {}, {}

    for col in signals.keys():
        if col not in proc.columns: continue
        series = proc[col].values
        
        if auto_tune:
            best = tune_holt(series, steps=steps)
            a, b, res = best["alpha"], best["beta"], best["model"]
        else:
            a, b = alpha, beta
            res = holt_forecast(series, alpha=a, beta=b, steps=steps)

        # Clipping
        lo_clip, hi_clip = signals[col]["clip"]
        res["forecast"] = np.clip(res["forecast"], lo_clip, hi_clip)
        res["lower"] = np.clip(res["lower"], lo_clip, hi_clip)
        res["upper"] = np.clip(res["upper"], lo_clip, hi_clip)

        results[col] = res
        chosen[col] = {"alpha": a, "beta": b, "rmse": float(res["rmse"])}

    return {"proc": proc, "last_ts": last_ts, "future_idx": future_idx, "results": results, "chosen": chosen}

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload sensor_log.csv", type=["csv"])
    cat_scale = st.selectbox("Map Color Scale", ["EPA AQI", "DENR PM2.5", "PAGASA Heat Index"])
    auto_tune = st.checkbox("Auto-tune per signal", value=True)
    alpha = st.slider("Manual Alpha", 0.05, 0.9, 0.3)
    beta = st.slider("Manual Beta", 0.05, 0.9, 0.15)
    steps = st.select_slider("Forecast Hours", options=[12, 24, 36, 48], value=48)
    tab_choice = st.radio("View", ["Single site", "City map"])

# ---------- Data Loading ----------
default_path = "sensor_log.csv"
if uploaded:
    raw = pd.read_csv(uploaded)
else:
    if os.path.exists(default_path):
        raw = pd.read_csv(default_path)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
site_codes = raw["Location"].unique()

# ---------- Main View ----------

if tab_choice == "Single site":
    site_code = st.selectbox("Select Barangay", site_codes, format_func=lambda x: SITE_NAME.get(x, x))
    bundle = process_site_data(raw, site_code, steps, alpha, beta, auto_tune)
    
    if bundle:
        tabs = st.tabs([meta["label"] for meta in signals.values()])
        for tab, (col, meta) in zip(tabs, signals.items()):
            with tab:
                if col in bundle["results"]:
                    res = bundle["results"][col]
                    fig = go.Figure()
                    # History
                    fig.add_trace(go.Scatter(x=bundle["proc"].index, y=bundle["proc"][col], name="Observed", line=dict(color=meta["color"])))
                    # Forecast
                    fig.add_trace(go.Scatter(x=bundle["future_idx"], y=res["forecast"], name="Forecast", line=dict(color=meta["color"], dash="dash")))
                    fig.update_layout(template="plotly_dark", title=f"{meta['label']} Trend")
                    st.plotly_chart(fig, use_container_width=True)

elif tab_choice == "City map":
    st.subheader(f"Dasmariñas Status: {cat_scale}")
    
    # Prep map data
    map_data = []
    for code in site_codes:
        b = process_site_data(raw, code, steps, alpha, beta, auto_tune)
        if b:
            last_row = b["proc"].iloc[-1]
            
            # Determine color and category based on selection
            if cat_scale == "PAGASA Heat Index" and "heat_index" in last_row:
                val = last_row["heat_index"]
                label, color = get_pagasa_hi_category(val)
            elif cat_scale == "DENR PM2.5" and "pm25" in last_row:
                val = last_row["pm25"]
                label, color = categorize_pm25_denr(val)
            else: # Default EPA AQI
                val = last_row.get("aqi", 0)
                label, color = categorize_aqi(val)
            
            rgb = hex_to_rgb_tuple(color)
            map_data.append({
                "location": code,
                "name": SITE_NAME.get(code, code),
                "value": round(val, 1),
                "category": label,
                "color": [*rgb, 200] # RGBA
            })

    # Assuming you have a CSV with lat/lon for these codes
    # For demo, let's assume we map 'map_data' to coordinates
    pts_path = "barangays_dasmarinas.csv"
    if os.path.exists(pts_path):
        coords = pd.read_csv(pts_path)
        df_map = pd.DataFrame(map_data).merge(coords, left_on="location", right_on="location_code")
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            df_map,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius=200,
            pickable=True,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=14.329, longitude=120.936, zoom=12),
            tooltip={"text": "{name}\nValue: {value}\nStatus: {category}"}
        ))
