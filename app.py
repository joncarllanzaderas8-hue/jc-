import os, json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from io import BytesIO

# Optional for PDF/plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- PDF GUARD: allow app to run even if reportlab isn't installed
REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
except Exception:
    REPORTLAB_OK = False

# -----------------------------
# Page & Auto-refresh
# -----------------------------
st.set_page_config(page_title="Dasmariñas Environmental Risk Monitor", layout="wide")
count = st_autorefresh(interval=5000, limit=100, key="autorefresh")

COORDS_JSON = "site_coords.json"

# -----------------------------
# Data Loading & Cleaning
# -----------------------------
@st.cache_data
def load_data(path: str = "sensor_log.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Ensure columns are numeric to prevent plotting errors
    for col in ['tempC', 'humidity', 'aqi']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse time
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # --- THE FIX: Remove extreme outliers that break the graph scale ---
    if 'tempC' in df.columns:
        df = df[(df['tempC'] > -10) & (df['tempC'] < 100)] 
    if 'humidity' in df.columns:
        df = df[(df['humidity'] >= 0) & (df['humidity'] <= 100)]
    # -----------------------------------------------------------------

    # Drop duplicate (timestamp, Location)
    if set(['timestamp','Location']).issubset(df.columns):
        df = df.drop_duplicates(subset=['timestamp','Location'])

    # Treat suspicious zeros as missing
    for c in ['aqi','mqRaw']:
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan
            
    return df

df = load_data()
df_hist = df.copy()
# -----------------------------
# Heat Index (NOAA/NWS Rothfusz + Steadman fallback)
# -----------------------------
# Ref: NOAA/NWS heat index equation
# https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
def heat_index_celsius(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    T = temp_c * 9.0/5.0 + 32.0  # to °F
    R = rh
    HI = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R
          - 6.83783e-3*T*T - 5.481717e-2*R*R
          + 1.22874e-3*T*T*R + 8.5282e-4*T*R*R - 1.99e-6*T*T*R*R)
    adj = np.zeros_like(HI, dtype=float)
    mask_low = (R < 13) & (T >= 80) & (T <= 112)
    adj[mask_low] = -((13 - R[mask_low])/4.0) * np.sqrt((17 - np.abs(T[mask_low]-95))/17.0)
    mask_high = (R > 85) & (T >= 80) & (T <= 87)
    adj[mask_high] = ((R[mask_high]-85)/10.0) * ((87 - T[mask_high])/5.0)
    HI = np.where(T < 80, T + 0.33*R - 0.70, HI + adj)
    return (HI - 32.0) * 5.0/9.0  # back to °C

# PAGASA Heat Index Categories (°C)
# Not Hazardous (<27), Caution (27–32), Extreme Caution (33–41), Danger (42–51), Extreme Danger (≥52)
def pagasa_hi_category(hi_c: float) -> str:
    if not np.isfinite(hi_c):
        return '—'
    if hi_c < 27: return "Not Hazardous"
    if hi_c <= 32: return "Caution (27–32°C)"
    if hi_c <= 41: return "Extreme Caution (33–41°C)"
    if hi_c <= 51: return "Danger (42–51°C)"
    return "Extreme Danger (≥52°C)"

# US EPA AQI Categories (index)
def epa_aqi_category(aqi_value: float) -> str:
    if aqi_value is None or (isinstance(aqi_value, float) and np.isnan(aqi_value)):
        return "—"
    aqi = float(aqi_value)
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

AQI_COLORS = {
    "Good": "#009966",
    "Moderate": "#FFDE33",
    "Unhealthy for Sensitive Groups": "#FF9933",
    "Unhealthy": "#CC0033",
    "Very Unhealthy": "#660099",
    "Hazardous": "#7E0023",
    "—": "#999999"
}

RISK_COLORS = {
    "Normal": "#2B8A3E",
    "Moderate": "#F59F00",
    "High": "#D9480F",
    "Unknown": "#868E96"
}

HI_BAND_SHAPES = [
    (None, 27, 'rgba(0,0,0,0)'),
    (27, 32, 'rgba(255,235,132,0.25)'),
    (32, 41, 'rgba(255,165,0,0.20)'),
    (41, 51, 'rgba(255,69,58,0.20)'),
    (51, None, 'rgba(156,39,176,0.15)')
]

# Pre-compute Heat Index for history
if set(['tempC','humidity']).issubset(df_hist.columns):
    df_hist['heat_index_C'] = heat_index_celsius(df_hist['tempC'].values, df_hist['humidity'].values)
    df_hist['HI_Category'] = df_hist['heat_index_C'].apply(pagasa_hi_category)
else:
    df_hist['heat_index_C'] = np.nan
    df_hist['HI_Category'] = "—"

if 'aqi' in df_hist.columns:
    df_hist['AQI_Category'] = df_hist['aqi'].apply(epa_aqi_category)
else:
    df_hist['AQI_Category'] = "—"

# -----------------------------
# Model
# -----------------------------
MODEL_PATH = 'dt_model.joblib'
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model file not found or cannot be loaded. Please ensure '{MODEL_PATH}' is in the directory. Details: {e}")
    st.stop()

label_map = {0: "Normal", 1: "Moderate", 2: "High"}

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("📡 IoT Connection Settings")
location_mode = st.sidebar.selectbox("Select Monitoring Site", ["A - Green Space", "B - Residential", "C - Commercial"])
data_source = st.sidebar.radio("Data Source", ["Latest Reading", "Manual Input"])

# Forecast options
st.sidebar.header("🕒 Forecast Settings")
forecast_minutes = st.sidebar.selectbox("Forecast Horizon", [30, 60, 1440], index=0,
                                        format_func=lambda m: f"{m//60}h" if m>=60 else f"{m}m")
forecast_method = st.sidebar.selectbox("Method", ["Naive (last value)", "Rolling Mean", "Linear Trend"], index=1)
show_pagasa_bands = st.sidebar.checkbox("Show PAGASA HI bands on chart", value=True)

# Map options & coordinates
st.sidebar.header("🗺️ Map Settings")
map_basis = st.sidebar.selectbox("Map risk based on",
                                 ["Current model risk", f"Forecasted risk in {forecast_minutes} min", "Forecasted risk in 24h"],
                                 index=2)

# -----------------------------
# Persisted Coordinates (site_coords.json)
# -----------------------------
def default_coords():
    return {
        'A': {'name': 'Green Space', 'lat': None, 'lon': None},
        'B': {'name': 'Residential', 'lat': None, 'lon': None},
        'C': {'name': 'Commercial', 'lat': None, 'lon': None},
    }

@st.cache_resource
def load_coords_from_disk():
    if os.path.exists(COORDS_JSON):
        try:
            with open(COORDS_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            base = default_coords()
            base.update({k:v for k,v in data.items() if k in base})
            return base
        except Exception:
            return default_coords()
    return default_coords()

if 'site_coords' not in st.session_state:
    st.session_state.site_coords = load_coords_from_disk()

for key in ['A','B','C']:
    with st.sidebar.expander(f"Set coordinates for site {key}"):
        lat_str = st.text_input(f"{key} Latitude",
                                value='' if st.session_state.site_coords[key]['lat'] is None
                                else str(st.session_state.site_coords[key]['lat']),
                                key=f"lat_{key}")
        lon_str = st.text_input(f"{key} Longitude",
                                value='' if st.session_state.site_coords[key]['lon'] is None
                                else str(st.session_state.site_coords[key]['lon']),
                                key=f"lon_{key}")
        try:
            st.session_state.site_coords[key]['lat'] = float(lat_str) if lat_str.strip() != '' else None
            st.session_state.site_coords[key]['lon'] = float(lon_str) if lon_str.strip() != '' else None
        except ValueError:
            st.warning(f"Invalid lat/lon for site {key}. Leave empty or input numeric values.")

save_col1, save_col2 = st.sidebar.columns(2)
if save_col1.button("💾 Save Coords"):
    try:
        with open(COORDS_JSON, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.site_coords, f, ensure_ascii=False, indent=2)
        st.sidebar.success("Saved to site_coords.json")
    except Exception as e:
        st.sidebar.error(f"Failed to save: {e}")
if save_col2.button("↩︎ Reset Coords"):
    st.session_state.site_coords = default_coords()
    st.sidebar.info("Reset to defaults (not saved yet)")

# -----------------------------
# Header & current reading
# -----------------------------
st.title("🌿 Dasmariñas Environmental Risk Monitor")
st.subheader(f"Locale: {location_mode} · Live Monitoring Dashboard")

loc_tag = location_mode[0]

if data_source == "Latest Reading":
    recent_loc = df_hist[df_hist['Location'] == loc_tag].sort_values('timestamp')
    if recent_loc.empty:
        st.warning("No data available for this location.")
        st.stop()
    current = recent_loc.dropna(subset=['tempC','humidity']).tail(1).iloc[0]
    temp = float(current['tempC'])
    hum = float(current['humidity'])
    aqi = float(current['aqi']) if 'aqi' in current and pd.notna(current['aqi']) \
          else float(df_hist['aqi'].median(skipna=True)) if 'aqi' in df_hist.columns else np.nan
    aqi_imputed = pd.isna(current['aqi']) if 'aqi' in current else True
else:
    temp = st.sidebar.slider("Temperature (°C)", 20.0, 45.0, 30.0)
    hum = st.sidebar.slider("Humidity (%)", 30.0, 100.0, 75.0)
    aqi = st.sidebar.slider("AQI (index)", 0.0, 300.0, 25.0)
    aqi_imputed = False

current_hi = float(heat_index_celsius(np.array([temp]), np.array([hum]))[0])
hi_cat = pagasa_hi_category(current_hi)
aqi_cat = epa_aqi_category(aqi)

input_df = pd.DataFrame([[temp, hum, aqi]], columns=['tempC','humidity','aqi'])
pred_idx = model.predict(input_df)[0]
risk_level = label_map.get(int(pred_idx), "Unknown")

# -----------------------------
# Forecast utilities
# -----------------------------
def median_minutes_delta(ts: pd.Series, default=5.0) -> float:
    ts_sorted = ts.sort_values().dropna()
    if len(ts_sorted) < 3:
        return default
    deltas = ts_sorted.diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return default
    return float(np.median(deltas))

def forecast_next(series: pd.Series, steps: int, method: str = "Rolling Mean", window: int = 12) -> np.ndarray:
    s = series.dropna().values
    if len(s) == 0:
        return np.array([np.nan]*steps)
    if method.startswith('Naive'):
        return np.repeat(s[-1], steps)
    if method.startswith('Rolling'):
        w = max(3, min(window, len(s)))
        avg = float(np.mean(s[-w:]))
        return np.repeat(avg, steps)
    # Linear Trend
    w = max(4, min(window, len(s)))
    y = s[-w:]
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    x_future = np.arange(len(y), len(y)+steps)
    yhat = coeffs[0]*x_future + coeffs[1]
    return yhat

def backtest_mae_rmse(series: pd.Series, steps_ahead: int, method: str, window: int = 12, max_points: int = 600):
    s = series.dropna()
    if len(s) < window + steps_ahead + 5:
        return np.nan, np.nan, 0
    s = s.tail(max_points)
    y_true, y_pred = [], []
    for end in range(window, len(s)-steps_ahead):
        hist = s.iloc[:end]
        pred = forecast_next(hist, steps=steps_ahead, method=method, window=window)[-1]
        y_true.append(float(s.iloc[end+steps_ahead-1]))
        y_pred.append(float(pred))
    if len(y_true) == 0:
        return np.nan, np.nan, 0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    return float(mae), rmse, len(y_true)

# Determine forecast steps from desired minutes
loc_df = df_hist[df_hist['Location'] == loc_tag].sort_values('timestamp')
median_min = median_minutes_delta(loc_df['timestamp']) if not loc_df.empty else 5.0
steps_horizon = int(max(1, round(forecast_minutes / max(1e-6, median_min))))
steps_24h = int(max(1, round(1440 / max(1e-6, median_min))))

# Forecast for HI & AQI
hi_series = loc_df['heat_index_C'] if 'heat_index_C' in loc_df else pd.Series(dtype=float)
aqi_series = loc_df['aqi'] if 'aqi' in loc_df else pd.Series(dtype=float)
temp_series = loc_df['tempC'] if 'tempC' in loc_df else pd.Series(dtype=float)
hum_series = loc_df['humidity'] if 'humidity' in loc_df else pd.Series(dtype=float)

hi_fore = forecast_next(hi_series, steps=steps_horizon, method=forecast_method, window=12)
aqi_fore = forecast_next(aqi_series, steps=steps_horizon, method=forecast_method, window=12)

hi_fore_val = float(hi_fore[-1]) if len(hi_fore) else float('nan')
aqi_fore_val = float(aqi_fore[-1]) if len(aqi_fore) else float('nan')
hi_fore_cat = pagasa_hi_category(hi_fore_val) if np.isfinite(hi_fore_val) else '—'
aqi_fore_cat = epa_aqi_category(aqi_fore_val) if np.isfinite(aqi_fore_val) else '—'

# Backtest metrics for selected horizon
hi_mae, hi_rmse, hi_n = backtest_mae_rmse(hi_series, steps_horizon, forecast_method, window=12)
aqi_mae, aqi_rmse, aqi_n = backtest_mae_rmse(aqi_series, steps_horizon, forecast_method, window=12)

# Build future timestamps for plotting
if not loc_df.empty and len(loc_df['timestamp'].dropna())>0:
    last_ts = loc_df['timestamp'].dropna().iloc[-1]
    dt_minutes = max(1, int(round(median_min)))
    future_index = pd.date_range(start=last_ts + timedelta(minutes=dt_minutes),
                                 periods=steps_horizon, freq=f"{dt_minutes}min")
    forecast_df = pd.DataFrame({
        'timestamp': future_index,
        'HI_forecast_C': hi_fore,
        'AQI_forecast': aqi_fore,
    })
else:
    forecast_df = pd.DataFrame()

# -----------------------------
# Metrics row
# -----------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Temperature", f"{temp:.1f} °C")
m2.metric("Humidity", f"{hum:.0f} %")
m3.metric("AQI", f"{aqi:.1f}", help=f"Now: {aqi_cat} | {forecast_minutes}m→ {aqi_fore_val:.1f} ({aqi_fore_cat})")
m4.metric("Heat Index", f"{current_hi:.1f} °C", help=f"PAGASA: {hi_cat} | {forecast_minutes}m→ {hi_fore_val:.1f} °C ({hi_fore_cat})")
with m5:
    chip_color = RISK_COLORS.get(risk_level, "#868E96")
    st.markdown(
        f'''<div style="background-color:{chip_color}; padding:10px; border-radius:10px; text-align:center;">
            <h3 style="color:white; margin:0;">{risk_level.upper()} RISK</h3>
        </div>''',
        unsafe_allow_html=True,
    )

if data_source == "Latest Reading" and aqi_imputed:
    st.caption("Note: AQI missing in latest record; used median AQI for stability.")

st.divider()

# -----------------------------
# Trends + forecast overlay
# -----------------------------
st.markdown("### 📈 Recent Environmental Trends & Forecast")
recent = loc_df.tail(20)
fig = go.Figure()

# PAGASA background bands
if show_pagasa_bands:
    for low, high, color in [(a,b,c) for a,b,c in HI_BAND_SHAPES]:
        y0 = -1e9 if low is None else low
        y1 = 1e9 if high is None else high
        fig.add_shape(type='rect', xref='paper', yref='y', x0=0, x1=1, y0=y0, y1=y1,
                      fillcolor=color, line=dict(width=0))

if 'tempC' in recent.columns:
    fig.add_trace(go.Scatter(x=recent['timestamp'], y=recent['tempC'], name="Temp (°C)", line=dict(color='firebrick')))
if 'aqi' in recent.columns:
    fig.add_trace(go.Scatter(x=recent['timestamp'], y=recent['aqi'], name="AQI (index)", line=dict(color='royalblue')))
if 'heat_index_C' in recent.columns:
    fig.add_trace(go.Scatter(x=recent['timestamp'], y=recent['heat_index_C'], name="Heat Index (°C)", line=dict(color='orange')))

# Forecast overlays (HI & AQI)
if not forecast_df.empty:
    fig.add_trace(go.Scatter(x=forecast_df['timestamp'], y=forecast_df['HI_forecast_C'],
                             name=f"HI forecast (+{forecast_minutes}m)", line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_df['timestamp'], y=forecast_df['AQI_forecast'],
                             name=f"AQI forecast (+{forecast_minutes}m)", line=dict(color='royalblue', dash='dot')))

fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Predictive Decision Support + Diagnostics
# -----------------------------
advice_map = {
    "Normal": "✅ Environment is stable. No immediate health risks detected.",
    "Moderate": "⚠️ Caution: Sensitive groups should limit prolonged outdoor exposure.",
    "High": "🚨 Alert: Poor environmental conditions. Avoid outdoor activities and close windows."
}

c1, c2 = st.columns([2, 1])
with c2:
    st.markdown("### 🤖 Predictive Decision Support")
    st.info(advice_map.get(risk_level, "Monitoring..."))
    st.markdown("**Model Diagnostic:**")
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        st.write(f"- Feature importances: tempC={fi[0]:.2f}, humidity={fi[1]:.2f}, aqi={fi[2]:.2f}")
    else:
        st.write("- Feature importances not available for this model.")
    st.markdown("**Forecast Diagnostic (backtest)**")
    st.write(f"- HI {forecast_minutes}m MAE={hi_mae:.2f}°C · RMSE={hi_rmse:.2f}°C · n={hi_n}")
    st.write(f"- AQI {forecast_minutes}m MAE={aqi_mae:.2f} · RMSE={aqi_rmse:.2f} · n={aqi_n}")
    if st.button("Refresh Real-Time Feed"):
        st.rerun()

with c1:
    st.markdown("### 🧭 Health Guidance Context")
    st.write("- **Heat Index (PAGASA):** Caution 27–32°C · Extreme Caution 33–41°C · Danger 42–51°C · Extreme Danger ≥52°C.")
    st.write("- **US EPA AQI:** Good (0–50) · Moderate (51–100) · Unhealthy for SG (101–150) · Unhealthy (151–200) · Very Unhealthy (201–300) · Hazardous (301+).")

# -----------------------------
# Risk Map – Dasmariñas (markers + optional barangay choropleth)
# -----------------------------
st.markdown("### 🗺️ Risk Map – Dasmariñas")

# Helper to compute risk for a given site and basis
def site_risk_for_basis(site_tag: str, basis: str):
    sdf = df_hist[df_hist['Location']==site_tag].sort_values('timestamp')
    if sdf.empty:
        return None
    latest = sdf.dropna(subset=['tempC','humidity']).tail(1)
    if latest.empty:
        return None
    t_now = float(latest['tempC'].iloc[0])
    h_now = float(latest['humidity'].iloc[0])
    a_now = float(latest['aqi'].iloc[0]) if 'aqi' in latest and pd.notna(latest['aqi'].iloc[0]) \
            else float(sdf['aqi'].median(skipna=True)) if 'aqi' in sdf.columns else np.nan
    median_m = median_minutes_delta(sdf['timestamp']) if not sdf.empty else 5.0
    if basis.startswith('Forecasted risk in 24h'):
        steps = int(max(1, round(1440 / max(1e-6, median_m))))
    elif basis.startswith('Forecasted risk in'):
        mins = forecast_minutes
        steps = int(max(1, round(mins / max(1e-6, median_m))))
    else:
        steps = 0

    if steps <= 0:
        X = pd.DataFrame([[t_now, h_now, a_now]], columns=['tempC','humidity','aqi'])
        ridx = model.predict(X)[0]
        rlabel = label_map.get(int(ridx), 'Unknown')
        hi_val = float(heat_index_celsius(np.array([t_now]), np.array([h_now]))[0])
        return {'risk': rlabel, 'hi_cat': pagasa_hi_category(hi_val), 'tempC': t_now, 'humidity': h_now, 'aqi': a_now}

    temp_f = forecast_next(sdf['tempC'], steps=steps, method=forecast_method, window=12)
    hum_f  = forecast_next(sdf['humidity'], steps=steps, method=forecast_method, window=12)
    aqi_f  = forecast_next(sdf['aqi'], steps=steps, method=forecast_method, window=12)
    hum_f = np.clip(hum_f, 0, 100)
    aqi_f = np.clip(aqi_f, 0, None)
    t_pred = float(temp_f[-1]) if len(temp_f) else np.nan
    h_pred = float(hum_f[-1]) if len(hum_f) else np.nan
    a_pred = float(aqi_f[-1]) if len(aqi_f) else np.nan
    Xf = pd.DataFrame([[t_pred, h_pred, a_pred]], columns=['tempC','humidity','aqi'])
    try:
        ridx = model.predict(Xf)[0]
        rlabel = label_map.get(int(ridx), 'Unknown')
    except Exception:
        rlabel = 'Unknown'
    hi_pred = float(heat_index_celsius(np.array([t_pred]), np.array([h_pred]))[0]) if np.isfinite(t_pred) and np.isfinite(h_pred) else np.nan
    return {'risk': rlabel, 'hi_cat': pagasa_hi_category(hi_pred), 'tempC': t_pred, 'humidity': h_pred, 'aqi': a_pred}

coords = st.session_state.site_coords
valid_sites = [(k,v) for k,v in coords.items() if v['lat'] is not None and v['lon'] is not None]

fig_map = go.Figure()
lats, lons = [], []
for tag, meta in valid_sites:
    stat = site_risk_for_basis(tag, map_basis)
    if stat is None:
        continue
    color = RISK_COLORS.get(stat['risk'], '#666666')
    caption = f"{tag} – {meta['name']}<br>Risk: {stat['risk']}<br>HI: {stat['hi_cat']}<br>T={stat['tempC']:.1f}°C, RH={stat['humidity']:.0f}%, AQI={stat['aqi']:.1f}"
    fig_map.add_trace(go.Scattermapbox(
        lat=[meta['lat']], lon=[meta['lon']],
        mode='markers+text',
        marker=dict(size=18, color=color),
        text=[tag], textposition='top center',
        hovertext=[caption], hoverinfo='text'
    ))
    lats.append(meta['lat']); lons.append(meta['lon'])

# ---- Optional Barangay Choropleth (GeoJSON upload)
st.sidebar.markdown("---")
st.sidebar.subheader("Barangay Choropleth (optional)")
geojson_file = st.sidebar.file_uploader("Upload barangay GeoJSON", type=["geojson","json"], accept_multiple_files=False)

geojson_data = None
feature_id_key = None
if geojson_file is not None:
    try:
        geojson_data = json.load(geojson_file)
        # Assign internal IDs to features for choropleth
        for i, feat in enumerate(geojson_data.get('features', [])):
            if 'properties' not in feat:
                feat['properties'] = {}
            feat['properties']['__id'] = i
        feature_id_key = "properties.__id"
    except Exception as e:
        st.sidebar.error(f"Failed to read GeoJSON: {e}")

# Optional mapping CSV: barangay -> site (A/B/C)
map_csv = st.sidebar.file_uploader("Optional CSV: barangay->site (columns: barangay,site)",
                                   type=["csv"], accept_multiple_files=False)
name_field_guess = st.sidebar.text_input("GeoJSON barangay name property (auto-detect if blank)", value="")

brgy_to_site = {}
if map_csv is not None:
    try:
        mdf = pd.read_csv(map_csv)
        if set(['barangay','site']).issubset(mdf.columns):
            for _, r in mdf.iterrows():
                brgy_to_site[str(r['barangay']).strip().lower()] = str(r['site']).strip().upper()
        else:
            st.sidebar.warning("CSV must have columns: barangay, site")
    except Exception as e:
        st.sidebar.error(f"Failed to read mapping CSV: {e}")

# Helper: centroid of Polygon/MultiPolygon (approx)
def feature_centroid(feat):
    try:
        geom = feat.get('geometry', {})
        gtype = geom.get('type')
        coords = geom.get('coordinates', [])
        pts = []
        if gtype == 'Polygon':
            ring = coords[0] if coords else []
            for pt in ring:
                pts.append(pt)  # [lon, lat]
        elif gtype == 'MultiPolygon':
            for poly in coords:
                ring = poly[0] if poly else []
                for pt in ring:
                    pts.append(pt)
        if not pts:
            return None, None
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        return float(np.mean(lats)), float(np.mean(lons))
    except Exception:
        return None, None

risk_to_num = {"Normal": 0, "Moderate": 1, "High": 2, "Unknown": 1}
colorscale = [[0.0, "#2B8A3E"], [0.5, "#F59F00"], [1.0, "#D9480F"]]

if geojson_data is not None:
    # Determine barangay name property
    props = geojson_data.get('features', [{}])[0].get('properties', {}) if geojson_data.get('features') else {}
    candidate_keys = [name_field_guess] if name_field_guess else []
    candidate_keys += ['brgy_name','barangay','Barangay','NAME_2','name','NAME']
    name_key = None
    for k in candidate_keys:
        if k and k in props:
            name_key = k
            break
    if name_key is None:
        st.sidebar.warning("Could not auto-detect barangay name property. Please fill it in.")

    # Build locations/z arrays
    loc_ids, z_vals, hover_texts = [], [], []
    assigned_sites = []
    for feat in geojson_data.get('features', []):
        fid = feat['properties'].get('__id')
        bname = str(feat['properties'].get(name_key, f"#{fid}")) if name_key else f"#{fid}"
        # Determine site: mapping -> else nearest
        site = None
        if brgy_to_site:
            site = brgy_to_site.get(bname.strip().lower())
            if site not in ['A','B','C']:
                site = None
        if site is None and len(valid_sites)>0:
            # nearest site by centroid
            latc, lonc = feature_centroid(feat)
            if latc is not None:
                dlist = []
                for tag, meta in valid_sites:
                    d = (latc - meta['lat'])**2 + (lonc - meta['lon'])**2
                    dlist.append((d, tag))
                dlist.sort()
                site = dlist[0][1]
        if site is None:
            site = 'A'  # fallback
        stat = site_risk_for_basis(site, map_basis)
        risk_label = stat['risk'] if stat else 'Unknown'
        loc_ids.append(fid)
        z_vals.append(risk_to_num.get(risk_label, 1))
        assigned_sites.append(site)
        hover_texts.append(f"{bname}<br>Site: {site}<br>Risk: {risk_label}")

    if loc_ids:
        fig_map.add_trace(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=loc_ids,
            z=z_vals,
            featureidkey=feature_id_key,
            colorscale=colorscale,
            zmin=0, zmax=2,
            marker_opacity=0.5,
            marker_line_width=0.6,
            hovertext=hover_texts,
            hoverinfo='text',
            showscale=False
        ))

# Finalize map view
if len(lats)>0:
    center_lat = float(np.mean(lats)); center_lon = float(np.mean(lons))
else:
    # fallback: approx Dasmariñas
    center_lat, center_lon = 14.33, 120.94

fig_map.update_layout(
    mapbox_style='open-street-map',
    mapbox_center=dict(lat=center_lat, lon=center_lon),
    mapbox_zoom=11,
    margin=dict(l=0, r=0, t=0, b=0), height=480
)
st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------
# Historical Analytics
# -----------------------------
with st.expander("🔎 View Historical Analytics for this Location"):
    st.write(loc_df.describe())
    if 'HI_Category' in df_hist.columns:
        st.write("**Heat Index Category Counts (PAGASA)**")
        st.write(df_hist[df_hist['Location']==loc_tag]['HI_Category'].value_counts())
    if 'AQI_Category' in df_hist.columns:
        st.write("**AQI Category Counts (US EPA)**")
        st.write(df_hist[df_hist['Location']==loc_tag]['AQI_Category'].value_counts())

# -----------------------------
# Export: 24h Forecast (CSV/PDF)
# -----------------------------
st.markdown("### 📤 Export 24‑Hour Forecast (CSV / PDF)")

# Build 24h forecast series for this site
if not loc_df.empty:
    dt_minutes = max(1, int(round(median_min)))
    future_index_24h = pd.date_range(start=loc_df['timestamp'].dropna().iloc[-1] + timedelta(minutes=dt_minutes),
                                     periods=steps_24h, freq=f"{dt_minutes}min")
    hi_fore_24h = forecast_next(hi_series, steps=steps_24h, method=forecast_method, window=12)
    aqi_fore_24h = forecast_next(aqi_series, steps=steps_24h, method=forecast_method, window=12)
    risk_fore_24h = []
    # To classify risk per step, we also need temp/hum forecasts
    temp_fore_24h = forecast_next(temp_series, steps=steps_24h, method=forecast_method, window=12) if len(temp_series)>0 else np.array([np.nan]*steps_24h)
    hum_fore_24h  = forecast_next(hum_series,  steps=steps_24h, method=forecast_method, window=12) if len(hum_series)>0 else np.array([np.nan]*steps_24h)
    hum_fore_24h = np.clip(hum_fore_24h, 0, 100)
    aqi_fore_24h = np.clip(aqi_fore_24h, 0, None)

    for i in range(steps_24h):
        t_pred = temp_fore_24h[i]
        h_pred = hum_fore_24h[i]
        a_pred = aqi_fore_24h[i]
        Xf = pd.DataFrame([[t_pred, h_pred, a_pred]], columns=['tempC','humidity','aqi'])
        try:
            ridx = model.predict(Xf)[0]
            rlabel = label_map.get(int(ridx), 'Unknown')
        except Exception:
            rlabel = 'Unknown'
        risk_fore_24h.append(rlabel)

    export_df = pd.DataFrame({
        'timestamp': future_index_24h,
        'HI_forecast_C': hi_fore_24h,
        'HI_PAGASA_Category': [pagasa_hi_category(v) for v in hi_fore_24h],
        'AQI_forecast': aqi_fore_24h,
        'AQI_Category': [epa_aqi_category(v) for v in aqi_fore_24h],
        'Risk_Class': risk_fore_24h,
    })
else:
    export_df = pd.DataFrame()

col_csv, col_pdf = st.columns(2)

# If reportlab isn't available, show a hint beside the PDF column
if not REPORTLAB_OK:
    with col_pdf:
        st.info('Install `reportlab` to enable PDF export: `pip install reportlab`')

if col_csv.button("⬇️ Download CSV (24h)"):
    if export_df.empty:
        st.warning("No data to export for this site.")
    else:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"forecast_{loc_tag}_24h_{ts_tag}.csv"
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv_bytes, file_name=fname, mime='text/csv')

# PDF export guarded by REPORTLAB_OK
if REPORTLAB_OK and col_pdf.button("⬇️ Download PDF Report (24h)"):
    if export_df.empty:
        st.warning("No data to export for this site.")
    else:
        # Build matplotlib plots (HI & AQI)
        fig1, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        hist = loc_df.tail(288)
        # History
        if 'heat_index_C' in hist.columns:
            ax[0].plot(hist['timestamp'], hist['heat_index_C'], label='HI (history)', color='tab:orange')
        if 'aqi' in hist.columns:
            ax[1].plot(hist['timestamp'], hist['aqi'], label='AQI (history)', color='tab:blue')
        # Forecast
        ax[0].plot(export_df['timestamp'], export_df['HI_forecast_C'], label='HI (24h forecast)', color='tab:orange', linestyle=':')
        ax[0].set_ylabel('HI (°C)')
        ax[0].legend(loc='upper left'); ax[0].grid(alpha=0.3)

        ax[1].plot(export_df['timestamp'], export_df['AQI_forecast'], label='AQI (24h forecast)', color='tab:blue', linestyle=':')
        ax[1].set_ylabel('AQI (index)')
        ax[1].legend(loc='upper left'); ax[1].grid(alpha=0.3)
        ax[1].set_xlabel('Time')

        buf = BytesIO()
        plt.tight_layout()
        fig1.savefig(buf, format='png', dpi=150)
        plt.close(fig1)
        buf.seek(0)

        # Create PDF
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M")
        pdf_name = f"forecast_{loc_tag}_24h_{ts_tag}.pdf"
        c = canvas.Canvas(pdf_name, pagesize=A4)
        W, H = A4
        y = H - 2*cm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, y, f"24-Hour Forecast Report – Site {loc_tag}")
        y -= 0.8*cm
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 0.6*cm
        c.drawString(2*cm, y, f"Forecast method: {forecast_method} · Horizon: 24h · Sampling step ≈ {int(round(median_min))} min")
        y -= 0.8*cm
        # Backtest metrics (24h)
        hi_mae24, hi_rmse24, hi_n24 = backtest_mae_rmse(hi_series, steps_24h, forecast_method)
        aqi_mae24, aqi_rmse24, aqi_n24 = backtest_mae_rmse(aqi_series, steps_24h, forecast_method)
        c.drawString(2*cm, y, f"Backtest – HI: MAE={np.nan_to_num(hi_mae24, nan=0):.2f}°C, RMSE={np.nan_to_num(hi_rmse24, nan=0):.2f}, n={hi_n24}")
        y -= 0.6*cm
        c.drawString(2*cm, y, f"Backtest – AQI: MAE={np.nan_to_num(aqi_mae24, nan=0):.2f}, RMSE={np.nan_to_num(aqi_rmse24, nan=0):.2f}, n={aqi_n24}")
        y -= 0.8*cm
        # Insert plot image
        img = ImageReader(buf)
        img_w = W - 4*cm
        img_h = img_w * 0.6
        c.drawImage(img, 2*cm, y - img_h, width=img_w, height=img_h)
        y = y - img_h - 0.6*cm
        # Table-like summary of last forecast point
        try:
            last_hi = export_df['HI_forecast_C'].iloc[-1]
            last_aqi = export_df['AQI_forecast'].iloc[-1]
            last_ts = export_df['timestamp'].iloc[-1]
            c.drawString(2*cm, y, f"End of horizon: {last_ts}")
            y -= 0.6*cm
            c.drawString(2*cm, y, f"HI at horizon: {last_hi:.1f}°C ({pagasa_hi_category(last_hi)})")
            y -= 0.6*cm
            c.drawString(2*cm, y, f"AQI at horizon: {last_aqi:.1f} ({epa_aqi_category(last_aqi)})")
        except Exception:
            pass
        c.showPage()
        c.save()

        with open(pdf_name, 'rb') as f:
            st.download_button("Download PDF", data=f, file_name=pdf_name, mime='application/pdf')

# -----------------------------
# Model Card
# -----------------------------
with st.expander("🪪 Model Card"):
    st.write("**Model**: Decision Tree Classifier (loaded from `dt_model.joblib`)")
    st.write("**Intended use**: Classify current environmental risk (Normal / Moderate / High) for the selected site.")
    st.write("**Inputs**: tempC, humidity, aqi (AQI index).")
    st.write("**Targets**: 3-class risk label.")
    st.write("**Training window**: _Update here with your training start & end timestamps._")
    if hasattr(model, 'classes_'):
        st.write(f"**Classes**: {list(model.classes_)}")
    else:
        st.write("**Classes**: Not available from model object.")
    st.write("**Evaluation**: _Add accuracy/F1 from your training notebook; include cross-validation details._")
    st.write("**Limitations**: Designed for shaded/light-wind conditions for HI; full sun can increase perceived temperature. AQI mapped as an index (dimensionless).")

st.caption("Developed for the Dasmariñas Environmental Monitoring Project. Standards: NOAA/NWS Heat Index; PAGASA HI categories; US EPA AQI.")
