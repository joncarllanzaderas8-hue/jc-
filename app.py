import os, json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from io import BytesIO
from plotly.subplots import make_subplots

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

df = pd.DataFrame()
REALTIME_CSV = 'sensor_realtime.csv'

# Read historical log and optional realtime feed separately
df_hist = pd.DataFrame()
df_live = pd.DataFrame()
try:
    if os.path.exists('sensor_log.csv'):
        df_hist = pd.read_csv('sensor_log.csv')
        # convert types
        for c in ['tempC','humidity','aqi']:
            if c in df_hist.columns:
                df_hist[c] = pd.to_numeric(df_hist[c], errors='coerce')
        if 'timestamp' in df_hist.columns:
            df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], errors='coerce')
            df_hist = df_hist.sort_values(by='timestamp')
        df_hist = df_hist.dropna(subset=['timestamp'])
except Exception as e:
    print(f"Error reading sensor_log.csv: {e}")

try:
    if os.path.exists(REALTIME_CSV):
        df_live = pd.read_csv(REALTIME_CSV)
        for c in ['tempC','humidity','aqi']:
            if c in df_live.columns:
                df_live[c] = pd.to_numeric(df_live[c], errors='coerce')
        if 'timestamp' in df_live.columns:
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], errors='coerce')
            df_live = df_live.sort_values(by='timestamp')
        df_live = df_live.dropna(subset=['timestamp'])
except Exception as e:
    print(f"Error reading {REALTIME_CSV}: {e}")

# Default dataframe used for plotting/historical views — combine history + live if available
if not df_live.empty and not df_hist.empty:
    df = pd.concat([df_hist, df_live], ignore_index=True).sort_values('timestamp')
elif not df_live.empty:
    df = df_live.copy()
else:
    df = df_hist.copy()

# --- 3. MULTI-VARIABLE CHARTING ---
if not df.empty:
    # Create figure with a secondary y-axis to handle different scales
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Temperature (Left Axis)
    if 'timestamp' in df.columns and 'tempC' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['tempC'], name="Temp (°C)", 
                       line=dict(color='#ff4b4b', width=2)),
            secondary_y=False,
        )

    # Add Humidity (Left Axis)
    if 'timestamp' in df.columns and 'humidity' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['humidity'], name="Humidity (%)", 
                       line=dict(color='#ffa500', width=2)),
            secondary_y=False,
        )

    # Add AQI (Right Axis) - This prevents AQI from squashing the Temp line
    if 'timestamp' in df.columns and 'aqi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['aqi'], name="AQI (index)", 
                       line=dict(color='#00d4ff', width=2)),
            secondary_y=True,
        )

    # Add Forecasted points (Dotted lines) if they exist in your data
    if 'AQI forecast (+30m)' in df.columns and 'timestamp' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['AQI forecast (+30m)'], 
                       name="AQI Forecast", line=dict(color='#00d4ff', dash='dot')),
            secondary_y=True,
        )
# Style the layout for the Dark Theme
    fig.update_layout(
        template="plotly_dark",
        title="Dasmariñas Environmental Trends",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Set axis titles
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="AQI (MQ135)", secondary_y=True)
    
    # prediction confidence annotation (if available)
    try:
        if pred_conf is not None:
            if pred_conf >= 0.8:
                conf_bg = "#2ecc71"
            elif pred_conf >= 0.6:
                conf_bg = "#f1c40f"
            else:
                conf_bg = "#e74c3c"
            fig.add_annotation(
                text=f"Prediction confidence: {pred_conf:.0%}",
                xref='paper', yref='paper', x=0.99, y=0.95,
                showarrow=False, bgcolor=conf_bg, font=dict(color='black')
            )
    except Exception:
        pass

    st.plotly_chart(fig, width='stretch')
else:
    st.warning("Waiting for valid data to display the chart...")
df_hist = df.copy()
# -----------------------------
# Data Quality Checks (QC) & Sensor Trust Scoring
# -----------------------------
def run_qc_and_trust(df_in: pd.DataFrame):
    df = df_in.copy()
    now = pd.to_datetime(datetime.utcnow())
    # thresholds
    T_MIN, T_MAX = -20.0, 60.0
    HUM_MIN, HUM_MAX = 0.0, 100.0
    AQI_MIN, AQI_MAX = 0.0, 500.0

    # flags for issues
    df['qc_missing'] = df[['tempC','humidity','aqi','timestamp']].isna().any(axis=1)
    df['qc_out_of_range'] = False
    if 'tempC' in df.columns:
        df.loc[df['tempC'].notna() & ((df['tempC'] < T_MIN) | (df['tempC'] > T_MAX)), 'qc_out_of_range'] = True
    if 'humidity' in df.columns:
        df.loc[df['humidity'].notna() & ((df['humidity'] < HUM_MIN) | (df['humidity'] > HUM_MAX)), 'qc_out_of_range'] = True
    if 'aqi' in df.columns:
        df.loc[df['aqi'].notna() & ((df['aqi'] < AQI_MIN) | (df['aqi'] > AQI_MAX)), 'qc_out_of_range'] = True

    # large jumps per location (delta threshold per minute)
    df['qc_jump'] = False
    if 'timestamp' in df.columns and 'tempC' in df.columns and 'Location' in df.columns:
        df = df.sort_values(['Location','timestamp'])
        grp = df.groupby('Location')
        for name, g in grp:
            if len(g) < 2:
                continue
            dt = g['timestamp'].diff().dt.total_seconds().fillna(0) / 60.0
            dtemp = g['tempC'].diff().abs().fillna(0)
            # flag if temp change > 8°C within 1 minute, scaled
            mask = (dt <= 5) & (dtemp > 8)
            df.loc[mask.index, 'qc_jump'] = mask

    df['qc_issue'] = df['qc_missing'] | df['qc_out_of_range'] | df['qc_jump']

    # per-site summary
    sites = []
    for tag, group in df.groupby('Location') if 'Location' in df.columns else [(None, df)]:
        total = len(group)
        missing_pct = float(group['qc_missing'].sum() / total) if total else 0.0
        oor_pct = float(group['qc_out_of_range'].sum() / total) if total else 0.0
        recent = group.loc[group['timestamp'] >= (pd.to_datetime(now) - pd.Timedelta(days=1))] if 'timestamp' in group.columns else group
        recent_anom_pct = float(recent['qc_issue'].sum() / len(recent)) if len(recent) else 0.0
        last_ts = group['timestamp'].max() if 'timestamp' in group.columns else pd.NaT
        freshness_min = float((pd.to_datetime(now) - pd.to_datetime(last_ts)).total_seconds()/60.0) if pd.notna(last_ts) else float('inf')
        # trust score: start 1.0, subtract penalties
        p_missing = 0.4 * missing_pct
        p_oor = 0.3 * oor_pct
        p_fresh = 0.2 * min(1.0, freshness_min / (24*60))
        p_recent = 0.1 * recent_anom_pct
        trust = max(0.0, 1.0 - (p_missing + p_oor + p_fresh + p_recent))
        sites.append({'site': tag if tag is not None else 'global', 'total_rows': total,
                      'missing_pct': missing_pct, 'out_of_range_pct': oor_pct,
                      'recent_anom_pct': recent_anom_pct, 'last_seen': last_ts,
                      'freshness_min': freshness_min, 'trust_score': trust})

    df_sites = pd.DataFrame(sites)
    if 'site' in df_sites.columns:
        df_sites = df_sites.sort_values('site')
    return df, df_sites


# run QC and compute trust
df_hist, sensors_summary = run_qc_and_trust(df_hist)

# Sensor Health UI: show per-site trust scores and simple chart
with st.expander("Sensor Health (QC & Trust Scores)", expanded=False):
    st.write("Per-site sensor health summary (trust score, freshness, missing%):")
    try:
        if sensors_summary is not None and not sensors_summary.empty:
            df_display = sensors_summary.copy()
            df_display['trust_pct'] = (df_display['trust_score'] * 100).round(1)
            st.dataframe(df_display[['site','trust_pct','missing_pct','out_of_range_pct','recent_anom_pct','last_seen']], use_container_width=True)
            try:
                chart_df = df_display.set_index('site')['trust_pct']
                st.bar_chart(chart_df)
            except Exception:
                pass
        else:
            st.info("No sensor summary available.")
    except Exception:
        st.warning("Unable to render sensor health summary.")

# 3. THE CHART (The "Separated" View)
if not df.empty and 'timestamp' in df.columns:
    # Create figure with a secondary y-axis for AQI
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Temperature (Left Axis)
    if 'tempC' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['tempC'], name="Temp (°C)", line=dict(color='red')),
            secondary_y=False,
        )

    # Humidity (Left Axis - Shares scale with Temp)
    if 'humidity' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['humidity'], name="Humidity (%)", line=dict(color='green')),
            secondary_y=False,
        )

    # AQI (Right Axis - Separated so it doesn't squash the others)
    if 'aqi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['aqi'], name="AQI", line=dict(color='blue')),
            secondary_y=True,
        )

    fig.update_layout(template="plotly_dark", title="Dasmariñas Real-Time Monitor")
    # add prediction confidence annotation if available
    try:
        if pred_conf is not None:
            if pred_conf >= 0.8:
                conf_bg = "#2ecc71"
            elif pred_conf >= 0.6:
                conf_bg = "#f1c40f"
            else:
                conf_bg = "#e74c3c"
            fig.add_annotation(text=f"Prediction confidence: {pred_conf:.0%}", xref='paper', yref='paper', x=0.99, y=0.95, showarrow=False, bgcolor=conf_bg, font=dict(color='black'))
    except Exception:
        pass

    st.plotly_chart(fig, width='stretch')
else:
    st.warning("Waiting for sensor data...")

# -----------------------------
# Heat Index (NOAA/NWS Rothfusz + Steadman fallback)
# -----------------------------
# Ref: NOAA/NWS heat index equation
# https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
def heat_index_celsius(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    T = temp_c * 9.0/5.0 + 32.0  
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
    return (HI - 32.0) * 5.0/9.0  

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
location_mode = st.sidebar.selectbox("Select Monitoring Site", ["A - Green Space (TUP CAVITE)", "B - Residential (PALIPARAN III)", "C - Commercial (SM DASMA)"])
data_source = st.sidebar.radio("Data Source", ["Latest Reading", "Manual Input"])

# Uploader for live CSV feed (optional). Saves to REALTIME_CSV and refreshes df_live.
uploaded_live = st.sidebar.file_uploader("Upload live CSV (sensor_realtime.csv)", type=["csv"], help="Upload a CSV with columns: Location,timestamp,tempC,humidity,aqi")
if uploaded_live is not None:
    try:
        data = uploaded_live.read()
        with open(REALTIME_CSV, 'wb') as fh:
            fh.write(data)
        st.sidebar.success(f"Saved live feed to {REALTIME_CSV}")
        try:
            df_live = pd.read_csv(REALTIME_CSV)
            for c in ['tempC','humidity','aqi']:
                if c in df_live.columns:
                    df_live[c] = pd.to_numeric(df_live[c], errors='coerce')
            if 'timestamp' in df_live.columns:
                df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], errors='coerce')
                df_live = df_live.sort_values(by='timestamp')
            df_live = df_live.dropna(subset=['timestamp'])
        except Exception as e:
            st.sidebar.error(f"Failed to parse uploaded CSV: {e}")
    except Exception as e:
        st.sidebar.error(f"Failed to save uploaded file: {e}")

if st.sidebar.button("Clear live feed file"):
    try:
        if os.path.exists(REALTIME_CSV):
            os.remove(REALTIME_CSV)
            df_live = pd.DataFrame()
            st.sidebar.info("Cleared live feed file.")
    except Exception as e:
        st.sidebar.error(f"Failed to clear live file: {e}")

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
        'A': {'name': 'Green Space (TUP CAVITE)', 'lat': None, 'lon': None},
        'B': {'name': 'Residential (PALIPARAN III)', 'lat': None, 'lon': None},
        'C': {'name': 'Commercial (SM DASMA)', 'lat': None, 'lon': None},
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
    ts_sorted = ts.dropna()
    # Convert to datetime if strings
    if ts_sorted.dtype == 'object':
        ts_sorted = pd.to_datetime(ts_sorted, errors='coerce').dropna()
    ts_sorted = ts_sorted.sort_values()
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
    # Ensure last_ts is datetime
    if isinstance(last_ts, str):
        last_ts = pd.to_datetime(last_ts, errors='coerce')
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
    # show last retrain/test eval if available
    if 'retrain_eval' in st.session_state:
        re = st.session_state['retrain_eval']
        try:
            st.markdown("**Retrain/Test Eval (last run)**")
            st.write(f"- Accuracy: {re['acc']*100:.2f}% (n_test={re.get('n_test','?')})")
            cm_arr = np.array(re['cm'])
            cm_df_main = pd.DataFrame(cm_arr, index=[f"act_{i}" for i in range(cm_arr.shape[0])], columns=[f"pred_{i}" for i in range(cm_arr.shape[1])])
            st.dataframe(cm_df_main)
            st.write("Classification Report:")
            st.json(re['report'])
            # Temporal results if available
            if re.get('temporal'):
                tr = re['temporal']
                try:
                    st.markdown("**Temporal Holdout Eval (last run)**")
                    st.write(f"- Temporal Accuracy: {tr['acc']*100:.2f}% (n_test={tr.get('n_test','?')})")
                    if tr.get('avg_conf') is not None:
                        st.write(f"- Avg predicted class confidence: {tr['avg_conf']*100:.2f}%")
                    cm_t = np.array(tr['cm'])
                    cm_df_t = pd.DataFrame(cm_t, index=[f"act_{i}" for i in range(cm_t.shape[0])], columns=[f"pred_{i}" for i in range(cm_t.shape[1])])
                    st.dataframe(cm_df_t)
                    st.write("Temporal Classification Report:")
                    st.json(tr['report'])
                except Exception:
                    pass
        except Exception:
            pass
    if st.button("Refresh Real-Time Feed"):
        st.rerun()

    # -----------------------------
    with st.expander("⚙️ Retrain & Test (debug)"):
        st.write("Run a proper train/test split and evaluate a Decision Tree on held-out data.")
        auto_label = st.checkbox("Auto-generate labels from `tempC`/`humidity`/`aqi` (creates temp_label/hum_label/aq_label/Category_Label)", value=True)
        temporal_split = st.checkbox("Use temporal split (train on older rows, test on newer)", value=False)
        if temporal_split:
            holdout_mode = st.radio("Temporal holdout mode", ["last_percent","last_days"], index=0, horizontal=True)
            if holdout_mode == 'last_percent':
                holdout_percent = st.slider("Test holdout (percent of latest rows)", min_value=1, max_value=50, value=10)
            else:
                holdout_days = st.number_input("Test holdout (last N days)", min_value=1, max_value=365, value=7)
        # External weather merge
        ext_weather_file = st.file_uploader("Optional external-weather CSV (timestamp + rain_forecast)", type=["csv"], accept_multiple_files=False)
        if ext_weather_file is not None:
            merge_mode = st.radio("Merge method for external weather", ["exact","nearest"], index=1, horizontal=True)
            tol_minutes = st.number_input("Nearest merge tolerance (minutes)", min_value=1, max_value=1440, value=30)
        cv_enable = st.checkbox("Enable k-fold cross-validation (k-fold)", value=True)
        cv_folds = st.slider("k (folds)", min_value=3, max_value=10, value=5)
        auto_save = st.checkbox("Auto-save trained model when accuracy >=", value=False)
        save_threshold = st.number_input("Auto-save threshold (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
        save_filename = st.text_input("Save filename", value="dt_model_retrained.joblib")
        run_diag = st.button("Run train/test evaluation")
        if run_diag:
            # Expected columns (user-provided guidance): temp_label, hum_label, aq_label, Category_Label
            feature_cols = ['temp_label', 'hum_label', 'aq_label']
            target_col = 'Category_Label'

            working_df = df_hist.copy()
            # Auto-generate labels if requested
            if auto_label:
                required_source = ['tempC', 'humidity', 'aqi']
                missing_src = [c for c in required_source if c not in working_df.columns]
                if missing_src:
                    st.error(f"Cannot auto-generate labels: missing source columns {missing_src} (need tempC, humidity, aqi).")
                    st.stop()

                def make_temp_label(t):
                    try:
                        t = float(t)
                    except Exception:
                        return np.nan
                    if t < 27:
                        return 0
                    if t <= 32:
                        return 1
                    return 2

                def make_hum_label(h):
                    try:
                        h = float(h)
                    except Exception:
                        return np.nan
                    if h < 40:
                        return 0
                    if h <= 70:
                        return 1
                    return 2

                def make_aqi_label(a):
                    try:
                        a = float(a)
                    except Exception:
                        return np.nan
                    if a <= 50:
                        return 0
                    if a <= 100:
                        return 1
                    return 2

                working_df['temp_label'] = working_df['tempC'].apply(make_temp_label)
                working_df['hum_label'] = working_df['humidity'].apply(make_hum_label)
                working_df['aq_label'] = working_df['aqi'].apply(make_aqi_label)
                # Compose Category_Label as rounded mean of the three labels (deterministic rule)
                working_df['Category_Label'] = working_df[['temp_label', 'hum_label', 'aq_label']].mean(axis=1).round().astype('Int64')

            missing = [c for c in feature_cols + [target_col] if c not in working_df.columns]
            if missing:
                st.error(f"Missing required columns for retrain/test: {missing}.\n\nExpected: {feature_cols + [target_col]}")
            else:
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.tree import DecisionTreeClassifier
                    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

                    raw = working_df[feature_cols + [target_col]].dropna()
                    if len(raw) < 10:
                        st.warning(f"Not enough rows for reliable split (found {len(raw)}). Need >= 10 rows.")
                    else:
                        X = raw[feature_cols]
                        y = raw[target_col].astype(int)
                        strat = y if len(np.unique(y))>1 and len(y)>50 else None
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
                        clf = DecisionTreeClassifier(random_state=42)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)

                        acc = accuracy_score(y_test, y_pred)
                        cm = confusion_matrix(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True)

                        # optionally run cross-validation on full raw set
                        cv_results = None
                        if cv_enable:
                            try:
                                from sklearn.model_selection import cross_val_score, StratifiedKFold
                                if len(np.unique(y)) > 1:
                                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                                    scores = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=cv, scoring='accuracy')
                                else:
                                    scores = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=cv_folds, scoring='accuracy')
                                cv_results = {'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'folds': [float(s) for s in scores]}
                                st.write(f"Cross-val ({cv_folds} folds) accuracy: {cv_results['mean']*100:.2f}% ± {cv_results['std']*100:.2f}%")
                            except Exception as e:
                                st.warning(f"Cross-validation failed: {e}")

                        # If external weather provided, merge into working_df
                        if ext_weather_file is not None:
                            try:
                                ext_df = pd.read_csv(ext_weather_file)
                                # try common timestamp column names
                                ts_cols = [c for c in ext_df.columns if 'time' in c.lower() or 'timestamp' in c.lower() or c.lower()=='ts']
                                if not ts_cols:
                                    st.warning("External CSV lacks a recognizable timestamp column. Expected column like 'timestamp' or 'time'. Skipping merge.")
                                else:
                                    ext_ts_col = ts_cols[0]
                                    ext_df[ext_ts_col] = pd.to_datetime(ext_df[ext_ts_col], errors='coerce')
                                    if 'timestamp' in working_df.columns:
                                        working_df['timestamp'] = pd.to_datetime(working_df['timestamp'], errors='coerce')
                                    if merge_mode == 'exact':
                                        working_df = pd.merge(working_df, ext_df, left_on='timestamp', right_on=ext_ts_col, how='left')
                                    else:
                                        # nearest merge using merge_asof
                                        working_df = working_df.sort_values('timestamp')
                                        ext_df = ext_df.sort_values(ext_ts_col)
                                        working_df = pd.merge_asof(working_df, ext_df, left_on='timestamp', right_on=ext_ts_col, direction='nearest', tolerance=pd.Timedelta(minutes=int(tol_minutes)))
                            except Exception as e:
                                st.warning(f"Failed to read/merge external weather CSV: {e}")

                        # Optionally perform a temporal split evaluation (time-series holdout)
                        temporal_results = None
                        if temporal_split:
                            if 'timestamp' not in working_df.columns or working_df['timestamp'].isna().all():
                                st.warning("Temporal split requested but no valid `timestamp` column available in data.")
                            else:
                                try:
                                    raw_ts = working_df.dropna(subset=feature_cols + [target_col, 'timestamp']).sort_values('timestamp')
                                    n_ts = len(raw_ts)
                                    if n_ts < 10:
                                        st.warning(f"Not enough timestamped rows for temporal split (found {n_ts}).")
                                    else:
                                        if 'holdout_mode' in locals() and holdout_mode == 'last_percent':
                                            test_size = max(1, int(round(n_ts * float(holdout_percent) / 100.0)))
                                            split_idx = n_ts - test_size
                                            train_ts = raw_ts.iloc[:split_idx]
                                            test_ts = raw_ts.iloc[split_idx:]
                                        else:
                                            # last N days
                                            latest = raw_ts['timestamp'].max()
                                            cutoff = latest - timedelta(days=int(holdout_days))
                                            train_ts = raw_ts[raw_ts['timestamp'] < cutoff]
                                            test_ts = raw_ts[raw_ts['timestamp'] >= cutoff]
                                        if len(test_ts) == 0:
                                            st.warning("Temporal split resulted in empty test set; adjust holdout settings or provide more data.")
                                        else:
                                            X_tr_ts = train_ts[feature_cols]
                                            y_tr_ts = train_ts[target_col].astype(int)
                                            X_te_ts = test_ts[feature_cols]
                                            y_te_ts = test_ts[target_col].astype(int)
                                            clf_ts = DecisionTreeClassifier(random_state=42)
                                            clf_ts.fit(X_tr_ts, y_tr_ts)
                                            y_pred_ts = clf_ts.predict(X_te_ts)
                                            acc_ts = accuracy_score(y_te_ts, y_pred_ts)
                                            cm_ts = confusion_matrix(y_te_ts, y_pred_ts)
                                            report_ts = classification_report(y_te_ts, y_pred_ts, output_dict=True)
                                            # average confidence if available
                                            avg_conf = None
                                            if hasattr(clf_ts, 'predict_proba'):
                                                try:
                                                    probs = clf_ts.predict_proba(X_te_ts)
                                                    avg_conf = float(np.mean(np.max(probs, axis=1)))
                                                except Exception:
                                                    avg_conf = None
                                            temporal_results = {
                                                'acc': float(acc_ts), 'cm': cm_ts.tolist(), 'report': report_ts,
                                                'n_test': int(len(y_te_ts)), 'avg_conf': avg_conf
                                            }
                                            st.write(f"Temporal-split accuracy: {acc_ts*100:.2f}% (n_test={len(y_te_ts)})")
                                            st.write("Temporal Confusion Matrix:")
                                            st.dataframe(pd.DataFrame(cm_ts, index=[f"act_{i}" for i in range(cm_ts.shape[0])], columns=[f"pred_{i}" for i in range(cm_ts.shape[1])]))
                                            st.write("Temporal Classification Report:")
                                            st.json(report_ts)
                                            if avg_conf is not None:
                                                st.write(f"Average predicted class confidence (temporal test): {avg_conf*100:.2f}%")
                                except Exception as e:
                                    st.warning(f"Temporal split evaluation failed: {e}")

                        # persist metrics to session state for display elsewhere
                        st.session_state['retrain_eval'] = {
                            'acc': float(acc),
                            'cm': cm.tolist(),
                            'report': report,
                            'n_test': int(len(y_test)),
                            'cv': cv_results,
                            'temporal': temporal_results
                        }

                        st.success(f"Accuracy on test set: {acc*100:.2f}% (n_test={len(y_test)})")
                        st.write("Confusion Matrix (rows=actual, cols=predicted):")
                        cm_df = pd.DataFrame(cm, index=[f"act_{i}" for i in range(cm.shape[0])], columns=[f"pred_{i}" for i in range(cm.shape[1])])
                        st.dataframe(cm_df)
                        st.write("Classification Report:")
                        st.json(report)
                        st.write("Trained Decision Tree feature importances:")
                        fi = getattr(clf, 'feature_importances_', None)
                        if fi is not None:
                            fi_map = {c: float(v) for c, v in zip(feature_cols, fi)}
                            st.write(fi_map)
                        else:
                            st.write("Feature importances not available.")
                        # Auto-save if requested and accuracy threshold met
                        if auto_save:
                            try:
                                if (acc * 100.0) >= float(save_threshold):
                                    joblib.dump(clf, save_filename)
                                    st.success(f"Trained model saved to {save_filename} (accuracy {acc*100:.2f}%)")
                                    st.session_state['last_saved_model'] = save_filename
                                else:
                                    st.info(f"Model not saved: accuracy {acc*100:.2f}% < threshold {save_threshold}%")
                            except Exception as e:
                                st.error(f"Failed to save model: {e}")
                        # Manual save option
                        if st.button("Save trained model now"):
                            try:
                                joblib.dump(clf, save_filename)
                                st.success(f"Trained model saved to {save_filename}")
                                st.session_state['last_saved_model'] = save_filename
                            except Exception as e:
                                st.error(f"Failed to save model: {e}")
                except Exception as e:
                    st.error(f"Failed to run retrain/test: {e}")

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
st.plotly_chart(fig_map, use_container_width='stretch')

# -----------------------------
# Historical Analytics
# -----------------------------
with st.expander("🔎 View Historical Analytics for this Location"):
    # Show only numeric columns in describe() to avoid timestamp conversion issues
    numeric_cols = loc_df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        st.write(loc_df[numeric_cols].describe())
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
    last_ts_24h = loc_df['timestamp'].dropna().iloc[-1]
    # Ensure last_ts_24h is datetime
    if isinstance(last_ts_24h, str):
        last_ts_24h = pd.to_datetime(last_ts_24h, errors='coerce')
    future_index_24h = pd.date_range(start=last_ts_24h + timedelta(minutes=dt_minutes),
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

import threading
import time
import queue
import json
import re
from datetime import datetime
from collections import deque
from IPython.display import clear_output, display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import warnings

# Configuration
SIGNALS = {
    'tempC':    {'label': 'Temperature',   'unit': '°C',  'color': '#ff6b6b', 'warn': 35,  'ylim': (24, 40)},
    'humidity': {'label': 'Humidity',      'unit': '%',   'color': '#4ecdc4', 'warn': 90,  'ylim': (55, 105)},
    'mqRaw':    {'label': 'MQ Gas Sensor', 'unit': 'raw', 'color': '#ffd93d', 'warn': 350, 'ylim': (50, 550)},
    'aqi':      {'label': 'AQI',           'unit': '',    'color': '#6bcb77', 'warn': 50,  'ylim': (0, 80)},
}

FORECAST_STEPS = 48   # 4 hours
HOLT_ALPHA = 0.30     # Level smoothing
HOLT_BETA = 0.15      # Trend smoothing
CI_Z = 1.645          # 90% Confidence Interval

def holt_forecast(series, alpha=HOLT_ALPHA, beta=HOLT_BETA, steps=FORECAST_STEPS):
    y = np.array(series, dtype=float)
    n = len(y)
    if n < 2: return np.full(steps, y[-1]), np.full(steps, y[-1]), np.full(steps, y[-1]), 0.0
    
    l, b = np.zeros(n), np.zeros(n)
    l[0], b[0] = y[0], y[1] - y[0]
    for t in range(1, n):
        l[t] = alpha * y[t] + (1 - alpha) * (l[t-1] + b[t-1])
        b[t] = beta  * (l[t] - l[t-1]) + (1 - beta) * b[t-1]
        
    forecasts = np.array([l[-1] + (h+1)*b[-1] for h in range(steps)])
    rmse = np.sqrt(np.mean((y[1:] - (l[:-1] + b[:-1]))**2))
    lower = forecasts - CI_Z * rmse * np.sqrt(np.arange(1, steps+1))
    upper = forecasts + CI_Z * rmse * np.sqrt(np.arange(1, steps+1))
    return forecasts, lower, upper, rmse

def render_dashboard(live_buf, new_row=None, alerts=None):
    df_live = pd.DataFrame(list(live_buf)).set_index('timestamp')
    last_ts = df_live.index[-1]
    future_idx = pd.date_range(last_ts + pd.Timedelta('5min'), periods=FORECAST_STEPS, freq='5min')
    
    fig = plt.figure(figsize=(18, 15), facecolor='#0f1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.3)
    
    for idx, (col, cfg) in enumerate(SIGNALS.items()):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        fc, lo, hi, rmse = holt_forecast(df_live[col].values)
        ax.plot(df_live.index[-72:], df_live[col].iloc[-72:], color=cfg['color'], label='Observed')
        ax.plot(future_idx, fc, color=cfg['color'], ls='--', label='Forecast')
        ax.fill_between(future_idx, lo, hi, color=cfg['color'], alpha=0.15)
        ax.set_title(f"{cfg['label']} (Now: {df_live[col].iloc[-1]:.1f})", color=cfg['color'])
        ax.grid(True, alpha=0.2)
        
    clear_output(wait=True)
    display(fig)
    plt.close(fig)
