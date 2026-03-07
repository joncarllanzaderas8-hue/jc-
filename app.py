import os, json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Dasmariñas Risk Monitor", layout="wide")

# -----------------------------
# Data Loading & QC
# -----------------------------
@st.cache_data
def load_data(path: str = "sensor_log.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for c in ['aqi','mqRaw']:
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan
    return df

df_hist = load_data()

def run_qc_only(df_in: pd.DataFrame):
    if df_in.empty: return df_in, pd.DataFrame()
    df = df_in.copy()
    # Basic Thresholds
    df['qc_issue'] = (df['tempC'] < -20) | (df['tempC'] > 60) | df['tempC'].isna()
    # Simple summary
    summary = df.groupby('Location').size().reset_index(name='rows')
    return df, summary

df_hist, sensors_summary = run_qc_only(df_hist)

# -----------------------------
# Calculation Functions
# -----------------------------
def heat_index_celsius(temp_c, rh):
    T = temp_c * 9.0/5.0 + 32.0
    HI = (-42.379 + 2.049*T + 10.143*rh - 0.224*T*rh - 6.837e-3*T*T 
          - 5.481e-2*rh*rh + 1.228e-3*T*T*rh + 8.528e-4*T*rh*rh - 1.99e-6*T*T*rh*rh)
    return (HI - 32.0) * 5.0/9.0

def pagasa_hi_category(hi_c):
    if hi_c < 27: return "Not Hazardous"
    if hi_c <= 32: return "Caution"
    if hi_c <= 41: return "Extreme Caution"
    if hi_c <= 51: return "Danger"
    return "Extreme Danger"

# -----------------------------
# Sidebar & Model Loading
# -----------------------------
st.sidebar.header("📡 Settings")
location_mode = st.sidebar.selectbox("Monitoring Site", ["A - Green Space", "B - Residential", "C - Commercial"])
data_source = st.sidebar.radio("Data Source", ["Latest Reading", "Manual Input"])

try:
    model = joblib.load('dt_model.joblib')
    label_map = {0: "Normal", 1: "Moderate", 2: "High"}
except:
    st.error("Model 'dt_model.joblib' not found.")
    st.stop()

# -----------------------------
# Current Status Logic
# -----------------------------
st.title("🌿 Dasmariñas Environmental Risk Monitor")
loc_tag = location_mode[0]

if data_source == "Latest Reading":
    recent_loc = df_hist[df_hist['Location'] == loc_tag].sort_values('timestamp')
    if recent_loc.empty:
        st.warning("No data for this location.")
        st.stop()
    current = recent_loc.tail(1).iloc[0]
    temp, hum = float(current['tempC']), float(current['humidity'])
    aqi = float(current['aqi']) if pd.notna(current['aqi']) else 25.0
else:
    temp = st.sidebar.slider("Temp (°C)", 20.0, 45.0, 30.0)
    hum = st.sidebar.slider("Humidity (%)", 30.0, 100.0, 75.0)
    aqi = st.sidebar.slider("AQI", 0.0, 300.0, 25.0)

current_hi = heat_index_celsius(temp, hum)
risk_level = label_map.get(int(model.predict([[temp, hum, aqi]])[0]), "Unknown")

# -----------------------------
# Main Display (The "Big Four" Metrics)
# -----------------------------
st.markdown("### 📡 Current Environmental Status")

# Create 4 columns for the primary sensor data
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Temperature", f"{temp:.1f} °C")
with m2:
    st.metric("Humidity", f"{hum:.0f} %")
with m3:
    st.metric("AQI (MQ135)", f"{aqi:.1f}", delta_color="inverse")
with m4:
    # RESTORED: Heat Index Calculation
    st.metric("Heat Index", f"{current_hi:.1f} °C", help="Feels-like temperature based on PAGASA/NOAA standards")

st.divider()

# -----------------------------
# Restored Health & Risk Guides
# -----------------------------
st.markdown(f"### 📋 Risk Assessment: {risk_level}")

# Display the Health Guide based on the Decision Tree Prediction
if risk_level == "Normal":
    st.success("**Condition: Normal**\n\n✅ Air quality and heat levels are within safe limits. No special action required for the general public.")
elif risk_level == "Moderate":
    st.warning("**Condition: Moderate Risk**\n\n⚠️ **Health Advice:** Sensitive groups (children, elderly, and those with respiratory issues) should limit heavy outdoor activities. Stay hydrated.")
elif risk_level == "High":
    st.error("**Condition: High Risk**\n\n🚨 **Health Advice:** Dangerous conditions detected. Avoid outdoor exertion. Keep windows closed if AQI is high and use cooling systems to prevent heatstroke.")

# -----------------------------
# PAGASA Heat Index Reference
# -----------------------------
with st.expander("ℹ️ About the Heat Index (PAGASA Categories)"):
    hi_cat = pagasa_hi_category(current_hi)
    st.write(f"**Current Category:** {hi_cat}")
    st.info("""
    - **Caution (27–32°C):** Fatigue is possible with prolonged exposure.
    - **Extreme Caution (33–41°C):** Heat cramps and heat exhaustion are possible.
    - **Danger (42–51°C):** Heat exhaustion is likely; heat stroke is possible.
    - **Extreme Danger (52°C+):** Heat stroke is imminent.
    """)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# --- 1. Page Config & Professional Dark Theme ---
st.set_page_config(page_title="Sensor Forecasting", layout="wide")
warnings.filterwarnings('ignore')

# Custom CSS for the Dark Notebook Look
st.markdown("""
    <style>
        .main { background-color: #0f1117; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #1a1d27;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            color: #aaa;
        }
        .stTabs [aria-selected="true"] { color: #ff4b4b !important; border-bottom: 2px solid #ff4b4b; }
        div[data-testid="stMetric"] {
            background-color: #1a1d27;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #333;
        }
    </style>
    """, unsafe_allow_html=True)

# Global Plot Styling
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#333',
    'axes.labelcolor':  '#aaa',
    'xtick.color':      '#aaa',
    'ytick.color':      '#aaa',
    'text.color':       '#ddd',
    'grid.color':       '#2a2d3a',
    'grid.linewidth':   0.6,
    'font.family':      'monospace',
})

# --- 2. Logic Functions ---
def preprocess_data(df_raw):
    df = df_raw.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['humidity'] = df['humidity'].clip(upper=100)
    if 'aqi' in df.columns:
        rolling_med = df['aqi'].rolling(5, center=True, min_periods=1).median()
        df['aqi'] = np.where(df['aqi'] == 0, rolling_med, df['aqi'])
    df = df.set_index('timestamp')
    df_rs = df[['tempC', 'humidity', 'mqRaw', 'aqi']].resample('5min').median()
    return df_rs.interpolate(method='time', limit=6).dropna()

# --- 3. UI Content ---
st.title("🌡️ Sensor Log — 4-Hour Forecasting")

uploaded_file = st.sidebar.file_uploader("Upload sensor_log.csv", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_processed = preprocess_data(df_raw)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temp", f"{df_processed['tempC'].iloc[-1]:.1f}°C")
    m2.metric("Humidity", f"{df_processed['humidity'].iloc[-1]:.1f}%")
    m3.metric("AQI", f"{df_processed['aqi'].iloc[-1]:.0f}")
    m4.metric("Last Sync", df_processed.index[-1].strftime('%H:%M'))

    tab_eda, tab_forecast, tab_export = st.tabs(["📊 EDA", "📈 Forecasts", "📥 Export"])

    with tab_eda:
        st.subheader("Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
        sns.heatmap(df_processed.corr(), annot=True, cmap='coolwarm', ax=ax_corr, cbar=False)
        st.pyplot(fig_corr)
        st.dataframe(df_processed.describe().T, use_container_width=True)

    with tab_forecast:
        st.subheader("Holt's Double Exponential Smoothing (4-Hour Window)")
        metrics = ['tempC', 'humidity', 'mqRaw', 'aqi']
        colors = ['#ff4b4b', '#1f77b4', '#00d4a1', '#ffaa00']
        
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3)
        forecast_results = {}

        # THE CORRECTED FOR LOOP
        for i, (col, color) in enumerate(zip(metrics, colors)):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            series = df_processed[col]
            
            # Use Damped Trend to make the forecast more "correct" and less erratic
            model = ExponentialSmoothing(series, trend='add', damped_trend=True).fit()
            forecast = model.forecast(48)
            forecast_results[col] = forecast
            
            # Plotting
            ax.plot(series.tail(100).index, series.tail(100), color=color, alpha=0.3)
            ax.plot(forecast.index, forecast, color=color, linewidth=2, linestyle='--')
            
            # 90% Confidence Intervals
            sigma = np.std(model.resid)
            ax.fill_between(forecast.index, forecast - 1.645*sigma, forecast + 1.645*sigma, color=color, alpha=0.1)
            
            ax.set_title(f"Predicted {col}", loc='left', fontweight='bold')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.grid(True, alpha=0.1)

        st.pyplot(fig)

    with tab_export:
        st.subheader("Forecast Data")
        export_df = pd.DataFrame(forecast_results)
        st.dataframe(export_df, use_container_width=True)
        st.download_button("📥 Download CSV", export_df.to_csv().encode('utf-8'), "sensor_forecast.csv")

else:
    st.info("Upload your `sensor_log.csv` file in the sidebar to generate the dashboard.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import serial
import serial.tools.list_ports
import json
import re
import time
from datetime import datetime
from collections import deque

# ── 1. CONFIGURATION (Directly from Notebook) ──────────────────────────────
SIGNALS = {
    'tempC':    {'label': 'Temperature',   'unit': '°C',  'color': '#ff6b6b', 'warn': 35,  'ylim': (24, 40)},
    'humidity': {'label': 'Humidity',      'unit': '%',   'color': '#4ecdc4', 'warn': 90,  'ylim': (55, 105)},
    'mqRaw':    {'label': 'MQ Gas Sensor', 'unit': 'raw', 'color': '#ffd93d', 'warn': 350, 'ylim': (50, 550)},
    'aqi':      {'label': 'AQI',           'unit': '',    'color': '#6bcb77', 'warn': 50,  'ylim': (0, 80)},
}

FORECAST_STEPS = 48
HOLT_ALPHA = 0.30
HOLT_BETA = 0.15
CI_Z = 1.645
LIVE_BUFFER_SIZE = 2000

# ── 2. CORE LOGIC (Directly from Notebook) ──────────────────────────────────
def auto_detect_port():
    """Finds the first port that looks like an Arduino (CH340, CP210, etc)."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ['arduino', 'ch340', 'cp210', 'ftdi', 'usb serial']):
            return p.device
    return None

def holt_forecast(series, alpha=HOLT_ALPHA, beta=HOLT_BETA, steps=FORECAST_STEPS):
    y = np.array(series, dtype=float)
    n = len(y)
    if n < 2:
        return np.full(steps, y[-1]), np.full(steps, y[-1]), np.full(steps, y[-1]), 0.0
    l, b = np.zeros(n), np.zeros(n)
    l[0], b[0] = y[0], y[1] - y[0]
    for t in range(1, n):
        l[t] = alpha * y[t] + (1 - alpha) * (l[t-1] + b[t-1])
        b[t] = beta  * (l[t] - l[t-1]) + (1 - beta) * b[t-1]
    forecasts = np.array([l[-1] + (h+1)*b[-1] for h in range(steps)])
    residuals = y[1:] - (l[:-1] + b[:-1])
    rmse = np.sqrt(np.mean(residuals**2)) if len(residuals) else 0.0
    lower = forecasts - CI_Z * rmse * np.sqrt(np.arange(1, steps+1))
    upper = forecasts + CI_Z * rmse * np.sqrt(np.arange(1, steps+1))
    return forecasts, lower, upper, rmse

def parse_serial_line(line: str) -> dict:
    line = line.strip()
    if not line: return None
    if line.startswith('{'):
        try:
            d = json.loads(line)
            if all(k in d for k in SIGNALS): return d
        except: pass
    parts = re.split(r'[,\t;]', line)
    if len(parts) >= 4:
        try:
            keys = list(SIGNALS.keys())
            return {keys[i]: float(parts[i]) for i in range(4)}
        except: pass
    return None

# ── 3. UI SETUP ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Barangay Forecast", layout="wide")
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27', 
    'axes.edgecolor': '#2a2d3a', 'text.color': '#ddd', 'font.family': 'monospace'
})

st.title("📊 Barangay Sensor Forecast Dashboard")

# Sidebar for controls
st.sidebar.header("Connection")
detected = auto_detect_port()
if detected:
    st.sidebar.success(f"Detected: {detected}")
else:
    st.sidebar.warning("No Arduino detected.")

mode = st.sidebar.radio("Mode", ["Live Arduino", "Simulation"])
run_toggle = st.sidebar.toggle("Start Live Feed")

# ── 4. EXECUTION LOOP ───────────────────────────────────────────────────────
if run_toggle:
    # Maintain state across Streamlit reruns
    if 'live_buffer' not in st.session_state:
        st.session_state.live_buffer = deque(maxlen=LIVE_BUFFER_SIZE)
    
    plot_spot = st.empty()
    ser = None

    if mode == "Live Arduino":
        if not detected:
            st.error("Cannot start: No Arduino found.")
            st.stop()
        try:
            ser = serial.Serial(detected, 9600, timeout=2.0)
            time.sleep(2) # Arduino Reboot delay
        except Exception as e:
            st.error(f"Serial Error: {e}")
            st.stop()

    try:
        while True:
            new_row = None
            if mode == "Simulation":
                time.sleep(1)
                last = st.session_state.live_buffer[-1] if st.session_state.live_buffer else {k: 25.0 for k in SIGNALS}
                new_row = {k: last.get(k, 25.0) + np.random.normal(0, 0.2) for k in SIGNALS}
                new_row['timestamp'] = datetime.now()
            elif ser and ser.in_waiting > 0:
                raw = ser.readline().decode('utf-8', errors='replace')
                parsed = parse_serial_line(raw)
                if parsed:
                    new_row = {**parsed, 'timestamp': datetime.now()}

            if new_row:
                st.session_state.live_buffer.append(new_row)
                df = pd.DataFrame(list(st.session_state.live_buffer)).set_index('timestamp')
                
                # Plotting logic matching notebook visuals
                fig = plt.figure(figsize=(15, 10))
                gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)
                
                for idx, (col, cfg) in enumerate(SIGNALS.items()):
                    ax = fig.add_subplot(gs[idx//2, idx%2])
                    hist_data = df[col].tail(100)
                    fc, lo, hi, rmse = holt_forecast(df[col].values)
                    
                    # Generate future time index
                    future_ts = pd.date_range(df.index[-1], periods=FORECAST_STEPS+1, freq='5min')[1:]
                    
                    ax.plot(hist_data.index, hist_data.values, color=cfg['color'], label="Actual")
                    ax.plot(future_ts, fc, '--', color=cfg['color'], alpha=0.8, label="Forecast")
                    ax.fill_between(future_ts, lo, hi, color=cfg['color'], alpha=0.1)
                    ax.set_title(f"{cfg['label']}: {df[col].iloc[-1]:.1f}{cfg['unit']}")
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.grid(True, alpha=0.3)
                
                plot_spot.pyplot(fig)
                plt.close(fig)
            
            time.sleep(0.1)
    finally:
        if ser: ser.close()
