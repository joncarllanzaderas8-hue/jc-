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

# ── 1. GLOBAL CONFIGURATION (Copied exactly from your notebook) ──────────────
SIGNALS = {
    'tempC':    {'label': 'Temperature',   'unit': '°C',  'color': '#ff6b6b', 'warn': 35,  'ylim': (24, 40)},
    'humidity': {'label': 'Humidity',      'unit': '%',   'color': '#4ecdc4', 'warn': 90,  'ylim': (55, 105)},
    'mqRaw':    {'label': 'MQ Gas Sensor', 'unit': 'raw', 'color': '#ffd93d', 'warn': 350, 'ylim': (50, 550)},
    'aqi':      {'label': 'AQI',           'unit': '',    'color': '#6bcb77', 'warn': 50,  'ylim': (0, 80)},
}

FORECAST_STEPS   = 48       
HOLT_ALPHA       = 0.30     
HOLT_BETA        = 0.15     
CI_Z             = 1.645    
MIN_OBS_FORECAST = 10       
LIVE_BUFFER_SIZE = 2000     

# ── 2. LOGIC HELPERS (Exact algorithms from your notebook) ──────────────────
def holt_forecast(series, alpha=HOLT_ALPHA, beta=HOLT_BETA, steps=FORECAST_STEPS):
    y = np.array(series, dtype=float)
    n = len(y)
    if n < 2:
        return np.full(steps, y[-1]), np.full(steps, y[-1]), np.full(steps, y[-1]), 0.0

    l = np.zeros(n); b = np.zeros(n)
    l[0] = y[0];     b[0] = y[1] - y[0]

    for t in range(1, n):
        l[t] = alpha * y[t] + (1 - alpha) * (l[t-1] + b[t-1])
        b[t] = beta  * (l[t] - l[t-1]) + (1 - beta) * b[t-1]

    forecasts = np.array([l[-1] + (h+1)*b[-1] for h in range(steps)])
    residuals = y[1:] - (l[:-1] + b[:-1])
    rmse      = np.sqrt(np.mean(residuals**2)) if len(residuals) else 0.0
    lower = forecasts - CI_Z * rmse * np.sqrt(np.arange(1, steps+1))
    upper = forecasts + CI_Z * rmse * np.sqrt(np.arange(1, steps+1))
    return forecasts, lower, upper, rmse

def parse_serial_line(line: str) -> dict | None:
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

# ── 3. STREAMLIT UI SETUP ──────────────────────────────────────────────────
st.set_page_config(page_title="Arduino Real-Time Forecast", layout="wide")

# Apply the dark theme style to Matplotlib
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#2a2d3a', 'axes.labelcolor': '#aaa',
    'xtick.color': '#888', 'ytick.color': '#888',
    'text.color': '#ddd', 'grid.color': '#2a2d3a',
    'font.family': 'monospace',
})

st.title("🌡️ Arduino Sensor — Real-Time Forecasting")

# Sidebar for controls
st.sidebar.header("Connection Settings")
port_list = [p.device for p in serial.tools.list_ports.comports()]
selected_port = st.sidebar.selectbox("Select Serial Port", ["Simulation"] + port_list)
run_button = st.sidebar.toggle("Start Forecasting Loop")

# ── 4. LIVE DASHBOARD RENDERER ──────────────────────────────────────────────
def render_st_dashboard(live_buf, placeholder, new_row=None, alerts=None):
    df_live = pd.DataFrame(list(live_buf)).set_index('timestamp')
    df_live = df_live[list(SIGNALS.keys())].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_live) < 2:
        return

    last_ts = df_live.index[-1]
    future_idx = pd.date_range(last_ts + pd.Timedelta('5min'), periods=FORECAST_STEPS, freq='5min')
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.2)
    HIST = min(len(df_live), 72)

    for idx, (col, cfg) in enumerate(SIGNALS.items()):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ser = df_live[col].iloc[-HIST:]
        fc, lo, hi, rmse = holt_forecast(df_live[col].values)

        ax.plot(ser.index, ser.values, color=cfg['color'], lw=1.8, label='Observed')
        ax.plot(future_idx, fc, color=cfg['color'], lw=2.0, ls='--', label='Forecast')
        ax.fill_between(future_idx, lo, hi, color=cfg['color'], alpha=0.15)
        ax.axvline(last_ts, color='white', lw=0.7, ls=':', alpha=0.4)
        ax.axhline(cfg['warn'], color='#ff6b6b', lw=0.8, ls='--', alpha=0.5)

        latest = df_live[col].iloc[-1]
        ax.set_title(f"{cfg['label']}: {latest:.1f}{cfg['unit']}", color=cfg['color'], fontweight='bold')
        ax.set_ylim(cfg['ylim'])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True)

    # Display the figure in the specific Streamlit placeholder
    placeholder.pyplot(fig)
    plt.close(fig)

# ── 5. MAIN EXECUTION LOOP ──────────────────────────────────────────────────
if run_button:
    # Initialize Buffer
    if 'live_buffer' not in st.session_state:
        st.session_state.live_buffer = deque(maxlen=LIVE_BUFFER_SIZE)
    
    plot_placeholder = st.empty()
    status_text = st.sidebar.empty()
    
    # Handle Serial Connection
    ser = None
    if selected_port != "Simulation":
        try:
            ser = serial.Serial(selected_port, 9600, timeout=1)
            time.sleep(2) # Arduino Reset
        except Exception as e:
            st.error(f"Failed to connect: {e}")
            run_button = False

    # The Loop
    try:
        while True:
            new_row = None
            
            if selected_port == "Simulation":
                # Generate fake data (Matching your notebook's simulation logic)
                time.sleep(1)
                last_val = st.session_state.live_buffer[-1] if st.session_state.live_buffer else {k: 25.0 for k in SIGNALS}
                row = {k: last_val.get(k, 25.0) + np.random.normal(0, 0.1) for k in SIGNALS}
                new_row = {'timestamp': datetime.now(), **row}
            else:
                if ser and ser.in_waiting > 0:
                    raw = ser.readline().decode('utf-8', errors='replace')
                    parsed = parse_serial_line(raw)
                    if parsed:
                        new_row = {'timestamp': datetime.now(), **parsed}

            if new_row:
                st.session_state.live_buffer.append(new_row)
                alerts = [f"{SIGNALS[k]['label']} high!" for k, v in new_row.items() if k in SIGNALS and v > SIGNALS[k]['warn']]
                
                # Update Dashboard
                render_st_dashboard(st.session_state.live_buffer, plot_placeholder, new_row, alerts)
                status_text.write(f"Last update: {new_row['timestamp'].strftime('%H:%M:%S')}")
            
            if not run_button: break
            time.sleep(0.1)

    except Exception as e:
        st.error(f"Loop Error: {e}")
    finally:
        if ser: ser.close()
else:
    st.info("Adjust settings in the sidebar and toggle 'Start' to begin visualization.")
