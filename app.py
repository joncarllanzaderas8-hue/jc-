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
from collections import deque
from datetime import datetime
import serial
import json
import threading
import time

# ── Sensor Config ───────────────────────────────
SIGNALS = {
    'tempC':    {'label': 'Temperature',   'unit': '°C',  'color': '#ff6b6b', 'warn': 35,  'ylim': (24, 40)},
    'humidity': {'label': 'Humidity',      'unit': '%',   'color': '#4ecdc4', 'warn': 90,  'ylim': (55, 105)},
    'mqRaw':    {'label': 'MQ Gas Sensor', 'unit': 'raw', 'color': '#ffd93d', 'warn': 350, 'ylim': (50, 550)},
    'aqi':      {'label': 'AQI',           'unit': '',    'color': '#6bcb77', 'warn': 150, 'ylim': (0, 200)}
}

FORECAST_STEPS = 48
HOLT_ALPHA = 0.4
HOLT_BETA = 0.2
LIVE_BUFFER_SIZE = 500

# ── Live Buffer ────────────────────────────────
live_buffer = deque(maxlen=LIVE_BUFFER_SIZE)

# ── Holt Forecast Function ─────────────────────
def holt_forecast(series, alpha=HOLT_ALPHA, beta=HOLT_BETA, steps=FORECAST_STEPS):
    level = series[0]
    trend = series[1] - series[0]
    levels = []
    trends = []
    for y in series:
        prev_level = level
        level = alpha*y + (1-alpha)*(level+trend)
        trend = beta*(level-prev_level) + (1-beta)*trend
        levels.append(level)
        trends.append(trend)
    forecasts = np.array([level+(i+1)*trend for i in range(steps)])
    residuals = series - np.array(levels)
    rmse = np.sqrt(np.mean(residuals**2))
    conf = 1.64 * rmse
    lower = forecasts - conf
    upper = forecasts + conf
    return forecasts, lower, upper, rmse

# ── Dashboard Plot ─────────────────────────────
def render_dashboard():
    if len(live_buffer) < 2:
        st.write("⏳ Waiting for data from Arduino...")
        return

    df_live = pd.DataFrame(list(live_buffer)).set_index('timestamp')
    df_live = df_live[list(SIGNALS.keys())].apply(pd.to_numeric, errors='coerce').dropna()
    if len(df_live) < 2:
        return

    last_ts = df_live.index[-1]
    future_idx = pd.date_range(last_ts + pd.Timedelta('5min'), periods=FORECAST_STEPS, freq='5min')

    fig = plt.figure(figsize=(18, 15), facecolor='#0f1117')
    now_str = datetime.now().strftime('%H:%M:%S')
    obs_str = f"{len(df_live)} obs"
    fc_end = future_idx[-1].strftime('%H:%M')
    title = f"⚡ LIVE Arduino Sensor Forecast  |  {now_str}  |  {obs_str}  |  horizon → {fc_end}"
    fig.suptitle(title, color='#6bcb77', fontsize=13, fontweight='bold', y=0.995)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.32)
    HIST = min(len(df_live), 72)

    for idx, (col, cfg) in enumerate(SIGNALS.items()):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ser = df_live[col].iloc[-HIST:]
        fc, lo, hi, rmse = holt_forecast(df_live[col].values)

        ax.plot(ser.index, ser.values, color=cfg['color'], lw=1.8, label='Observed', zorder=3)
        ax.plot(future_idx, fc, color=cfg['color'], lw=2.0, ls='--', label='Forecast', zorder=5)
        ax.fill_between(future_idx, lo, hi, color=cfg['color'], alpha=0.18, label='90% CI', zorder=2)
        ax.axvline(last_ts, color='white', lw=0.7, ls=':', alpha=0.4)
        warn_val = cfg['warn']
        ax.axhline(warn_val, color='#ff6b6b', lw=0.8, ls='--', alpha=0.5)

        latest = df_live[col].iloc[-1]
        warn = latest > warn_val
        badge_col = '#ff6b6b' if warn else cfg['color']

        ax.set_title(f"{cfg['label']}  ·  now: {latest:.1f} {cfg['unit']}" + ("  ⚠" if warn else ""),
                     color=badge_col, fontsize=11, fontweight='bold', pad=7)

        ax.annotate(f"{fc[-1]:.1f}{cfg['unit']}", xy=(future_idx[-1], fc[-1]), xytext=(-50,6),
                    textcoords='offset points', color=cfg['color'], fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=cfg['color'], lw=0.8))

        ax.set_ylim(cfg['ylim'])
        ax.set_xlabel('Time', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.grid(True)
        rmse_txt = f"RMSE {rmse:.2f}"
        ax.legend(fontsize=7, framealpha=0.25, facecolor='#222', edgecolor='#444', labelcolor='white',
                  loc='upper left', title=rmse_txt, title_fontsize=7)

    st.pyplot(fig)

# ── Arduino Serial Reading ─────────────────────
def read_serial(port='COM5', baud=9600):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    data = json.loads(line)
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    live_buffer.append(data)
                except:
                    continue
    except Exception as e:
        st.error(f"Serial Error: {e}")

# ── Streamlit App ─────────────────────────────
st.title("Arduino Sensor Forecast Dashboard")

arduino_port = st.text_input("Arduino COM Port", "COM5")
baud_rate = st.number_input("Baud Rate", 9600)

if st.button("Start Live Reading"):

    threading.Thread(target=read_serial, args=(arduino_port, baud_rate), daemon=True).start()
    st.success(f"Started reading from {arduino_port} at {baud_rate} baud!")

# Auto-refresh the dashboard every 5 seconds
while True:
    render_dashboard()
    time.sleep(5)
