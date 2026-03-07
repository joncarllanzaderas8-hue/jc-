import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# --- Page Config ---
st.set_page_config(page_title="Sensor Forecasting Dashboard", layout="wide")
warnings.filterwarnings('ignore')

# --- Custom Styling (Matching Notebook) ---
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
    'lines.linewidth':  1.8,
    'font.family':      'monospace',
})

# --- Title ---
st.title("🌡️ Sensor Log — 4-Hour Forecasting Dashboard")
st.markdown("""
> **Method:** Holt's Double Exponential Smoothing (DES)  
> **Preprocessing:** sort, humidity cap, AQI zero-fix, 5-min resampling, gap interpolation  
> **Forecast horizon:** 48 steps × 5 min = **4 hours** with 90% confidence intervals
""")

# --- Sidebar: Data Loading ---
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload sensor_log.csv", type=["csv"])

def preprocess_data(df_raw):
    df = df_raw.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    # 1. Sort
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Humidity Cap
    df['humidity'] = df['humidity'].clip(upper=100)
    
    # 3. AQI Zero-fix
    rolling_med = df['aqi'].rolling(5, center=True, min_periods=1).median()
    df['aqi'] = np.where(df['aqi'] == 0, rolling_med, df['aqi'])
    
    # 4. Resample to 5min
    df = df.set_index('timestamp')
    df_rs = df[['tempC', 'humidity', 'mqRaw', 'aqi']].resample('5min').median()
    
    # 5. Interpolate (limit 6 steps = 30 mins)
    df_rs = df_rs.interpolate(method='time', limit=6).dropna()
    return df_rs

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    df_processed = preprocess_data(df_raw)
    
    # --- UI: Metrics ---
    last_ts = df_processed.index[-1]
    cols = st.columns(4)
    cols[0].metric("Latest Temp", f"{df_processed['tempC'].iloc[-1]:.1f}°C")
    cols[1].metric("Latest Humidity", f"{df_processed['humidity'].iloc[-1]:.1f}%")
    cols[2].metric("Latest AQI", f"{df_processed['aqi'].iloc[-1]:.1f}")
    cols[3].metric("Last Sync", last_ts.strftime('%H:%M'))

    # --- Forecasting Logic ---
    forecast_steps = 48  # 4 hours
    results = {}
    
    for col in ['tempC', 'humidity', 'mqRaw', 'aqi']:
        series = df_processed[col]
        # Fit Holt's Linear (DES)
        model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
        forecast = model.forecast(forecast_steps)
        
        # Calculate naive 90% Confidence Interval (Std Dev of residuals)
        residuals = model.resid
        sigma = np.std(residuals)
        margin = 1.645 * sigma  # 90% CI
        
        results[col] = {
            'history': series,
            'forecast': forecast,
            'upper': forecast + margin,
            'lower': forecast - margin
        }

    # --- Plotting ---
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
    
    metrics_info = [
        ('tempC', 'Temperature (°C)', '#ff4b4b', gs[0, 0]),
        ('humidity', 'Humidity (%)', '#1f77b4', gs[0, 1]),
        ('mqRaw', 'Gas Sensor (Raw)', '#00d4a1', gs[1, 0]),
        ('aqi', 'AQI Index', '#ffaa00', gs[1, 1])
    ]

    for key, title, color, spec in metrics_info:
        ax = fig.add_subplot(spec)
        data = results[key]
        
        # Plot last 12 hours of history (144 points)
        hist_display = data['history'].tail(144)
        ax.plot(hist_display.index, hist_display, color=color, alpha=0.4, label='Historical')
        
        # Plot Forecast
        ax.plot(data['forecast'].index, data['forecast'], color=color, linewidth=2.5, label='Forecast (4h)')
        ax.fill_between(data['forecast'].index, data['lower'], data['upper'], color=color, alpha=0.15)
        
        # Formatting
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    st.pyplot(fig)

    # --- Data Table ---
    with st.expander("View Processed Data"):
        st.dataframe(df_processed.tail(20), use_container_width=True)

else:
    st.info("Please upload a `sensor_log.csv` file to start the dashboard.")
