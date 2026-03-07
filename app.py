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

# --- Custom Styling ---
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
st.title("🌡️ Sensor Log — 4-Hour Forecasting")

# --- Sidebar: Data Loading ---
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload sensor_log.csv", type=["csv"])

def preprocess_data(df_raw):
    try:
        df = df_raw.copy()
        # Handle mixed date formats safely
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp')
        
        # Preprocessing logic from notebook
        df['humidity'] = df['humidity'].clip(upper=100)
        df['aqi'] = df['aqi'].replace(0, np.nan).ffill().fillna(0)
        
        df = df.set_index('timestamp')
        # Resample to 5min and interpolate gaps (max 30 mins)
        df_rs = df[['tempC', 'humidity', 'mqRaw', 'aqi']].resample('5min').mean()
        df_rs = df_rs.interpolate(method='linear', limit=6).dropna()
        return df_rs
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    df_processed = preprocess_data(df_raw)
    
    if not df_processed.empty:
        # --- UI: Metrics ---
        last_ts = df_processed.index[-1]
        cols = st.columns(4)
        cols[0].metric("Temp", f"{df_processed['tempC'].iloc[-1]:.1f}°C")
        cols[1].metric("Humidity", f"{df_processed['humidity'].iloc[-1]:.1f}%")
        cols[2].metric("AQI", f"{df_processed['aqi'].iloc[-1]:.0f}")
        cols[3].metric("Last Update", last_ts.strftime('%H:%M'))

        # --- Forecasting & Plotting ---
        try:
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
                series = df_processed[key]
                
                # Fit Model
                model = ExponentialSmoothing(series, trend='add').fit()
                forecast = model.forecast(48) # 4 hours
                
                # Plotting
                ax.plot(series.tail(100).index, series.tail(100), color=color, alpha=0.4)
                ax.plot(forecast.index, forecast, color=color, linewidth=2, label='Forecast')
                
                ax.set_title(title, loc='left', color='#ddd')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            st.pyplot(fig)
            
            # Updated Table syntax for 2026 Streamlit versions
            st.subheader("Raw Data Preview")
            st.dataframe(df_processed.tail(10), width="100%") 
            
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
    else:
        st.warning("Processed data is empty. Check your CSV format.")
else:
    st.info("👋 Awaiting CSV upload...")
