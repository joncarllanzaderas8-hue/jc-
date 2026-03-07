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

st.title("🌡️ Sensor Log — 4-Hour Forecasting")

# --- Sidebar: Data Loading ---
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload sensor_log.csv", type=["csv"])

def preprocess_data(df_raw):
    try:
        df = df_raw.copy()
        # Convert timestamp and handle errors
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp')
        
        # Notebook specific cleaning
        df['humidity'] = df['humidity'].clip(upper=100)
        df['aqi'] = df['aqi'].replace(0, np.nan).ffill().fillna(0)
        
        df = df.set_index('timestamp')
        # Resample to 5-minute intervals
        df_rs = df[['tempC', 'humidity', 'mqRaw', 'aqi']].resample('5min').mean()
        # Interpolate gaps up to 30 mins
        df_rs = df_rs.interpolate(method='linear', limit=6).dropna()
        return df_rs
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return pd.DataFrame()

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    df_processed = preprocess_data(df_raw)
    
    if not df_processed.empty:
        # --- Metrics ---
        last_ts = df_processed.index[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Temp", f"{df_processed['tempC'].iloc[-1]:.1f}°C")
        m2.metric("Humidity", f"{df_processed['humidity'].iloc[-1]:.1f}%")
        m3.metric("AQI", f"{df_processed['aqi'].iloc[-1]:.0f}")
        m4.metric("Last Sync", last_ts.strftime('%H:%M'))

        # --- Forecasting ---
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
                
                # Holt's Linear Trend (DES)
                model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
                forecast = model.forecast(48) # 48 * 5min = 4 hours
                
                # Plot last 12 hours of history + forecast
                hist_display = series.tail(144)
                ax.plot(hist_display.index, hist_display, color=color, alpha=0.3, label='History')
                ax.plot(forecast.index, forecast, color=color, linewidth=2, label='Forecast')
                
                # Confidence intervals (90%)
                sigma = np.std(model.resid)
                margin = 1.645 * sigma
                ax.fill_between(forecast.index, forecast - margin, forecast + margin, color=color, alpha=0.1)

                ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.2)

            st.pyplot(fig)
            
            st.subheader("Recent Data Log")
            # FIX: Use use_container_width=True instead of '100%'
            st.dataframe(df_processed.tail(15), use_container_width=True)

        except Exception as e:
            st.error(f"Forecasting calculation failed: {e}")
    else:
        st.warning("The uploaded file could not be processed. Please check the 'timestamp' column.")
else:
    st.info("Please upload your `sensor_log.csv` file in the sidebar to begin.")
