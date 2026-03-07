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
