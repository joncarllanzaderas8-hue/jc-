import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

# --- Page Config ---
st.set_page_config(page_title="Sensor Forecasting Dashboard", layout="wide")
warnings.filterwarnings('ignore')

# --- Notebook-Matched Styling ---
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

st.title("🌡️ Sensor Log — Analysis & 4-Hour Forecasting")

# --- Sidebar ---
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload sensor_log.csv", type=["csv"])

def preprocess_pipeline(df_raw):
    df = df_raw.copy()
    # 1. Sort & Fix timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # 2. Humidity Cap
    df['humidity'] = df['humidity'].clip(upper=100)
    
    # 3. AQI Noise Fix
    if 'aqi' in df.columns:
        rolling_med = df['aqi'].rolling(5, center=True, min_periods=1).median()
        df['aqi'] = np.where(df['aqi'] == 0, rolling_med, df['aqi'])
    
    # 4. Resample & Interpolate
    df = df.set_index('timestamp')
    cols_to_keep = [c for c in ['tempC', 'humidity', 'mqRaw', 'aqi'] if c in df.columns]
    df_rs = df[cols_to_keep].resample('5min').median()
    df_rs = df_rs.interpolate(method='time', limit=6).dropna()
    return df_rs

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_processed = preprocess_pipeline(df_raw)
    
    # --- Tabbed Navigation ---
    tab1, tab2, tab3 = st.tabs(["📊 EDA & Correlation", "📈 4-Hour Forecast", "🎯 Evaluation & Export"])

    with tab1:
        st.subheader("Exploratory Data Analysis")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Correlation Matrix**")
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_processed.corr(), annot=True, cmap='coolwarm', ax=ax_corr, cbar=False)
            st.pyplot(fig_corr)
        with c2:
            st.markdown("**Sensor Statistics**")
            st.dataframe(df_processed.describe().T, use_container_width=True)

    with tab2:
        st.subheader("Holt's Double Exponential Smoothing")
        metrics = [c for c in ['tempC', 'humidity', 'mqRaw', 'aqi'] if c in df_processed.columns]
        
        forecast_results = {}
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3)

        for i, col in enumerate(metrics):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            series = df_processed[col]
            
            # DES Model
            model = ExponentialSmoothing(series, trend='add').fit()
            forecast = model.forecast(48)
            forecast_results[col] = forecast
            
            # Plotting
            ax.plot(series.tail(100).index, series.tail(100), alpha=0.3, label='History')
            ax.plot(forecast.index, forecast, linewidth=2, label='Forecast')
            
            # 90% Confidence Interval
            sigma = np.std(model.resid)
            ax.fill_between(forecast.index, forecast - 1.645*sigma, forecast + 1.645*sigma, alpha=0.1)
            ax.set_title(f"Forecast: {col}", loc='left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        st.pyplot(fig)

    with tab3:
        # Confusion Matrix Logic
        if 'Class' in df_raw.columns:
            st.subheader("Confusion Matrix")
            # This is a placeholder for your classification logic
            st.info("Class labels detected. Displaying label distribution.")
            st.bar_chart(df_raw['Class'].value_counts())
        else:
            st.warning("No 'Class' column found for Confusion Matrix.")

        st.divider()
        st.subheader("Export Forecasts")
        export_df = pd.DataFrame(forecast_results)
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=export_df.to_csv().encode('utf-8'),
            file_name='sensor_forecast.csv',
            mime='text/csv'
        )
        st.dataframe(export_df, use_container_width=True)

else:
    st.info("Please upload `sensor_log.csv` to begin.")
