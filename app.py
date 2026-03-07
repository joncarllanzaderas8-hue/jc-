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

# --- Page Configuration ---
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

# --- 1. Data Loading & Sidebar ---
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload sensor_log.csv", type=["csv"])

def preprocess_pipeline(df_raw):
    """Replicates the 5-step pipeline from the notebook"""
    df = df_raw.copy()
    # [1] Sort & Fix timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # [2] Humidity Cap (Sensor saturation)
    df['humidity'] = df['humidity'].clip(upper=100)
    
    # [3] AQI Noise Fix (Rolling Median)
    rolling_med = df['aqi'].rolling(5, center=True, min_periods=1).median()
    df['aqi'] = np.where(df['aqi'] == 0, rolling_med, df['aqi'])
    
    # [4] Resample to 5-min buckets
    df = df.set_index('timestamp')
    df_rs = df[['tempC', 'humidity', 'mqRaw', 'aqi']].resample('5min').median()
    
    # [5] Interpolate (limit=6 steps/30 mins)
    df_rs = df_rs.interpolate(method='time', limit=6).dropna()
    return df_rs

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_processed = preprocess_pipeline(df_raw)
    
    # --- 2. Exploratory Data Analysis (EDA) ---
    st.header("📊 Exploratory Data Analysis")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Correlation Matrix")
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(df_processed.corr(), annot=True, cmap='coolwarm', ax=ax_corr, cbar=False)
        st.pyplot(fig_corr)
        
    with col2:
        st.subheader("Sensor Statistics")
        st.dataframe(df_processed.describe().T[['mean', 'std', 'min', 'max']], use_container_width=True)

    # --- 3. Run Forecasts for All Signals ---
    st.header("📈 4-Hour Predictive Forecasts")
    
    forecast_data = {}
    metrics = [
        ('tempC', 'Temperature (°C)', '#ff4b4b'),
        ('humidity', 'Humidity (%)', '#1f77b4'),
        ('mqRaw', 'Gas Sensor (Raw)', '#00d4a1'),
        ('aqi', 'AQI Index', '#ffaa00')
    ]
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3)

    for i, (col, title, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        series = df_processed[col]
        
        # Fit Holt's Double Exponential Smoothing (DES)
        model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
        forecast = model.forecast(48) # 48 steps * 5 mins = 4 hours
        forecast_data[col] = forecast
        
        # Plotting
        ax.plot(series.tail(100).index, series.tail(100), color=color, alpha=0.3, label='History')
        ax.plot(forecast.index, forecast, color=color, linewidth=2, label='Forecast')
        
        # Confidence Interval (Approx 90%)
        sigma = np.std(model.resid)
        ax.fill_between(forecast.index, forecast - 1.645*sigma, forecast + 1.645*sigma, color=color, alpha=0.1)
        
        ax.set_title(title, loc='left', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True, alpha=0.1)

    st.pyplot(fig)

    # --- 4. Individual Signal Deep-Dive ---
    st.header("🔍 Individual Signal Deep-Dive")
    selected_metric = st.selectbox("Select metric to inspect:", [m[0] for m in metrics])
    st.line_chart(df_processed[selected_metric].tail(200))

    # --- 5. Export Forecast to CSV ---
    st.header("📥 Data Export")
    export_df = pd.DataFrame(forecast_data)
    csv = export_df.to_csv().encode('utf-8')
    
    st.download_button(
        label="Download 4-Hour Forecast (CSV)",
        data=csv,
        file_name='forecast_results.csv',
        mime='text/csv'
    )

    # --- 6. Confusion Matrix (If Labels Exist) ---
    if 'Class' in df_raw.columns:
        st.header("🎯 Classification Evaluation")
        st.info("Confusion Matrix based on provided 'Class' labels in raw data.")
        # Placeholder for actual vs predicted logic if available in your model
        # cm = confusion_matrix(y_true, y_pred)
        # st.write("Confusion matrix visualization would appear here.")

else:
    st.info("Waiting for `sensor_log.csv` upload...")
