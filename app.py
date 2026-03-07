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
    # Resample and interpolate
    df_rs = df[['tempC', 'humidity', 'mqRaw', 'aqi']].resample('5min').median()
    return df_rs.interpolate(method='time', limit=6).dropna()

# --- 3. Main UI ---
st.title("🌡️ Sensor Log — 4-Hour Forecasting")

uploaded_file = st.sidebar.file_uploader("Upload sensor_log.csv", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_rs = preprocess_data(df_raw)
    last_ts = df_rs.index[-1]

    # Forecast calculation
    results = {}
    metrics_meta = {
        'tempC':   {'label': 'Temperature', 'unit': '°C', 'color': '#ff4b4b'},
        'humidity':{'label': 'Humidity',    'unit': '%',  'color': '#1f77b4'},
        'mqRaw':   {'label': 'Gas (Raw)',   'unit': '',   'color': '#00d4a1'},
        'aqi':     {'label': 'AQI Index',   'unit': '',   'color': '#ffaa00'}
    }

    for col, meta in metrics_meta.items():
        if col in df_rs.columns:
            # Model fitting (Double Exponential Smoothing)
            model = ExponentialSmoothing(df_rs[col], trend='add', damped_trend=True).fit()
            forecast = model.forecast(48)
            sigma = np.std(model.resid)
            results[col] = {
                **meta,
                'forecast': forecast,
                'upper': forecast + (1.645 * sigma),
                'lower': forecast - (1.645 * sigma)
            }

    # Extract the future index safely
    future_idx = results['tempC']['forecast'].index

    tab_eda, tab_forecast, tab_export = st.tabs(["📊 EDA", "📈 Forecasts", "📥 Export"])

    with tab_eda:
        st.subheader("Correlation & Stats")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
        sns.heatmap(df_rs.corr(), annot=True, cmap='coolwarm', ax=ax_corr, cbar=False)
        st.pyplot(fig_corr)
        st.dataframe(df_rs.describe().T, use_container_width=True)

    with tab_forecast:
        # Configuration for History
        HISTORY_HOURS = 6
        history_start = last_ts - pd.Timedelta(hours=HISTORY_HOURS)
        hist = df_rs[df_rs.index >= history_start]

        fig = plt.figure(figsize=(18, 14), facecolor='#0f1117')
        
        # FIX: Using .iloc[-1] to avoid AttributeError
        fig.suptitle(
            f'Sensor Log — 4-Hour Forecast\n'
            f'History: last {HISTORY_HOURS} h  |  '
            f'{last_ts.strftime("%Y-%m-%d %H:%M")} → {future_idx.to_series().iloc[-1].strftime("%H:%M")}',
            color='white', fontsize=15, fontweight='bold', y=0.99)

        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.32)

        for idx, (col, res) in enumerate(results.items()):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])

            # Historical data
            ax.plot(hist.index, hist[col], color=res['color'], lw=1.8, label='Observed', zorder=3)

            # Forecasted line
            ax.plot(future_idx, res['forecast'], color=res['color'], lw=2.2, ls='--', label='Forecast', zorder=5)

            # Confidence band (90%)
            ax.fill_between(future_idx, res['lower'], res['upper'], color=res['color'], alpha=0.18, label='90% CI', zorder=2)

            # NOW divider line
            ax.axvline(last_ts, color='white', lw=0.8, ls=':', alpha=0.5, zorder=4)
            ax.text(last_ts, ax.get_ylim()[1], ' NOW', color='#888', fontsize=8, va='top')

            # Final value annotation (FIX: using .iloc[-1])
            last_val = res['forecast'].iloc[-1]
            ax.annotate(f"{last_val:.1f} {res['unit']}",
                        xy=(future_idx.to_series().iloc[-1], last_val),
                        xytext=(-48, 6), textcoords='offset points',
                        color=res['color'], fontsize=9, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=res['color'], lw=0.8))

            ax.set_title(f"{res['label']} ({res['unit']})", color='white', fontsize=13, fontweight='bold', pad=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
            ax.grid(True, alpha=0.15)
            ax.legend(fontsize=8, framealpha=0.25, facecolor='#222', edgecolor='#444', labelcolor='white', loc='upper left')

        st.pyplot(fig)

    with tab_export:
        st.subheader("Download Forecast Data")
        export_df = pd.DataFrame({k: v['forecast'] for k, v in results.items()})
        st.download_button("📥 Download Forecast CSV", export_df.to_csv().encode('utf-8'), "sensor_forecast.csv", "text/csv")
        st.dataframe(export_df, use_container_width=True)

else:
    st.info("👋 Dashboard ready. Please upload your `sensor_log.csv` file to begin analysis.")
