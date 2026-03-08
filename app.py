import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# Import your custom modules
from preprocessing import preprocess_site, SITE_NAME
from des import holt_forecast, tune_holt
from aqi import categorize_aqi

# -----------------------------
# 1. Setup & Model Loading
# -----------------------------
st.set_page_config(page_title="Dasmariñas Risk Monitor", layout="wide")

@st.cache_resource
def load_models():
    try:
        model = joblib.load('dt_model.joblib')
        return model
    except:
        st.error("Model 'dt_model.joblib' not found. Please ensure it is in the directory.")
        return None

risk_model = load_models()
label_map = {0: "Normal", 1: "Moderate", 2: "High"}

# -----------------------------
# 2. Calculation Helpers
# -----------------------------
def calculate_heat_index(temp_c, rh):
    if temp_c < 27: return temp_c
    T = temp_c * 9.0/5.0 + 32.0
    HI = (-42.379 + 2.049*T + 10.143*rh - 0.224*T*rh - 6.837e-3*T*T 
          - 5.481e-2*rh*rh + 1.228e-3*T*T*rh + 8.528e-4*T*rh*rh - 1.99e-6*T*T*rh*rh)
    return (HI - 32.0) * 5.0/9.0

# -----------------------------
# 3. Sidebar: Data Upload
# -----------------------------
with st.sidebar:
    st.header("📂 Data Input")
    uploaded_file = st.file_uploader("Upload sensor_log.csv", type=["csv"])
    
    st.divider()
    st.header("⚙️ Forecast Settings")
    auto_tune = st.checkbox("Auto-tune Model", value=True)
    history_hrs = st.slider("History Window (Hours)", 3, 24, 6)
    forecast_steps = st.select_slider("Forecast Horizon", options=[12, 24, 48], value=24)

# -----------------------------
# 4. Main Logic
# -----------------------------
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    
    # Site Selection
    available_sites = raw_df['Location'].unique()
    site_choice = st.selectbox("Select Monitoring Site", available_sites)
    
    # Filter and Preprocess
    site_df = raw_df[raw_df['Location'] == site_choice].copy()
    proc = preprocess_site(site_df) # Cleaning from your module
    
    # Inject Heat Index into historical data
    proc["heat_index"] = proc.apply(lambda r: calculate_heat_index(r["tempC"], r["humidity"]), axis=1)

    # --- PART A: ACTUAL CURRENT RISK ---
    current_data = proc.iloc[-1]
    curr_temp, curr_hum, curr_aqi = current_data['tempC'], current_data['humidity'], current_data['aqi']
    curr_hi = current_data['heat_index']
    
    # Predict Risk
    risk_idx = risk_model.predict([[curr_temp, curr_hum, curr_aqi]])[0]
    current_risk = label_map.get(int(risk_idx), "Unknown")

    st.title(f"📍 {site_choice} Status: {current_risk}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temperature", f"{curr_temp:.1f}°C")
    m2.metric("Humidity", f"{curr_hum:.0f}%")
    m3.metric("AQI", f"{curr_aqi:.1f}")
    m4.metric("Heat Index", f"{curr_hi:.1f}°C")

    # --- PART B: FORECASTING ---
    st.divider()
    st.subheader("🔮 4-Hour Predictive Forecast")
    
    signals = {
        "tempC": {"label": "Temperature", "color": "#ff6b6b"},
        "aqi": {"label": "AQI", "color": "#6bcb77"},
        "heat_index": {"label": "Heat Index", "color": "#ff8c00"}
    }

    forecast_results = {}
    for col in signals:
        series = proc[col].values
        if auto_tune:
            res = tune_holt(series, steps=forecast_steps)["model"]
        else:
            res = holt_forecast(series, alpha=0.3, beta=0.1, steps=forecast_steps)
        forecast_results[col] = res

    # --- PART C: FORECASTED RISK ---
    # We take the 4-hour mark (last step of forecast) and predict the risk
    future_temp = forecast_results["tempC"]["forecast"][-1]
    future_aqi = forecast_results["aqi"]["forecast"][-1]
    # Simplified humidity assumption for future risk (latest humidity)
    future_risk_idx = risk_model.predict([[future_temp, curr_hum, future_aqi]])[0]
    future_risk = label_map.get(int(future_risk_idx), "Unknown")

    st.info(f"The predicted risk level in 4 hours is: **{future_risk}**")

    # Display Chart
    fig = go.Figure()
    # Adding Heat Index Forecast as example
    fig.add_trace(go.Scatter(x=proc.index[-24:], y=proc["heat_index"].tail(24), name="Actual Heat Index"))
    
    future_times = pd.date_range(proc.index[-1], periods=forecast_steps+1, freq='5min')[1:]
    fig.add_trace(go.Scatter(x=future_times, y=forecast_results["heat_index"]["forecast"], 
                             name="Forecast", line=dict(dash='dash', color='orange')))
    
    fig.update_layout(template="plotly_dark", title="Heat Index Trend & Forecast")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a `sensor_log.csv` file to begin.")
