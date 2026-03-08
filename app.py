import os
import io
import json
import smtplib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh
from plotly.subplots import make_subplots

from preprocessing import preprocess_site, SITE_NAME  # retains your A/B/C defaults
from des import holt_forecast, tune_holt
from aqi import categorize_aqi  # EPA AQI categorizer (index -> (label, color))

COORDS_JSON = "site_coords.json"

df = pd.DataFrame()
REALTIME_CSV = 'sensor_realtime.csv'

# Read historical log and optional realtime feed separately
df_hist = pd.DataFrame()
df_live = pd.DataFrame()
try:
    if os.path.exists('sensor_log.csv'):
        df_hist = pd.read_csv('sensor_log.csv')
        # convert types
        for c in ['tempC','humidity','aqi']:
            if c in df_hist.columns:
                df_hist[c] = pd.to_numeric(df_hist[c], errors='coerce')
        if 'timestamp' in df_hist.columns:
            df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], errors='coerce')
            df_hist = df_hist.sort_values(by='timestamp')
        df_hist = df_hist.dropna(subset=['timestamp'])
except Exception as e:
    print(f"Error reading sensor_log.csv: {e}")

try:
    if os.path.exists(REALTIME_CSV):
        df_live = pd.read_csv(REALTIME_CSV)
        for c in ['tempC','humidity','aqi']:
            if c in df_live.columns:
                df_live[c] = pd.to_numeric(df_live[c], errors='coerce')
        if 'timestamp' in df_live.columns:
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], errors='coerce')
            df_live = df_live.sort_values(by='timestamp')
        df_live = df_live.dropna(subset=['timestamp'])
except Exception as e:
    print(f"Error reading {REALTIME_CSV}: {e}")

# Default dataframe used for plotting/historical views — combine history + live if available
if not df_live.empty and not df_hist.empty:
    df = pd.concat([df_hist, df_live], ignore_index=True).sort_values('timestamp')
elif not df_live.empty:
    df = df_live.copy()
else:
    df = df_hist.copy()

# --- 3. MULTI-VARIABLE CHARTING ---
if not df.empty:
    # Create figure with a secondary y-axis to handle different scales
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Temperature (Left Axis)
    if 'timestamp' in df.columns and 'tempC' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['tempC'], name="Temp (°C)", 
                       line=dict(color='#ff4b4b', width=2)),
            secondary_y=False,
        )

    # Add Humidity (Left Axis)
    if 'timestamp' in df.columns and 'humidity' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['humidity'], name="Humidity (%)", 
                       line=dict(color='#ffa500', width=2)),
            secondary_y=False,
        )

    # Add AQI (Right Axis) - This prevents AQI from squashing the Temp line
    if 'timestamp' in df.columns and 'aqi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['aqi'], name="AQI (index)", 
                       line=dict(color='#00d4ff', width=2)),
            secondary_y=True,
        )

    # Add Forecasted points (Dotted lines) if they exist in your data
    if 'AQI forecast (+30m)' in df.columns and 'timestamp' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['AQI forecast (+30m)'], 
                       name="AQI Forecast", line=dict(color='#00d4ff', dash='dot')),
            secondary_y=True,
        )
# Style the layout for the Dark Theme
    fig.update_layout(
        template="plotly_dark",
        title="Dasmariñas Environmental Trends",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Set axis titles
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="AQI (MQ135)", secondary_y=True)
    
    # prediction confidence annotation (if available)
    try:
        if pred_conf is not None:
            if pred_conf >= 0.8:
                conf_bg = "#2ecc71"
            elif pred_conf >= 0.6:
                conf_bg = "#f1c40f"
            else:
                conf_bg = "#e74c3c"
            fig.add_annotation(
                text=f"Prediction confidence: {pred_conf:.0%}",
                xref='paper', yref='paper', x=0.99, y=0.95,
                showarrow=False, bgcolor=conf_bg, font=dict(color='black')
            )
    except Exception:
        pass

    st.plotly_chart(fig, width='stretch')
else:
    st.warning("Waiting for valid data to display the chart...")
df_hist = df.copy()

with st.sidebar:
    with st.expander("📖 Dashboard User Guide"):
        st.markdown("""
        ### 1. Select Your Location
        Use the **'Site'** dropdown to choose a specific area. The map and charts will update automatically.
        
        ### 2. Choose a Metric
        Switch between **Temperature, Humidity, Heat Index, and AQI**. 
        * *Heat Index:* Tells you the 'Real Feel'—critical for heat stroke prevention!

        ### 3. Understanding the Forecast
        The graph shows **Actual Data** (solid line) vs. **Holt's DES Forecast** (dashed line).
        
        **Model Parameters:**
        * **Alpha (Level):** Controls how much the forecast reacts to the most recent data point.
        * **Beta (Trend):** Controls how much the forecast follows the 'slope' (upward/downward trend).

        **Accuracy Metrics (How good is the prediction?):**
        * **RMSE (Root Mean Square Error):** Measures average error magnitude. It punishes large outliers. **Lower is better.**
        * **MAE (Mean Absolute Error):** The average "off-by" amount. If MAE is 0.5, your forecast is typically off by 0.5 units.
        * **MAPE (Mean Absolute Percentage Error):** The error as a percentage. 
            * *Below 10%:* Excellent.
            * *10% - 25%:* Good/Dependable.
            * *> 50% :* Inaccurate forecasting.
        * **Residuals:** The "leftover" error. If these are small and random, your model is doing a great job!
        * **Confidence Interval:** The shaded 90% range where future values are likely to fall.

        ### 4. The Map Colors
        * **Green/Yellow:** Safe levels.
        * **Orange/Red:** High Heat or Poor Air Quality.
        * **Black/Empty:** No sensor data currently reporting from that zone.
        * **still in progress...**
        """)
        
with st.sidebar:
    st.divider()
    with st.expander("❓ How to read this Dashboard"):
        # 1. First, show the Sensor Guide and Heat Index table using Markdown
        st.markdown("""
        ### 📡 Sensor Guide
        * **Temp (°C):** Actual air temperature.
        * **Humidity (%):** Amount of moisture in the air.
        * **Heat Index:** The 'Real Feel' temperature (Calculated).
        * **AQI:** Air Quality Index (Lower is better).

        ### 🌡️ Heat Index Categories (PAGASA)
        | Range | Label | Precaution |
        | :--- | :--- | :--- |
        | 27-32°C | **Caution** | Fatigue possible |
        | 33-41°C | **Extreme Caution** | Heat cramps likely |
        | 42-51°C | **DANGER** | Heat stroke probable |
        | >52°C | **Extreme Danger** | Heat stroke imminent |

        ### 🌬️ AQI Health Levels
        """)

        # 2. Use a real DataFrame and st.table for the AQI labels so it looks professional
        aqi_data = {
            "Range": ["0-50", "51-100", "101-150", "151-200", "201-300", "301+"],
            "Health Level": ["Good", "Moderate", "Unhealthy*", "Unhealthy", "Very Unhealthy", "Hazardous"],
            "Precaution": ["None", "Limit effort", "Reduce exertion", "Avoid outdoors", "Stay indoors", "Emergency"]
        }
        st.table(pd.DataFrame(aqi_data))
        
        st.caption("*Unhealthy for Sensitive Groups")

import pandas as pd
import io

# --- 1. Create a Sample Template for Users to Download ---
def generate_sample_csv():
    example_data = {
        "timestamp": ["2024-05-20 10:00:00", "2024-05-20 10:05:00", "2024-05-20 10:10:00"],
        "Location": ["Salitran", "Salitran", "Salitran"], # 3 items
        "tempC": [30.5, 30.7, 31.0],                     # 3 items
        "humidity": [70, 71, 72],                        # 3 items
        "aqi": [42.1, 43.5, 44.0],                       # 3 items
        "mqRaw": [105, 108, 110]                         # 3 items
    }
    df_sample = pd.DataFrame(example_data)
    return df_sample.to_csv(index=False).encode('utf-8')

with st.sidebar:
    with st.expander("📝 CSV Upload Guide"):
        st.markdown("""
        **Location Flexibility:**
        You are not limited to A, B, or C! You can enter any Barangay or Landmark name in the `Location` column.
        
        **Examples:** 
        * `Salitran`
        * `Langkaan II`
        * `DBB-1`
        * `SM Dasma Area`
        
        **Required Columns:**
        1. `timestamp`: (YYYY-MM-DD HH:MM:SS)
        2. `Location`: (A, B, or C)
        3. `tempC`: Temperature in Celsius
        4. `humidity`: Humidity % (0-100)
        5. `aqi`: Air Quality Index value
        
        **Notes:** * Use **5-minute intervals** for best forecast accuracy.
        * Ensure no empty rows between data points.
        """)
        
        # Download button for the template
        st.download_button(
            label="📥 Download CSV Template",
            data=generate_sample_csv(),
            file_name="sensor_log_template.csv",
            mime="text/csv",
        )

    st.divider()
    
# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload sensor_log.csv", type=["csv"])
    default_path = os.path.join("sensor_log.csv")

    auto_tune = st.checkbox(
        "Auto‑tune α, β per signal",
        value=False,
        help="Grid search per site & per signal to minimize RMSE",
    )
    alpha = st.slider("Alpha (level)", 0.05, 0.9, 0.30, 0.05, disabled=auto_tune)
    beta = st.slider("Beta (trend)", 0.05, 0.9, 0.15, 0.05, disabled=auto_tune)

    history_hours = st.slider("History window (hours)", 3, 24, 6, 1)
    steps = st.select_slider(
        "Forecast horizon",
        options=[12, 24, 36, 48],
        value=48,
        help="steps × 5 min (48 = 4 hours)",
    )

    cat_scale = st.selectbox(
    "Category scale",
    ["EPA AQI (uses AQI value)", "PAGASA Heat Index (Celsius)"],
    index=0,
    help="EPA uses the AQI value (0-500); PAGASA uses the Heat Index categories (Caution to Extreme Danger)."
)

    tab_choice = st.radio("View", ["Single site", "Compare sites", "City map"], index=0)
    show_residuals = st.checkbox("Show residuals panel (deep‑dive)", value=True)

import os, json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Dasmariñas Risk Monitor", layout="wide")

# -----------------------------
# Data Loading & QC
# -----------------------------
@st.cache_data
def load_data(path: str = "sensor_log.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for c in ['aqi','mqRaw']:
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan
    return df

df_hist = load_data()

raw = df_hist 

def run_qc_only(df_in: pd.DataFrame):
    if df_in.empty: return df_in, pd.DataFrame()
    df = df_in.copy()
    # Basic Thresholds
    df['qc_issue'] = (df['tempC'] < -20) | (df['tempC'] > 60) | df['tempC'].isna()
    # Simple summary
    summary = df.groupby('Location').size().reset_index(name='rows')
    return df, summary

df_hist, sensors_summary = run_qc_only(df_hist)

# -----------------------------
# Calculation Functions
# -----------------------------
def heat_index_celsius(temp_c, rh):
    T = temp_c * 9.0/5.0 + 32.0
    HI = (-42.379 + 2.049*T + 10.143*rh - 0.224*T*rh - 6.837e-3*T*T 
          - 5.481e-2*rh*rh + 1.228e-3*T*T*rh + 8.528e-4*T*rh*rh - 1.99e-6*T*T*rh*rh)
    return (HI - 32.0) * 5.0/9.0

def pagasa_hi_category(hi_c):
    if hi_c < 27: return "Not Hazardous"
    if hi_c <= 32: return "Caution"
    if hi_c <= 41: return "Extreme Caution"
    if hi_c <= 51: return "Danger"
    return "Extreme Dander"

st.sidebar.header("📡 Settings")
location_mode = st.sidebar.selectbox("Monitoring Site", ["A - Green Space", "B - Residential", "C - Commercial"])
data_source = st.sidebar.radio("Data Source", ["Latest Reading", "Manual Input"])

try:
    model = joblib.load('environment_dt_model.joblib')
    label_map = {0: "Normal", 1: "Moderate", 2: "High"}
except:
    st.error("Model 'environment_dt_model.joblib' not found.")
    st.stop()
    
# -----------------------------
# Current Status Logic
# -----------------------------
st.title("🌿 Dasmariñas Environmental Risk Monitor")
loc_tag = location_mode[0]

if data_source == "Latest Reading":
    recent_loc = df_hist[df_hist['Location'] == loc_tag].sort_values('timestamp')
    if recent_loc.empty:
        st.warning("No data for this location.")
        st.stop()
    current = recent_loc.tail(1).iloc[0]
    temp, hum = float(current['tempC']), float(current['humidity'])
    aqi = float(current['aqi']) if pd.notna(current['aqi']) else 25.0
else:
    temp = st.sidebar.slider("Temp (°C)", 20.0, 45.0, 30.0)
    hum = st.sidebar.slider("Humidity (%)", 30.0, 100.0, 75.0)
    aqi = st.sidebar.slider("AQI", 0.0, 300.0, 25.0)

current_hi = heat_index_celsius(temp, hum)
risk_level = label_map.get(int(model.predict([[temp, hum, aqi]])[0]), "Unknown")

# -----------------------------
# Main Display (The "Big Four" Metrics)
# -----------------------------
st.markdown("### 📡 Recent Environmental Status")

# Create 4 columns for the primary sensor data
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Temperature", f"{temp:.1f} °C")
with m2:
    st.metric("Humidity", f"{hum:.0f} %")
with m3:
    st.metric("AQI (MQ135)", f"{aqi:.1f}", delta_color="inverse")
with m4:
    # RESTORED: Heat Index Calculation
    st.metric("Heat Index", f"{current_hi:.1f} °C", help="Feels-like temperature based on PAGASA/NOAA standards")

st.divider()

# -----------------------------
# Restored Health & Risk Guides
# -----------------------------
st.markdown(f"### 📋 Risk Assessment: {risk_level}")

# Display the Health Guide based on the Decision Tree Prediction
if risk_level == "Normal":
    st.success("**Condition: Normal**\n\n✅ Air quality and heat levels are within safe limits. No special action required for the general public.")
elif risk_level == "Moderate":
    st.warning("**Condition: Moderate Risk**\n\n⚠️ **Health Advice:** Sensitive groups (children, elderly, and those with respiratory issues) should limit heavy outdoor activities. Stay hydrated.")
elif risk_level == "High":
    st.error("**Condition: High Risk**\n\n🚨 **Health Advice:** Dangerous conditions detected. Avoid outdoor exertion. Keep windows closed if AQI is high and use cooling systems to prevent heatstroke.")


st.header("PREDICTIVE MODELLING")

    # --- FIX 1: CALCULATE HEAT INDEX COLUMN BEFORE FORECASTING ---
def your_forecast_function(proc, signals, steps, auto_tune, alpha, beta):
    # This whole block must be indented at least 4 spaces
    if "tempC" in proc.columns and "humidity" in proc.columns:
        proc["heat_index"] = proc.apply(
            lambda r: calculate_heat_index(r["tempC"], r["humidity"]), axis=1
        )

        last_ts = proc.index[-1]
        future_idx = pd.date_range(last_ts + pd.Timedelta("5min"), periods=steps, freq="5min")

        results = {}
        chosen = {}

        for col, meta in signals.items():
            series = proc[col].values if col in proc.columns else None
            
            if series is None or len(series) == 0 or np.all(np.isnan(series)):
                continue 

            if auto_tune:
                best = tune_holt(series, steps=steps)
                a, b = best["alpha"], best["beta"]
                res = best["model"]
            else:
                a, b = alpha, beta
                res = holt_forecast(series, alpha=a, beta=b, steps=steps)

            lo_clip, hi_clip = meta["clip"]
            if lo_clip is not None:
                res["forecast"] = np.maximum(res["forecast"], lo_clip)
                res["lower"] = np.maximum(res["lower"], lo_clip)
            if hi_clip is not None:
                res["forecast"] = np.minimum(res["forecast"], hi_clip)
                res["upper"] = np.minimum(res["upper"], hi_clip)

            results[col] = res
            chosen[col] = {"alpha": a, "beta": b, "rmse": float(res["rmse"])}

        # This return must be indented to align with the 'if' above
        return {
            "proc": proc,
            "last_ts": last_ts,
            "future_idx": future_idx,
            "results": results,
            "chosen": chosen,
        }
# ---------- Auto-detect sites & dynamic labels ----------
@st.cache_data(show_spinner=False)
def detect_sites_and_labels(df: pd.DataFrame) -> tuple[list[str], dict, dict]:
    """
    Returns (site_codes, code->label, label->code).
    Uses existing SITE_NAME for known codes; for others, creates 'Site <code>'.
    """
    codes = list(pd.Series(df["Location"].dropna().unique()).astype(str))
    code_to_label = {}
    for code in codes:
        if code in SITE_NAME:
            code_to_label[code] = SITE_NAME[code]
        else:
            code_to_label[code] = f"Site {code}"
    label_to_code = {v: k for k, v in code_to_label.items()}
    return codes, code_to_label, label_to_code


site_codes, CODE2LABEL, LABEL2CODE = detect_sites_and_labels(raw)

# ---------- Signals & helpers ----------
signals = {
    "tempC":      {"label": "Temperature", "unit": "°C", "color": "#ff6b6b", "clip": (None, None)},
    "humidity":   {"label": "Humidity",    "unit": "%",  "color": "#4ecdc4", "clip": (0, 100)},
    "heat_index": {"label": "Heat Index",  "unit": "°C", "color": "#ff8c00", "clip": (27, None)}, 
    "aqi":        {"label": "AQI",         "unit": "",   "color": "#6bcb77", "clip": (0, None)},
}

# ---------- Category utilities ----------
def calculate_heat_index(temp_c, humidity):
    """Calculates Heat Index in Celsius based on NOAA/PAGASA formula."""
    if temp_c < 27:
        return temp_c  # Heat Index is not defined below 27°C
    
    T = (temp_c * 9/5) + 32
    R = humidity
    
    hi_f = (-42.379 + (2.04901523 * T) + (10.14333127 * R) - 
            (0.22475541 * T * R) - (0.00683783 * T**2) - 
            (0.05481717 * R**2) + (0.00122874 * T**2 * R) + 
            (0.00085282 * T * R**2) - (0.00000199 * T**2 * R**2))
    
    return (hi_f - 32) * 5/9

# -----------------------------
# Forecast Accuracy Metrics
# -----------------------------
def compute_accuracy_metrics(actual, predicted):
    """
    Computes RMSE, MAE, and MAPE for forecast accuracy.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    error = actual - predicted

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    mask = actual != 0
    mape = np.mean(np.abs(error[mask] / actual[mask])) * 100 if np.any(mask) else np.nan

    return rmse, mae, mape
    
def heat_index_celsius(temp_c, rh):
    T = temp_c * 9.0/5.0 + 32.0
    HI = (-42.379 + 2.049*T + 10.143*rh - 0.224*T*rh - 6.837e-3*T*T 
          - 5.481e-2*rh*rh + 1.228e-3*T*T*rh + 8.528e-4*T*rh*rh - 1.99e-6*T*T*rh*rh)
    return (HI - 32.0) * 5.0/9.0

def pagasa_hi_category(hi_c):
    if hi_c < 27: return "Not Hazardous"
    if hi_c <= 32: return "Caution"
    if hi_c <= 41: return "Extreme Caution"
    if hi_c <= 51: return "Danger"
    return "Extreme Danger"

def categorize_pm25_denr(pm25_value: float) -> tuple[str, str]:
    """
    DENR DAO 2020-14 PM2.5 breakpoints (µg/m³):
      Good (0–25), Fair (25.1–35), USG (35.1–45), Very Unhealthy (45.1–55),
      Acutely Unhealthy (55.1–90), Emergency (> 91)
    Color palette chosen to roughly align with intuitive traffic-light + extended bands.
    """
    try:
        v = float(pm25_value)
    except Exception:
        return ("Unknown", "#888888")

    if v <= 25.0:
        return ("Good", "#00e400")               # green
    if v <= 35.0:
        return ("Fair", "#ffff00")               # yellow
    if v <= 45.0:
        return ("USG", "#ff7e00")                # orange
    if v <= 55.0:
        return ("Very Unhealthy", "#ff0000")     # red
    if v <= 90.0:
        return ("Acutely Unhealthy", "#8f3f97")  # purple-ish
    return ("Emergency", "#7e0023")              # maroon

def label_categories_vector(values: np.ndarray, scale: str) -> list[str]:
    """
    Return list of category labels for a numeric array:
      - "EPA AQI..." scale expects AQI values
      - "DENR PM2.5..." scale expects PM2.5 concentrations
    """
    labels = []
    for v in values:
        if scale.startswith("EPA"):
            cat, _ = categorize_aqi(float(v))
        else:
            cat, _ = categorize_pm25_denr(float(v))
        labels.append(cat)
    return labels


def hex_to_rgb_tuple(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


# ---------- Site processing ----------
@st.cache_data(show_spinner=False)
def process_site(df: pd.DataFrame, site_code: str, steps: int,
                 alpha: float | None, beta: float | None, auto_tune: bool):
    site_df = df[df["Location"] == site_code].copy()
    if site_df.empty:
        return None

    # This calls the code from your 2nd image
    proc = preprocess_site(site_df) 
    if proc.empty:
        return None

    # --- ADD THIS LOGIC HERE ---
    # We must calculate the index AFTER preprocessing but BEFORE the forecast loop
    if "tempC" in proc.columns and "humidity" in proc.columns:
        proc["heat_index"] = proc.apply(
            lambda r: calculate_heat_index(r["tempC"], r["humidity"]), axis=1
        )
    # ---------------------------

    last_ts = proc.index[-1]
    future_idx = pd.date_range(last_ts + pd.Timedelta("5min"), periods=steps, freq="5min")

    results = {}
    chosen = {}

    for col, meta in signals.items():
        # Now 'heat_index' will be found here because we added it above
        series = proc[col].values if col in proc.columns else None
        
        if series is None or len(series) == 0:
            continue
        
        # ... rest of your forecast code ...

        if auto_tune:
            best = tune_holt(series, steps=steps)
            a, b = best["alpha"], best["beta"]
            res = best["model"]
        else:
            a, b = alpha, beta
            res = holt_forecast(series, alpha=a, beta=b, steps=steps)

        lo_clip, hi_clip = meta["clip"]
        if lo_clip is not None:
            res["forecast"] = np.maximum(res["forecast"], lo_clip)
            res["lower"] = np.maximum(res["lower"], lo_clip)
        if hi_clip is not None:
            res["forecast"] = np.minimum(res["forecast"], hi_clip)
            res["upper"] = np.minimum(res["upper"], hi_clip)

        results[col] = res
        chosen[col] = {"alpha": a, "beta": b, "rmse": float(res["rmse"])}

    return {
        "proc": proc,
        "last_ts": last_ts,
        "future_idx": future_idx,
        "results": results,
        "chosen": chosen,
    }


# ---------- Helper: choose numeric series for selected category scale ----------
def pick_category_series(bundle: dict, scale: str) -> tuple[np.ndarray, pd.DatetimeIndex, str]:
    """
    Returns (forecast_values, future_index, source_name)
      - EPA AQI scale -> use bundle['results']['aqi'] forecast
      - DENR PM2.5 scale -> use bundle['results']['pm25'] forecast if present; else fallback to AQI
    """
    if scale.startswith("DENR"):
        res_pm = bundle["results"].get("pm25")
        if res_pm is not None and len(res_pm["forecast"]) > 0:
            return res_pm["forecast"], bundle["future_idx"], "pm25"
        # fallback
    res_aqi = bundle["results"].get("aqi")
    if res_aqi is None:
        return np.array([]), bundle["future_idx"], "none"
    return res_aqi["forecast"], bundle["future_idx"], "aqi"


def first_crossing_index(series: np.ndarray, threshold: float) -> int | None:
    mask = series > threshold
    return int(np.argmax(mask)) if mask.any() else None


# =====================================================================================
# Single site
# =====================================================================================
if tab_choice == "Single site":
    site_choice = st.selectbox("Site", options=[CODE2LABEL[c] for c in site_codes])
    site_code = LABEL2CODE[site_choice]

    bundle = process_site(raw, site_code, steps, alpha, beta, auto_tune)
    if bundle is None:
        st.error("No rows found for the selected site.")
        st.stop()

    proc = bundle["proc"]
    last_ts = bundle["last_ts"]
    future_idx = bundle["future_idx"]
    results_site = bundle["results"]
    chosen = bundle["chosen"]

    st.caption("Smoothing parameters (by signal):")
    st.json(
        {
            k: {
                "alpha": round(v["alpha"], 3) if v["alpha"] is not None else None,
                "beta": round(v["beta"], 3) if v["beta"] is not None else None,
                "rmse": round(v["rmse"], 3) if np.isfinite(v["rmse"]) else None,
            }
            for k, v in chosen.items()
        },
        expanded=False,
    )

    tabs = st.tabs([f"{meta['label']}" for meta in signals.values()])

    for tab, (col, meta) in zip(tabs, signals.items()):
        with tab:
            # 1. Safety Check: Does the data exist for this specific sensor?
            if col not in proc.columns or col not in results_site:
                st.warning(f"No data for {meta['label']} at {site_choice}.")
                continue

            st.subheader(f"{site_choice} — {meta['label']}")
            
            # 2. Prepare Data
            hist = proc.iloc[-12 * history_hours:]  # 12 points per hour
            fc = results_site[col]["forecast"]
            fitted = results_site[col]["fitted"]
            
            # Calculate metrics BEFORE displaying them
            actual_vals = proc[col].iloc[:len(fitted)].values
            rmse_calc, mae_calc, mape_calc = compute_accuracy_metrics(actual_vals, fitted)

            # 3. Main Forecast Plot
            fig = go.Figure()
            # Observed Data
            fig.add_trace(go.Scatter(x=hist.index, y=hist[col], mode="lines", name="Observed", line=dict(color=meta["color"], width=2)))
            # Forecasted Data
            fig.add_trace(go.Scatter(x=future_idx, y=fc, mode="lines", name="Forecast", line=dict(color=meta["color"], width=2, dash="dash")))
            # Confidence Intervals
            fig.add_trace(go.Scatter(x=future_idx, y=results_site[col]["upper"], mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=future_idx, y=results_site[col]["lower"], mode="lines", fill="tonexty", fillcolor="rgba(160,160,160,0.2)", line=dict(width=0), name="90% CI"))
            
            fig.add_vline(x=last_ts, line_width=1, line_dash="dot", line_color="white")
            fig.update_layout(height=420, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # 4. KPI Metrics Cards
            last_val = hist[col].iloc[-1]
            fc1 = fc[11] if len(fc) >= 12 else np.nan
            fc2 = fc[23] if len(fc) >= 24 else np.nan
            fc4 = fc[-1] if len(fc) > 0 else np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last observed", f"{last_val:.2f} {meta['unit']}")
            c2.metric("+1 hour", f"{fc1:.2f} {meta['unit']}" if np.isfinite(fc1) else "—")
            c3.metric("+2 hours", f"{fc2:.2f} {meta['unit']}" if np.isfinite(fc2) else "—")
            c4.metric("+4 hours", f"{fc4:.2f} {meta['unit']}" if np.isfinite(fc4) else "—")

            fitted = results_site[col]["fitted"]
            actual_vals = proc[col].iloc[:len(fitted)].values
            rmse_calc, mae_calc, mape_calc = compute_accuracy_metrics(actual_vals, fitted)
            
            st.markdown("---")
            st.subheader("🎯 Accuracy Evaluation")
            
            # Ito yung code para maging small at side-by-side ang metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.caption("RMSE")
                st.markdown(f"#### {rmse_calc:.3f}")
            
            with col_b:
                st.caption("MAE")
                st.markdown(f"#### {mae_calc:.3f}")
            
            with col_c:
                st.caption("MAPE")
                st.markdown(f"#### {mape_calc:.2f}%")
            
            
            # 6. Residuals Section
            if show_residuals:
                st.markdown("**Residuals (Observed − Fitted One-Step Ahead)**")
                fitted_idx = proc.index[: len(fitted)]
                resid = proc[col].iloc[: len(fitted)] - pd.Series(fitted, index=fitted_idx)
                
                fig_r = px.bar(
                    x=resid.index,
                    y=resid.values,
                    labels={"x": "Time", "y": "Residual"},
                    height=200,
                    template="plotly_dark",
                )
                st.plotly_chart(fig_r, use_container_width=True)

    # ---------- Export CSV with per-step category (for selected scale) ----------
    st.markdown("---")
    st.subheader("Export forecast (with categories)")

    export_rows = []
    for col, meta in signals.items():
        res = results_site.get(col)
        if res is None:
            continue

        # compute category per step using selected scale's numeric series
        if cat_scale.startswith("DENR") and source_name == "pm25" and col == "pm25":
            cat_list = label_categories_vector(res["forecast"], cat_scale)
        elif cat_scale.startswith("EPA") and col == "aqi":
            cat_list = label_categories_vector(res["forecast"], cat_scale)
        elif col == "heat_index":
          # The function defined on line 169 is pagasa_hi_category
            cat_list = [pagasa_hi_category(v) for v in res["forecast"]]
        else:
            # not the "category-driving" series; leave blank
            cat_list = [""] * len(res["forecast"])
            
        for i, ts in enumerate(future_idx):
            export_rows.append(
                {
                    "site_label": site_choice,
                    "location_code": site_code,
                    "timestamp": ts,
                    "signal": col,
                    "forecast": round(res["forecast"][i], 3),
                    "lower_90ci": round(res["lower"][i], 3),
                    "upper_90ci": round(res["upper"][i], 3),
                    "category_scale": "DENR PM2.5" if cat_scale.startswith("DENR") else "EPA AQI",
                    "category": cat_list[i],
                    "rmse_in_sample": round(res["rmse"], 3),
                }
            )

    exp_df = pd.DataFrame(export_rows)
    st.download_button(
        "⬇️ Download this site's 4‑hour forecast (CSV)",
        data=exp_df.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{site_code}.csv",
        mime="text/csv",
    )


# =====================================================================================
# Compare sites (auto-detected)
# =====================================================================================
if tab_choice == "Compare sites":
    st.subheader("Multi‑site comparison (side‑by‑side)")
    signal_choice = st.selectbox("Signal", list(signals.keys()), format_func=lambda k: signals[k]["label"])

    # Build bundles for all detected sites
    site_bundles = {}
    for code in site_codes:
        site_bundles[code] = process_site(raw, code, steps, alpha, beta, auto_tune)

    cols = st.columns(3 if len(site_codes) >= 3 else max(1, len(site_codes)))
    # Render in pages of 3 columns if many sites
    per_row = len(cols)

    for start in range(0, len(site_codes), per_row):
        row_codes = site_codes[start : start + per_row]
        row_cols = st.columns(len(row_codes))
        for i, code in enumerate(row_codes):
            bundle = site_bundles.get(code)
            label = CODE2LABEL.get(code, f"Site {code}")
            with row_cols[i]:
                st.markdown(f"**{label}**")
                if bundle is None:
                    st.warning(f"No data for {label}")
                    continue

                proc = bundle["proc"]
                last_ts = bundle["last_ts"]
                future_idx = bundle["future_idx"]
                res = bundle["results"].get(signal_choice)
                if res is None or signal_choice not in proc.columns:
                    st.warning(f"No {signals[signal_choice]['label']} series for {label}.")
                    continue

                hist = proc.iloc[-12 * history_hours:]  # last N hours

                # Category badge for "current" based on selected scale
                if cat_scale.startswith("DENR"):
                    if "MQ135" in proc.columns:
                        current_val = float(proc["MQ135"].iloc[-1])
                        cat, color = categorize_pm25_denr(current_val)
                        st.markdown(
                            f"Current PM2.5: **{current_val:.1f} µg/m³** — "
                            f"<span style='color:{color}'>{cat}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("MQ135 not available; using AQI categories.")
                        current_val = float(proc["aqi"].iloc[-1]) if "aqi" in proc.columns else np.nan
                        if np.isfinite(current_val):
                            cat, color = categorize_aqi(current_val)
                            st.markdown(
                                f"Current AQI: **{current_val:.1f}** — "
                                f"<span style='color:{color}'>{cat}</span>",
                                unsafe_allow_html=True,
                            )
                else:
                    current_val = float(proc["aqi"].iloc[-1]) if "aqi" in proc.columns else np.nan
                    if np.isfinite(current_val):
                        cat, color = categorize_aqi(current_val)
                        st.markdown(
                            f"Current AQI: **{current_val:.1f}** — "
                            f"<span style='color:{color}'>{cat}</span>",
                            unsafe_allow_html=True,
                        )

                # Figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist[signal_choice], mode="lines", name="Observed"))
                fig.add_trace(go.Scatter(x=future_idx, y=res["forecast"], mode="lines", name="Forecast",
                                         line=dict(dash="dash")))
                fig.add_trace(go.Scatter(x=future_idx, y=res["upper"], mode="lines", line=dict(width=0),
                                         showlegend=False))
                fig.add_trace(go.Scatter(x=future_idx, y=res["lower"], mode="lines", fill="tonexty",
                                         fillcolor="rgba(160,160,160,0.2)", line=dict(width=0), name="90% CI"))
                fig.add_vline(x=last_ts, line_width=1, line_dash="dot", line_color="white")
                fig.update_layout(height=360, template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)


# =====================================================================================
# City map (Dasmariñas): color each barangay by selected category scale
# =====================================================================================
if tab_choice == "City map":
    st.subheader("Dasmariñas barangay map — category coloring")

    # Build a 'current status' per site using selected category scale
    latest_by_site = {}
    for code in site_codes:
        b = process_site(raw, code, steps, alpha, beta, auto_tune)
        if b is None:
            continue
        proc = b["proc"]
        if cat_scale.startswith("DENR") and "pm25" in proc.columns:
            val = float(proc["pm25"].iloc[-1])
            cat, color = categorize_pm25_denr(val)
        else:
            if "aqi" not in proc.columns:
                continue
            val = float(proc["aqi"].iloc[-1])
            cat, color = categorize_aqi(val)
        latest_by_site[code] = {"label": CODE2LABEL[code], "value": val, "cat": cat, "color": color}

    # Files (optional)
    gj_path = os.path.join("dasmarinas_barangays.geojson")      # GeoJSON polygons
    bind_path = os.path.join("site_binding.csv")                # polygon_name,location_code
    pts_path = os.path.join("barangays_dasmarinas.csv")         # name,lat,lon,location_code

    def attach_color_to_feature(feat, latest_dict):
        name = feat.get("properties", {}).get("name") or feat.get("properties", {}).get("NAME")
        code = name_to_code.get(name)
        if code and code in latest_dict:
            cat = latest_dict[code]["cat"]
            color_hex = latest_dict[code]["color"]
            feat["properties"]["aqi_cat"] = cat
            feat["properties"]["color_hex"] = color_hex
        else:
            feat["properties"]["aqi_cat"] = "Unknown"
            feat["properties"]["color_hex"] = "#888888"

    def gj_color_getter(f):
        hx = f["properties"]["color_hex"]
        r, g, b = hex_to_rgb_tuple(hx)
        return [r, g, b, 160]

    if os.path.exists(gj_path) and os.path.exists(bind_path):
        st.caption("Using GeoJSON polygons + site binding")
        with open(gj_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
        bind_df = pd.read_csv(bind_path)  # columns: polygon_name, location_code
        name_to_code = dict(zip(bind_df["polygon_name"], bind_df["location_code"]))

        for feat in geojson.get("features", []):
            attach_color_to_feature(feat, latest_by_site)

        layer = pdk.Layer(
            "GeoJsonLayer",
            geojson,
            stroked=True,
            filled=True,
            get_fill_color=gj_color_getter,
            get_line_color=[255, 255, 255],
            lineWidthMinPixels=1,
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=14.329, longitude=120.936, zoom=12)
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}\n{aqi_cat}"})
        st.pydeck_chart(deck, use_container_width=True)

    elif os.path.exists(pts_path):
        st.caption("Using barangay point markers (lat/lon)")
        pts = pd.read_csv(pts_path)  # name,lat,lon,location_code

        def map_cat(row):
            code = str(row["location_code"])
            return latest_by_site.get(code, {}).get("cat", "Unknown")

        def map_color(row):
            code = str(row["location_code"])
            return latest_by_site.get(code, {}).get("color", "#888888")

        pts["aqi_cat"] = pts.apply(map_cat, axis=1)
        pts["color_hex"] = pts.apply(map_color, axis=1)
        pts["r"] = pts["color_hex"].apply(lambda h: hex_to_rgb_tuple(h)[0])
        pts["g"] = pts["color_hex"].apply(lambda h: hex_to_rgb_tuple(h)[1])
        pts["b"] = pts["color_hex"].apply(lambda h: hex_to_rgb_tuple(h)[2])

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position=["lon", "lat"],
            get_radius=120,
            get_fill_color=["r", "g", "b", 180],
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=14.329, longitude=120.936, zoom=12)
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}\n{aqi_cat}"})
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.warning(
            "To enable the map, add either:\n"
            "1) `data/dasmarinas_barangays.geojson` **and** `data/site_binding.csv` (columns: polygon_name,location_code),\n"
            "   or\n"
            "2) `data/barangays_dasmarinas.csv` with columns: name,lat,lon,location_code."
        )
        st.info(
            "Tip: `location_code` can be any code present in your CSV (auto‑detected). "
            "We color each barangay by the **selected** category scale (EPA AQI or DENR PM2.5)."
        )


# ---------- Footer note ----------
st.info(
    "Notes: Preprocessing = sort → humidity cap at 100 → AQI zero‑fix (rolling median, w=5) → "
    "5‑min resample (median) → interpolate ≤ 30 min. Forecasts use DES with optional auto‑tuning and "
    "90% CIs; humidity clipped to [0,100] and AQI ≥ 0. Categories: EPA/AirNow AQI (0–500) or "
    "DENR PM2.5 (DAO 2020‑14), selectable at left."
)

import streamlit as st
import time
import pandas as pd
import numpy as np

# Set page config for a wider layout
st.set_page_config(page_title="Sensor Data Processor", layout="wide")

def process_data(df_raw):
    """Encapsulates your processing logic with logging."""
    df = df_raw.copy()
    logs = []

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # ── Step 1: Sort by timestamp ───────────────
    ooo = (df['timestamp'].diff().dt.total_seconds() < 0).sum()
    df = df.sort_values('timestamp').reset_index(drop=True)
    logs.append(f"✅ Sorted: {ooo} out-of-order records fixed.")

    # ── Step 2: Cap humidity at 100% ────────────
    sat = (df['humidity'] >= 100).sum()
    df['humidity'] = df['humidity'].clip(upper=100)
    logs.append(f"✅ Humidity: {sat} readings capped at 100%.")

    # ── Step 3: Replace AQI = 0 with rolling median ──────────
    zero_aqi = (df['aqi'] == 0).sum()
    rolling_med = df['aqi'].rolling(5, center=True, min_periods=1).median()
    df['aqi'] = np.where(df['aqi'] == 0, rolling_med, df['aqi'])
    logs.append(f"✅ AQI: {zero_aqi} zero-readings replaced with rolling median.")

    # ── Step 4: Resample to uniform 5-min grid ───────────
    df = df.set_index('timestamp')
    df_rs = df[['tempC', 'humidity', 'mqRaw', 'aqi']].resample('5min').median()
    logs.append(f"✅ Resample: {len(df_rs)} uniform 5-minute buckets created.")

    # ── Step 5: Interpolate gaps ≤ 30 min (6 steps) ───────────────────────────
    gap_mask = df_rs.isnull().any(axis=1).sum()
    df_rs = df_rs.interpolate(method='time', limit=6)
    df_rs = df_rs.dropna()
    logs.append(f"✅ Interpolation: {gap_mask} missing buckets filled/cleaned.")
    
    return df_rs, logs

# --- UI Layout ---
st.title("📊 Sensor Data Cleaning Pipeline")
st.markdown("Upload your raw `sensor_log.csv` to apply sorting, capping, and resampling.")

# Sidebar for Upload
with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader(
        "Upload sensor_log.csv", 
        type=["csv"], 
        key="sensor_data_uploader" # Unique key to prevent DuplicateID error
    )

if uploaded_file is not None:
    # Read Data
    df_input = pd.read_csv(uploaded_file)
    
    st.subheader("Processing Logs")
    
    # Run Processing
    with st.spinner("Processing sensor data..."):
        df_final, processing_logs = process_data(df_input)
    
    # Display Logs in a neat success box
    for log in processing_logs:
        st.write(log)

    st.divider()

    # Layout for Results
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Cleaned Data (Preview)")
        st.dataframe(df_final.head(10), use_container_width=True)
        
        # Download Button
        csv = df_final.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Processed CSV",
            data=csv,
            file_name="cleaned_sensor_data.csv",
            mime="text/csv",
        )

    with col2:
        st.subheader("Statistics")
        st.write(df_final.describe().round(3))

else:
    st.info("Please upload a CSV file in the sidebar to begin.")
    st.warning("Note: Your CSV must contain columns: `timestamp`, `tempC`, `humidity`, `mqRaw`, and `aqi`.")

# Example of correct alignment
for i in range(100):
    # Do some work here
    time.sleep(0.1)  # All lines inside the 'for' must line up

# ---------- Arduino Code Download (Sidebar) ----------
arduino_code = """
#include <DHT.h>

#define DHT_PIN     2
#define DHT_TYPE    DHT22
#define MQ_PIN      A0
#define INTERVAL_MS 5000   // send every 5 seconds

DHT dht(DHT_PIN, DHT_TYPE);
unsigned long lastSend = 0;

void setup() {
  Serial.begin(9600);
  dht.begin();
}

void loop() {
  if (millis() - lastSend < INTERVAL_MS) return;
  lastSend = millis();

  float tempC    = dht.readTemperature();
  float humidity = dht.readHumidity();
  int   mqRaw    = analogRead(MQ_PIN);

  // Simple AQI proxy — calibrate to your sensor
  float aqi = (mqRaw - 100) * 0.12;
  if (aqi < 0) aqi = 0;

  if (isnan(tempC) || isnan(humidity)) return;  // skip bad reads

  Serial.print('{');
  Serial.print("\\"tempC\\":"); Serial.print(tempC, 1); Serial.print(',');
  Serial.print("\\"humidity\\":"); Serial.print(humidity, 1); Serial.print(',');
  Serial.print("\\"mqRaw\\":"); Serial.print(mqRaw); Serial.print(',');
  Serial.print("\\"aqi\\":"); Serial.print(aqi, 1);
  Serial.println('}');
}
"""

with st.sidebar:
    st.divider()
    st.subheader("🔧 Sensor Setup")

    st.write(
        "Download the **Arduino sketch** to program your sensor node "
        "(DHT22 + MQ gas sensor)."
    )

    st.download_button(
        label="⬇️ Download Arduino Code (.ino)",
        data=arduino_code,
        file_name="dasmarinas_sensor_node.ino",
        mime="text/plain",
    )

    with st.expander("👁 View Arduino Source Code"):
        st.code(arduino_code, language="cpp")
