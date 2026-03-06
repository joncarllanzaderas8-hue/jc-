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

from preprocessing import preprocess_site, SITE_NAME  # retains your A/B/C defaults
from des import holt_forecast, tune_holt
from aqi import categorize_aqi  # EPA AQI categorizer (index -> (label, color))


# ---------- Page config ----------
st.set_page_config(page_title="Barangay Microclimate Forecast", layout="wide")
st.title("Barangay Microclimate Monitoring — 4‑Hour Forecasts")
st.caption("Holt's Double Exponential Smoothing (DES) — 5‑minute resolution, 90% confidence bands")

def calculate_heat_index(temp_c, humidity):
    T = (temp_c * 9/5) + 32
    RH = humidity
    hi = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    if hi > 80:
        hi = -42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH \
             - 6.83783e-3*T*T - 5.481717e-2*RH*RH + 1.22874e-3*T*T*RH \
             + 8.5282e-4*T*RH*RH - 1.99e-6*T*T*RH*RH
    return (hi - 32) * 5/9

def get_pagasa_hi_category(hi_c):
    if hi_c < 27: return "Not Hazardous", "#00e400"
    if hi_c <= 32: return "Caution", "#ffff00"
    if hi_c <= 41: return "Extreme Caution", "#ff7e00"
    if hi_c <= 51: return "Danger", "#ff0000"
    return "Extreme Danger", "#7e0023"
    
# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload sensor_log.csv", type=["csv"])
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
        ["EPA AQI (uses AQI value)", "DENR PM2.5 (uses PM2.5 concentration)"],
        index=0,
        help="EPA uses the AQI value (0–500); DENR uses PM2.5 μg/m³ (DAO 2020‑14).",
    )

    tab_choice = st.radio("View", ["Single site", "Compare sites", "City map"], index=0)
    show_residuals = st.checkbox("Show residuals panel (deep‑dive)", value=True)


# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes | None, fallback_path: str) -> pd.DataFrame:
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        if not os.path.exists(fallback_path):
            st.error(f"CSV not found at {fallback_path}. Upload a file or place it under data/ as sensor_log.csv.")
            st.stop()
        df = pd.read_csv(fallback_path)

    if "timestamp" not in df.columns:
        st.error("Missing 'timestamp' column in CSV.")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


raw = load_data(uploaded.getvalue() if uploaded else None, default_path)


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
    "tempC":    {"label": "Temperature", "unit": "°C",      "color": "#ff6b6b", "clip": (None, None)},
    "humidity": {"label": "Humidity",    "unit": "%",       "color": "#4ecdc4", "clip": (0, 100)},
    "heat_index": {"label": "Heat Index", "unit": "°C",     "color": "#ff8c00", "clip": (None, None)}, # <--- Add this
    "mqRaw":    {"label": "MQ Gas Raw",  "unit": "raw",     "color": "#ffd93d", "clip": (None, None)},
    "aqi":      {"label": "AQI",         "unit": "",        "color": "#6bcb77", "clip": (0, None)},
}
if "pm25" in raw.columns:
    signals["pm25"] = {"label": "PM2.5", "unit": "µg/m³", "color": "#a175ff", "clip": (0, None)}


# ---------- Category utilities ----------
def process_site(df: pd.DataFrame, site_code: str, steps: int,
                 alpha: float | None, beta: float | None, auto_tune: bool):
    # ... existing code ...
    proc = preprocess_site(site_df)

    # --- ADD THIS LINE HERE ---
    if "tempC" in proc.columns and "humidity" in proc.columns:
        proc["heat_index"] = proc.apply(lambda r: calculate_heat_index(r["tempC"], r["humidity"]), axis=1)
    # ---------------------------

    if proc.empty:
        return None
        
def calculate_heat_index(temp_c, humidity):
    T = (temp_c * 9/5) + 32
    RH = humidity
    hi = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    if hi > 80:
        hi = -42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH \
             - 6.83783e-3*T*T - 5.481717e-2*RH*RH + 1.22874e-3*T*T*RH \
             + 8.5282e-4*T*RH*RH - 1.99e-6*T*T*RH*RH
    return (hi - 32) * 5/9

def get_pagasa_hi_category(hi_c):
    if hi_c < 27: return "Not Hazardous", "#00e400"
    if hi_c <= 32: return "Caution", "#ffff00"
    if hi_c <= 41: return "Extreme Caution", "#ff7e00"
    if hi_c <= 51: return "Danger", "#ff0000"
    return "Extreme Danger", "#7e0023"

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

    # ... rest of function ...

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

    proc = preprocess_site(site_df)
    if proc.empty:
        return None

    last_ts = proc.index[-1]
    future_idx = pd.date_range(last_ts + pd.Timedelta("5min"), periods=steps, freq="5min")

    results = {}
    chosen = {}

    for col, meta in signals.items():
        series = proc[col].values if col in proc.columns else None
        if series is None or len(series) == 0:
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
            if col not in proc.columns or col not in results_site:
                st.warning(f"No data for {meta['label']} at {site_choice}.")
                continue

            st.subheader(f"{site_choice} — {meta['label']}")
            hist = proc.iloc[-12 * history_hours:]  # 12 points per hour at 5-min

            # Plot (history + forecast + CI)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist[col],
                    mode="lines",
                    name="Observed",
                    line=dict(color=meta["color"], width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future_idx,
                    y=results_site[col]["forecast"],
                    mode="lines",
                    name="Forecast",
                    line=dict(color=meta["color"], width=2, dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future_idx,
                    y=results_site[col]["upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future_idx,
                    y=results_site[col]["lower"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(160,160,160,0.2)",
                    line=dict(width=0),
                    name="90% CI",
                )
            )
            fig.add_vline(x=last_ts, line_width=1, line_dash="dot", line_color="white")
            fig.update_layout(height=420, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # KPIs
            last_val = hist[col].iloc[-1]
            fc = results_site[col]["forecast"]
            fc1 = fc[11] if len(fc) >= 12 else np.nan
            fc2 = fc[23] if len(fc) >= 24 else np.nan
            fc4 = fc[-1] if len(fc) > 0 else np.nan
            rmse = results_site[col]["rmse"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last observed", f"{last_val:.2f} {meta['unit']}")
            c2.metric("+1 hour", f"{fc1:.2f} {meta['unit']}" if np.isfinite(fc1) else "—")
            c3.metric("+2 hours", f"{fc2:.2f} {meta['unit']}" if np.isfinite(fc2) else "—")
            c4.metric("+4 hours", f"{fc4:.2f} {meta['unit']}" if np.isfinite(fc4) else "—")
            st.caption(f"In‑sample RMSE: {rmse:.3f} {meta['unit']}")

            # Residuals
            if show_residuals:
                st.markdown("**Residuals (Observed − Fitted One‑Step Ahead)**")
                fitted = results_site[col]["fitted"]
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

    # ---------- Category labels & first crossing (based on selected scale) ----------
    st.markdown("---")
    st.subheader("Categories in the forecast (based on selected scale)")

    fc_vals, fc_index, source_name = pick_category_series(bundle, cat_scale)
    if source_name == "none" or len(fc_vals) == 0:
        st.info("No forecast series available for the selected scale. (Tip: EPA uses AQI; DENR uses PM2.5.)")
    else:
        # Default threshold depends on scale: EPA USG=100; DENR USG~35
        default_thr = 100 if cat_scale.startswith("EPA") else 35
        thr = st.number_input(
            f"Compute first crossing time for threshold {'AQI' if source_name=='aqi' else 'PM2.5'} >",
            min_value=0.0, max_value=500.0, value=float(default_thr), step=1.0
        )
        idx = first_crossing_index(fc_vals, thr)
        if idx is None:
            st.info("No crossing in the next 4h.")
        else:
            t_cross = fc_index[idx]
            st.success(
                f"First crossing at **{t_cross:%Y-%m-%d %H:%M}** "
                f"({('AQI' if source_name=='aqi' else 'PM2.5')} ~ {fc_vals[idx]:.1f})"
            )

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
            cat_list = [get_pagasa_hi_category(v)[0] for v in res["forecast"]]
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
                    if "pm25" in proc.columns:
                        current_val = float(proc["pm25"].iloc[-1])
                        cat, color = categorize_pm25_denr(current_val)
                        st.markdown(
                            f"Current PM2.5: **{current_val:.1f} µg/m³** — "
                            f"<span style='color:{color}'>{cat}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("PM2.5 not available; using AQI categories.")
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

import os, json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_autorefresh import st_autorefresh
from io import BytesIO


# -----------------------------
# Page & Auto-refresh
# -----------------------------
st.set_page_config(page_title="Dasmariñas Environmental Risk Monitor", layout="wide")
count = st_autorefresh(interval=5000, limit=100, key="autorefresh")

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
# -----------------------------
# Data Loading & Cleaning
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

def run_qc_only(df_in: pd.DataFrame):
    if df_in.empty: return df_in, pd.DataFrame()
    df = df_in.copy()
    # Basic Thresholds
    df['qc_issue'] = (df['tempC'] < -20) | (df['tempC'] > 60) | df['tempC'].isna()
    # Simple summary
    summary = df.groupby('Location').size().reset_index(name='rows')
    return df, summary

df_hist, sensors_summary = run_qc_only(df_hist)

def run_qc_and_trust(df_in: pd.DataFrame):
    df = df_in.copy()
    now = pd.to_datetime(datetime.utcnow())
    # thresholds
    T_MIN, T_MAX = -20.0, 60.0
    HUM_MIN, HUM_MAX = 0.0, 100.0
    AQI_MIN, AQI_MAX = 0.0, 500.0

    # flags for issues
    df['qc_missing'] = df[['tempC','humidity','aqi','timestamp']].isna().any(axis=1)
    df['qc_out_of_range'] = False
    if 'tempC' in df.columns:
        df.loc[df['tempC'].notna() & ((df['tempC'] < T_MIN) | (df['tempC'] > T_MAX)), 'qc_out_of_range'] = True
    if 'humidity' in df.columns:
        df.loc[df['humidity'].notna() & ((df['humidity'] < HUM_MIN) | (df['humidity'] > HUM_MAX)), 'qc_out_of_range'] = True
    if 'aqi' in df.columns:
        df.loc[df['aqi'].notna() & ((df['aqi'] < AQI_MIN) | (df['aqi'] > AQI_MAX)), 'qc_out_of_range'] = True

    # large jumps per location (delta threshold per minute)
    df['qc_jump'] = False
    if 'timestamp' in df.columns and 'tempC' in df.columns and 'Location' in df.columns:
        df = df.sort_values(['Location','timestamp'])
        grp = df.groupby('Location')
        for name, g in grp:
            if len(g) < 2:
                continue
            dt = g['timestamp'].diff().dt.total_seconds().fillna(0) / 60.0
            dtemp = g['tempC'].diff().abs().fillna(0)
            # flag if temp change > 8°C within 1 minute, scaled
            mask = (dt <= 5) & (dtemp > 8)
            df.loc[mask.index, 'qc_jump'] = mask

    df['qc_issue'] = df['qc_missing'] | df['qc_out_of_range'] | df['qc_jump']

    # per-site summary
    sites = []
    for tag, group in df.groupby('Location') if 'Location' in df.columns else [(None, df)]:
        total = len(group)
        missing_pct = float(group['qc_missing'].sum() / total) if total else 0.0
        oor_pct = float(group['qc_out_of_range'].sum() / total) if total else 0.0
        recent = group.loc[group['timestamp'] >= (pd.to_datetime(now) - pd.Timedelta(days=1))] if 'timestamp' in group.columns else group
        recent_anom_pct = float(recent['qc_issue'].sum() / len(recent)) if len(recent) else 0.0
        last_ts = group['timestamp'].max() if 'timestamp' in group.columns else pd.NaT
        freshness_min = float((pd.to_datetime(now) - pd.to_datetime(last_ts)).total_seconds()/60.0) if pd.notna(last_ts) else float('inf')
        # trust score: start 1.0, subtract penalties
        p_missing = 0.4 * missing_pct
        p_oor = 0.3 * oor_pct
        p_fresh = 0.2 * min(1.0, freshness_min / (24*60))
        p_recent = 0.1 * recent_anom_pct
        trust = max(0.0, 1.0 - (p_missing + p_oor + p_fresh + p_recent))
        sites.append({'site': tag if tag is not None else 'global', 'total_rows': total,
                      'missing_pct': missing_pct, 'out_of_range_pct': oor_pct,
                      'recent_anom_pct': recent_anom_pct, 'last_seen': last_ts,
                      'freshness_min': freshness_min, 'trust_score': trust})

    df_sites = pd.DataFrame(sites)
    if 'site' in df_sites.columns:
        df_sites = df_sites.sort_values('site')
    return df, df_sites

# Sensor Health UI: show per-site trust scores and simple chart
with st.expander("Sensor Health (QC & Trust Scores)", expanded=False):
    st.write("Per-site sensor health summary (trust score, freshness, missing%):")
    try:
        if sensors_summary is not None and not sensors_summary.empty:
            df_display = sensors_summary.copy()
            df_display['trust_pct'] = (df_display['trust_score'] * 100).round(1)
            st.dataframe(df_display[['site','trust_pct','missing_pct','out_of_range_pct','recent_anom_pct','last_seen']], use_container_width=True)
            try:
                chart_df = df_display.set_index('site')['trust_pct']
                st.bar_chart(chart_df)
            except Exception:
                pass
        else:
            st.info("No sensor summary available.")
    except Exception:
        st.warning("Unable to render sensor health summary.")

# -----------------------------
# Heat Index (NOAA/NWS Rothfusz + Steadman fallback)
# -----------------------------
# Ref: NOAA/NWS heat index equation
# https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
def heat_index_celsius(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    T = temp_c * 9.0/5.0 + 32.0  # to °F
    R = rh
    HI = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R
          - 6.83783e-3*T*T - 5.481717e-2*R*R
          + 1.22874e-3*T*T*R + 8.5282e-4*T*R*R - 1.99e-6*T*T*R*R)
    adj = np.zeros_like(HI, dtype=float)
    mask_low = (R < 13) & (T >= 80) & (T <= 112)
    adj[mask_low] = -((13 - R[mask_low])/4.0) * np.sqrt((17 - np.abs(T[mask_low]-95))/17.0)
    mask_high = (R > 85) & (T >= 80) & (T <= 87)
    adj[mask_high] = ((R[mask_high]-85)/10.0) * ((87 - T[mask_high])/5.0)
    HI = np.where(T < 80, T + 0.33*R - 0.70, HI + adj)
    return (HI - 32.0) * 5.0/9.0  # back to °C

# PAGASA Heat Index Categories (°C)
# Not Hazardous (<27), Caution (27–32), Extreme Caution (33–41), Danger (42–51), Extreme Danger (≥52)
def pagasa_hi_category(hi_c: float) -> str:
    if not np.isfinite(hi_c):
        return '—'
    if hi_c < 27: return "Not Hazardous"
    if hi_c <= 32: return "Caution (27–32°C)"
    if hi_c <= 41: return "Extreme Caution (33–41°C)"
    if hi_c <= 51: return "Danger (42–51°C)"
    return "Extreme Danger (≥52°C)"

# US EPA AQI Categories (index)
def epa_aqi_category(aqi_value: float) -> str:
    if aqi_value is None or (isinstance(aqi_value, float) and np.isnan(aqi_value)):
        return "—"
    aqi = float(aqi_value)
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

AQI_COLORS = {
    "Good": "#009966",
    "Moderate": "#FFDE33",
    "Unhealthy for Sensitive Groups": "#FF9933",
    "Unhealthy": "#CC0033",
    "Very Unhealthy": "#660099",
    "Hazardous": "#7E0023",
    "—": "#999999"
}

RISK_COLORS = {
    "Normal": "#2B8A3E",
    "Moderate": "#F59F00",
    "High": "#D9480F",
    "Unknown": "#868E96"
}

HI_BAND_SHAPES = [
    (None, 27, 'rgba(0,0,0,0)'),
    (27, 32, 'rgba(255,235,132,0.25)'),
    (32, 41, 'rgba(255,165,0,0.20)'),
    (41, 51, 'rgba(255,69,58,0.20)'),
    (51, None, 'rgba(156,39,176,0.15)')
]

# Pre-compute Heat Index for history
if set(['tempC','humidity']).issubset(df_hist.columns):
    df_hist['heat_index_C'] = heat_index_celsius(df_hist['tempC'].values, df_hist['humidity'].values)
    df_hist['HI_Category'] = df_hist['heat_index_C'].apply(pagasa_hi_category)
else:
    df_hist['heat_index_C'] = np.nan
    df_hist['HI_Category'] = "—"

if 'aqi' in df_hist.columns:
    df_hist['AQI_Category'] = df_hist['aqi'].apply(epa_aqi_category)
else:
    df_hist['AQI_Category'] = "—"

# -----------------------------
# Model
# -----------------------------
MODEL_PATH = 'dt_model.joblib'
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model file not found or cannot be loaded. Please ensure '{MODEL_PATH}' is in the directory. Details: {e}")
    st.stop()


label_map = {0: "Normal", 1: "Moderate", 2: "High"}

# -----------------------------
# Header & current reading
# -----------------------------
st.title("🌿 Dasmariñas Environmental Risk Monitor")
st.subheader(f"Locale: {location_mode} · Live Monitoring Dashboard")

loc_tag = location_mode[0]

if data_source == "Latest Reading":
    recent_loc = df_hist[df_hist['Location'] == loc_tag].sort_values('timestamp')
    if recent_loc.empty:
        st.warning("No data available for this location.")
        st.stop()
    current = recent_loc.dropna(subset=['tempC','humidity']).tail(1).iloc[0]
    temp = float(current['tempC'])
    hum = float(current['humidity'])
    aqi = float(current['aqi']) if 'aqi' in current and pd.notna(current['aqi']) \
          else float(df_hist['aqi'].median(skipna=True)) if 'aqi' in df_hist.columns else np.nan
    aqi_imputed = pd.isna(current['aqi']) if 'aqi' in current else True
else:
    temp = st.sidebar.slider("Temperature (°C)", 20.0, 45.0, 30.0)
    hum = st.sidebar.slider("Humidity (%)", 30.0, 100.0, 75.0)
    aqi = st.sidebar.slider("AQI (index)", 0.0, 300.0, 25.0)
    aqi_imputed = False

current_hi = float(heat_index_celsius(np.array([temp]), np.array([hum]))[0])
hi_cat = pagasa_hi_category(current_hi)
aqi_cat = epa_aqi_category(aqi)

input_df = pd.DataFrame([[temp, hum, aqi]], columns=['tempC','humidity','aqi'])
pred_idx = model.predict(input_df)[0]
risk_level = label_map.get(int(pred_idx), "Unknown")

# -----------------------------
# Forecast utilities
# -----------------------------
def median_minutes_delta(ts: pd.Series, default=5.0) -> float:
    ts_sorted = ts.sort_values().dropna()
    if len(ts_sorted) < 3:
        return default
    deltas = ts_sorted.diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return default
    return float(np.median(deltas))

def forecast_next(series: pd.Series, steps: int, method: str = "Rolling Mean", window: int = 12) -> np.ndarray:
    s = series.dropna().values
    if len(s) == 0:
        return np.array([np.nan]*steps)
    if method.startswith('Naive'):
        return np.repeat(s[-1], steps)
    if method.startswith('Rolling'):
        w = max(3, min(window, len(s)))
        avg = float(np.mean(s[-w:]))
        return np.repeat(avg, steps)
    # Linear Trend
    w = max(4, min(window, len(s)))
    y = s[-w:]
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    x_future = np.arange(len(y), len(y)+steps)
    yhat = coeffs[0]*x_future + coeffs[1]
    return yhat

def backtest_mae_rmse(series: pd.Series, steps_ahead: int, method: str, window: int = 12, max_points: int = 600):
    s = series.dropna()
    if len(s) < window + steps_ahead + 5:
        return np.nan, np.nan, 0
    s = s.tail(max_points)
    y_true, y_pred = [], []
    for end in range(window, len(s)-steps_ahead):
        hist = s.iloc[:end]
        pred = forecast_next(hist, steps=steps_ahead, method=method, window=window)[-1]
        y_true.append(float(s.iloc[end+steps_ahead-1]))
        y_pred.append(float(pred))
    if len(y_true) == 0:
        return np.nan, np.nan, 0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    return float(mae), rmse, len(y_true)

# Determine forecast steps from desired minutes
loc_df = df_hist[df_hist['Location'] == loc_tag].sort_values('timestamp')
median_min = median_minutes_delta(loc_df['timestamp']) if not loc_df.empty else 5.0
steps_horizon = int(max(1, round(forecast_minutes / max(1e-6, median_min))))
steps_24h = int(max(1, round(1440 / max(1e-6, median_min))))

# Forecast for HI & AQI
hi_series = loc_df['heat_index_C'] if 'heat_index_C' in loc_df else pd.Series(dtype=float)
aqi_series = loc_df['aqi'] if 'aqi' in loc_df else pd.Series(dtype=float)
temp_series = loc_df['tempC'] if 'tempC' in loc_df else pd.Series(dtype=float)
hum_series = loc_df['humidity'] if 'humidity' in loc_df else pd.Series(dtype=float)

hi_fore = forecast_next(hi_series, steps=steps_horizon, method=forecast_method, window=12)
aqi_fore = forecast_next(aqi_series, steps=steps_horizon, method=forecast_method, window=12)

hi_fore_val = float(hi_fore[-1]) if len(hi_fore) else float('nan')
aqi_fore_val = float(aqi_fore[-1]) if len(aqi_fore) else float('nan')
hi_fore_cat = pagasa_hi_category(hi_fore_val) if np.isfinite(hi_fore_val) else '—'
aqi_fore_cat = epa_aqi_category(aqi_fore_val) if np.isfinite(aqi_fore_val) else '—'

# Backtest metrics for selected horizon
hi_mae, hi_rmse, hi_n = backtest_mae_rmse(hi_series, steps_horizon, forecast_method, window=12)
aqi_mae, aqi_rmse, aqi_n = backtest_mae_rmse(aqi_series, steps_horizon, forecast_method, window=12)

# Build future timestamps for plotting
if not loc_df.empty and len(loc_df['timestamp'].dropna())>0:
    last_ts = loc_df['timestamp'].dropna().iloc[-1]
    # Ensure last_ts is datetime
    if isinstance(last_ts, str):
        last_ts = pd.to_datetime(last_ts, errors='coerce')
    dt_minutes = max(1, int(round(median_min)))
    future_index = pd.date_range(start=last_ts + timedelta(minutes=dt_minutes),
                                 periods=steps_horizon, freq=f"{dt_minutes}min")
    forecast_df = pd.DataFrame({
        'timestamp': future_index,
        'HI_forecast_C': hi_fore,
        'AQI_forecast': aqi_fore,
    })
else:
    forecast_df = pd.DataFrame()

# -----------------------------
# Metrics row
# -----------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Temperature", f"{temp:.1f} °C")
m2.metric("Humidity", f"{hum:.0f} %")
m3.metric("AQI", f"{aqi:.1f}", help=f"Now: {aqi_cat} | {forecast_minutes}m→ {aqi_fore_val:.1f} ({aqi_fore_cat})")
m4.metric("Heat Index", f"{current_hi:.1f} °C", help=f"PAGASA: {hi_cat} | {forecast_minutes}m→ {hi_fore_val:.1f} °C ({hi_fore_cat})")
with m5:
    chip_color = RISK_COLORS.get(risk_level, "#868E96")
    st.markdown(
        f'''<div style="background-color:{chip_color}; padding:10px; border-radius:10px; text-align:center;">
            <h3 style="color:white; margin:0;">{risk_level.upper()} RISK</h3>
        </div>''',
        unsafe_allow_html=True,
    )

if data_source == "Latest Reading" and aqi_imputed:
    st.caption("Note: AQI missing in latest record; used median AQI for stability.")

st.divider()

# -----------------------------
# Predictive Decision Support + Diagnostics
# -----------------------------
advice_map = {
    "Normal": "✅ Environment is stable. No immediate health risks detected.",
    "Moderate": "⚠️ Caution: Sensitive groups should limit prolonged outdoor exposure.",
    "High": "🚨 Alert: Poor environmental conditions. Avoid outdoor activities and close windows."
}

c1, c2 = st.columns([2, 1])
with c2:
    st.markdown("### 🤖 Predictive Decision Support")
    st.info(advice_map.get(risk_level, "Monitoring..."))
    st.markdown("*Model Diagnostic:*")
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        st.write(f"- Feature importances: tempC={fi[0]:.2f}, humidity={fi[1]:.2f}, aqi={fi[2]:.2f}")
    else:
        st.write("- Feature importances not available for this model.")
    st.markdown("*Forecast Diagnostic (backtest)*")
    st.write(f"- HI {forecast_minutes}m MAE={hi_mae:.2f}°C · RMSE={hi_rmse:.2f}°C · n={hi_n}")
    st.write(f"- AQI {forecast_minutes}m MAE={aqi_mae:.2f} · RMSE={aqi_rmse:.2f} · n={aqi_n}")
    # show last retrain/test eval if available
    if 'retrain_eval' in st.session_state:
        re = st.session_state['retrain_eval']
        try:
            st.markdown("*Retrain/Test Eval (last run)*")
            st.write(f"- Accuracy: {re['acc']*100:.2f}% (n_test={re.get('n_test','?')})")
            cm_arr = np.array(re['cm'])
            cm_df_main = pd.DataFrame(cm_arr, index=[f"act_{i}" for i in range(cm_arr.shape[0])], columns=[f"pred_{i}" for i in range(cm_arr.shape[1])])
            st.dataframe(cm_df_main)
            st.write("Classification Report:")
            st.json(re['report'])
            # Temporal results if available
            if re.get('temporal'):
                tr = re['temporal']
                try:
                    st.markdown("*Temporal Holdout Eval (last run)*")
                    st.write(f"- Temporal Accuracy: {tr['acc']*100:.2f}% (n_test={tr.get('n_test','?')})")
                    if tr.get('avg_conf') is not None:
                        st.write(f"- Avg predicted class confidence: {tr['avg_conf']*100:.2f}%")
                    cm_t = np.array(tr['cm'])
                    cm_df_t = pd.DataFrame(cm_t, index=[f"act_{i}" for i in range(cm_t.shape[0])], columns=[f"pred_{i}" for i in range(cm_t.shape[1])])
                    st.dataframe(cm_df_t)
                    st.write("Temporal Classification Report:")
                    st.json(tr['report'])
                except Exception:
                    pass
        except Exception:
            pass
    if st.button("Refresh Real-Time Feed"):
        st.rerun()

    # -----------------------------
    with st.expander("⚙️ Retrain & Test (debug)"):
        st.write("Run a proper train/test split and evaluate a Decision Tree on held-out data.")
        auto_label = st.checkbox("Auto-generate labels from tempC`/humidity`/`aqi` (creates temp_label/hum_label/aq_label/Category_Label)", value=True)
        temporal_split = st.checkbox("Use temporal split (train on older rows, test on newer)", value=False)
        if temporal_split:
            holdout_mode = st.radio("Temporal holdout mode", ["last_percent","last_days"], index=0, horizontal=True)
            if holdout_mode == 'last_percent':
                holdout_percent = st.slider("Test holdout (percent of latest rows)", min_value=1, max_value=50, value=10)
            else:
                holdout_days = st.number_input("Test holdout (last N days)", min_value=1, max_value=365, value=7)
        # External weather merge
        ext_weather_file = st.file_uploader("Optional external-weather CSV (timestamp + rain_forecast)", type=["csv"], accept_multiple_files=False)
        if ext_weather_file is not None:
            merge_mode = st.radio("Merge method for external weather", ["exact","nearest"], index=1, horizontal=True)
            tol_minutes = st.number_input("Nearest merge tolerance (minutes)", min_value=1, max_value=1440, value=30)
        cv_enable = st.checkbox("Enable k-fold cross-validation (k-fold)", value=True)
        cv_folds = st.slider("k (folds)", min_value=3, max_value=10, value=5)
        auto_save = st.checkbox("Auto-save trained model when accuracy >=", value=False)
        save_threshold = st.number_input("Auto-save threshold (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
        save_filename = st.text_input("Save filename", value="dt_model_retrained.joblib")
        run_diag = st.button("Run train/test evaluation")
        if run_diag:
            # Expected columns (user-provided guidance): temp_label, hum_label, aq_label, Category_Label
            feature_cols = ['temp_label', 'hum_label', 'aq_label']
            target_col = 'Category_Label'

            working_df = df_hist.copy()
            # Auto-generate labels if requested
            if auto_label:
                required_source = ['tempC', 'humidity', 'aqi']
                missing_src = [c for c in required_source if c not in working_df.columns]
                if missing_src:
                    st.error(f"Cannot auto-generate labels: missing source columns {missing_src} (need tempC, humidity, aqi).")
                    st.stop()

                def make_temp_label(t):
                    try:
                        t = float(t)
                    except Exception:
                        return np.nan
                    if t < 27:
                        return 0
                    if t <= 32:
                        return 1
                    return 2

                def make_hum_label(h):
                    try:
                        h = float(h)
                    except Exception:
                        return np.nan
                    if h < 40:
                        return 0
                    if h <= 70:
                        return 1
                    return 2

                def make_aqi_label(a):
                    try:
                        a = float(a)
                    except Exception:
                        return np.nan
                    if a <= 50:
                        return 0
                    if a <= 100:
                        return 1
                    return 2

                working_df['temp_label'] = working_df['tempC'].apply(make_temp_label)
                working_df['hum_label'] = working_df['humidity'].apply(make_hum_label)
                working_df['aq_label'] = working_df['aqi'].apply(make_aqi_label)
                # Compose Category_Label as rounded mean of the three labels (deterministic rule)
                working_df['Category_Label'] = working_df[['temp_label', 'hum_label', 'aq_label']].mean(axis=1).round().astype('Int64')

            missing = [c for c in feature_cols + [target_col] if c not in working_df.columns]
            if missing:
                st.error(f"Missing required columns for retrain/test: {missing}.\n\nExpected: {feature_cols + [target_col]}")
            else:
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.tree import DecisionTreeClassifier
                    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

                    raw = working_df[feature_cols + [target_col]].dropna()
                    if len(raw) < 10:
                        st.warning(f"Not enough rows for reliable split (found {len(raw)}). Need >= 10 rows.")
                    else:
                        X = raw[feature_cols]
                        y = raw[target_col].astype(int)
                        strat = y if len(np.unique(y))>1 and len(y)>50 else None
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
                        clf = DecisionTreeClassifier(random_state=42)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)

                        acc = accuracy_score(y_test, y_pred)
                        cm = confusion_matrix(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True)

                        # optionally run cross-validation on full raw set
                        cv_results = None
                        if cv_enable:
                            try:
                                from sklearn.model_selection import cross_val_score, StratifiedKFold
                                if len(np.unique(y)) > 1:
                                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                                    scores = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=cv, scoring='accuracy')
                                else:
                                    scores = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=cv_folds, scoring='accuracy')
                                cv_results = {'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'folds': [float(s) for s in scores]}
                                st.write(f"Cross-val ({cv_folds} folds) accuracy: {cv_results['mean']*100:.2f}% ± {cv_results['std']*100:.2f}%")
                            except Exception as e:
                                st.warning(f"Cross-validation failed: {e}")

                        # If external weather provided, merge into working_df
                        if ext_weather_file is not None:
                            try:
                                ext_df = pd.read_csv(ext_weather_file)
                                # try common timestamp column names
                                ts_cols = [c for c in ext_df.columns if 'time' in c.lower() or 'timestamp' in c.lower() or c.lower()=='ts']
                                if not ts_cols:
                                    st.warning("External CSV lacks a recognizable timestamp column. Expected column like 'timestamp' or 'time'. Skipping merge.")
                                else:
                                    ext_ts_col = ts_cols[0]
                                    ext_df[ext_ts_col] = pd.to_datetime(ext_df[ext_ts_col], errors='coerce')
                                    if 'timestamp' in working_df.columns:
                                        working_df['timestamp'] = pd.to_datetime(working_df['timestamp'], errors='coerce')
                                    if merge_mode == 'exact':
                                        working_df = pd.merge(working_df, ext_df, left_on='timestamp', right_on=ext_ts_col, how='left')
                                    else:
                                        # nearest merge using merge_asof
                                        working_df = working_df.sort_values('timestamp')
                                        ext_df = ext_df.sort_values(ext_ts_col)
                                        working_df = pd.merge_asof(working_df, ext_df, left_on='timestamp', right_on=ext_ts_col, direction='nearest', tolerance=pd.Timedelta(minutes=int(tol_minutes)))
                            except Exception as e:
                                st.warning(f"Failed to read/merge external weather CSV: {e}")

                        # Optionally perform a temporal split evaluation (time-series holdout)
                        temporal_results = None
                        if temporal_split:
                            if 'timestamp' not in working_df.columns or working_df['timestamp'].isna().all():
                                st.warning("Temporal split requested but no valid timestamp column available in data.")
                            else:
                                try:
                                    raw_ts = working_df.dropna(subset=feature_cols + [target_col, 'timestamp']).sort_values('timestamp')
                                    n_ts = len(raw_ts)
                                    if n_ts < 10:
                                        st.warning(f"Not enough timestamped rows for temporal split (found {n_ts}).")
                                    else:
                                        if 'holdout_mode' in locals() and holdout_mode == 'last_percent':
                                            test_size = max(1, int(round(n_ts * float(holdout_percent) / 100.0)))
                                            split_idx = n_ts - test_size
                                            train_ts = raw_ts.iloc[:split_idx]
                                            test_ts = raw_ts.iloc[split_idx:]
                                        else:
                                            # last N days
                                            latest = raw_ts['timestamp'].max()
                                            cutoff = latest - timedelta(days=int(holdout_days))
                                            train_ts = raw_ts[raw_ts['timestamp'] < cutoff]
                                            test_ts = raw_ts[raw_ts['timestamp'] >= cutoff]
                                        if len(test_ts) == 0:
                                            st.warning("Temporal split resulted in empty test set; adjust holdout settings or provide more data.")
                                        else:
                                            X_tr_ts = train_ts[feature_cols]
                                            y_tr_ts = train_ts[target_col].astype(int)
                                            X_te_ts = test_ts[feature_cols]
                                            y_te_ts = test_ts[target_col].astype(int)
                                            clf_ts = DecisionTreeClassifier(random_state=42)
                                            clf_ts.fit(X_tr_ts, y_tr_ts)
                                            y_pred_ts = clf_ts.predict(X_te_ts)
                                            acc_ts = accuracy_score(y_te_ts, y_pred_ts)
                                            cm_ts = confusion_matrix(y_te_ts, y_pred_ts)
                                            report_ts = classification_report(y_te_ts, y_pred_ts, output_dict=True)
                                            # average confidence if available
                                            avg_conf = None
                                            if hasattr(clf_ts, 'predict_proba'):
                                                try:
                                                    probs = clf_ts.predict_proba(X_te_ts)
                                                    avg_conf = float(np.mean(np.max(probs, axis=1)))
                                                except Exception:
                                                    avg_conf = None
                                            temporal_results = {
                                                'acc': float(acc_ts), 'cm': cm_ts.tolist(), 'report': report_ts,
                                                'n_test': int(len(y_te_ts)), 'avg_conf': avg_conf
                                            }
                                            st.write(f"Temporal-split accuracy: {acc_ts*100:.2f}% (n_test={len(y_te_ts)})")
                                            st.write("Temporal Confusion Matrix:")
                                            st.dataframe(pd.DataFrame(cm_ts, index=[f"act_{i}" for i in range(cm_ts.shape[0])], columns=[f"pred_{i}" for i in range(cm_ts.shape[1])]))
                                            st.write("Temporal Classification Report:")
                                            st.json(report_ts)
                                            if avg_conf is not None:
                                                st.write(f"Average predicted class confidence (temporal test): {avg_conf*100:.2f}%")
                                except Exception as e:
                                    st.warning(f"Temporal split evaluation failed: {e}")

                        # persist metrics to session state for display elsewhere
                        st.session_state['retrain_eval'] = {
                            'acc': float(acc),
                            'cm': cm.tolist(),
                            'report': report,
                            'n_test': int(len(y_test)),
                            'cv': cv_results,
                            'temporal': temporal_results
                        }

                        st.success(f"Accuracy on test set: {acc*100:.2f}% (n_test={len(y_test)})")
                        st.write("Confusion Matrix (rows=actual, cols=predicted):")
                        cm_df = pd.DataFrame(cm, index=[f"act_{i}" for i in range(cm.shape[0])], columns=[f"pred_{i}" for i in range(cm.shape[1])])
                        st.dataframe(cm_df)
                        st.write("Classification Report:")
                        st.json(report)
                        st.write("Trained Decision Tree feature importances:")
                        fi = getattr(clf, 'feature_importances_', None)
                        if fi is not None:
                            fi_map = {c: float(v) for c, v in zip(feature_cols, fi)}
                            st.write(fi_map)
                        else:
                            st.write("Feature importances not available.")
                        # Auto-save if requested and accuracy threshold met
                        if auto_save:
                            try:
                                if (acc * 100.0) >= float(save_threshold):
                                    joblib.dump(clf, save_filename)
                                    st.success(f"Trained model saved to {save_filename} (accuracy {acc*100:.2f}%)")
                                    st.session_state['last_saved_model'] = save_filename
                                else:
                                    st.info(f"Model not saved: accuracy {acc*100:.2f}% < threshold {save_threshold}%")
                            except Exception as e:
                                st.error(f"Failed to save model: {e}")
                        # Manual save option
                        if st.button("Save trained model now"):
                            try:
                                joblib.dump(clf, save_filename)
                                st.success(f"Trained model saved to {save_filename}")
                                st.session_state['last_saved_model'] = save_filename
                            except Exception as e:
                                st.error(f"Failed to save model: {e}")
                except Exception as e:
                    st.error(f"Failed to run retrain/test: {e}")

with c1:
    st.markdown("### 🧭 Health Guidance Context")
    st.write("- *Heat Index (PAGASA):* Caution 27–32°C · Extreme Caution 33–41°C · Danger 42–51°C · Extreme Danger ≥52°C.")
    st.write("- *US EPA AQI:* Good (0–50) · Moderate (51–100) · Unhealthy for SG (101–150) · Unhealthy (151–200) · Very Unhealthy (201–300) · Hazardous (301+).")
