# ── Standard library ──────────────────────────────────────────────────────
import threading
import time
import queue
import json
import re
import sys
from datetime import datetime
from collections import deque

# ── Streamlit ─────────────────────────────────────────────────────────────
import streamlit as st

# ── Scientific stack ──────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# ── Signals to track ─────────────────────────────────────────────────────
SIGNALS = {
    'tempC':    {'label': 'Temperature',   'unit': '°C',  'color': '#ff6b6b', 'warn': 35,  'ylim': (24, 40)},
    'humidity': {'label': 'Humidity',      'unit': '%',   'color': '#4ecdc4', 'warn': 90,  'ylim': (55, 105)},
    'mqRaw':    {'label': 'MQ Gas Sensor', 'unit': 'raw', 'color': '#ffd93d', 'warn': 350, 'ylim': (50, 550)},
    'aqi':      {'label': 'AQI',           'unit': '',    'color': '#6bcb77', 'warn': 150, 'ylim': (0, 200)}
}

FORECAST_STEPS = 48
HOLT_ALPHA = 0.4
HOLT_BETA = 0.2
MIN_OBS_FORECAST = 10
LIVE_BUFFER_SIZE = 500

# ── Preprocess history ────────────────────────────────────────────────────
def preprocess_history(df_raw):
    df = df_raw.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df = df.sort_values('timestamp').set_index('timestamp')
    return df


# ── Holt Forecast ─────────────────────────────────────────────────────────
def holt_forecast(series, alpha=HOLT_ALPHA, beta=HOLT_BETA, steps=FORECAST_STEPS):

    level = series[0]
    trend = series[1] - series[0]

    levels = []
    trends = []

    for y in series:
        prev_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

        levels.append(level)
        trends.append(trend)

    forecasts = np.array([level + (i + 1) * trend for i in range(steps)])

    residuals = series - np.array(levels)
    rmse = np.sqrt(np.mean(residuals ** 2))

    conf = 1.64 * rmse
    lower = forecasts - conf
    upper = forecasts + conf

    return forecasts, lower, upper, rmse


# ── Dashboard ─────────────────────────────────────────────────────────────
def render_dashboard(live_buf, new_row=None, alerts=None):

    if len(live_buf) < 2:
        st.write("⏳ Waiting for enough data to plot...")
        return

    df_live = pd.DataFrame(list(live_buf)).set_index('timestamp')
    df_live = df_live[list(SIGNALS.keys())].apply(pd.to_numeric, errors='coerce').dropna()

    if len(df_live) < 2:
        return

    last_ts = df_live.index[-1]

    future_idx = pd.date_range(
        last_ts + pd.Timedelta('5min'),
        periods=FORECAST_STEPS,
        freq='5min'
    )

    fig = plt.figure(figsize=(18, 15), facecolor='#0f1117')

    now_str = datetime.now().strftime('%H:%M:%S')
    obs_str = f"{len(df_live)} obs"
    fc_end = future_idx[-1].strftime('%H:%M')

    title = (
        f"⚡ LIVE  Arduino Sensor Forecast  |  {now_str}  |  "
        f"{obs_str}  |  horizon → {fc_end}"
    )

    fig.suptitle(
        title,
        color='#6bcb77' if not alerts else '#ff6b6b',
        fontsize=13,
        fontweight='bold',
        y=0.995
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.32)

    HIST = min(len(df_live), 72)

    for idx, (col, cfg) in enumerate(SIGNALS.items()):

        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        ser = df_live[col].iloc[-HIST:]

        fc, lo, hi, rmse = holt_forecast(df_live[col].values)

        ax.plot(
            ser.index,
            ser.values,
            color=cfg['color'],
            lw=1.8,
            label='Observed',
            zorder=3
        )

        ax.plot(
            future_idx,
            fc,
            color=cfg['color'],
            lw=2.0,
            ls='--',
            label='Forecast',
            zorder=5
        )

        ax.fill_between(
            future_idx,
            lo,
            hi,
            color=cfg['color'],
            alpha=0.18,
            label='90% CI',
            zorder=2
        )

        ax.axvline(last_ts, color='white', lw=0.7, ls=':', alpha=0.4)

        warn_val = cfg['warn']

        ax.axhline(
            warn_val,
            color='#ff6b6b',
            lw=0.8,
            ls='--',
            alpha=0.5
        )

        latest = df_live[col].iloc[-1]
        warn = latest > warn_val

        badge_col = '#ff6b6b' if warn else cfg['color']

        ax.set_title(
            f"{cfg['label']}  ·  now: {latest:.1f} {cfg['unit']}"
            + ("  ⚠" if warn else ""),
            color=badge_col,
            fontsize=11,
            fontweight='bold',
            pad=7
        )

        ax.annotate(
            f"{fc[-1]:.1f}{cfg['unit']}",
            xy=(future_idx[-1], fc[-1]),
            xytext=(-50, 6),
            textcoords='offset points',
            color=cfg['color'],
            fontsize=9,
            fontweight='bold',
            arrowprops=dict(
                arrowstyle='->',
                color=cfg['color'],
                lw=0.8
            )
        )

        ax.set_ylim(cfg['ylim'])
        ax.set_xlabel('Time', fontsize=9)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

        ax.grid(True)

        rmse_txt = f"RMSE {rmse:.2f}"

        ax.legend(
            fontsize=7,
            framealpha=0.25,
            facecolor='#222',
            edgecolor='#444',
            labelcolor='white',
            loc='upper left',
            title=rmse_txt,
            title_fontsize=7
        )

    st.pyplot(fig)


# ── Load historical data ─────────────────────────────────────────────────
CSV_PATH = "sensor_log.csv"

df_history = pd.read_csv(CSV_PATH)

df_clean = preprocess_history(df_history)

live_buffer = deque(maxlen=LIVE_BUFFER_SIZE)

for ts, row in df_clean.tail(100).iterrows():
    live_buffer.append({
        'timestamp': ts,
        'tempC': row['tempC'],
        'humidity': row['humidity'],
        'mqRaw': row['mqRaw'],
        'aqi': row['aqi']
    })


st.title("Arduino Sensor Forecast Dashboard")

render_dashboard(live_buffer)
