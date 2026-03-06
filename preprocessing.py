
import numpy as np
import pandas as pd

SITE_NAME = {'A': 'Green space', 'B': 'Residential', 'C': 'Commercial'}

def preprocess_site(df_site: pd.DataFrame, resample_freq='5min', interp_limit=6):
    """Sort → cap humidity → AQI zero-fix (rolling median, w=5) → 5‑min median resample → interpolate ≤30 min → drop NaNs."""
    df_site = df_site.copy()
    df_site = df_site.sort_values('timestamp')
    if 'humidity' in df_site:
        df_site['humidity'] = df_site['humidity'].clip(upper=100)
    if 'aqi' in df_site:
        rolling_med = df_site['aqi'].rolling(5, center=True, min_periods=1).median()
        df_site['aqi'] = np.where(df_site['aqi'] == 0, rolling_med, df_site['aqi'])
    df_rs = (df_site.set_index('timestamp')[['tempC','humidity','mqRaw','aqi']]
                     .resample(resample_freq).median())
    df_rs = df_rs.interpolate(method='time', limit=interp_limit).dropna()
    return df_rs
