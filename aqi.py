
# AQI categorization helper (US EPA-like bands)
# Returns (category, color_hex)

def categorize_aqi(aqi_value: float):
    if aqi_value is None:
        return ("Unknown", "#888888")
    try:
        v = float(aqi_value)
    except Exception:
        return ("Unknown", "#888888")
    if v <= 50:   return ("Good",      "#00e400")
    if v <= 100:  return ("Moderate",  "#ffff00")
    if v <= 150:  return ("USG",       "#ff7e00")  # Unhealthy for Sensitive Groups
    if v <= 200:  return ("Unhealthy", "#ff0000")
    if v <= 300:  return ("Very Unhealthy", "#8f3f97")
    return ("Hazardous", "#7e0023")
