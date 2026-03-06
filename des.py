
import numpy as np

def holt_forecast(series, alpha=0.3, beta=0.15, steps=48):
    """Holt DES (linear trend). Returns dict with arrays and rmse."""
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n == 0:
        return {'forecast': np.array([]), 'lower': np.array([]), 'upper': np.array([]), 'rmse': float('nan'), 'fitted': np.array([])}
    l = np.zeros(n); b = np.zeros(n)
    l[0] = y[0]
    b[0] = y[1] - y[0] if n > 1 else 0.0
    for t in range(1, n):
        l[t] = alpha * y[t] + (1 - alpha) * (l[t-1] + b[t-1])
        b[t] = beta  * (l[t] - l[t-1]) + (1 - beta) * b[t-1]
    forecasts = np.array([l[-1] + (h+1)*b[-1] for h in range(steps)])
    fitted = l + b
    if n > 1:
        residuals = y[1:] - fitted[:-1]
        rmse = float(np.sqrt(np.mean(residuals**2))) if residuals.size>0 else float('nan')
    else:
        rmse = float('nan')
    z = 1.645
    h = np.arange(1, steps+1)
    lower = forecasts - z * rmse * np.sqrt(h)
    upper = forecasts + z * rmse * np.sqrt(h)
    return {'forecast': forecasts, 'lower': lower, 'upper': upper, 'rmse': rmse, 'fitted': fitted}

# Grid search for alpha/beta

def tune_holt(series, alpha_grid=None, beta_grid=None, steps=48):
    y = np.asarray(series, dtype=float)
    if alpha_grid is None:
        alpha_grid = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    if beta_grid is None:
        beta_grid  = [0.05,0.1,0.15,0.2,0.25,0.3,0.4]
    best = {'rmse': float('inf'), 'alpha': None, 'beta': None, 'model': None}
    for a in alpha_grid:
        for b in beta_grid:
            res = holt_forecast(y, alpha=a, beta=b, steps=steps)
            rmse = res['rmse'] if np.isfinite(res['rmse']) else float('inf')
            if rmse < best['rmse']:
                best = {'rmse': rmse, 'alpha': a, 'beta': b, 'model': res}
    return best
