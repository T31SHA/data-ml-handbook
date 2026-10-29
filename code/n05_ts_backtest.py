import numpy as np, pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(4)
t = np.arange(240)
y = pd.Series(50 + 0.15 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 240),
              index=pd.period_range("2006-01", periods=240, freq="M"))
def mase(a, f, train, m=12):
    return np.mean(np.abs(a - f)) / np.mean(np.abs(train[m:] - train[:-m]))
res = {"seasonal_naive": [], "holt_winters": [], "sarima": []}
for origin in range(180, 228, 12):                  # rolling-origin, 12-step horizon
    tr, te = y[:origin].values, y[origin:origin + 12].values
    res["seasonal_naive"].append(mase(te, tr[-12:], tr))
    hw = ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=12).fit()
    res["holt_winters"].append(mase(te, hw.forecast(12), tr))
    sa = SARIMAX(tr, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    res["sarima"].append(mase(te, sa.forecast(12), tr))
for k, v in res.items():
    print(f"{k:15s} MASE per fold={np.round(v,2)}  mean={np.mean(v):.3f}")
