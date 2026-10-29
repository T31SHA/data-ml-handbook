# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
import numpy as np, pandas as pd, lightgbm as lgb
rng = np.random.default_rng(0); idx = pd.date_range("2024-01-01", periods=730, freq="D")
y = pd.Series(100 + 0.05 * np.arange(730) + 15 * np.sin(2 * np.pi * idx.dayofweek / 7) + 10 * np.sin(2 * np.pi * idx.dayofyear / 365) + rng.normal(0, 4, 730), idx)
df = pd.DataFrame({"y": y}); H = 28
for L in [H, H + 7, 364]: df[f"lag_{L}"] = df.y.shift(L)              # lags >= horizon: no leakage
df["roll_mean_28"] = df.y.shift(H).rolling(28).mean(); df["dow"] = idx.dayofweek; df["doy"] = idx.dayofyear
train, test = df.iloc[:-H].dropna(), df.iloc[-H:]
m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, verbose=-1).fit(train.drop(columns="y"), train.y)
pred = m.predict(test.drop(columns="y"))
X_tr, X_te = train.drop(columns="y"), test.drop(columns="y")
m2 = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, verbose=-1).fit(X_tr, train.y - train.roll_mean_28)
pred2 = m2.predict(X_te) + test.roll_mean_28                             # trees can't extrapolate: model the residual
snaive = df.y.shift(H).iloc[-H:].values                                  # naive: value 28 days earlier (same weekday)
mae = lambda a, b: np.mean(np.abs(np.asarray(a) - np.asarray(b)))
scale = mae(train.y.values[7:], train.y.values[:-7])                     # in-sample seasonal-naive MAE (lag 7)
for name, p in [("seasonal naive (t-28)", snaive), ("LightGBM on raw y", pred), ("LightGBM on y - level", pred2)]:
    print(f"{name:24s} MAE={mae(test.y, p):5.2f}  MASE={mae(test.y, p)/scale:.2f}")
