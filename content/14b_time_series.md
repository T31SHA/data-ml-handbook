# Time Series Forecasting in Depth

tagline: Decomposition, stationarity, ARIMA and exponential smoothing, honest backtesting and forecast evaluation.

## Structure of a series

$$y_t=T_t+S_t+R_t\quad\text{(additive)}\qquad y_t=T_t\cdot S_t\cdot R_t\quad\text{(multiplicative)}$$

Use the multiplicative form (or log-transform) when seasonal swings grow with the level. STL decomposition is a robust default for inspection.

## Stationarity and differencing

A series is weakly stationary if its mean and autocovariances do not depend on $t$. Differencing $\nabla y_t=y_t-y_{t-1}$ removes trends; seasonal differencing $\nabla_m y_t=y_t-y_{t-m}$ removes stable seasonality. ADF tests the null of a unit root; KPSS tests the null of stationarity. Use them together.

## ARIMA and exponential smoothing

ARIMA$(p,d,q)$ models the $d$-times differenced series as

$$\phi(B)\nabla^d y_t=\theta(B)\varepsilon_t,\quad \phi(B)=1-\sum_{i=1}^p\phi_iB^i,\quad\theta(B)=1+\sum_{j=1}^q\theta_jB^j,$$

with $B$ the backshift operator. Seasonal ARIMA adds the same polynomials in $B^m$. Read the ACF/PACF of the differenced series: a sharp PACF cut-off at lag $p$ suggests AR$(p)$, a sharp ACF cut-off at lag $q$ suggests MA$(q)$.

Holt-Winters (additive) updates level, trend and season recursively:

$$\ell_t=\alpha(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1}),\quad b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)b_{t-1},\quad s_t=\gamma(y_t-\ell_t)+(1-\gamma)s_{t-m}.$$

## Backtesting

![](figures/v2_backtest.svg)

Never use random K-fold on time series. Use rolling-origin evaluation: fit on data up to an origin, forecast the horizon, move the origin forward. Report the metric per fold, not only the mean.

Scale-free error, comparable across series:

$$\text{MASE}=\frac{\frac1h\sum_{t}|y_t-\hat y_t|}{\frac{1}{n-m}\sum_{t=m+1}^{n}|y_t-y_{t-m}|}.$$

MASE below 1 beats the in-sample seasonal naive forecast.

@@code code/n05_ts_backtest.py@@
@@out code/out/n05_ts_backtest.txt@@

!!! pitfall "Pitfall: leakage through features"
    Rolling means, target encodings and "future" exogenous variables (actual weather, not the forecast available at the time) leak information. Build every feature as-of the forecast origin.

## Global ML models

For many related series (SKUs, stores), one gradient-boosted model over lag, rolling and calendar features usually beats per-series ARIMA. Use direct multi-horizon models or recursive forecasting with care: recursive errors compound.

!!! tip "Pro tip"
    Forecast quantiles, not just means. Train with the pinball loss $L_\tau(y,\hat q)=\max(\tau(y-\hat q),(\tau-1)(y-\hat q))$ and check calibration: about 90% of actuals should fall below the 0.9 quantile.

## Cheat card: Forecasting

- Plot, decompose (STL), check stationarity with ADF and KPSS.
- Baselines first: naive, seasonal naive, ETS. Many fancy models lose to them.
- Rolling-origin backtests; MASE or sMAPE per fold.
- All features as-of the forecast origin.
- Global GBM for many series; quantile forecasts for decisions.
