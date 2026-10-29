import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
rng = np.random.default_rng(2)
n = 50000
x = rng.normal(size=(n, 3))
ps_true = 1 / (1 + np.exp(-(0.8 * x[:, 0] - 0.5 * x[:, 1])))
t = rng.binomial(1, ps_true)
y = 2.0 * t + 1.5 * x[:, 0] + x[:, 1] + rng.normal(size=n)  # true ATE = 2
print(f"naive difference in means : {y[t==1].mean()-y[t==0].mean():.3f}")
e = LogisticRegression().fit(x, t).predict_proba(x)[:, 1].clip(0.01, 0.99)
ipw = np.mean(t * y / e) - np.mean((1 - t) * y / (1 - e))
print(f"IPW                       : {ipw:.3f}")
m1 = LinearRegression().fit(x[t==1], y[t==1]).predict(x)
m0 = LinearRegression().fit(x[t==0], y[t==0]).predict(x)
psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
print(f"AIPW (doubly robust)      : {psi.mean():.3f} +/- {1.96*psi.std()/np.sqrt(n):.3f}")
