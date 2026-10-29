import numpy as np
from scipy import stats
rng = np.random.default_rng(1)
n = 20000
pre = rng.gamma(2.0, 10.0, 2 * n)                  # pre-period spend (covariate)
grp = np.r_[np.zeros(n), np.ones(n)]
y = 0.8 * pre + rng.normal(0, 8, 2 * n) + 0.4 * grp  # true lift = 0.4
def diff_ci(v):
    a, b = v[grp == 0], v[grp == 1]
    d = b.mean() - a.mean(); se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    return d, se
d, se = diff_ci(y)
theta = np.cov(y, pre)[0, 1] / pre.var(ddof=1)
y_cv = y - theta * (pre - pre.mean())
d2, se2 = diff_ci(y_cv)
print(f"naive : lift={d:.3f} se={se:.3f}  p={2*stats.norm.sf(abs(d/se)):.4f}")
print(f"CUPED : lift={d2:.3f} se={se2:.3f}  p={2*stats.norm.sf(abs(d2/se2)):.4f}")
print(f"variance reduction = {1-(se2/se)**2:.1%}  (theory 1-rho^2 = {np.corrcoef(y,pre)[0,1]**2:.1%})")
# Sample-ratio mismatch check: expected 50/50, observed counts
obs = np.array([50_210, 49_420])
chi2, p = stats.chisquare(obs)
print(f"SRM chi2={chi2:.2f} p={p:.4f} -> {'INVESTIGATE before reading results' if p < 0.001 else 'ok'}")
