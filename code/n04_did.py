import numpy as np, pandas as pd
import statsmodels.formula.api as smf
rng = np.random.default_rng(3)
units, periods = 200, 10
df = pd.DataFrame([(u, p) for u in range(units) for p in range(periods)], columns=["unit", "t"])
df["treated"] = (df.unit < 100).astype(int)
df["post"] = (df.t >= 6).astype(int)
ufx = rng.normal(0, 2, units)
df["y"] = ufx[df.unit] + 0.3 * df.t + 1.2 * df.treated * df.post + rng.normal(0, 1, len(df))
m = smf.ols("y ~ treated:post + C(unit) + C(t)", df).fit(cov_type="cluster", cov_kwds={"groups": df.unit})
print(f"DiD estimate = {m.params['treated:post']:.3f}  (cluster-robust se {m.bse['treated:post']:.3f}; truth 1.2)")
# Pre-trend (event-study) check: leads should be ~0
for k in range(3, 6):
    df[f"lead{k}"] = ((df.t == k) & (df.treated == 1)).astype(int)
m2 = smf.ols("y ~ treated:post + lead3 + lead4 + lead5 + C(unit) + C(t)", df).fit(cov_type="cluster", cov_kwds={"groups": df.unit})
print("leads:", {k: round(m2.params[k], 3) for k in ["lead3", "lead4", "lead5"]})
