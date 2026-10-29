import numpy as np
def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected); a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
rng = np.random.default_rng(0); train = rng.normal(50, 10, 50_000)
for name, prod in [("same distribution", rng.normal(50, 10, 20_000)), ("mean +2", rng.normal(52, 10, 20_000)),
                   ("mean +5, wider", rng.normal(55, 13, 20_000))]:
    v = psi(train, prod); print(f"{name:18s} PSI={v:.3f} -> {'stable' if v < .1 else 'moderate' if v < .25 else 'SIGNIFICANT'}")
from scipy.stats import ks_2samp
print(f"KS test p (mean +2): {ks_2samp(train, rng.normal(52, 10, 20_000)).pvalue:.1e}  <- tiny p at large n: use an effect size (PSI) too")
