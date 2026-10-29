import numpy as np
from scipy import stats
rng = np.random.default_rng(7)
ref = rng.normal(0, 1, 20000)
def psi(ref, cur, bins=10):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref); b = np.histogram(cur, edges)[0] / len(cur)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return np.sum((b - a) * np.log(b / a))
for name, cur in [("same", rng.normal(0, 1, 5000)), ("shift 0.25sd", rng.normal(.25, 1, 5000)),
                  ("scale x1.5", rng.normal(0, 1.5, 5000)), ("shift 1sd", rng.normal(1, 1, 5000))]:
    ks = stats.ks_2samp(ref, cur)
    print(f"{name:13s} PSI={psi(ref,cur):.3f}  KS={ks.statistic:.3f} p={ks.pvalue:.1e}")
