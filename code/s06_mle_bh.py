import numpy as np
from scipy import stats, optimize
from statsmodels.stats.multitest import multipletests
x = np.random.default_rng(0).normal(3, 1.5, 500)
nll = lambda th: -stats.norm(th[0], np.exp(th[1])).logpdf(x).sum()     # optimise log σ (unconstrained)
res = optimize.minimize(nll, [0, 0])
print(f"numeric MLE μ={res.x[0]:.4f} σ={np.exp(res.x[1]):.4f} | closed form μ={x.mean():.4f} σ={x.std(ddof=0):.4f}")
rng = np.random.default_rng(1)
p = np.r_[rng.uniform(0, 1, 90), rng.uniform(0, 0.001, 10)]            # 90 nulls, 10 real effects
for m in ["bonferroni", "holm", "fdr_bh"]:
    rej = multipletests(p, alpha=0.05, method=m)[0]
    print(f"{m:10s}: discoveries={rej.sum():2d}  true={rej[90:].sum()}  false={rej[:90].sum()}")
print("uncorrected:", (p < .05).sum(), "discoveries, false =", (p[:90] < .05).sum())
