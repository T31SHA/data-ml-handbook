import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(0)
a, b = rng.normal(100, 15, 40), rng.normal(108, 25, 45)
t = stats.ttest_ind(a, b, equal_var=False)                  # Welch by default in practice
d = (b.mean() - a.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
print(f"Welch t={t.statistic:.2f} p={t.pvalue:.4f}  Cohen's d={d:.2f}")
print("Mann-Whitney p =", round(stats.mannwhitneyu(a, b).pvalue, 4))
tab = pd.crosstab(rng.choice(["A", "B"], 500), rng.choice(["buy", "no"], 500, p=[.3, .7]))
chi2, p, dof, exp = stats.chi2_contingency(tab); print(f"chi2={chi2:.2f} dof={dof} p={p:.3f}")
g1, g2, g3 = rng.normal(5, 1, 30), rng.normal(5.2, 1, 30), rng.normal(6, 1, 30)
print("ANOVA F=%.2f p=%.5f" % stats.f_oneway(g1, g2, g3))
x = rng.lognormal(3, .8, 200); boot = [np.median(rng.choice(x, len(x))) for _ in range(5000)]
print("median %.2f, 95%% bootstrap CI [%.2f, %.2f]" % (np.median(x), *np.percentile(boot, [2.5, 97.5])))
