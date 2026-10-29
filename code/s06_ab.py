import numpy as np
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
n = NormalIndPower().solve_power(proportion_effectsize(0.11, 0.10), alpha=0.05, power=0.8)
print(f"needed per arm for 10% -> 11%: {int(np.ceil(n)):,}  (rule of thumb 16·p(1−p)/δ² = {16*.1*.9/.01**2:,.0f})")
conv, nobs = np.array([1_580, 1_450]), np.array([14_500, 14_500])        # B, A
z, p = proportions_ztest(conv, nobs)
lo, hi = confint_proportions_2indep(conv[0], nobs[0], conv[1], nobs[1])
print(f"rates B={conv[0]/nobs[0]:.4f} A={conv[1]/nobs[1]:.4f}  z={z:.2f} p={p:.4f}  diff 95% CI [{lo:.4f}, {hi:.4f}]")
from scipy.stats import chisquare
print("SRM check p =", round(chisquare([14_500, 14_210]).pvalue, 4), "(< 0.001 would mean broken randomisation)")
rng = np.random.default_rng(0)                                            # CUPED variance reduction
pre = rng.normal(50, 10, 10_000); post = pre * 0.8 + rng.normal(0, 5, 10_000)
theta = np.cov(pre, post)[0, 1] / pre.var(); cuped = post - theta * (pre - pre.mean())
print(f"CUPED: variance {post.var():.1f} -> {cuped.var():.1f} ({1-cuped.var()/post.var():.0%} reduction)")
