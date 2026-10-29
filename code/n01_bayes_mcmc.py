import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
# Conjugate Beta-Binomial: prior Beta(2,2), observe 37 conversions out of 120
a0, b0, k, n = 2, 2, 37, 120
post = stats.beta(a0 + k, b0 + n - k)
print(f"posterior mean={post.mean():.4f}  95% CrI=({post.ppf(.025):.4f}, {post.ppf(.975):.4f})")
# Random-walk Metropolis for the same posterior on the logit scale
def logpost(z):
    p = 1 / (1 + np.exp(-z))
    return (a0 + k - 1) * np.log(p) + (b0 + n - k - 1) * np.log(1 - p) + np.log(p * (1 - p))  # Jacobian
z, chain, acc = 0.0, [], 0
for t in range(20000):
    prop = z + rng.normal(0, 0.4)
    if np.log(rng.uniform()) < logpost(prop) - logpost(z):
        z, acc = prop, acc + 1
    chain.append(z)
p = 1 / (1 + np.exp(-np.array(chain[2000:])))
print(f"MCMC mean={p.mean():.4f}  95% CrI=({np.quantile(p,.025):.4f}, {np.quantile(p,.975):.4f})  accept={acc/20000:.2f}")
# Effective sample size via initial positive autocorrelation sum
x = p - p.mean(); ac = np.correlate(x, x, "full")[len(x)-1:] / (x @ x)
s = 0
for r in ac[1:]:
    if r < 0: break
    s += r
print(f"ESS ~ {len(p)/(1+2*s):.0f} of {len(p)} draws")
