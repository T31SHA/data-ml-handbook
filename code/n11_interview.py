import numpy as np
# P1: reservoir sampling (uniform k from a stream of unknown length)
def reservoir(stream, k, rng):
    R = []
    for i, x in enumerate(stream):
        if i < k: R.append(x)
        else:
            j = rng.integers(i + 1)
            if j < k: R[j] = x
    return R
rng = np.random.default_rng(8)
counts = np.zeros(20)
for _ in range(20000):
    for x in reservoir(range(20), 5, rng): counts[x] += 1
print("reservoir inclusion prob (expect 0.25):", np.round(counts / 20000, 3)[:8], "...")
# P2: streaming mean/variance (Welford) is numerically stable
def welford(xs):
    n = m = M2 = 0.0
    for x in xs:
        n += 1; d = x - m; m += d / n; M2 += d * (x - m)
    return m, M2 / (n - 1)
x = 1e9 + rng.normal(0, 1, 100000)
print("welford var=%.4f  numpy var=%.4f  naive E[x^2]-E[x]^2=%.1f" % (welford(x)[1], x.var(ddof=1), (x**2).mean() - x.mean()**2))
# P3: AUC equals P(score_pos > score_neg) (Mann-Whitney)
from sklearn.metrics import roc_auc_score
y = rng.integers(0, 2, 2000); s = y + rng.normal(0, 1.2, 2000)
pos, neg = s[y == 1], s[y == 0]
print(f"AUC sklearn={roc_auc_score(y,s):.4f}  pairwise={np.mean(pos[:,None] > neg[None,:]):.4f}")
