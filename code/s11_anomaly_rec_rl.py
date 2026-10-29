# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
rng = np.random.default_rng(0)
X = np.r_[rng.normal(0, 1, (2000, 5)), rng.normal(0, 1, (20, 5)) + rng.choice([-4, 4], (20, 5))]; lab = np.r_[np.zeros(2000), np.ones(20)]
iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(X)
print(f"IsolationForest PR-AUC={average_precision_score(lab, -iso.score_samples(X)):.3f} (baseline {lab.mean():.3f})")
# matrix factorisation by ALS on a toy ratings matrix
R = np.array([[5, 4, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [0, 1, 5, 4], [1, 0, 4, 0]], float); M = R > 0
k, lam = 2, 0.1; U, V = rng.normal(0, .1, (5, k)), rng.normal(0, .1, (4, k))
for _ in range(30):
    for u in range(5): U[u] = np.linalg.solve(V[M[u]].T @ V[M[u]] + lam * np.eye(k), V[M[u]].T @ R[u, M[u]])
    for i in range(4): V[i] = np.linalg.solve(U[M[:, i]].T @ U[M[:, i]] + lam * np.eye(k), U[M[:, i]].T @ R[M[:, i], i])
print("ALS reconstruction (0 = unseen in input):\n", (U @ V.T).round(1))
# tabular Q-learning on a 1-D corridor: reward at the right end
nS, Q, alpha, gamma, eps = 6, np.zeros((6, 2)), 0.5, 0.9, 0.2
for ep in range(300):
    s = 0
    while s != nS - 1:
        a = rng.integers(2) if rng.random() < eps else int(Q[s].argmax())
        s2 = max(0, s - 1) if a == 0 else s + 1; r = 1.0 if s2 == nS - 1 else -0.01
        Q[s, a] += alpha * (r + gamma * Q[s2].max() * (s2 != nS - 1) - Q[s, a]); s = s2
print("greedy policy (1=right):", Q.argmax(1)[:-1], " V(s0)=", Q[0].max().round(3))
