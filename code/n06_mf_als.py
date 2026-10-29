import numpy as np
rng = np.random.default_rng(5)
U, I, k = 300, 200, 8
P0, Q0 = rng.normal(size=(U, k)), rng.normal(size=(I, k))
R = P0 @ Q0.T + rng.normal(0, .5, (U, I))
mask = rng.uniform(size=(U, I)) < 0.3                 # 30% observed
test = mask & (rng.uniform(size=(U, I)) < 0.2); train = mask & ~test
P, Q, lam = rng.normal(0, .1, (U, k)), rng.normal(0, .1, (I, k)), 5.0
for it in range(15):                                  # alternating least squares
    for u in range(U):
        idx = train[u]; Qu = Q[idx]
        P[u] = np.linalg.solve(Qu.T @ Qu + lam * np.eye(k), Qu.T @ R[u, idx])
    for i in range(I):
        idx = train[:, i]; Pi = P[idx]
        Q[i] = np.linalg.solve(Pi.T @ Pi + lam * np.eye(k), Pi.T @ R[idx, i])
    if it % 5 == 4:
        E = (P @ Q.T - R)
        print(f"iter {it+1:2d} train RMSE={np.sqrt((E[train]**2).mean()):.3f}  test RMSE={np.sqrt((E[test]**2).mean()):.3f}")
# ranking metric: recall@10 on held-out items with true score in user's top 20%
S = P @ Q.T; S[train] = -np.inf
rec = []
for u in range(U):
    rel = set(np.where(test[u] & (R[u] > np.quantile(R[u], .8)))[0])
    if rel: rec.append(len(rel & set(np.argsort(-S[u])[:10])) / min(10, len(rel)))
print(f"recall@10 = {np.mean(rec):.3f} over {len(rec)} users")
