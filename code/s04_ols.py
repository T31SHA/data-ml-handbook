import numpy as np
rng = np.random.default_rng(0)
X = np.c_[np.ones(200), rng.normal(size=(200, 2))]          # bias column + 2 features
w_true = np.array([1.0, 2.0, -3.0]); y = X @ w_true + rng.normal(0, 0.5, 200)
w_normal = np.linalg.solve(X.T @ X, X.T @ y)                  # normal equation (solve, never inv)
w_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)               # QR/SVD based: numerically safer
w = np.zeros(3)                                                # gradient descent on MSE
for _ in range(500): w -= 0.1 * (2 / len(y)) * X.T @ (X @ w - y)
print("normal eq:", w_normal.round(3), "\nlstsq    :", w_lstsq.round(3), "\nGD       :", w.round(3))
print("condition number of XᵀX:", round(np.linalg.cond(X.T @ X), 2))
