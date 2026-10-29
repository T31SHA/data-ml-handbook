import numpy as np
rng = np.random.default_rng(1)
X = rng.normal(size=(50, 3)); y = (rng.random(50) < 0.5).astype(float); w = rng.normal(size=3)
sig = lambda z: 1 / (1 + np.exp(-z))
def loss(w):                                    # mean binary cross-entropy
    p = sig(X @ w); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
analytic = X.T @ (sig(X @ w) - y) / len(y)      # derived:  ∇L = Xᵀ(σ(Xw) − y) / n
eps = 1e-6
numeric = np.array([(loss(w + eps * e) - loss(w - eps * e)) / (2 * eps) for e in np.eye(3)])
print("analytic:", analytic.round(6)); print("numeric :", numeric.round(6))
print("max relative error:", float(np.max(np.abs(analytic - numeric) / (np.abs(numeric) + 1e-12))))
