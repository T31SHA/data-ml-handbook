import numpy as np
prev, sens, fpr = 0.01, 0.99, 0.05
post = sens * prev / (sens * prev + fpr * (1 - prev))
print(f"P(disease | +) = {post:.3f}")                      # base-rate fallacy
P = np.array([[.7, .3, 0], [.3, .3, .4], [.2, .3, .5]])   # Sunny, Cloudy, Rainy
vals, vecs = np.linalg.eig(P.T)
pi = np.real(vecs[:, np.isclose(vals, 1)][:, 0]); pi /= pi.sum()
print("stationary π:", pi.round(4), " check πP = π:", np.allclose(pi @ P, pi))
print("after 50 steps from 'Sunny':", np.linalg.matrix_power(P, 50)[0].round(4))
rng = np.random.default_rng(0); u = rng.random((1_000_000, 2))
print("Monte Carlo π ≈", 4 * np.mean((u**2).sum(1) < 1), "(error ~ 1/√N)")
