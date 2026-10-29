import numpy as np
rng = np.random.default_rng(0)
n, d, dk = 4, 8, 8
X = rng.normal(size=(n, d)); Wq, Wk, Wv = (rng.normal(size=(d, dk)) / np.sqrt(d) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
mask = np.triu(np.ones((n, n)), 1).astype(bool)                     # causal: no peeking at future tokens
S = Q @ K.T / np.sqrt(dk); S[mask] = -np.inf
A = softmax(S); out = A @ V
print("attention weights (rows sum to 1):\n", A.round(3)); print("output shape:", out.shape)
for dim in [16, 256, 4096]:                                         # why divide by sqrt(d)
    q, k = rng.normal(size=(10000, dim)), rng.normal(size=(10000, dim))
    print(f"d={dim:5d}  std(q·k)={np.std((q*k).sum(1)):7.2f}  std(q·k/√d)={np.std((q*k).sum(1)/np.sqrt(dim)):.2f}")
