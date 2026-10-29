import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(8, 4)); y = rng.integers(0, 3, 8); Y = np.eye(3)[y]
W1, b1, W2, b2 = rng.normal(size=(4, 5)) * .5, np.zeros(5), rng.normal(size=(5, 3)) * .5, np.zeros(3)
def forward(W1, b1, W2, b2):
    z1 = X @ W1 + b1; h = np.maximum(0, z1); z2 = h @ W2 + b2
    p = np.exp(z2 - z2.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return z1, h, p, -np.mean(np.log(p[np.arange(len(y)), y]))
z1, h, p, L = forward(W1, b1, W2, b2)
dz2 = (p - Y) / len(y)                 # softmax + cross-entropy gradient
dW2, db2 = h.T @ dz2, dz2.sum(0)
dh = dz2 @ W2.T; dz1 = dh * (z1 > 0)    # ReLU gate
dW1, db1 = X.T @ dz1, dz1.sum(0)
eps, i, j = 1e-6, 2, 1                  # gradient check one weight
Wp, Wm = W1.copy(), W1.copy(); Wp[i, j] += eps; Wm[i, j] -= eps
num = (forward(Wp, b1, W2, b2)[3] - forward(Wm, b1, W2, b2)[3]) / (2 * eps)
print(f"loss={L:.4f}  dW1[{i},{j}] analytic={dW1[i, j]:.8f} numeric={num:.8f}")
import torch
t = [torch.tensor(a, requires_grad=True) for a in (W1, b1, W2, b2)]
out = torch.relu(torch.tensor(X) @ t[0] + t[1]) @ t[2] + t[3]
torch.nn.functional.cross_entropy(out, torch.tensor(y)).backward()
print("matches PyTorch autograd:", np.allclose(t[0].grad.numpy(), dW1), np.allclose(t[2].grad.numpy(), dW2))
