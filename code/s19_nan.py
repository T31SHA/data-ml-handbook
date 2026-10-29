import torch, torch.nn as nn
torch.manual_seed(0)
p = torch.tensor([1.0, 0.0]); y = torch.tensor([1.0, 1.0])            # a probability of exactly 0 for a positive
print("manual BCE on probabilities:", (-(y * torch.log(p) + (1 - y) * torch.log(1 - p))).mean().item())
logits = torch.tensor([30.0, -30.0])                                     # same predictions as logits
print("BCEWithLogitsLoss (stable): ", round(nn.BCEWithLogitsLoss()(logits, y).item(), 3))
X = torch.randn(512, 10) * 100; yr = X.sum(1, keepdim=True)              # unscaled inputs, large targets
for lr in [1e-1, 1e-6]:
    m = nn.Linear(10, 1); opt = torch.optim.SGD(m.parameters(), lr=lr)
    for step in range(50):
        opt.zero_grad(); loss = ((m(X) - yr) ** 2).mean(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")); opt.step()
        if not torch.isfinite(loss): print(f"lr={lr}: loss became {loss.item()} at step {step} (grad norm before: {gn:.1e})"); break
    else: print(f"lr={lr}: finite after 50 steps, loss={loss.item():.1f}")
torch.autograd.set_detect_anomaly(True)                                   # finds the op that produced NaN
try:
    a = torch.tensor([0.0], requires_grad=True); torch.sqrt(a * 0 - 1).sum().backward()
except RuntimeError as e: print("anomaly mode:", str(e).splitlines()[0][:90])
