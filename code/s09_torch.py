import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
torch.manual_seed(0)
X = torch.randn(4000, 20); w = torch.randn(20, 3); y = (X @ w + 0.5 * torch.randn(4000, 3)).argmax(1)   # 3 classes
train_ds, val_ds = random_split(TensorDataset(X, y), [3200, 800], generator=torch.Generator().manual_seed(0))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True); val_dl = DataLoader(val_ds, batch_size=256)
class MLP(nn.Module):
    def __init__(self, d_in=20, h=64, n_cls=3, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p), nn.Linear(h, n_cls))
    def forward(self, x): return self.net(x)                    # return LOGITS
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
loss_fn = nn.CrossEntropyLoss()
print(f"params={sum(p.numel() for p in model.parameters())}  initial loss≈ln(3)={torch.log(torch.tensor(3.)):.3f}")
best, bad = float("inf"), 0
for epoch in range(30):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval(); tot, correct, n = 0.0, 0, 0
    with torch.inference_mode():
        for xb, yb in val_dl:
            out = model(xb.to(device)); tot += loss_fn(out, yb.to(device)).item() * len(yb)
            correct += (out.argmax(1).cpu() == yb).sum().item(); n += len(yb)
    val = tot / n
    if epoch % 5 == 0: print(f"epoch {epoch:2d} train_loss={loss.item():.3f} val_loss={val:.3f} val_acc={correct/n:.3f}")
    if val < best - 1e-4: best, bad = val, 0; torch.save(model.state_dict(), "data/best.pt")
    else:
        bad += 1
        if bad == 5: print(f"early stop at epoch {epoch}"); break
model.load_state_dict(torch.load("data/best.pt")); print(f"best val_loss={best:.3f}")
