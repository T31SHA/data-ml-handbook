import torch, torch.nn as nn
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base; self.base.requires_grad_(False)                   # frozen W0
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))            # zero init => starts identical
        self.scale, self.drop = alpha / r, nn.Dropout(dropout)
    def forward(self, x): return self.base(x) + self.drop(x) @ self.A.T @ self.B.T * self.scale
layer = LoRALinear(nn.Linear(1024, 1024))
x = torch.randn(2, 1024)
print("identical at init:", torch.allclose(layer(x), layer.base(x)))
tr = sum(p.numel() for p in layer.parameters() if p.requires_grad); tot = sum(p.numel() for p in layer.parameters())
print(f"trainable {tr:,} of {tot:,} ({tr/tot:.2%})")
