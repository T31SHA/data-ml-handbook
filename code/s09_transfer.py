import torch, torch.nn as nn
from torchvision import models
m = models.resnet18(weights=None)                 # use weights="IMAGENET1K_V1" with internet access
for p in m.parameters(): p.requires_grad = False   # freeze backbone
m.fc = nn.Linear(m.fc.in_features, 5)              # new head (trainable)
train = sum(p.numel() for p in m.parameters() if p.requires_grad); total = sum(p.numel() for p in m.parameters())
print(f"trainable {train:,} / {total:,} params ({train/total:.2%})")
for p in m.layer4.parameters(): p.requires_grad = True       # stage 2: unfreeze last block
opt = torch.optim.AdamW([{"params": m.layer4.parameters(), "lr": 1e-4}, {"params": m.fc.parameters(), "lr": 1e-3}])
x = torch.randn(2, 3, 224, 224); print("logits shape:", tuple(m(x).shape))
