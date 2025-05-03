# train_cfg.py

import os
import random
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
train_ds   = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
loader     = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)

# — Model + class‑embedding (must match training dimensions) —
unet_base_channel = 64
num_classes       = 100

unet = UNet(
    source_channel=3,
    unet_base_channel=unet_base_channel,
    num_groups=32
).to(device)

emb = torch.nn.Embedding(num_classes, unet_base_channel * 4).to(device)

# 2) OPTIMIZER / SCHEDULER / AMP INITIALIZATION
p_uncond   = 0.2
num_epochs = 200
T          = 1000
save_every = 10
log_file   = "train_loss.log"

opt       = Adam(list(emb.parameters()) + list(unet.parameters()), lr=2e-4, eps=1e-08)
scheduler = CosineAnnealingLR(opt, T_max=num_epochs * len(loader))
scaler    = GradScaler(device=str(device))

# 3) PRECOMPUTE NOISE SCHEDULE
alphas     = torch.linspace(0.9999, 0.98, T, dtype=torch.float64, device=device)
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_ab    = torch.sqrt(alpha_bars).float()
sqrt_1m_ab = torch.sqrt(1 - alpha_bars).float()

# 4) CLEAN OLD LOG
if os.path.exists(log_file):
    os.remove(log_file)

# 5) TRAINING LOOP (DDPM + CFG)
for epoch in range(1, num_epochs + 1):
    unet.train()
    losses = []

    for batch_idx, (x0, y) in enumerate(loader, start=1):
        x0, y = x0.to(device), y.to(device)
        B     = x0.shape[0]

        # 1) sample random t and noise ε 
        t   = torch.randint(0, T, (B,), device=device)
        eps = torch.randn_like(x0)

        # 2) forward diffusion: x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε 
        a_t   = sqrt_ab[t][:, None, None, None]
        one_t = sqrt_1m_ab[t][:, None, None, None]
        xt    = a_t * x0 + one_t * eps

        # 3) get class‑embedding + dropout for CFG
        y_emb = emb(y)  # (B, emb_dim)
        mask  = (torch.rand(B, device=device) >= p_uncond).float().unsqueeze(1)
        y_emb = y_emb * mask

        # 4) predict noise and step optimizer (with AMP)
        opt.zero_grad()
        with autocast():
            pred = unet(xt, t, y_emb)
            loss = F.mse_loss(pred, eps, reduction="mean")
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        losses.append(loss.item())
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(loader)} | Loss {loss.item():.4f}", end="\r")

    avg = sum(losses) / len(losses)
    print(f"\nEpoch {epoch} done — Avg Loss: {avg:.4f}")

    # append to log
    with open(log_file, "a") as f:
        for l in losses:
            f.write(f"{l:.6f}\n")

    # checkpoint every N epochs
    if epoch % save_every == 0 or epoch == num_epochs:
        torch.save(unet.state_dict(),      f"guided_unet_{epoch}.pt")
        torch.save(emb.state_dict(),       f"guided_embedding_{epoch}.pt")

print("Training complete")