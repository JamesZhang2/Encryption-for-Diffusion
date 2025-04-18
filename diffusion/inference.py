import torch
import matplotlib.pyplot as plt
import tqdm
from unet import UNet
from torch.nn import Embedding

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000

# Load trained model + embedding
unet = UNet(3, 64, 32).to(device)
emb = Embedding(100, 256).to(device)
unet.load_state_dict(torch.load("guided_unet_200.pt", map_location=device))
emb.load_state_dict(torch.load("guided_embedding_200.pt", map_location=device))
unet.eval(); emb.eval()

# Precompute alphas
alphas = torch.linspace(0.9999, 0.98, T, dtype=torch.float64, device=device)
alpha_bars = torch.cumprod(alphas, dim=0)

def run_inference(unet, emb, class_name, class_list, s, num_row=10, num_col=10):
    unet.eval()

    alpha_bars_prev = torch.cat((torch.ones(1).to(device), alpha_bars[:-1]))
    sigma_t_squared = (1.0 - alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    sigma_t = torch.sqrt(sigma_t_squared)

    x = torch.randn(num_row*num_col, 3, 32, 32).to(device)

    with torch.no_grad():
        class_id_list = [i for i, v in enumerate(class_list) if v == class_name]
        if len(class_id_list) == 0:
            raise Exception("class name doesn't exist")
        y = class_id_list[0]
        y_batch = torch.tensor(y, device=device).repeat(num_row*num_col)
        y_batch = torch.cat((y_batch, y_batch), dim=0)
        y_emb_batch = emb(y_batch)
        mask = torch.cat((
            torch.ones(num_row*num_col, device=device),
            torch.zeros(num_row*num_col, device=device)))
        y_emb_batch = y_emb_batch * mask[:, None]

        for t in tqdm.tqdm(reversed(range(T)), total=T):
            t_batch = torch.tensor(t, device=device).repeat(num_row*num_col)
            t_batch = torch.cat((t_batch, t_batch), dim=0)
            x_batch = torch.cat((x, x), dim=0)
            eps_batch = unet(x_batch, t_batch, y_emb_batch)
            eps_cond, eps_uncond = torch.split(eps_batch, len(eps_batch)//2, dim=0)
            eps = (1.0 + s) * eps_cond - s * eps_uncond
            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = (1.0 / torch.sqrt(alphas[t])).float() * (
                x - ((1.0 - alphas[t]) / torch.sqrt(1.0 - alpha_bars[t])).float() * eps
            ) + sigma_t[t].float() * z

    x = x.permute(0, 2, 3, 1)
    x = torch.clamp(x, min=0.0, max=1.0)
    fig, axes = plt.subplots(num_row, num_col, figsize=(5,5))
    for i in range(num_row*num_col):
        image = x[i].cpu().numpy()
        row, col = i // num_col, i % num_col
        ax = axes[row, col]
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(image)

    plt.tight_layout()
    plt.show()