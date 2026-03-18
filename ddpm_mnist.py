import argparse
import os
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torchvision
from torchvision import datasets, transforms

from positional_embeddings import PositionalEmbedding

# ---------------------------------------------------------------------------
# 0. UTILITAIRES : Moyenne Mobile Exponentielle (EMA)
# ---------------------------------------------------------------------------
class EMA:
    """Maintient une copie lissée des poids du modèle pour une meilleure génération."""
    def __init__(self, beta=0.9999):
        self.beta = beta

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# ---------------------------------------------------------------------------
# 1. ARCHITECTURE : U-Net
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """Bloc d'auto-attention pour aider le modèle à comprendre la structure globale."""
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([channels])

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H * W).transpose(1, 2)
        x_norm = self.ln(x_reshaped)
        attn_val, _ = self.mha(x_norm, x_norm, x_norm)
        attn_val = attn_val.transpose(1, 2).view(B, C, H, W)
        return x + attn_val

class Block(nn.Module):
    """Un bloc de base avec Convolution, GroupNorm, Dropout et injection du temps."""
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout_rate=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch) # GroupNorm remplace BatchNorm
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        self.act = nn.GELU()
        
    def forward(self, x, t_emb):
        # Première convolution + normalisation
        h = self.act(self.norm1(self.conv1(x)))
        
        # Injection du temps
        time_val = self.time_mlp(t_emb).view(-1, h.shape[1], 1, 1)
        h = h + time_val
        
        # Dropout et seconde convolution
        h = self.dropout(h)
        h = self.act(self.norm2(self.conv2(h)))
        return h

class UNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        time_dim = base_ch * 4
        
        self.time_emb = PositionalEmbedding(base_ch, "sinusoidal")
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.inc = nn.Conv2d(1, base_ch, 3, padding=1)
        
        self.down1 = Block(base_ch, base_ch * 2, time_dim)
        self.pool1 = nn.MaxPool2d(2) 
        
        self.down2 = Block(base_ch * 2, base_ch * 4, time_dim)
        self.pool2 = nn.MaxPool2d(2) 
        
        # Milieu avec attention
        self.mid1 = Block(base_ch * 4, base_ch * 4, time_dim)
        self.attn = SelfAttention(base_ch * 4)
        self.mid2 = Block(base_ch * 4, base_ch * 4, time_dim)
        
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.up_block2 = Block(base_ch * 6, base_ch * 2, time_dim) 
        
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.up_block1 = Block(base_ch * 3, base_ch, time_dim)
        
        self.outc = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_emb(t))
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        p2 = self.pool1(x2)
        
        x3 = self.down2(p2, t_emb)
        p3 = self.pool2(x3)
        
        m = self.mid1(p3, t_emb)
        m = self.attn(m)
        m = self.mid2(m, t_emb)
        
        u2 = self.up2(m)
        u2 = torch.cat([u2, x3], dim=1) 
        u2 = self.up_block2(u2, t_emb)
        
        u1 = self.up1(u2)
        u1 = torch.cat([u1, x2], dim=1) 
        u1 = self.up_block1(u1, t_emb)
        
        return self.outc(u1)

# ---------------------------------------------------------------------------
# 2. SCHEDULER
# ---------------------------------------------------------------------------
class ImageNoiseScheduler():
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        device = x_t.device
        s1 = self.sqrt_inv_alphas_cumprod.to(device)[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one.to(device)[t]
        return s1.view(-1, 1, 1, 1) * x_t - s2.view(-1, 1, 1, 1) * noise

    def q_posterior(self, x_0, x_t, t):
        device = x_t.device
        s1 = self.posterior_mean_coef1.to(device)[t]
        s2 = self.posterior_mean_coef2.to(device)[t]
        return s1.view(-1, 1, 1, 1) * x_0 + s2.view(-1, 1, 1, 1) * x_t

    def get_variance(self, t, device):
        if t == 0:
            return 0
        variance = self.betas.to(device)[t] * (1. - self.alphas_cumprod_prev.to(device)[t]) / (1. - self.alphas_cumprod.to(device)[t])
        return variance.clip(1e-20)

    def step(self, model_output, timestep, sample):
        device = sample.device
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t, device) ** 0.5) * noise

        return pred_prev_sample + variance

    def add_noise(self, x_start, x_noise, timesteps):
        device = x_start.device
        s1 = self.sqrt_alphas_cumprod.to(device)[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps]
        return s1.view(-1, 1, 1, 1) * x_start + s2.view(-1, 1, 1, 1) * x_noise

    def __len__(self):
        return self.num_timesteps

# ---------------------------------------------------------------------------
# 3. BOUCLE PRINCIPALE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="mnist_unet_opti")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--hidden_channels", type=int, default=64) # Modifié par défaut
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    model = UNet(base_ch=config.hidden_channels).to(device)
    
    # Création du modèle EMA (une copie indépendante)
    ema = EMA(beta=0.9999) # Utilisation du decay de 0.9999
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    noise_scheduler = ImageNoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    global_step = 0
    losses = []
    
    print("Début de l'entraînement...")
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(dataloader):
            images = batch[0].to(device)
            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (images.shape[0],)).long().to(device)

            noisy = noise_scheduler.add_noise(images, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Mise à jour des poids de l'EMA à chaque itération
            ema.update_model_average(ema_model, model)

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
            
        progress_bar.close()

    print("Sauvegarde des modèles...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")
    torch.save(ema_model.state_dict(), f"{outdir}/ema_model.pth")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Génération des images finales (avec le modèle EMA)...")
    # On utilise ema_model pour l'inférence
    ema_model.eval()
    sample = torch.randn(config.eval_batch_size, 1, 28, 28).to(device)
    timesteps = list(range(len(noise_scheduler)))[::-1]
    
    for i, t in enumerate(tqdm(timesteps)):
        t_tensor = torch.full((config.eval_batch_size,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            residual = ema_model(sample, t_tensor)
        sample = noise_scheduler.step(residual, t, sample)

    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    grid = torchvision.utils.make_grid(sample, nrow=8, normalize=True, value_range=(-1, 1))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title("Chiffres générés (MNIST - EMA)")
    plt.tight_layout()
    plt.savefig(f"{imgdir}/final_generation.png")
    plt.close()
    
    print(f"Entraînement terminé ! Résultats sauvegardés dans {outdir}")