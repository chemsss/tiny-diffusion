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
# 1. ARCHITECTURE : U-Net Conditionnel
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
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
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout_rate=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()
        
    def forward(self, x, emb):
        h = self.act(self.norm1(self.conv1(x)))
        # On injecte l'embedding combiné (temps + classe)
        emb_val = self.time_mlp(emb).view(-1, h.shape[1], 1, 1)
        h = h + emb_val
        h = self.dropout(h)
        h = self.act(self.norm2(self.conv2(h)))
        return h

class ConditionalUNet(nn.Module):
    def __init__(self, base_ch=64, num_classes=10):
        super().__init__()
        time_dim = base_ch * 4
        
        # Encodage du temps
        self.time_emb = PositionalEmbedding(base_ch, "sinusoidal")
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encodage des classes (num_classes + 1 pour la classe "nulle" du CFG)
        self.class_emb = nn.Embedding(num_classes + 1, time_dim)

        self.inc = nn.Conv2d(1, base_ch, 3, padding=1)
        
        self.down1 = Block(base_ch, base_ch * 2, time_dim)
        self.pool1 = nn.MaxPool2d(2) 
        
        self.down2 = Block(base_ch * 2, base_ch * 4, time_dim)
        self.pool2 = nn.MaxPool2d(2) 
        
        self.mid1 = Block(base_ch * 4, base_ch * 4, time_dim)
        self.attn = SelfAttention(base_ch * 4)
        self.mid2 = Block(base_ch * 4, base_ch * 4, time_dim)
        
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.up_block2 = Block(base_ch * 6, base_ch * 2, time_dim) 
        
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.up_block1 = Block(base_ch * 3, base_ch, time_dim)
        
        self.outc = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x, t, y):
        # On calcule les deux embeddings
        t_emb = self.time_mlp(self.time_emb(t))
        c_emb = self.class_emb(y)
        
        # Fusion par simple addition
        emb = t_emb + c_emb
        
        x1 = self.inc(x)
        x2 = self.down1(x1, emb)
        p2 = self.pool1(x2)
        
        x3 = self.down2(p2, emb)
        p3 = self.pool2(x3)
        
        m = self.mid1(p3, emb)
        m = self.attn(m)
        m = self.mid2(m, emb)
        
        u2 = self.up2(m)
        u2 = torch.cat([u2, x3], dim=1) 
        u2 = self.up_block2(u2, emb)
        
        u1 = self.up1(u2)
        u1 = torch.cat([u1, x2], dim=1) 
        u1 = self.up_block1(u1, emb)
        
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
        # Les variances de la chaîne inverse sont fixées à des constantes, comme conseillé par l'article Ho, Jonathan, et al. pour stabiliser l'entraînement
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
# 3. BOUCLE PRINCIPALE (Avec entraînement conditionnel CFG)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="mnist_unet_cond")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Force du conditionnement à la génération")
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    # L'index 10 sera notre "classe nulle" pour le Classifier-Free Guidance
    NULL_CLASS = 10
    model = ConditionalUNet(base_ch=config.hidden_channels, num_classes=10).to(device)
    
    ema = EMA(beta=0.9999)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    noise_scheduler = ImageNoiseScheduler(num_timesteps=config.num_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print("Début de l'entraînement...")
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(dataloader):
            images, labels = batch[0].to(device), batch[1].to(device)
            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (images.shape[0],)).long().to(device)

            noisy = noise_scheduler.add_noise(images, noise, timesteps)
            
            # --- CLASSIFIER-FREE GUIDANCE (ENTRAÎNEMENT) ---
            # 10% du temps, on remplace le vrai label par le label NULL
            mask = torch.rand(labels.shape[0], device=device) < 0.1
            labels[mask] = NULL_CLASS
            
            # Forward pass avec le bruit conditionné par la classe
            noise_pred = model(noisy, timesteps, labels)
            
            # Optimisation sur la borne variationnelle simplifiée (MSE unweighted)
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            ema.update_model_average(ema_model, model)

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.detach().item()})
            
        progress_bar.close()

    print("Sauvegarde des modèles...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(ema_model.state_dict(), f"{outdir}/ema_model.pth")

    # --- CLASSIFIER-FREE GUIDANCE (GÉNÉRATION) ---
    print("Génération finale (demande explicite des chiffres 0 à 9)...")
    ema_model.eval()
    
    # On va générer exactement les 10 chiffres (0, 1, 2, ..., 9)
    y_target = torch.arange(10, device=device, dtype=torch.long)
    y_null = torch.full((10,), NULL_CLASS, device=device, dtype=torch.long)
    
    sample = torch.randn(10, 1, 28, 28).to(device)
    timesteps = list(range(len(noise_scheduler)))[::-1]
    
    for t in tqdm(timesteps):
        t_tensor = torch.full((10,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            # Prédiction conditionnée (avec le label cible)
            noise_cond = ema_model(sample, t_tensor, y_target)
            # Prédiction inconditionnelle (avec le label nul)
            noise_uncond = ema_model(sample, t_tensor, y_null)
            
            # Extrapolation CFG : Inconditionnel + Scale * (Conditionnel - Inconditionnel)
            noise_pred = noise_uncond + config.guidance_scale * (noise_cond - noise_uncond)
            
        sample = noise_scheduler.step(noise_pred, t, sample)

    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    grid = torchvision.utils.make_grid(sample, nrow=5, normalize=True, value_range=(-1, 1))
    
    plt.figure(figsize=(10, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title(f"Génération Conditionnelle (Guidance Scale = {config.guidance_scale})")
    plt.tight_layout()
    plt.savefig(f"{imgdir}/final_conditional_generation.png")
    plt.close()
    
    print(f"Entraînement terminé ! Grille des chiffres de 0 à 9 sauvegardée dans {outdir}")