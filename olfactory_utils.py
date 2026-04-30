# ============================================================
# olfactory_utils.py
# Shared utilities for olfactory VAE, COIP, and diffusion training.
# ============================================================

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import ViTModel, ViTImageProcessor


# ============================================================
# General helpers
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_metadata_splits(config):
    df = pd.read_csv(config["metadata_csv"])
    df["file_idx"] = df.index

    if "object_idx" not in df.columns:
        raise ValueError("metadata.csv must contain an object_idx column.")

    unique_objects = df[["object_idx"]].drop_duplicates()

    train_objs = unique_objects.sample(frac=0.8, random_state=config["random_seed"])
    remaining_objs = unique_objects.drop(train_objs.index)

    val_objs = remaining_objs.sample(frac=0.5, random_state=config["random_seed"])
    test_objs = remaining_objs.drop(val_objs.index)

    train_df = df[df["object_idx"].isin(train_objs["object_idx"])].reset_index(drop=True)
    val_df = df[df["object_idx"].isin(val_objs["object_idx"])].reset_index(drop=True)
    test_df = df[df["object_idx"].isin(test_objs["object_idx"])].reset_index(drop=True)

    if config.get("run_small", False):
        train_df = train_df.sample(
            n=min(config["train_limit"], len(train_df)),
            random_state=config["random_seed"]
        ).reset_index(drop=True)

        val_df = val_df.sample(
            n=min(config["val_limit"], len(val_df)),
            random_state=config["random_seed"]
        ).reset_index(drop=True)

    return train_df, val_df, test_df


# ============================================================
# Dataset
# ============================================================

class OlfactoryMatrixDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        image_dir,
        signals_dir,
        processor,
        global_min=None,
        global_max=None
    ):
        self.df = metadata_df.copy()

        if "file_idx" in self.df.columns:
            self.indices = self.df["file_idx"].tolist()
        elif "global_id" in self.df.columns:
            self.indices = self.df["global_id"].tolist()
        else:
            self.indices = self.df.index.tolist()

        self.image_dir = image_dir
        self.signals_dir = signals_dir
        self.processor = processor
        self.global_min = global_min
        self.global_max = global_max

        if "object_idx" not in self.df.columns:
            raise ValueError("metadata_df must contain object_idx.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_idx = int(self.indices[idx])

        img_path = os.path.join(self.image_dir, f"sample_{file_idx:04d}.jpg")
        sig_path = os.path.join(self.signals_dir, f"signal_{file_idx:04d}.npy")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(sig_path):
            raise FileNotFoundError(f"Signal not found: {sig_path}")

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        matrix = np.load(sig_path)

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        elif matrix.ndim == 0:
            matrix = np.array([[matrix]])

        signal = torch.tensor(matrix, dtype=torch.float32)

        if signal.ndim != 2:
            raise ValueError(f"Signal should be 2D, got {signal.shape}.")

        if signal.shape[0] == 32:
            pass
        elif signal.shape[1] == 32:
            signal = signal.transpose(0, 1)
        else:
            raise ValueError(
                f"Expected [32, time] or [time, 32], got {signal.shape}."
            )

        if self.global_min is not None and self.global_max is not None:
            signal = (signal - self.global_min) / (
                self.global_max - self.global_min + 1e-8
            )

        object_idx = int(self.df.iloc[idx]["object_idx"])

        return pixel_values, signal, object_idx


def calculate_sensor_constants(loader):
    global_min = None
    global_max = None

    for _, signals, _ in tqdm(loader, desc="Calculating signal min/max"):
        batch_min = signals.amin(dim=(0, 2))
        batch_max = signals.amax(dim=(0, 2))

        if global_min is None:
            global_min = batch_min
            global_max = batch_max
        else:
            global_min = torch.minimum(global_min, batch_min)
            global_max = torch.maximum(global_max, batch_max)

    return global_min.view(32, 1), global_max.view(32, 1)


def build_loaders(config, use_saved_sensor_metadata=False):
    processor = ViTImageProcessor.from_pretrained(config["vit_model_name"])

    train_df, val_df, test_df = prepare_metadata_splits(config)

    temp_train_dataset = OlfactoryMatrixDataset(
        train_df,
        image_dir=config["image_dir"],
        signals_dir=config["signal_dir"],
        processor=processor,
        global_min=None,
        global_max=None
    )

    temp_train_loader = DataLoader(
        temp_train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    if use_saved_sensor_metadata:
        meta = torch.load(config["sensor_metadata_path"], map_location="cpu")
        global_min = meta["min"]
        global_max = meta["max"]
    else:
        global_min, global_max = calculate_sensor_constants(temp_train_loader)

    train_dataset = OlfactoryMatrixDataset(
        train_df,
        image_dir=config["image_dir"],
        signals_dir=config["signal_dir"],
        processor=processor,
        global_min=global_min,
        global_max=global_max
    )

    val_dataset = OlfactoryMatrixDataset(
        val_df,
        image_dir=config["image_dir"],
        signals_dir=config["signal_dir"],
        processor=processor,
        global_min=global_min,
        global_max=global_max
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    sample_images, sample_signals, sample_object_ids = next(iter(train_loader))
    signal_length = sample_signals.shape[-1]

    return train_loader, val_loader, signal_length, global_min, global_max


# ============================================================
# Models
# ============================================================

class VisionEncoder(nn.Module):
    def __init__(self, latent_dim=512, model_name="google/vit-base-patch16-224"):
        super().__init__()

        self.backbone = ViTModel.from_pretrained(model_name)

        self.projection = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        latent_vector = self.projection(cls_token)
        return latent_vector


class OlfactoryEncoder(nn.Module):
    def __init__(self, latent_dim=512, num_sensors=32, signal_length=1):
        super().__init__()

        input_dim = num_sensors * signal_length

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, signal):
        return self.net(signal)


class COIPModel(nn.Module):
    def __init__(self, vision_encoder, olfactory_encoder):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.olfactory_encoder = olfactory_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, pixel_values, signals):
        image_features = self.vision_encoder(pixel_values)
        signal_features = self.olfactory_encoder(signals)

        image_features = F.normalize(image_features, dim=-1)
        signal_features = F.normalize(signal_features, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100)

        return image_features, signal_features, logit_scale


class SignalVAE(nn.Module):
    def __init__(self, latent_dim=128, signal_length=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.signal_length = signal_length

        self.encoder = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 128 * signal_length)

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.squeeze(-1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(z.size(0), 128, self.signal_length)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class LatentDenoisingModel(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + 1, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, z_t, t, condition):
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t = t.float().to(z_t.device)
        condition = condition.to(z_t.device)

        x = torch.cat([z_t, t, condition], dim=-1)

        return self.net(x)


# ============================================================
# Training helpers
# ============================================================

def vae_loss(recon_x, x, mu, logvar, beta=1e-4):
    if recon_x.shape != x.shape:
        raise ValueError(f"VAE output shape {recon_x.shape} does not match target shape {x.shape}")

    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_loss

    return loss, recon_loss, kl_loss


def train_vae_loop(vae, train_loader, optimizer, device, epochs=1):
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        pbar = tqdm(train_loader, desc=f"VAE Epoch {epoch + 1}/{epochs}")

        for _, signals, _ in pbar:
            signals = signals.to(device)

            recon, mu, logvar = vae(signals)
            loss, recon_loss, kl_loss = vae_loss(recon, signals, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            pbar.set_postfix({
                "loss": loss.item(),
                "recon": recon_loss.item(),
                "kl": kl_loss.item()
            })

        print(
            f"VAE Epoch {epoch + 1}: "
            f"loss={total_loss / len(train_loader):.4f}, "
            f"recon={total_recon / len(train_loader):.4f}, "
            f"kl={total_kl / len(train_loader):.4f}"
        )

        if device.type == "mps":
            torch.mps.empty_cache()


def object_level_contrastive_loss(logits, object_ids):
    object_ids = object_ids.to(logits.device)
    positive_mask = object_ids.unsqueeze(0) == object_ids.unsqueeze(1)
    positive_mask = positive_mask.float()

    log_probs = F.log_softmax(logits, dim=1)
    positive_count = positive_mask.sum(dim=1).clamp(min=1)

    return (-(positive_mask * log_probs).sum(dim=1) / positive_count).mean()


def get_coip_optimizer_object(model, lr=2e-5, weight_decay=0.1):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name == "logit_scale" or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr
    )


@torch.no_grad()
def validate_object_retrieval(model, val_loader, device):
    model.eval()

    all_img_emb = []
    all_sig_emb = []
    all_object_ids = []

    for images, signals, object_ids in tqdm(val_loader, desc="Validating object-level retrieval"):
        images = images.to(device)
        signals = signals.to(device)

        img_emb, sig_emb, _ = model(images, signals)

        all_img_emb.append(img_emb.cpu())
        all_sig_emb.append(sig_emb.cpu())
        all_object_ids.append(object_ids.cpu())

    all_img_emb = torch.cat(all_img_emb, dim=0)
    all_sig_emb = torch.cat(all_sig_emb, dim=0)
    all_object_ids = torch.cat(all_object_ids, dim=0)

    similarity = all_img_emb @ all_sig_emb.T
    ranking = similarity.argsort(dim=1, descending=True)

    correct_at_1 = 0
    correct_at_5 = 0

    for i in range(similarity.size(0)):
        true_object = all_object_ids[i]
        top1_idx = ranking[i, 0]
        top5_idx = ranking[i, :5]

        if all_object_ids[top1_idx] == true_object:
            correct_at_1 += 1

        if (all_object_ids[top5_idx] == true_object).any():
            correct_at_5 += 1

    recall_at_1 = correct_at_1 / similarity.size(0)
    recall_at_5 = correct_at_5 / similarity.size(0)

    return recall_at_1, recall_at_5


def train_coip_loop(model, train_loader, val_loader, optimizer, device, epochs=1):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"COIP Epoch {epoch + 1}/{epochs}")

        for images, signals, object_ids in pbar:
            images = images.to(device)
            signals = signals.to(device)
            object_ids = object_ids.to(device)

            img_emb, sig_emb, logit_scale = model(images, signals)

            logits_per_image = logit_scale * (img_emb @ sig_emb.T)
            logits_per_signal = logits_per_image.T

            loss_img = object_level_contrastive_loss(logits_per_image, object_ids)
            loss_sig = object_level_contrastive_loss(logits_per_signal, object_ids)

            loss = (loss_img + loss_sig) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        obj_r1, obj_r5 = validate_object_retrieval(model, val_loader, device)

        print(
            f"COIP Epoch {epoch + 1}: "
            f"loss={avg_loss:.4f}, "
            f"Object Recall@1={obj_r1:.4f}, "
            f"Object Recall@5={obj_r5:.4f}"
        )

        if device.type == "mps":
            torch.mps.empty_cache()


def add_noise_to_latent(z, t):
    noise = torch.randn_like(z)
    z_t = (1 - t) * z + t * noise
    return z_t, noise


def train_denoising_loop(denoising_model, vae, coip_model, train_loader, optimizer, device, epochs=1):
    vae.eval()
    coip_model.eval()

    for epoch in range(epochs):
        denoising_model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Denoising Epoch {epoch + 1}/{epochs}")

        for images, signals, _ in pbar:
            images = images.to(device)
            signals = signals.to(device)

            with torch.no_grad():
                image_features, _, _ = coip_model(images, signals)
                mu, _ = vae.encode(signals)
                z = mu

            batch_size = z.size(0)
            t = torch.rand(batch_size, 1, device=device)

            z_t, noise = add_noise_to_latent(z, t)

            predicted_noise = denoising_model(z_t, t, image_features)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Denoising Epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}")

        if device.type == "mps":
            torch.mps.empty_cache()


def freeze_vit_layers(coip_model, unfreeze_last_vit_layers=2):
    for param in coip_model.vision_encoder.backbone.parameters():
        param.requires_grad = False

    if unfreeze_last_vit_layers > 0:
        for layer in coip_model.vision_encoder.backbone.encoder.layer[-unfreeze_last_vit_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    for param in coip_model.vision_encoder.projection.parameters():
        param.requires_grad = True


def load_vision_encoder_from_coip(coip_model, state_dict):
    filtered = {
        k.replace("vision_encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("vision_encoder.")
    }
    coip_model.vision_encoder.load_state_dict(filtered, strict=True)
