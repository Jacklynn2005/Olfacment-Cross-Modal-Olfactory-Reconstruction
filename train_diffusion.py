import argparse
from pathlib import Path

import torch

from olfactory_utils import (
    load_config,
    set_seed,
    get_device,
    build_loaders,
    VisionEncoder,
    OlfactoryEncoder,
    COIPModel,
    SignalVAE,
    LatentDenoisingModel,
    train_denoising_loop,
)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--coip_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/diffusion_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["random_seed"])

    device = get_device()
    print("Using device:", device)

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, signal_length, global_min, global_max = build_loaders(
        config,
        use_saved_sensor_metadata=True
    )

    print("Detected signal length:", signal_length)

    vae = SignalVAE(
        latent_dim=config["signal_latent_dim"],
        signal_length=signal_length
    ).to(device)

    vae_ckpt = torch.load(args.vae_path, map_location=device)
    vae.load_state_dict(extract_state_dict(vae_ckpt), strict=True)
    vae.eval()

    vision_encoder = VisionEncoder(
        latent_dim=config["image_latent_dim"],
        model_name=config["vit_model_name"]
    ).to(device)

    olfactory_encoder = OlfactoryEncoder(
        latent_dim=config["image_latent_dim"],
        num_sensors=32,
        signal_length=signal_length
    ).to(device)

    coip_model = COIPModel(
        vision_encoder=vision_encoder,
        olfactory_encoder=olfactory_encoder
    ).to(device)

    coip_ckpt = torch.load(args.coip_path, map_location=device)
    coip_model.load_state_dict(extract_state_dict(coip_ckpt), strict=True)
    coip_model.eval()

    denoising_model = LatentDenoisingModel(
        latent_dim=config["signal_latent_dim"],
        cond_dim=config["image_latent_dim"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        denoising_model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    train_denoising_loop(
        denoising_model=denoising_model,
        vae=vae,
        coip_model=coip_model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"]
    )

    save_path = save_dir / config["save_name"]
    torch.save(
        {
            "model_state_dict": denoising_model.state_dict(),
            "signal_length": signal_length,
            "config": config,
            "vae_path": args.vae_path,
            "coip_path": args.coip_path,
        },
        save_path
    )

    print("Saved diffusion checkpoint:", save_path)


if __name__ == "__main__":
    main()
