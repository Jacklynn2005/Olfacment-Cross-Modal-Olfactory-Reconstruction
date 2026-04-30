import argparse
from pathlib import Path

import torch

from olfactory_utils import (
    load_config,
    set_seed,
    get_device,
    build_loaders,
    SignalVAE,
    train_vae_loop,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vae_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["random_seed"])

    device = get_device()
    print("Using device:", device)

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, signal_length, global_min, global_max = build_loaders(
        config,
        use_saved_sensor_metadata=False
    )

    torch.save(
        {"min": global_min, "max": global_max},
        save_dir / config["metadata_save_name"]
    )

    print("Saved sensor metadata:", save_dir / config["metadata_save_name"])
    print("Detected signal length:", signal_length)

    vae = SignalVAE(
        latent_dim=config["signal_latent_dim"],
        signal_length=signal_length
    ).to(device)

    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    train_vae_loop(
        vae=vae,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"]
    )

    save_path = save_dir / config["save_name"]
    torch.save(
        {
            "model_state_dict": vae.state_dict(),
            "signal_length": signal_length,
            "config": config,
        },
        save_path
    )

    print("Saved VAE checkpoint:", save_path)


if __name__ == "__main__":
    main()
