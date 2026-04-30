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
    get_coip_optimizer_object,
    train_coip_loop,
    freeze_vit_layers,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/coip_config.yaml")
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

    if config["freeze_vit"]:
        print(
            "Freezing ViT backbone, unfreezing last "
            f"{config['unfreeze_last_vit_layers']} layer(s)."
        )
        freeze_vit_layers(
            coip_model,
            unfreeze_last_vit_layers=config["unfreeze_last_vit_layers"]
        )

    trainable_params = sum(p.numel() for p in coip_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in coip_model.parameters())

    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable ratio: {trainable_params / total_params:.2%}")

    optimizer = get_coip_optimizer_object(
        coip_model,
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    train_coip_loop(
        model=coip_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"]
    )

    save_path = save_dir / config["save_name"]
    torch.save(
        {
            "model_state_dict": coip_model.state_dict(),
            "signal_length": signal_length,
            "config": config,
            "vae_path": args.vae_path,
        },
        save_path
    )

    print("Saved COIP checkpoint:", save_path)


if __name__ == "__main__":
    main()
