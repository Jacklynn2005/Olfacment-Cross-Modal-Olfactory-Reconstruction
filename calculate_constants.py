import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


def load_ids(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Cannot find ID file: {path}")

    ids = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(int(line))

    return ids


def load_signal(signal_dir, file_idx):
    signal_path = Path(signal_dir) / f"signal_{file_idx:04d}.npy"

    if not signal_path.exists():
        raise FileNotFoundError(f"Cannot find signal file: {signal_path}")

    matrix = np.load(signal_path)

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim == 0:
        matrix = np.array([[matrix]])

    signal = torch.tensor(matrix, dtype=torch.float32)

    if signal.ndim != 2:
        raise ValueError(
            f"Signal should be 2D, got shape {signal.shape}. "
            f"Problem file: {signal_path}"
        )

    if signal.shape[0] == 32:
        pass
    elif signal.shape[1] == 32:
        signal = signal.transpose(0, 1)
    else:
        raise ValueError(
            f"Expected signal shape [32, time] or [time, 32], got {signal.shape}. "
            f"Problem file: {signal_path}"
        )

    return signal


def main():
    parser = argparse.ArgumentParser(
        description="Calculate per-sensor normalization constants from training signals."
    )

    parser.add_argument(
        "--train_ids",
        type=str,
        required=True,
        help="Path to train_ids.txt."
    )

    parser.add_argument(
        "--signal_dir",
        type=str,
        default="ny_smells_local/signals_raw",
        help="Directory containing signal_XXXX.npy files."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="sensor_metadata.pt",
        help="Output path for sensor_metadata.pt."
    )

    args = parser.parse_args()

    train_ids = load_ids(args.train_ids)

    if len(train_ids) == 0:
        raise ValueError("train_ids.txt is empty.")

    global_min = None
    global_max = None

    for file_idx in tqdm(train_ids, desc="Calculating sensor constants"):
        signal = load_signal(args.signal_dir, file_idx)

        batch_min = signal.amin(dim=1)
        batch_max = signal.amax(dim=1)

        if global_min is None:
            global_min = batch_min
            global_max = batch_max
        else:
            global_min = torch.minimum(global_min, batch_min)
            global_max = torch.maximum(global_max, batch_max)

    global_min = global_min.view(32, 1)
    global_max = global_max.view(32, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "min": global_min,
            "max": global_max,
        },
        output_path
    )

    print("Saved sensor metadata:", output_path)
    print("Min shape:", tuple(global_min.shape))
    print("Max shape:", tuple(global_max.shape))


if __name__ == "__main__":
    main()
