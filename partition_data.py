import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Partition metadata into train/val/test IDs using object-level grouping."
    )

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata.csv."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./indices/",
        help="Directory to save train_ids.txt, val_ids.txt, and test_ids.txt."
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting."
    )

    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of unique object IDs used for training."
    )

    parser.add_argument(
        "--val_frac_of_remaining",
        type=float,
        default=0.5,
        help="Fraction of remaining object IDs used for validation. The rest are used for testing."
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot find metadata file: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if "object_idx" not in df.columns:
        raise ValueError(
            "metadata.csv must contain an 'object_idx' column. "
            "This is required to prevent object-level data leakage."
        )

    if "file_idx" not in df.columns:
        df["file_idx"] = df.index

    unique_objects = df[["object_idx"]].drop_duplicates()

    train_objs = unique_objects.sample(
        frac=args.train_frac,
        random_state=args.random_seed
    )

    remaining_objs = unique_objects.drop(train_objs.index)

    val_objs = remaining_objs.sample(
        frac=args.val_frac_of_remaining,
        random_state=args.random_seed
    )

    test_objs = remaining_objs.drop(val_objs.index)

    train_df = df[df["object_idx"].isin(train_objs["object_idx"])]
    val_df = df[df["object_idx"].isin(val_objs["object_idx"])]
    test_df = df[df["object_idx"].isin(test_objs["object_idx"])]

    train_ids = train_df["file_idx"].astype(int).tolist()
    val_ids = val_df["file_idx"].astype(int).tolist()
    test_ids = test_df["file_idx"].astype(int).tolist()

    (output_dir / "train_ids.txt").write_text(
        "\n".join(map(str, train_ids)) + "\n",
        encoding="utf-8"
    )

    (output_dir / "val_ids.txt").write_text(
        "\n".join(map(str, val_ids)) + "\n",
        encoding="utf-8"
    )

    (output_dir / "test_ids.txt").write_text(
        "\n".join(map(str, test_ids)) + "\n",
        encoding="utf-8"
    )

    print("Partition complete.")
    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples:   {len(val_ids)}")
    print(f"Test samples:  {len(test_ids)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
