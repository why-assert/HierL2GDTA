"""
Randomly split a CSV dataset into train/val/test at 7:2:1 ratio.

Usage:
    python tools/split_dataset.py \
        --input data.csv \
        --output_dir data/mydataset \
        --seed 42

Output files in output_dir:
splited.csv — full dataset with a split column
train.csv / val.csv / test.csv
"""

import argparse
import random
from pathlib import Path

import pandas as pd


def split_dataset(
    input_csv: str,
    output_dir: str,
    seed: int = 42,
):
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if len(df) == 0:
        raise ValueError("Input CSV is empty; cannot split.")

    # Drop any existing split column to avoid reusing old splits.
    if "split" in df.columns:
        df = df.drop(columns=["split"])

    # Fixed seed for reproducible splits.
    rng = random.Random(seed)

    indices = list(df.index)
    rng.shuffle(indices)

    total_num = len(indices)

    # 70% train, 20% val, 10% test.
    train_num = int(total_num * 0.7)
    val_num = int(total_num * 0.2)

    train_indices = indices[:train_num]
    val_indices = indices[train_num:train_num + val_num]
    test_indices = indices[train_num + val_num:]

    split_map = {}

    for idx in train_indices:
        split_map[idx] = "train"

    for idx in val_indices:
        split_map[idx] = "val"

    for idx in test_indices:
        split_map[idx] = "test"

    # Assign split labels back by original row index.
    df["split"] = df.index.map(split_map)

    splited_path = output_path / "splited.csv"
    df.to_csv(splited_path, index=False)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    print("Dataset split completed.")
    print(f"Input file: {input_path}")
    print(f"Output dir: {output_path}")
    print(f"Random seed: {seed}")
    print()
    print(f"Total samples: {total_num}")
    print(f"Train: {len(train_df)} ({len(train_df) / total_num:.2%})")
    print(f"Val:   {len(val_df)} ({len(val_df) / total_num:.2%})")
    print(f"Test:  {len(test_df)} ({len(test_df) / total_num:.2%})")
    print()
    print(f"Full split file: {splited_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly split a CSV into train/val/test at 7:2:1."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV path.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/split",
        help="Output directory, default data/split.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed, default 42.",
    )

    args = parser.parse_args()

    split_dataset(
        input_csv=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()