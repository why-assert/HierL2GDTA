"""
Train a HierL2G DTA model on a pre-split CSV with cached features.

Usage:
    python src/train.py \
        --data_file data/dataset/splited.csv \
        --cache_root feature_cache/dataset \
        --save_dir checkpoints/exp_name \
        --use_amp

The CSV must contain ``smiles``, ``protein``, ``split``, and a target
column (``pKd`` / ``pKi`` / ``affinity``, auto-detected or set via
``--target_col``). Features are loaded from ``cache_root`` (supports
both split-subdir and flat layouts). Training supports AMP, early
stopping, and checkpoint resumption.
"""

import argparse
import json
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEFAULT_MODEL_CONFIG
from utils import log, set_seed, torch_load
from data import load_split_csv, BindingPKDDataset, collate_fn
from engine import train_model, evaluate_model
from metrics import metrics_str
from model import create_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model on a CSV with train/val/test splits."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to CSV containing a split column.",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        required=True,
        help="Feature directory; supports split subdirs or flat layout.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Output dir for checkpoints, configs, and predictions.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="Target column. Defaults to trying pKd, pki, affinity.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate, default 1e-4.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=1400)
    parser.add_argument("--max_c_len", type=int, default=16)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_file = args.data_file
    cache_root = args.cache_root
    save_dir = args.save_dir

    if not os.path.isabs(data_file):
        data_file = os.path.join(project_root, data_file)
    if not os.path.isabs(cache_root):
        cache_root = os.path.join(project_root, cache_root)
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(project_root, save_dir)

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_col = "split"
    feature_idx_col = "__feature_idx__"
    resume = not args.no_resume

    log(" Program start ")
    log(f"data_file={data_file}")
    log(f"cache_root={cache_root}")
    log(f"save_dir={save_dir}")
    log(f"Device: {device}")
    log(f"NumPy version: {np.__version__}")
    log(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    df, target_col = load_split_csv(
        csv_path=data_file,
        split_col=split_col,
        feature_idx_col=feature_idx_col,
        target_col=args.target_col,
    )

    required_columns = [
        "smiles",
        "protein",
        target_col,
        split_col,
        feature_idx_col,
    ]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"CSV must contain column: {column}")

    log(f"Target column: {target_col}")
    log(f"Total samples: {len(df):,}")
    log(f"Split counts:\n{df[split_col].value_counts().to_string()}")

    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col] == "val"].copy()
    test_df = df[df[split_col] == "test"].copy()

    if len(train_df) == 0:
        raise ValueError("split=train has 0 samples.")
    if len(val_df) == 0:
        raise ValueError("split=val has 0 samples.")
    if len(test_df) == 0:
        raise ValueError("split=test has 0 samples.")

    target_mean = float(train_df[target_col].mean())
    target_std = float(train_df[target_col].std())

    if not np.isfinite(target_std) or target_std < 1e-8:
        raise ValueError(
            f"Train target std is invalid: {target_std}. "
            "Cannot perform target standardization."
        )

    log(
        f"Train target standardization: "
        f"mean={target_mean:.6f}, std={target_std:.6f}"
    )
    log(f"Train samples: {len(train_df):,}")
    log(f"Val samples:   {len(val_df):,}")
    log(f"Test samples:  {len(test_df):,}")

    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config["max_seq_len"] = args.max_seq_len
    model_config["max_c_len"] = args.max_c_len

    train_config = {
        "data_file": data_file,
        "cache_root": cache_root,
        "save_dir": save_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_seq_len": args.max_seq_len,
        "max_c_len": args.max_c_len,
        "verbose_interval": 50,
        "num_workers": args.num_workers,
        "use_amp": args.use_amp,
        "device": str(device),
        "seed": args.seed,
        "split_col": split_col,
        "target_col": target_col,
        "feature_idx_col": feature_idx_col,
        "resume": resume,
        "patience": args.patience,
        "min_delta": 1e-4,
        "grad_clip_norm": 5.0,
        "target_mean": target_mean,
        "target_std": target_std,
        "num_c_tokens": args.max_c_len,
    }

    with open(
        os.path.join(save_dir, "train_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(train_config, file, ensure_ascii=False, indent=2)

    with open(
        os.path.join(save_dir, "model_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(model_config, file, ensure_ascii=False, indent=2)

    dynamic_collate_fn = partial(
        collate_fn,
        max_seq_len=args.max_seq_len,
        max_c_len=args.max_c_len,
    )

    loader_common_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": dynamic_collate_fn,
        "drop_last": False,
        "num_workers": args.num_workers,
    }

    if args.num_workers > 0:
        loader_common_kwargs["persistent_workers"] = True
        loader_common_kwargs["prefetch_factor"] = 2

    train_dataset = BindingPKDDataset(
        data_df=train_df,
        cache_root=cache_root,
        target_col=target_col,
        feature_idx_col=feature_idx_col,
        split_col=split_col,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_dataset = BindingPKDDataset(
        data_df=val_df,
        cache_root=cache_root,
        target_col=target_col,
        feature_idx_col=feature_idx_col,
        split_col=split_col,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_dataset = BindingPKDDataset(
        data_df=test_df,
        cache_root=cache_root,
        target_col=target_col,
        feature_idx_col=feature_idx_col,
        split_col=split_col,
        target_mean=target_mean,
        target_std=target_std,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_common_kwargs,
    )

    log(" Training start ")

    model = create_model(model_config).to(device)

    best_path, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=save_dir,
        train_config=train_config,
        model_config=model_config,
        target_mean=target_mean,
        target_std=target_std,
        verbose_interval=50,
        patience=args.patience,
        min_delta=1e-4,
        grad_clip_norm=5.0,
        resume=resume,
        use_amp=args.use_amp,
    )

    training_summary = {
        "best_checkpoint": best_path,
        "best_val_loss": float(best_val_loss),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }

    with open(
        os.path.join(save_dir, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(training_summary, file, ensure_ascii=False, indent=2)

    log(f"Training finished, best_val_loss={best_val_loss:.6f}")

    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"Best checkpoint not found, cannot run test: {best_path}"
        )

    log(" Test evaluation ")

    best_model = create_model(model_config).to(device)
    checkpoint = torch_load(best_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        best_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        best_model.load_state_dict(checkpoint)

    test_metrics, all_targets, all_preds = evaluate_model(
        model=best_model,
        data_loader=test_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    log(" Test metrics ")
    log(metrics_str(test_metrics))

    if len(all_targets) != len(test_df):
        raise ValueError(
            "Number of test predictions does not match test samples: "
            f"predictions={len(all_targets)}, samples={len(test_df)}"
        )

    prediction_df = test_df.copy()
    prediction_df[f"pred_{target_col}"] = all_preds
    prediction_df[f"true_{target_col}"] = all_targets

    prediction_path = os.path.join(save_dir, "test_predictions.csv")
    prediction_df.to_csv(prediction_path, index=False)

    with open(
        os.path.join(save_dir, "test_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(test_metrics, file, ensure_ascii=False, indent=2)

    log(f"Test predictions saved: {prediction_path}")
    log(" Program end ")


if __name__ == "__main__":
    main()