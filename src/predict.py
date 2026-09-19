"""
Predict DTA affinity for a CSV of SMILES / protein pairs.

Usage:
    python src/predict.py \
        --input data/predict/my_data.csv \
        --model_path checkpoints/exp_name/best_model.pt \
        --device cuda

CSV must contain at least ``smiles`` and ``protein`` columns.
If a target column (pKd / pKi / affinity) is present, metrics will
also be computed. Features are cached under
``feature_cache/predict/<csv_basename>/`` and can be reused.
Output predictions go to ``predict/<csv_name>_predictions.csv``.
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project root: this script lives in src/, so parent is project root
_project_root = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(_project_root / "lib" / "esm")

sys.path.append(os.path.dirname(__file__))

from config import DEFAULT_MODEL_CONFIG
from utils import log, set_seed, torch_load
from data import (
    resolve_target_column,
    BindingPKDDataset,
    collate_fn,
    _find_inf_rows_in_csv,
)
from model import create_model
from metrics import compute_metrics, metrics_str


def resolve_path(path_str):
    p = Path(path_str)
    if p.is_absolute():
        return p
    return _project_root / p


def load_predict_csv(csv_path, target_col=None):
    """Load and validate a prediction CSV.

    Required columns: smiles, protein
    Optional: target column (auto-detected if not specified).
    Adds a __feature_idx__ column using the original row index.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    original_count = len(df)

    if "smiles" not in df.columns:
        raise ValueError("CSV must contain a 'smiles' column.")
    if "protein" not in df.columns:
        raise ValueError("CSV must contain a 'protein' column.")

    # Try to resolve target column (optional for prediction)
    resolved_target_col = None
    try:
        resolved_target_col = resolve_target_column(df, target_col)
        log(f"Target column found: {resolved_target_col} (metrics will be computed)")
    except ValueError:
        if target_col is not None:
            raise
        log("No target column found; running in prediction-only mode.")

    # Filter inf / -inf in numeric columns
    inf_row_mask, _ = _find_inf_rows_in_csv(df)
    inf_count = int(inf_row_mask.sum())
    if inf_count > 0:
        log(f"Filtered {inf_count} rows containing inf / -inf.")
        df = df.loc[~inf_row_mask].copy()

    # Add feature index column (use original row index)
    df["__feature_idx__"] = df.index.astype(int)

    log(f"Loaded {len(df)} / {original_count} rows from {csv_path}")
    return df, resolved_target_col


def extract_predict_features(df, cache_dir, device, batch_size=32):
    """Extract features for prediction data.

    All samples are treated as a single 'predict' split internally.
    Reuses the extract_features module's batched extraction logic.
    """
    from extract_features import (
        extract_and_cache_features_batched,
        ensure_dir,
    )
    from encoders.esm_encoder import ESMEncoder
    from encoders.chemberta_encoder import ChemBERTaEncoder

    cache_dir = Path(cache_dir)
    ensure_dir(cache_dir)

    progress_file = cache_dir / "progress.csv"

    # Add a virtual split column
    df = df.copy()
    df["split"] = "predict"

    # Initialize ChemBERTa encoder
    chemberta_snapshots = (
        _project_root
        / "lib"
        / "ChemBERTa"
        / "models"
        / "models--DeepChem--ChemBERTa-77M-MTR"
        / "snapshots"
    )
    snapshot_dirs = [d for d in chemberta_snapshots.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        raise FileNotFoundError(
            f"No ChemBERTa snapshot found in {chemberta_snapshots}. "
            "Please download the model first."
        )
    local_chemberta_path = snapshot_dirs[0]

    chem_encoder = ChemBERTaEncoder(
        model_name=str(local_chemberta_path),
        device=device,
        batch_size=32,
        augment=0,
        cache=True,
        local_files_only=True,
    )

    # Initialize ESM encoder
    esm_encoder = ESMEncoder(
        variant="normal",
        device=device,
        batch_size=4,
        cache=True,
    )

    extract_and_cache_features_batched(
        data_df=df,
        esm_encoder=esm_encoder,
        chem_encoder=chem_encoder,
        cache_dir=str(cache_dir),
        progress_file=progress_file,
        batch_size=batch_size,
        protein_col="protein",
        split_col="split",
        num_c_tokens=16,
    )

    return str(cache_dir)


@torch.inference_mode()
def predict(model, data_loader, device, target_mean=0.0, target_std=1.0):
    """Run inference and return predictions on original scale."""
    model.eval()
    all_preds = []

    for batch in tqdm(data_loader, desc="Predicting", ncols=120):
        a, b, c, a_mask, b_mask, c_mask, targets = batch
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        a_mask = a_mask.to(device, non_blocking=True)
        b_mask = b_mask.to(device, non_blocking=True)
        c_mask = c_mask.to(device, non_blocking=True)

        outputs = model(
            a, b, c,
            a_mask=a_mask,
            b_mask=b_mask,
            c_mask=c_mask,
        )
        all_preds.extend(outputs.detach().cpu().reshape(-1).tolist())

    # Denormalize to original scale
    preds = np.array(all_preds, dtype=np.float64)
    original_preds = preds * float(target_std) + float(target_mean)
    return original_preds.tolist()


def load_model_and_config(checkpoint_path, device):
    """Load model from checkpoint, restoring model_config if available."""
    checkpoint = torch_load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint (fallback to default)
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        log("Loaded model config from checkpoint.")
    else:
        model_config = dict(DEFAULT_MODEL_CONFIG)
        log("No model_config in checkpoint; using defaults.")

    # Extract target normalization params
    target_mean = 0.0
    target_std = 1.0
    train_config = None
    if isinstance(checkpoint, dict) and "train_config" in checkpoint:
        train_config = checkpoint["train_config"]
        target_mean = float(train_config.get("target_mean", 0.0))
        target_std = float(train_config.get("target_std", 1.0))
        log(f"Target normalization: mean={target_mean:.4f}, std={target_std:.4f}")

    model = create_model(model_config).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    log("Model weights loaded.")
    return model, model_config, target_mean, target_std


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict DTA affinity from a CSV of SMILES/protein pairs."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV (must contain smiles, protein columns).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Checkpoint file (.pt) or directory containing best_model.pt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: predict/<csv_name>_predictions.csv",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Feature cache directory. "
        "Default: feature_cache/predict/<csv_basename>/",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="Target column name. Auto-detected if not specified.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=1400)
    parser.add_argument("--max_c_len", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_extract",
        action="store_true",
        help="Skip feature extraction (assume features already cached).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    input_path = resolve_path(args.input)
    model_path = resolve_path(args.model_path)

    # Resolve checkpoint: if a directory is given, auto-find best_model.pt
    if model_path.is_dir():
        candidate = model_path / "best_model.pt"
        if not candidate.exists():
            candidate = model_path / "last_model.pt"
        if not candidate.exists():
            raise FileNotFoundError(
                f"No best_model.pt or last_model.pt found in {model_path}. "
                "Please specify the checkpoint file directly."
            )
        model_path = candidate
        log(f"Auto-resolved checkpoint: {model_path}")

    # Determine output path: default to predict/<csv_name>_predictions.csv
    if args.output:
        output_path = resolve_path(args.output)
    else:
        output_path = _project_root / "predict" / f"{input_path.stem}_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine cache dir
    if args.cache_dir:
        cache_dir = resolve_path(args.cache_dir)
    else:
        cache_dir = _project_root / "feature_cache" / "predict" / input_path.stem
    log(f"Cache dir: {cache_dir}")

    # Load CSV
    df, target_col = load_predict_csv(str(input_path), args.target_col)

    # Extract features
    if not args.no_extract:
        log("Extracting features...")
        extract_predict_features(
            df=df,
            cache_dir=str(cache_dir),
            device=device,
            batch_size=args.batch_size,
        )
    else:
        log("Skipping feature extraction (--no_extract).")

    # Add split column for dataset compatibility
    df = df.copy()
    df["split"] = "predict"

    # If no target column, add a dummy one (dataset requires target_col)
    dataset_target_col = target_col
    if dataset_target_col is None:
        df["__dummy_target__"] = 0.0
        dataset_target_col = "__dummy_target__"

    # Create dataset and dataloader (use 'predict' as the split subset)
    dataset = BindingPKDDataset(
        data_df=df,
        cache_root=str(cache_dir),
        target_col=dataset_target_col,
    )
    dynamic_collate_fn = partial(
        collate_fn,
        max_seq_len=args.max_seq_len,
        max_c_len=args.max_c_len,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dynamic_collate_fn,
        pin_memory=(device == "cuda"),
    )

    # Load model
    log(f"Loading model from {model_path}")
    model, model_config, target_mean, target_std = load_model_and_config(
        str(model_path), device
    )

    # Predict
    log("Running prediction...")
    predictions = predict(
        model=model,
        data_loader=data_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    # Build output dataframe
    result_df = df.copy()
    result_df["prediction"] = predictions

    # Reorder: put prediction after smiles, protein, and target if present
    base_cols = ["smiles", "protein"]
    if target_col and target_col in result_df.columns:
        base_cols.append(target_col)
    base_cols.append("prediction")
    other_cols = [c for c in result_df.columns if c not in base_cols]
    result_df = result_df[base_cols + other_cols]

    # Compute metrics if target column exists
    if target_col and target_col in result_df.columns:
        targets = result_df[target_col].values.astype(np.float64)
        preds_arr = np.array(predictions, dtype=np.float64)
        valid_mask = np.isfinite(targets) & np.isfinite(preds_arr)
        if valid_mask.sum() > 0:
            metrics = compute_metrics(targets[valid_mask], preds_arr[valid_mask])
            log("Prediction metrics:")
            log(metrics_str(metrics))

    # Save
    result_df.to_csv(output_path, index=False)
    log(f"Predictions saved to: {output_path}")
    log(f"Total samples: {len(predictions)}")


if __name__ == "__main__":
    main()
