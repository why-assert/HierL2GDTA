"""
Dataset and DataLoader utilities for DTA training.

Handles CSV loading, target column resolution, feature file loading,
and batch collation with mask-aware padding.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import log


def _find_inf_rows_in_csv(df):
    """Return a boolean mask for rows containing inf or -inf values."""
    inf_row_mask = pd.Series(False, index=df.index)
    inf_columns = {}
    for column in df.columns:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        if not numeric_values.notna().any():
            continue
        values = numeric_values.to_numpy(dtype=np.float64)
        column_inf_mask = np.isinf(values)
        if column_inf_mask.any():
            row_mask = pd.Series(column_inf_mask, index=df.index)
            inf_row_mask |= row_mask
            inf_columns[column] = int(column_inf_mask.sum())
    return inf_row_mask, inf_columns


def resolve_target_column(df, requested_target_col=None):
    """Resolve target column name.

    If ``requested_target_col`` is given, verify it exists and return it.
    Otherwise auto-detect by scanning column names (case-insensitive) for
    ``pKd``, ``pKi``, or ``affinity`` in that order of priority.
    """
    if requested_target_col is not None:
        if requested_target_col not in df.columns:
            raise ValueError(
                f"Requested target column does not exist: "
                f"{requested_target_col}. CSV columns: {list(df.columns)}"
            )
        return requested_target_col

    candidates = ["pKd", "pKi", "affinity"]
    lower_columns = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_columns:
            matched = lower_columns[candidate.lower()]
            log(f"Auto-selected target column: {matched}")
            return matched

    raise ValueError(
        "No target column found. Use --target_col to specify one. "
        f"CSV columns: {list(df.columns)}"
    )


def load_split_csv(
    csv_path,
    split_col="split",
    feature_idx_col="__feature_idx__",
    target_col=None,
):
    """Load and clean a CSV with train/val/test split info."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    original_count = len(df)
    if split_col not in df.columns:
        raise ValueError(
            f"CSV must contain split column: {split_col}. "
            "Generate train/val/test splits first."
        )
    target_col = resolve_target_column(df, target_col)

    # Filter inf / -inf in all numeric columns
    inf_row_mask, inf_columns = _find_inf_rows_in_csv(df)
    inf_count = int(inf_row_mask.sum())
    if inf_count > 0:
        log(
            f"Detected {inf_count} CSV rows containing inf / -inf; "
            "filtering them out during data loading."
        )
        for column, count in inf_columns.items():
            log(f"  Bad column: {column}, count: {count}")
        df = df.loc[~inf_row_mask].copy()

    # Handle feature index column
    if feature_idx_col not in df.columns:
        df[feature_idx_col] = df.index.astype(int)
        log(
            f"CSV has no {feature_idx_col}; "
            "using current CSV row index as feature file index."
        )
    else:
        feature_idx_numeric = pd.to_numeric(
            df[feature_idx_col],
            errors="coerce",
        )
        valid_feature_idx_mask = (
            feature_idx_numeric.notna()
            & np.isfinite(feature_idx_numeric.to_numpy(dtype=np.float64))
        )
        valid_feature_idx_mask = pd.Series(
            valid_feature_idx_mask,
            index=df.index,
        )
        invalid_count = int((~valid_feature_idx_mask).sum())
        if invalid_count > 0:
            log(
                f"Filtered {invalid_count} rows with "
                "NaN / inf / -inf feature index."
            )
            df = df.loc[valid_feature_idx_mask].copy()
            feature_idx_numeric = feature_idx_numeric.loc[
                valid_feature_idx_mask
            ]
        df[feature_idx_col] = feature_idx_numeric.astype(np.int64)

    # Validate split values
    df[split_col] = df[split_col].astype(str).str.strip().str.lower()
    valid_splits = {"train", "val", "test"}
    invalid_mask = ~df[split_col].isin(valid_splits)
    if invalid_mask.any():
        invalid_values = sorted(df.loc[invalid_mask, split_col].unique())
        raise ValueError(
            "split column may only contain train, val, test. "
            f"Found invalid values: {invalid_values}"
        )

    # Clean target values
    target_numeric = pd.to_numeric(df[target_col], errors="coerce")
    valid_target_mask = np.isfinite(
        target_numeric.to_numpy(dtype=np.float64)
    )
    valid_target_mask = pd.Series(valid_target_mask, index=df.index)
    invalid_target_count = int((~valid_target_mask).sum())
    if invalid_target_count > 0:
        log(
            f"Filtered {invalid_target_count} rows with "
            "NaN / inf / -inf target."
        )
        df = df.loc[valid_target_mask].copy()
        target_numeric = target_numeric.loc[valid_target_mask]
    df[target_col] = target_numeric.astype(np.float64)

    total_dropped = original_count - len(df)
    if total_dropped > 0:
        log(
            f"CSV cleaning done: original={original_count:,}, "
            f"kept={len(df):,}, dropped={total_dropped:,}"
        )
    else:
        log("CSV check done, no inf / -inf / invalid targets found.")
    if len(df) == 0:
        raise ValueError("No samples left after filtering invalid data.")
    return df, target_col


class BindingPKDDataset(Dataset):
    """Feature dataset supporting split subdirs and flat dirs."""

    def __init__(
        self,
        data_df,
        cache_root,
        target_col,
        feature_idx_col="__feature_idx__",
        split_col="split",
        target_mean=None,
        target_std=None,
    ):
        super().__init__()
        self.data_df = data_df.reset_index(drop=True)
        self.cache_root = cache_root
        self.target_col = target_col
        self.feature_idx_col = feature_idx_col
        self.split_col = split_col
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.data_df)

    @staticmethod
    def _load_pt(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")
        try:
            value = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            value = torch.load(path, map_location="cpu")

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().to(dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            if value.dtype == object:
                try:
                    value = np.stack(value.tolist())
                except Exception as exc:
                    raise TypeError(
                        f"Feature file {path} is an object ndarray that "
                        "cannot be stacked."
                    ) from exc
            try:
                value = np.asarray(value, dtype=np.float32)
                value = np.ascontiguousarray(value)
                tensor = torch.from_numpy(value.copy())
            except Exception as exc:
                raise TypeError(
                    f"NumPy array in feature file {path} could not be "
                    f"converted. shape={value.shape}, dtype={value.dtype}"
                ) from exc
        elif isinstance(value, (list, tuple)):
            try:
                value = np.asarray(value, dtype=np.float32)
            except Exception:
                try:
                    value = np.stack([
                        np.asarray(item, dtype=np.float32)
                        for item in value
                    ])
                except Exception as exc:
                    raise TypeError(
                        f"list/tuple in feature file {path} could not be "
                        "converted."
                    ) from exc
            value = np.ascontiguousarray(value)
            tensor = torch.from_numpy(value.copy())
        else:
            raise TypeError(
                f"Feature file {path} must contain a Tensor, NumPy array, "
                f"list, or tuple, got {type(value)}."
            )

        if tensor.ndim != 2:
            raise ValueError(
                f"Feature file {path} must be 2-D "
                f"[seq_len, feature_dim], got {tuple(tensor.shape)}."
            )
        if tensor.size(0) == 0:
            raise ValueError(f"Feature file {path} has zero sequence length.")
        if tensor.size(1) == 0:
            raise ValueError(f"Feature file {path} has zero feature dim.")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Feature file {path} contains NaN or Inf.")
        return tensor.contiguous()

    def _resolve_feature_paths(self, row, feature_idx):
        split_name = str(row[self.split_col]).strip().lower()
        split_root = os.path.join(self.cache_root, split_name)
        split_paths = {
            "a": os.path.join(split_root, f"a_{feature_idx}.pt"),
            "b": os.path.join(split_root, f"b_{feature_idx}.pt"),
            "c": os.path.join(split_root, f"c_{feature_idx}.pt"),
        }
        flat_paths = {
            "a": os.path.join(self.cache_root, f"a_{feature_idx}.pt"),
            "b": os.path.join(self.cache_root, f"b_{feature_idx}.pt"),
            "c": os.path.join(self.cache_root, f"c_{feature_idx}.pt"),
        }
        if all(os.path.exists(path) for path in split_paths.values()):
            return split_paths
        if all(os.path.exists(path) for path in flat_paths.values()):
            return flat_paths
        raise FileNotFoundError(
            f"Features incomplete for feature_idx={feature_idx}.\n"
            f"Tried split dir: {split_root}\n"
            f"Tried flat dir: {self.cache_root}"
        )

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        feature_idx = int(row[self.feature_idx_col])
        paths = self._resolve_feature_paths(row, feature_idx)
        a = self._load_pt(paths["a"])
        b = self._load_pt(paths["b"])
        c = self._load_pt(paths["c"])
        target_value = float(row[self.target_col])
        if self.target_mean is not None and self.target_std is not None:
            target_value = (
                (target_value - self.target_mean)
                / self.target_std
            )
        target = torch.tensor([target_value], dtype=torch.float32)
        return a, b, c, target


def collate_fn(batch, max_seq_len=None, max_c_len=16):
    if len(batch) == 0:
        raise ValueError("collate_fn received an empty batch.")

    def pad_tensors_and_build_mask(tensors, max_length=None):
        if not tensors:
            raise ValueError("No tensors available for padding.")
        for index, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Feature {index} is not a torch.Tensor, "
                    f"got {type(tensor)}."
                )
            if tensor.ndim != 2:
                raise ValueError(
                    "Each feature must be [seq_len, feature_dim], "
                    f"got shape {tuple(tensor.shape)}."
                )
            if tensor.size(0) == 0:
                raise ValueError(
                    "Batch contains a feature sequence of length 0."
                )

        batch_max_len = max(tensor.size(0) for tensor in tensors)
        if max_length is None:
            target_len = batch_max_len
        else:
            if max_length <= 0:
                raise ValueError(
                    f"max_length must be > 0, got {max_length}."
                )
            target_len = min(batch_max_len, max_length)

        embedding_dim = tensors[0].size(1)
        if any(tensor.size(1) != embedding_dim for tensor in tensors):
            raise ValueError(
                "Feature dims within the same branch are inconsistent."
            )

        padded = torch.zeros(
            len(tensors),
            target_len,
            embedding_dim,
            dtype=torch.float32,
        )
        lengths = torch.zeros(len(tensors), dtype=torch.long)
        for i, tensor in enumerate(tensors):
            current = tensor[:target_len].to(
                dtype=torch.float32,
                device="cpu",
            )
            current_len = current.size(0)
            if current_len == 0:
                raise ValueError(
                    "A sequence became length 0 after truncation; "
                    "check max_seq_len / max_c_len."
                )
            padded[i, :current_len] = current
            lengths[i] = current_len

        positions = torch.arange(target_len, dtype=torch.long)
        positions = positions.unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        return padded, lengths, mask

    a_list = [item[0] for item in batch]
    b_list = [item[1] for item in batch]
    c_list = [item[2] for item in batch]
    targets = torch.stack([item[3] for item in batch])

    a_padded, _, a_mask = pad_tensors_and_build_mask(
        a_list,
        max_length=max_seq_len,
    )
    b_padded, _, b_mask = pad_tensors_and_build_mask(
        b_list,
        max_length=max_seq_len,
    )
    c_padded, _, c_mask = pad_tensors_and_build_mask(
        c_list,
        max_length=max_c_len,
    )

    result = (
        a_padded,
        b_padded,
        c_padded,
        a_mask,
        b_mask,
        c_mask,
        targets.to(dtype=torch.float32),
    )
    for index, value in enumerate(result):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"collate_fn return value {index} is not a torch.Tensor, "
                f"got {type(value)}."
            )
    return result