"""Regression metrics: MSE, RMSE, MAE, R², and Pearson correlation."""

import numpy as np
import torch


def mean_absolute_error_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def pearson_corr_np(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def metric_array(value, name):
    """Convert metric input to a 1-D float64 NumPy array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        result = np.array(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} could not be converted to a numeric array, "
            f"got type {type(value)}."
        ) from exc
    return result


def compute_metrics(y_true, y_pred):
    y_true = metric_array(y_true, "y_true")
    y_pred = metric_array(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch between targets and predictions: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    if len(y_true) == 0:
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "pearson": np.nan,
        }
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error_np(y_true, y_pred),
        "r2": r2_score_np(y_true, y_pred),
        "pearson": pearson_corr_np(y_true, y_pred),
    }


def metrics_str(metrics):
    def format_value(value):
        if value is None:
            return "nan"
        try:
            if np.isnan(value):
                return "nan"
        except TypeError:
            pass
        return f"{value:.4f}"

    return (
        f"MSE={format_value(metrics['mse'])}, "
        f"RMSE={format_value(metrics['rmse'])}, "
        f"MAE={format_value(metrics['mae'])}, "
        f"R2={format_value(metrics['r2'])}, "
        f"Pearson={format_value(metrics['pearson'])}"
    )


def align_pred_target(pred, target):
    if pred.ndim == 1 and target.ndim == 2 and target.size(-1) == 1:
        pred = pred.unsqueeze(-1)
    if target.ndim == 1 and pred.ndim == 2 and pred.size(-1) == 1:
        target = target.unsqueeze(-1)
    return pred, target