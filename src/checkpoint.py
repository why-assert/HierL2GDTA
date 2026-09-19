"""Checkpoint save/load utilities with RNG state restoration."""

import os

import torch

from utils import log, get_rng_state, set_rng_state, torch_load


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_loss,
    bad_epochs,
    train_loss,
    val_metrics,
    train_config,
    model_config,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scaler_state_dict": (
            scaler.state_dict() if scaler is not None else None
        ),
        "best_val_loss": best_val_loss,
        "bad_epochs": bad_epochs,
        "train_loss": train_loss,
        "val_metrics": val_metrics,
        "train_config": train_config,
        "model_config": model_config,
        "rng_state": get_rng_state(),
    }
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device="cpu",
    strict=True,
    restore_rng=True,
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    log(f"Loading checkpoint: {path}")
    checkpoint = torch_load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint, strict=strict)
        log("Loaded model from a bare state_dict.")
        return 1, float("inf"), 0

    checkpoint_model_config = checkpoint.get("model_config")
    if isinstance(checkpoint_model_config, dict):
        old_seq_len = checkpoint_model_config.get("max_seq_len")
        old_c_len = checkpoint_model_config.get("max_c_len")
        current_seq_len = getattr(model, "max_seq_len", None)
        current_c_len = getattr(model, "max_c_len", None)
        if old_seq_len is not None or old_c_len is not None:
            log(
                "Checkpoint model config: "
                f"max_seq_len={old_seq_len}, max_c_len={old_c_len}"
            )
            log(
                "Current model config: "
                f"max_seq_len={current_seq_len}, max_c_len={current_c_len}"
            )

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    log("Model weights loaded.")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        log("Optimizer state loaded.")

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        log("Scheduler state loaded.")

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        log("GradScaler state loaded.")

    if restore_rng:
        set_rng_state(checkpoint.get("rng_state"))
        log("RNG state restored.")

    last_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = last_epoch + 1
    best_val_loss = float(
        checkpoint.get("best_val_loss", float("inf"))
    )
    bad_epochs = int(checkpoint.get("bad_epochs", 0))

    log(f"Checkpoint epoch={last_epoch}")
    log(f"Resuming from epoch={start_epoch}")
    log(f"best_val_loss={best_val_loss:.6f}")
    log(f"bad_epochs={bad_epochs}")

    return start_epoch, best_val_loss, bad_epochs