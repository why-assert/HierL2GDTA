"""
Training and evaluation loops for HierL2G DTA model.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import log
from metrics import compute_metrics, metrics_str, align_pred_target
from checkpoint import save_checkpoint, load_checkpoint


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    scaler=None,
    use_amp=False,
    verbose_interval=50,
    grad_clip_norm=5.0,
):
    model.train()
    total_sse = 0.0
    total_count = 0

    for batch_idx, batch in enumerate(
        tqdm(
            data_loader,
            desc="Train",
            ncols=120,
            leave=False,
            disable=True,
        ),
        start=1,
    ):
        a, b, c, a_mask, b_mask, c_mask, targets = batch
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        a_mask = a_mask.to(device, non_blocking=True)
        b_mask = b_mask.to(device, non_blocking=True)
        c_mask = c_mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = use_amp and scaler is not None

        if amp_enabled:
            with torch.cuda.amp.autocast():
                outputs = model(
                    a, b, c,
                    a_mask=a_mask,
                    b_mask=b_mask,
                    c_mask=c_mask,
                )
                outputs, targets = align_pred_target(outputs, targets)
                batch_loss = F.mse_loss(outputs, targets, reduction="mean")
                batch_sse = F.mse_loss(outputs, targets, reduction="sum")

            scaler.scale(batch_loss).backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_norm,
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                a, b, c,
                a_mask=a_mask,
                b_mask=b_mask,
                c_mask=c_mask,
            )
            outputs, targets = align_pred_target(outputs, targets)
            batch_loss = F.mse_loss(outputs, targets, reduction="mean")
            batch_sse = F.mse_loss(outputs, targets, reduction="sum")
            batch_loss.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_norm,
                )
            optimizer.step()

        total_sse += float(batch_sse.item())
        total_count += int(targets.numel())

        if verbose_interval > 0 and batch_idx % verbose_interval == 0:
            log(
                f"[Train] batch {batch_idx}/{len(data_loader)} | "
                f"current_batch_loss={batch_loss.item():.6f}"
            )

    return total_sse / max(1, total_count)


@torch.inference_mode()
def evaluate_model(
    model,
    data_loader,
    device,
    target_mean=0.0,
    target_std=1.0,
):
    model.eval()
    total_sse = 0.0
    total_count = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(
        data_loader,
        desc="Evaluate",
        ncols=120,
        leave=False,
    ):
        a, b, c, a_mask, b_mask, c_mask, targets = batch
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        a_mask = a_mask.to(device, non_blocking=True)
        b_mask = b_mask.to(device, non_blocking=True)
        c_mask = c_mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(
            a, b, c,
            a_mask=a_mask,
            b_mask=b_mask,
            c_mask=c_mask,
        )
        outputs, targets = align_pred_target(outputs, targets)
        batch_sse = F.mse_loss(outputs, targets, reduction="sum")
        total_sse += float(batch_sse.item())
        total_count += int(targets.numel())
        all_preds.extend(outputs.detach().cpu().reshape(-1).tolist())
        all_targets.extend(targets.detach().cpu().reshape(-1).tolist())

    if not all_preds:
        raise ValueError("evaluate_model produced no predictions.")

    normalized_preds = np.array(all_preds, dtype=np.float64)
    normalized_targets = np.array(all_targets, dtype=np.float64)
    original_preds = normalized_preds * float(target_std) + float(target_mean)
    original_targets = normalized_targets * float(target_std) + float(target_mean)

    metrics = compute_metrics(original_targets, original_preds)
    metrics["loss"] = float(total_sse / max(1, total_count))
    return metrics, original_targets.tolist(), original_preds.tolist()


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay,
    save_dir,
    train_config,
    model_config,
    target_mean,
    target_std,
    verbose_interval=50,
    patience=12,
    min_delta=1e-4,
    grad_clip_norm=5.0,
    resume=True,
    use_amp=False,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-3,
        min_lr=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(use_amp and device.type == "cuda")
    )
    best_path = os.path.join(save_dir, "best_model.pt")
    last_path = os.path.join(save_dir, "last_model.pt")

    start_epoch = 1
    best_val_loss = float("inf")
    bad_epochs = 0

    log("Initializing training")
    log(
        f"lr={lr}, weight_decay={weight_decay}, epochs={epochs}, "
        f"patience={patience}, use_amp={use_amp}"
    )

    if resume and os.path.exists(last_path):
        log("Found last checkpoint; resuming training.")
        start_epoch, best_val_loss, bad_epochs = load_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            strict=True,
            restore_rng=True,
        )

    if start_epoch > epochs:
        log(f"start_epoch={start_epoch} > epochs={epochs}, skipping training.")
        return best_path, best_val_loss

    for epoch in range(start_epoch, epochs + 1):
        log(f"\n Epoch {epoch}/{epochs} start ")
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            verbose_interval=verbose_interval,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics, _, _ = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start_time

        log(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.6f} | "
            f"val_loss_normalized={val_loss:.6f} | "
            f"{metrics_str(val_metrics)} | "
            f"lr={current_lr:.8f} | "
            f"time={epoch_time:.1f}s"
        )

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        save_checkpoint(
            path=last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_loss=best_val_loss,
            bad_epochs=bad_epochs,
            train_loss=train_loss,
            val_metrics=val_metrics,
            train_config=train_config,
            model_config=model_config,
        )

        if improved:
            save_checkpoint(
                path=best_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
                bad_epochs=bad_epochs,
                train_loss=train_loss,
                val_metrics=val_metrics,
                train_config=train_config,
                model_config=model_config,
            )
            log(
                f"Saved new best checkpoint -> {best_path} "
                f"(best_val_loss={best_val_loss:.6f})"
            )
        else:
            log(f"val_loss did not improve, bad_epochs={bad_epochs}/{patience}")

        log(f" Epoch {epoch}/{epochs} end ")

        if bad_epochs >= patience:
            log("Early stopping triggered.")
            break

    return best_path, best_val_loss