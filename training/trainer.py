"""Training loop with Focal Loss, cosine LR, early stopping, and W&B.

This module provides the core training engine:
- ``FocalLoss``: handles class imbalance by down-weighting
  easy examples
- ``train_one_epoch`` / ``validate``: single-epoch forward/
  backward passes
- ``train_fold``: full two-phase training for one CV fold
  (Phase 1 = linear probe, Phase 2 = fine-tune)
"""

from __future__ import annotations

import csv
import math
import time

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from training.config import TrainingConfig

# Optional: wandb import guarded.
try:
    import wandb  # type: ignore[import-untyped]

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ── Focal Loss ───────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification.

    Reduces the loss contribution from easy examples so that
    the model focuses on hard, misclassified samples.  This is
    critical for our 58:1 class imbalance.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  0 = standard cross-entropy.
        2.0 is the typical default.
    weight : torch.Tensor | None
        Per-class weights (from EDA class_weights).
    label_smoothing : float
        Label smoothing factor.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # Use CrossEntropyLoss internally for label smoothing.
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            reduction='none',
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model outputs, shape ``(B, C)``.
        targets : torch.Tensor
            Ground-truth class indices, shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ── Learning Rate Scheduler ─────────────────────────────────


def _cosine_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
) -> float:
    """Apply cosine LR schedule with linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    epoch : int
        Current epoch (0-indexed).
    total_epochs : int
    warmup_epochs : int
    base_lr : float

    Returns
    -------
    float
        Current learning rate.
    """
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(
            total_epochs - warmup_epochs,
            1,
        )
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ── Single Epoch Routines ───────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns
    -------
    tuple[float, float]
        ``(average_loss, accuracy)``
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predicted = logits.argmax(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, npt.NDArray[Any], npt.NDArray[Any]]:
    """Run validation pass.

    Returns
    -------
    tuple[float, float, np.ndarray, np.ndarray]
        ``(avg_loss, balanced_accuracy, predictions,
        true_labels)``
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        predicted = logits.argmax(dim=1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    total = max(len(all_labels), 1)
    avg_loss = total_loss / total
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return (
        avg_loss,
        float(bal_acc),
        np.array(all_preds),
        np.array(all_labels),
    )


# ── Per-fold Training ───────────────────────────────────────


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    cfg: TrainingConfig,
    fold: int,
    device: torch.device,
) -> tuple[float, Path]:
    """Train one fold through both phases.

    Phase 1: Linear probing (frozen backbone).
    Phase 2: Fine-tuning (last N blocks unfrozen).

    Parameters
    ----------
    model : nn.Module
        Fresh model (frozen backbone + trainable head).
    train_loader : DataLoader
    val_loader : DataLoader
    cfg : TrainingConfig
    fold : int
        Current fold number (for logging).
    device : torch.device

    Returns
    -------
    tuple[float, Path]
        ``(best_balanced_accuracy, path_to_best_checkpoint)``
    """
    from training.model import unfreeze_last_n_blocks

    fold_dir = cfg.output_dir / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Build focal loss with class weights.
    class_weights_tensor = torch.tensor(
        cfg.class_weights,
        dtype=torch.float32,
    ).to(device)
    criterion = FocalLoss(
        gamma=cfg.focal_gamma,
        weight=class_weights_tensor,
        label_smoothing=cfg.label_smoothing,
    )

    # CSV log for this fold.
    log_path = fold_dir / 'training_log.csv'
    log_fields = [
        'epoch',
        'phase',
        'train_loss',
        'train_acc',
        'val_loss',
        'val_bal_acc',
        'lr',
    ]
    log_fh = log_path.open('w', newline='', encoding='utf-8')
    log_writer = csv.DictWriter(log_fh, fieldnames=log_fields)
    log_writer.writeheader()

    best_bal_acc = 0.0
    best_ckpt_path = fold_dir / 'best_model.pth'
    epochs_without_improvement = 0

    def _run_phase(
        phase_name: str,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        base_lr: float,
    ) -> None:
        nonlocal best_bal_acc, epochs_without_improvement

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # LR schedule.
            lr = _cosine_lr_with_warmup(
                optimizer,
                epoch,
                num_epochs,
                cfg.warmup_epochs,
                base_lr,
            )

            # Train.
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            # Validate.
            val_loss, val_bal_acc, _val_preds, _val_true = validate(
                model, val_loader, criterion, device
            )

            epoch_time = time.time() - epoch_start

            # Log.
            row = {
                'epoch': epoch,
                'phase': phase_name,
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{train_acc:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_bal_acc': f'{val_bal_acc:.4f}',
                'lr': f'{lr:.6f}',
            }
            log_writer.writerow(row)
            log_fh.flush()

            print(
                f'  [{phase_name}] Epoch {epoch + 1:2d}/'
                f'{num_epochs} | '
                f'Train Loss: {train_loss:.4f} | '
                f'Val Loss: {val_loss:.4f} | '
                f'Val Bal Acc: {val_bal_acc * 100:.1f}% | '
                f'LR: {lr:.6f} | '
                f'{epoch_time:.0f}s'
            )

            # W&B logging.
            if cfg.use_wandb and _HAS_WANDB:
                wandb.log(
                    {
                        f'fold_{fold}/{phase_name}/train_loss': (train_loss),
                        f'fold_{fold}/{phase_name}/val_loss': (val_loss),
                        f'fold_{fold}/{phase_name}/val_bal_acc': (val_bal_acc),
                        f'fold_{fold}/{phase_name}/lr': lr,
                    }
                )

            # Checkpointing.
            if val_bal_acc > best_bal_acc:
                best_bal_acc = val_bal_acc
                torch.save(model.state_dict(), best_ckpt_path)
                epochs_without_improvement = 0
                print(
                    f'    ✓ New best: {val_bal_acc * 100:.1f}%'
                    f' → saved to {best_ckpt_path.name}'
                )
            else:
                epochs_without_improvement += 1

            # Early stopping.
            if epochs_without_improvement >= cfg.patience:
                print(
                    f'    ⏹ Early stopping after '
                    f'{cfg.patience} epochs without '
                    f'improvement'
                )
                break

    # ── Phase 1: Linear Probing ──────────────────────────
    print(f'\n── Fold {fold} | Phase 1: Linear Probing ──')
    optimizer_p1 = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    _run_phase(
        'phase1',
        optimizer_p1,
        cfg.num_epochs_phase1,
        cfg.learning_rate,
    )

    # ── Phase 2: Fine-tuning ─────────────────────────────
    if cfg.unfreeze_last_n > 0:
        print(
            f'\n── Fold {fold} | Phase 2: Fine-tuning '
            f'(last {cfg.unfreeze_last_n} blocks) ──'
        )
        epochs_without_improvement = 0  # Reset for Phase 2.
        optimizer_p2 = unfreeze_last_n_blocks(
            model,
            cfg.unfreeze_last_n,
            cfg.finetune_lr,
            cfg.weight_decay,
        )
        _run_phase(
            'phase2',
            optimizer_p2,
            cfg.num_epochs_phase2,
            cfg.finetune_lr,
        )

    log_fh.close()
    print(
        f'\n✅ Fold {fold} complete — Best Bal Acc: {best_bal_acc * 100:.1f}%'
    )
    return best_bal_acc, best_ckpt_path
