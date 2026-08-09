"""DINOv2 ViT-S/14 with linear classification head.

Architecture::

    Input (3 x 224 x 224)
          │
          ▼
    ┌─────────────────────┐
    │  DINOv2 ViT-S/14    │  Pre-trained, frozen in Phase 1
    │  (12 blocks)        │  embed_dim = 384
    └────────┬────────────┘
             │ 384-dim CLS token
             ▼
    ┌─────────────────────┐
    │  Classification Head│  Always trainable
    │  LayerNorm(384)     │
    │  Linear(384 → 128)  │
    │  GELU               │
    │  Dropout(0.3)       │
    │  Linear(128 → 7)    │
    └────────┬────────────┘
             │ 7 logits
             ▼
        Prediction
"""

from __future__ import annotations

import torch

from torch import nn

from training.config import TrainingConfig


def build_model(cfg: TrainingConfig) -> nn.Module:
    """Build DINOv2 ViT-S/14 with classification head.

    Parameters
    ----------
    cfg : TrainingConfig
        Must include ``backbone``, ``embed_dim``,
        ``num_classes``, and ``freeze_backbone``.

    Returns
    -------
    nn.Module
        Ready-to-train model.
    """
    backbone: nn.Module = torch.hub.load(
        'facebookresearch/dinov2',
        cfg.backbone,
        pretrained=True,
    )

    if cfg.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    # Replace the identity head with our classification head.
    backbone.head = nn.Sequential(  # type: ignore[attr-defined]
        nn.LayerNorm(cfg.embed_dim),
        nn.Linear(cfg.embed_dim, 128),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, cfg.num_classes),
    )

    # Ensure head is always trainable.
    for param in backbone.head.parameters():  # type: ignore[attr-defined]
        param.requires_grad = True

    return backbone


def unfreeze_last_n_blocks(
    model: nn.Module,
    n: int,
    finetune_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Unfreeze the last *n* transformer blocks for Phase 2.

    Creates a new optimizer with differential learning rates:
    - Unfrozen backbone blocks: ``finetune_lr``
    - Classification head: ``finetune_lr * 10``

    Parameters
    ----------
    model : nn.Module
        The DINOv2 model with classification head.
    n : int
        Number of blocks to unfreeze from the end.
    finetune_lr : float
        Base learning rate for unfrozen blocks.
    weight_decay : float
        AdamW weight decay.

    Returns
    -------
    torch.optim.Optimizer
        New AdamW optimizer with parameter groups.
    """
    # DINOv2 stores blocks in model.blocks (nn.ModuleList).
    blocks = model.blocks  # type: ignore[attr-defined]
    total_blocks = len(blocks)

    # Unfreeze last n blocks.
    for block in blocks[total_blocks - n :]:
        for param in block.parameters():
            param.requires_grad = True

    # Also unfreeze the final LayerNorm of the backbone.
    if hasattr(model, 'norm'):
        for param in model.norm.parameters():  # type: ignore[union-attr]
            param.requires_grad = True

    # Build parameter groups with differential LR.
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {
            'params': backbone_params,
            'lr': finetune_lr,
        },
        {
            'params': head_params,
            'lr': finetune_lr * 10,
        },
    ]

    return torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """Count trainable and total parameters.

    Returns
    -------
    tuple[int, int]
        ``(trainable, total)``
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
