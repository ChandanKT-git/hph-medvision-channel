"""Skin lesion classification skill for HiperHealth.

Uses a fine-tuned DINOv2 ViT-S/14 backbone to classify
dermatoscopic images into 7 ISIC 2018 lesion categories,
with Grad-CAM visual explanations and temperature-scaled
confidence calibration.
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any

import torch

from torch import nn

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────

_SKILL_NAME = 'medvision.skin_analysis'

_LABELS_FILE = Path(__file__).parent / 'labels.json'

_MODEL_ID = 'dinov2_vits14_skin_v1'

_MODEL_URL = (
    'https://github.com/ChandanKT-git/'
    'hph-medvision-channel/releases/download/'
    'v0.1.0-models/dinov2_vits14_skin.pth'
)

_MODEL_SHA256 = (
    '7d39f05a39cf36873751cc1ef7bf7b7c2ef33eb5dcd439f3ec9b6351eaaa203c'
)

_NUM_CLASSES = 7
_EMBED_DIM = 384

_FITTED_TEMPERATURE = 1.3495

_DEFAULT_CONFIDENCE_THRESHOLD = 0.7

_DANGEROUS_CLASSES = frozenset(
    {
        'melanoma',
        'basal_cell_carcinoma',
    }
)

_DANGEROUS_THRESHOLD = 0.1


# ── Helper Functions ───────────────────────────────────


def _load_label_map() -> dict[str, Any]:
    """Load the SNOMED CT label mappings from labels.json.

    Returns
    -------
    dict[str, Any]
        Parsed JSON with ``'classes'`` list.  Each entry
        has ``'index'``, ``'name'``, ``'display_name'``,
        ``'snomed_code'``, ``'snomed_display'``.
    """
    with _LABELS_FILE.open('r', encoding='utf-8') as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _build_classification_head(
    embed_dim: int,
    num_classes: int,
) -> nn.Sequential:
    """Reconstruct the classification head architecture.

    Must match the architecture in ``training/model.py``
    exactly, or ``load_state_dict`` will fail with a key
    mismatch.

    Parameters
    ----------
    embed_dim : int
        Input dimension from the backbone (384 for ViT-S/14).
    num_classes : int
        Number of output classes (7 for ISIC 2018).

    Returns
    -------
    nn.Sequential
        The classification head module.
    """
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, 128),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes),
    )


def _load_model(weights_path: Path) -> nn.Module:
    """Load the DINOv2 backbone with trained head weights.

    Parameters
    ----------
    weights_path : Path
        Path to the saved ``state_dict`` (``.pth`` file).

    Returns
    -------
    nn.Module
        Model in eval mode, ready for inference.
    """
    model: nn.Module = torch.hub.load(  # type: ignore[no-untyped-call]
        'facebookresearch/dinov2',
        'dinov2_vits14',
        pretrained=True,
    )

    model.head = _build_classification_head(
        _EMBED_DIM,
        _NUM_CLASSES,
    )

    state_dict = torch.load(
        weights_path,
        map_location='cpu',
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model
