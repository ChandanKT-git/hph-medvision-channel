"""Training configuration for DINOv2 skin lesion classifier.

All hyperparameters live here so experiments only require
changing one file.  Command-line arguments override defaults
via ``TrainingConfig.from_args()``.
"""

from __future__ import annotations

import json

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Central configuration for the training pipeline.

    Parameters
    ----------
    data_dir : Path
        Root directory containing HAM10000 images + metadata.
    output_dir : Path
        Where to write checkpoints, logs, and evaluation
        artefacts.
    num_classes : int
        Number of target classes (7 for ISIC).
    backbone : str
        ``torch.hub`` model name for DINOv2.
    embed_dim : int
        Embedding dimension of the backbone (384 for ViT-S).
    freeze_backbone : bool
        If True, freeze all backbone parameters during Phase 1
        (linear probing).
    unfreeze_last_n : int
        Number of transformer blocks to unfreeze in Phase 2
        (fine-tuning).  0 means no fine-tuning phase.
    batch_size : int
        Mini-batch size.  Reduce to 16 if OOM on Colab free.
    num_epochs_phase1 : int
        Epochs for Phase 1 (linear probing, frozen backbone).
    num_epochs_phase2 : int
        Epochs for Phase 2 (fine-tuning last N blocks).
    learning_rate : float
        Initial LR for Phase 1 (AdamW).
    finetune_lr : float
        LR for Phase 2 (much smaller to avoid catastrophic
        forgetting).
    weight_decay : float
        AdamW weight decay.
    label_smoothing : float
        Label smoothing factor for the loss function.
    focal_gamma : float
        Gamma for focal loss.  0 = plain cross-entropy.
    warmup_epochs : int
        Number of linear-warmup epochs before cosine decay.
    patience : int
        Early-stopping patience (epochs without improvement).
    n_folds : int
        Number of CV folds.
    dev_mode : bool
        If True, use a single 80/20 split instead of K-fold.
    target_size : int
        Spatial size for model input (224 for DINOv2).
    num_workers : int
        DataLoader worker processes.
    use_weighted_sampler : bool
        Use ``WeightedRandomSampler`` to counter class
        imbalance.
    use_wandb : bool
        Enable Weights & Biases logging.
    wandb_project : str
        W&B project name.
    seed : int
        Global random seed for reproducibility.
    class_names : list[str]
        Ordered class abbreviations matching label indices.
    class_full_names : list[str]
        Full human-readable class names.
    """

    # ── Paths ────────────────────────────────────────────
    data_dir: Path = field(
        default_factory=lambda: Path('./training/data'),
    )
    output_dir: Path = field(
        default_factory=lambda: Path('./outputs'),
    )

    # ── Model Architecture ───────────────────────────────
    num_classes: int = 7
    backbone: str = 'dinov2_vits14'
    embed_dim: int = 384
    freeze_backbone: bool = True
    unfreeze_last_n: int = 4

    # ── Training Hyperparameters ─────────────────────────
    batch_size: int = 32
    num_epochs_phase1: int = 15
    num_epochs_phase2: int = 25
    learning_rate: float = 1e-3
    finetune_lr: float = 1e-5
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    focal_gamma: float = 1.0
    warmup_epochs: int = 3
    patience: int = 10
    use_mixup: bool = True
    mixup_alpha: float = 0.4

    # ── Cross-Validation ─────────────────────────────────
    n_folds: int = 5
    dev_mode: bool = False
    # Truncates dataset to tiny subset for testing pipeline
    fast_dev_mode: bool = False

    # ── Data Handling ────────────────────────────────────
    target_size: int = 224
    num_workers: int = 2
    use_weighted_sampler: bool = True
    tta_runs: int = 5  # Test-time augmentation passes

    # ── Experiment Tracking ──────────────────────────────
    use_wandb: bool = True
    wandb_project: str = 'medvision-skin'

    # ── Reproducibility ──────────────────────────────────
    seed: int = 42

    # ── Class Mapping ────────────────────────────────────
    # Index order matches the label integers (0-6).
    class_names: list[str] = field(
        default_factory=lambda: [
            'mel',
            'nv',
            'bcc',
            'akiec',
            'bkl',
            'df',
            'vasc',
        ],
    )
    class_full_names: list[str] = field(
        default_factory=lambda: [
            'melanoma',
            'melanocytic_nevi',
            'basal_cell_carcinoma',
            'actinic_keratosis',
            'benign_keratosis',
            'dermatofibroma',
            'vascular_lesion',
        ],
    )

    # ── Class Weights (from EDA) ─────────────────────────
    class_weights: list[float] = field(
        default_factory=lambda: [
            1.134,  # mel
            0.462,  # nv
            1.668,  # bcc
            2.092,  # akiec
            1.141,  # bkl
            3.527,  # df
            3.174,  # vasc
        ],
    )

    def save(self, path: Path | None = None) -> Path:
        """Serialise config to JSON for reproducibility.

        Parameters
        ----------
        path : Path | None
            Destination file.  Defaults to
            ``self.output_dir / 'training_config.json'``.

        Returns
        -------
        Path
            The written file.
        """
        if path is None:
            path = self.output_dir / 'training_config.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # Convert Path objects to strings for JSON.
        for key in ('data_dir', 'output_dir'):
            data[key] = str(data[key])
        with path.open('w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        return path

    @classmethod
    def from_json(cls, path: Path) -> TrainingConfig:
        """Load config from a JSON file.

        Parameters
        ----------
        path : Path
            Path to a ``training_config.json``.

        Returns
        -------
        TrainingConfig
        """
        with path.open(encoding='utf-8') as fh:
            data = json.load(fh)
        data['data_dir'] = Path(data['data_dir'])
        data['output_dir'] = Path(data['output_dir'])
        return cls(**data)
