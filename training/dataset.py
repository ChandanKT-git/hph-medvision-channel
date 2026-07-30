"""ISIC / HAM10000 dataset loading with MONAI transforms.

Handles:
- Auto-detection of HAM10000 metadata CSV format
- Image loading from split directories (part_1 + part_2)
- MONAI-based training augmentation & validation transforms
- ``StratifiedGroupKFold`` splitting (grouped by ``lesion_id``
  to prevent data leakage from duplicate lesion images)
- ``WeightedRandomSampler`` for class imbalance
"""

from __future__ import annotations

import csv

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from monai.transforms import (  # type: ignore[attr-defined]
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    RandAdjustContrast,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandRotate,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from training.config import TrainingConfig

# ── Constants ────────────────────────────────────────────────

# ImageNet normalisation — must match
# shared/preprocessing.py exactly.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# HAM10000 metadata dx column → label index mapping.
# Index order matches ``TrainingConfig.class_names``.
_DX_TO_IDX: dict[str, int] = {
    'mel': 0,
    'nv': 1,
    'bcc': 2,
    'akiec': 3,
    'bkl': 4,
    'df': 5,
    'vasc': 6,
}


# ── Dataset Class ────────────────────────────────────────────


class ISICDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch dataset for ISIC / HAM10000 images.

    Parameters
    ----------
    image_paths : list[Path]
        Absolute paths to ``.jpg`` image files.
    labels : np.ndarray
        Integer labels (0-6) aligned with *image_paths*.
    transform : Compose | None
        MONAI transform pipeline.
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: npt.NDArray[Any],
        transform: Compose | None = None,
    ) -> None:
        self._paths = image_paths
        self._labels = labels
        self._transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, int]:
        bgr = cv2.imread(str(self._paths[idx]), cv2.IMREAD_COLOR)
        if bgr is None:
            msg = f'Failed to read image: {self._paths[idx]}'
            raise RuntimeError(msg)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img: npt.NDArray[Any] = rgb.astype(np.float32)

        if self._transform is not None:
            img = self._transform(img)  # type: ignore[assignment]

        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)  # type: ignore[assignment]

        return img, int(self._labels[idx])  # type: ignore[return-value]


# ── Data Loading ─────────────────────────────────────────────


def load_ham10000(
    data_dir: Path,
) -> tuple[list[Path], npt.NDArray[Any], npt.NDArray[Any]]:
    """Load HAM10000 image paths, labels, and lesion IDs.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``HAM10000_metadata``
        and image part directories.

    Returns
    -------
    tuple[list[Path], np.ndarray, np.ndarray]
        ``(image_paths, labels, lesion_ids)``
        where ``lesion_ids`` is an integer-encoded array for
        use with ``StratifiedGroupKFold``.

    Raises
    ------
    FileNotFoundError
        If the metadata CSV is not found.
    """
    # Locate metadata CSV.
    csv_path = data_dir / 'HAM10000_metadata'
    if not csv_path.exists():
        csv_path = data_dir / 'HAM10000_metadata.csv'
    if not csv_path.exists():
        # Try nested
        for match in data_dir.rglob('HAM10000_metadata*'):
            if match.is_file():
                csv_path = match
                break
    if not csv_path.exists():
        msg = (
            f'HAM10000 metadata not found in {data_dir}. '
            f'Expected HAM10000_metadata or '
            f'HAM10000_metadata.csv'
        )
        raise FileNotFoundError(msg)

    # Collect image directories.
    image_dirs: list[Path] = []
    for name in (
        'HAM10000_images_part_1',
        'HAM10000_images_part_2',
    ):
        for match in sorted(data_dir.rglob(name)):
            if match.is_dir():
                image_dirs.append(match)

    if not image_dirs:
        msg = f'No image directories found in {data_dir}'
        raise FileNotFoundError(msg)

    # Parse CSV.
    paths: list[Path] = []
    labels: list[int] = []
    lesion_strs: list[str] = []

    with csv_path.open(encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_id = row['image_id']
            dx = row['dx'].strip().lower()

            if dx not in _DX_TO_IDX:
                continue

            # Search image directories.
            img_path: Path | None = None
            for img_dir in image_dirs:
                candidate = img_dir / f'{image_id}.jpg'
                if candidate.is_file():
                    img_path = candidate
                    break

            if img_path is not None:
                paths.append(img_path)
                labels.append(_DX_TO_IDX[dx])
                lesion_strs.append(row['lesion_id'])

    # Encode lesion IDs as integers for sklearn.
    unique_lesions = sorted(set(lesion_strs))
    lesion_map = {lid: i for i, lid in enumerate(unique_lesions)}
    lesion_ids = np.array(
        [lesion_map[lid] for lid in lesion_strs],
        dtype=np.int64,
    )

    print(f'[data] Loaded {len(paths)} images from {csv_path}')
    print(f'[data] {len(unique_lesions)} unique lesions')
    print(f'[data] Image dirs: {image_dirs}')

    # Print class distribution.
    counts = Counter(labels)
    for cls_idx in range(len(_DX_TO_IDX)):
        name = list(_DX_TO_IDX.keys())[cls_idx]
        count = counts.get(cls_idx, 0)
        pct = count / len(labels) * 100
        print(f'  [{cls_idx}] {name:8s}: {count:5d} ({pct:5.1f}%)')

    return (
        paths,
        np.array(labels, dtype=np.int64),
        lesion_ids,
    )


# ── MONAI Transform Pipelines ───────────────────────────────


def build_train_transforms(size: int = 224) -> Compose:
    """MONAI training transforms with medical augmentation.

    Augmentation pipeline:
    1. ``Resize(256)`` — slightly larger for random cropping
    2. ``RandSpatialCrop(224)`` — random crop to target size
    3. ``RandFlip`` — horizontal + vertical (orientation
       invariance)
    4. ``RandRotate(±15°)`` — small rotations
    5. ``RandGaussianSmooth`` — simulates out-of-focus
       dermatoscope
    6. ``RandAdjustContrast`` — simulates lighting variance
    7. ``RandGaussianNoise`` — simulates sensor noise
    8. ``ScaleIntensity`` + ``NormalizeIntensity`` — ImageNet
       normalisation

    Parameters
    ----------
    size : int
        Output spatial size.

    Returns
    -------
    monai.transforms.Compose
    """
    return Compose(
        [
            EnsureChannelFirst(channel_dim=-1),
            Resize(spatial_size=(256, 256)),
            RandSpatialCrop(
                roi_size=(size, size),
                random_size=False,
            ),
            RandFlip(spatial_axis=0, prob=0.5),
            RandFlip(spatial_axis=1, prob=0.5),
            RandRotate(
                range_x=0.26,  # ~15 degrees
                prob=0.5,
                padding_mode='zeros',
            ),
            RandGaussianSmooth(
                sigma_x=(0.1, 0.6),
                sigma_y=(0.1, 0.6),
                prob=0.2,
            ),
            RandAdjustContrast(gamma=(0.8, 1.2), prob=0.2),
            RandGaussianNoise(mean=0.0, std=0.02, prob=0.15),
            ScaleIntensity(),
            NormalizeIntensity(
                subtrahend=_IMAGENET_MEAN,
                divisor=_IMAGENET_STD,
                channel_wise=True,
            ),
        ]
    )


def build_val_transforms(size: int = 224) -> Compose:
    """MONAI validation transforms (no augmentation).

    Identical to ``shared.preprocessing.build_inference_transforms``
    so that training validation and runtime inference see
    exactly the same inputs.

    Parameters
    ----------
    size : int
        Output spatial size.

    Returns
    -------
    monai.transforms.Compose
    """
    return Compose(
        [
            EnsureChannelFirst(channel_dim=-1),
            Resize(spatial_size=(size, size)),
            ScaleIntensity(),
            NormalizeIntensity(
                subtrahend=_IMAGENET_MEAN,
                divisor=_IMAGENET_STD,
                channel_wise=True,
            ),
        ]
    )


# ── Cross-Validation Splits ─────────────────────────────────


def create_fold_splits(
    labels: npt.NDArray[Any],
    lesion_ids: npt.NDArray[Any],
    cfg: TrainingConfig,
) -> list[tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
    """Create train/val index splits.

    Uses ``StratifiedGroupKFold`` (grouped by ``lesion_id``)
    to prevent data leakage from duplicate lesion images.

    In dev mode, returns a single 80/20 stratified split.

    Parameters
    ----------
    labels : np.ndarray
        Integer class labels.
    lesion_ids : np.ndarray
        Integer-encoded lesion IDs for grouping.
    cfg : TrainingConfig
        Configuration (n_folds, dev_mode, seed).

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of ``(train_indices, val_indices)`` tuples.
    """
    if cfg.dev_mode:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=cfg.seed,
        )
        indices = np.arange(len(labels))
        splits = list(sss.split(indices, labels))
        print(
            f'[split] Dev mode: single 80/20 split '
            f'(train={len(splits[0][0])}, '
            f'val={len(splits[0][1])})'
        )
        return splits

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed,
    )
    indices = np.arange(len(labels))
    splits = list(sgkf.split(indices, labels, groups=lesion_ids))
    for i, (train_idx, val_idx) in enumerate(splits):
        print(f'[split] Fold {i}: train={len(train_idx)}, val={len(val_idx)}')
    return splits


# ── DataLoader Factory ───────────────────────────────────────


def create_dataloaders(
    image_paths: list[Path],
    labels: npt.NDArray[Any],
    train_idx: npt.NDArray[Any],
    val_idx: npt.NDArray[Any],
    cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    """Build train and validation DataLoaders for one fold.

    Parameters
    ----------
    image_paths : list[Path]
        All image paths.
    labels : np.ndarray
        All integer labels.
    train_idx : np.ndarray
        Training set indices.
    val_idx : np.ndarray
        Validation set indices.
    cfg : TrainingConfig
        Configuration.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        ``(train_loader, val_loader)``
    """
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = labels[val_idx]

    train_ds = ISICDataset(
        train_paths,
        train_labels,
        build_train_transforms(cfg.target_size),
    )
    val_ds = ISICDataset(
        val_paths,
        val_labels,
        build_val_transforms(cfg.target_size),
    )

    # Weighted sampler for class imbalance.
    sampler = None
    shuffle = True
    if cfg.use_weighted_sampler:
        class_counts = np.bincount(
            train_labels,
            minlength=cfg.num_classes,
        )
        weights = 1.0 / (class_counts.astype(float) + 1e-6)
        sample_weights = weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # Sampler handles ordering.

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
