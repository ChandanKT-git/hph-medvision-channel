"""Main entry point for DINOv2 skin lesion training.

Usage
-----
::

    # Dev mode (single 80/20 split, fast):
    python -m training.train --data-dir ./training/data --dev

    # Full 5-fold CV:
    python -m training.train --data-dir ./training/data

    # Custom settings:
    python -m training.train \\
        --data-dir ./training/data \\
        --output-dir ./outputs \\
        --batch-size 16 \\
        --num-epochs-phase1 10 \\
        --no-wandb

Colab Quick Start
-----------------
::

    # Cell 1: Clone & install
    !git clone -b model-training \\
        https://github.com/ChandanKT-git/hph-medvision-channel.git
    %cd hph-medvision-channel
    !pip install -e . -q

    # Cell 2: Mount Google Drive (where your data is stored)
    from google.colab import drive
    drive.mount('/content/drive')

    # Cell 3: Fast Dev Mode (1 epoch, 64 images - finishes in seconds)
    !python -m training.train \\
        --data-dir "/content/drive/MyDrive/path/to/your/data" \\
        --fast-dev --no-wandb

    # Cell 4: Train (dev mode - full dataset, 1 split)
    !python -m training.train \\
        --data-dir "/content/drive/MyDrive/path/to/your/data" \\
        --dev --no-wandb

    # Cell 5: Full 5-fold (when you're ready for the final run)
    !python -m training.train \\
        --data-dir "/content/drive/MyDrive/path/to/your/data"
"""

from __future__ import annotations

import argparse
import time

from pathlib import Path

import numpy as np
import torch

from training.config import TrainingConfig
from training.dataset import (
    create_dataloaders,
    create_fold_splits,
    load_ham10000,
)
from training.evaluate import (
    aggregate_fold_metrics,
    collect_predictions,
    compute_metrics,
    print_summary,
    save_evaluation_report,
)
from training.model import build_model, count_parameters
from training.trainer import train_fold

# Optional wandb.
try:
    import wandb  # type: ignore[import-untyped]

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=('Train DINOv2 ViT-S/14 on ISIC skin lesion dataset'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./training/data',
        help='Root data directory with HAM10000',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Where to save checkpoints and reports',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (reduce to 16 if OOM)',
    )
    parser.add_argument(
        '--num-epochs-phase1',
        type=int,
        default=15,
        help='Epochs for Phase 1 (linear probing)',
    )
    parser.add_argument(
        '--num-epochs-phase2',
        type=int,
        default=25,
        help='Epochs for Phase 2 (fine-tuning)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate for Phase 1',
    )
    parser.add_argument(
        '--finetune-lr',
        type=float,
        default=1e-5,
        help='Learning rate for Phase 2',
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of CV folds',
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Dev mode: single 80/20 split',
    )
    parser.add_argument(
        '--fast-dev',
        action='store_true',
        help='Fast dev mode: 1 epoch, tiny dataset',
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='DataLoader workers',
    )
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()

    # Build config from CLI args.
    cfg = TrainingConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        num_epochs_phase1=1 if args.fast_dev else args.num_epochs_phase1,
        num_epochs_phase2=1 if args.fast_dev else args.num_epochs_phase2,
        learning_rate=args.lr,
        finetune_lr=args.finetune_lr,
        n_folds=args.n_folds,
        dev_mode=args.dev or args.fast_dev,
        fast_dev_mode=args.fast_dev,
        use_wandb=not args.no_wandb,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # ── Setup ────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu',
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  DINOv2 ViT-S/14 — ISIC Skin Lesion Training')
    print('=' * 60)
    print(f'Device:      {device}')
    if device.type == 'cuda':
        print(f'GPU:         {torch.cuda.get_device_name(0)}')
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'VRAM:        {vram:.1f} GB')
    mode = 'Dev (single split)' if cfg.dev_mode else (f'{cfg.n_folds}-fold CV')
    print(f'Mode:        {mode}')
    print(f'Batch size:  {cfg.batch_size}')
    print(
        f'Phase 1:     {cfg.num_epochs_phase1} epochs (LR={cfg.learning_rate})'
    )
    print(
        f'Phase 2:     {cfg.num_epochs_phase2} epochs '
        f'(LR={cfg.finetune_lr}, '
        f'unfreeze last {cfg.unfreeze_last_n} blocks)'
    )
    print()

    # Save config for reproducibility.
    cfg.save()
    print(f'Config saved to {cfg.output_dir}/training_config.json')

    # ── W&B Init ─────────────────────────────────────────
    if cfg.use_wandb and _HAS_WANDB:
        wandb.init(
            project=cfg.wandb_project,
            config={
                'backbone': cfg.backbone,
                'batch_size': cfg.batch_size,
                'lr_phase1': cfg.learning_rate,
                'lr_phase2': cfg.finetune_lr,
                'focal_gamma': cfg.focal_gamma,
                'label_smoothing': cfg.label_smoothing,
                'n_folds': cfg.n_folds,
                'dev_mode': cfg.dev_mode,
                'unfreeze_last_n': cfg.unfreeze_last_n,
            },
            name=f'dinov2-isic-{"dev" if cfg.dev_mode else "5fold"}',
        )
    elif cfg.use_wandb and not _HAS_WANDB:
        print('[warn] wandb not installed. Continuing without tracking.')
        cfg.use_wandb = False

    # ── Load Data ────────────────────────────────────────
    print('\nLoading dataset...')
    image_paths, labels, lesion_ids = load_ham10000(
        cfg.data_dir,
    )
    print(f'Total: {len(image_paths)} images')

    # ── Create Fold Splits ───────────────────────────────
    splits = create_fold_splits(labels, lesion_ids, cfg)

    # ── Training Loop Across Folds ───────────────────────
    fold_metrics: list[dict] = []  # type: ignore[type-arg]
    fold_results: list[tuple[float, Path]] = []
    total_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f'\n{"=" * 60}')
        print(f'  FOLD {fold_idx} / {len(splits) - 1}')
        print('=' * 60)

        # Create loaders.
        train_loader, val_loader = create_dataloaders(
            image_paths,
            labels,
            train_idx,
            val_idx,
            cfg,
        )

        # Build fresh model.
        print('Loading DINOv2 ViT-S/14...')
        model = build_model(cfg)
        model = model.to(device)
        trainable, total = count_parameters(model)
        print(
            f'Trainable: {trainable:,} / {total:,} '
            f'({trainable / total * 100:.2f}%)'
        )

        # Train this fold.
        best_acc, best_ckpt = train_fold(
            model,
            train_loader,
            val_loader,
            cfg,
            fold_idx,
            device,
        )
        fold_results.append((best_acc, best_ckpt))

        # Evaluate this fold with best checkpoint.
        model.load_state_dict(torch.load(best_ckpt))
        model.to(device)
        preds, true_labels, logits = collect_predictions(
            model,
            val_loader,
            device,
            tta_runs=cfg.tta_runs,
        )
        metrics = compute_metrics(
            true_labels,
            preds,
            cfg.class_names,
            logits=logits,
        )
        fold_metrics.append(metrics)

        print(f'\nFold {fold_idx} evaluation:')
        print(
            f'  Balanced Accuracy: {metrics["balanced_accuracy"] * 100:.1f}%'
        )
        print(f"  Cohen's Kappa:     {metrics['cohen_kappa']:.3f}")
        print(f'  Macro F1:          {metrics["macro_f1"]:.3f}')
        if 'auc_roc' in metrics:
            print(f'  AUC-ROC (OVR):     {metrics["auc_roc"]:.3f}')

    total_time = time.time() - total_start

    # ── Aggregate Results ────────────────────────────────
    if len(fold_metrics) > 1:
        agg = aggregate_fold_metrics(fold_metrics)
    else:
        agg = fold_metrics[0]

    # Save final report.
    save_evaluation_report(agg, cfg.output_dir)
    print_summary(agg, cfg)

    # Find and copy best fold's model.
    best_fold_idx = int(np.argmax([r[0] for r in fold_results]))
    best_overall_path = fold_results[best_fold_idx][1]
    final_model_path = cfg.output_dir / 'best_model.pth'

    import shutil

    shutil.copy2(best_overall_path, final_model_path)
    print(f'Best model (fold {best_fold_idx}): {final_model_path}')
    print(
        f'Best balanced accuracy: {fold_results[best_fold_idx][0] * 100:.1f}%'
    )

    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f'Total training time: {minutes}m {seconds}s')

    # W&B summary.
    if cfg.use_wandb and _HAS_WANDB:
        if isinstance(
            agg.get('balanced_accuracy'),
            dict,
        ):
            wandb.summary['best_bal_acc_mean'] = agg['balanced_accuracy'][
                'mean'
            ]
            wandb.summary['best_bal_acc_std'] = agg['balanced_accuracy']['std']
        else:
            wandb.summary['best_bal_acc'] = agg.get('balanced_accuracy')
        wandb.summary['best_fold'] = best_fold_idx
        wandb.summary['training_time_min'] = total_time / 60
        wandb.finish()

    print('\n✅ Training complete!')


if __name__ == '__main__':
    main()
