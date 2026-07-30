"""Post-training evaluation and metrics reporting.

Computes comprehensive metrics after training:
- Balanced accuracy (primary)
- Per-class precision / recall / F1
- Confusion matrix
- Cohen's kappa
- ROC-AUC (one-vs-rest)
- Cross-fold aggregation (mean +/- std)
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader

from training.config import TrainingConfig


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Collect all predictions, true labels, and logits.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(predictions, true_labels, logits)``
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_logits: list[npt.NDArray[Any]] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)

        predicted = logits.argmax(dim=1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_logits.append(logits.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.concatenate(all_logits, axis=0),
    )


def compute_metrics(
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list[str]
        Ordered class names for the report.

    Returns
    -------
    dict[str, Any]
        Dictionary of all metrics.
    """
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Per-class report as dict.
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        'balanced_accuracy': float(bal_acc),
        'cohen_kappa': float(kappa),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'per_class': {
            name: {
                'precision': float(
                    report[name]['precision']  # type: ignore[index]
                ),
                'recall': float(
                    report[name]['recall']  # type: ignore[index]
                ),
                'f1': float(
                    report[name]['f1-score']  # type: ignore[index]
                ),
                'support': int(
                    report[name]['support']  # type: ignore[index]
                ),
            }
            for name in class_names
            if name in report
        },
    }


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate metrics across folds (mean +/- std).

    Parameters
    ----------
    fold_metrics : list[dict[str, Any]]
        List of per-fold metric dicts from
        ``compute_metrics()``.

    Returns
    -------
    dict[str, Any]
        Aggregated metrics.
    """
    bal_accs = [m['balanced_accuracy'] for m in fold_metrics]
    kappas = [m['cohen_kappa'] for m in fold_metrics]

    result: dict[str, Any] = {
        'n_folds': len(fold_metrics),
        'balanced_accuracy': {
            'mean': float(np.mean(bal_accs)),
            'std': float(np.std(bal_accs)),
            'per_fold': bal_accs,
        },
        'cohen_kappa': {
            'mean': float(np.mean(kappas)),
            'std': float(np.std(kappas)),
        },
    }

    # Aggregate per-class metrics.
    if fold_metrics and 'per_class' in fold_metrics[0]:
        class_names = list(fold_metrics[0]['per_class'].keys())
        per_class_agg: dict[str, dict[str, Any]] = {}
        for cls_name in class_names:
            precisions = []
            recalls = []
            f1s = []
            for m in fold_metrics:
                if cls_name in m['per_class']:
                    c = m['per_class'][cls_name]
                    precisions.append(c['precision'])
                    recalls.append(c['recall'])
                    f1s.append(c['f1'])
            per_class_agg[cls_name] = {
                'precision': {
                    'mean': float(np.mean(precisions)),
                    'std': float(np.std(precisions)),
                },
                'recall': {
                    'mean': float(np.mean(recalls)),
                    'std': float(np.std(recalls)),
                },
                'f1': {
                    'mean': float(np.mean(f1s)),
                    'std': float(np.std(f1s)),
                },
            }
        result['per_class'] = per_class_agg

    return result


def save_evaluation_report(
    metrics: dict[str, Any],
    output_dir: Path,
    filename: str = 'evaluation_report.json',
) -> Path:
    """Save metrics to a JSON file.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary.
    output_dir : Path
        Output directory.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open('w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)
    print(f'[eval] Report saved to {path}')
    return path


def print_summary(
    metrics: dict[str, Any],
    cfg: TrainingConfig,
) -> None:
    """Pretty-print evaluation summary to console.

    Parameters
    ----------
    metrics : dict
        Aggregated metrics from
        ``aggregate_fold_metrics()``.
    cfg : TrainingConfig
        For class name lookup.
    """
    print('\n' + '=' * 60)
    print('  EVALUATION SUMMARY')
    print('=' * 60)

    if 'balanced_accuracy' in metrics:
        ba = metrics['balanced_accuracy']
        if isinstance(ba, dict):
            print(
                f'Balanced Accuracy: '
                f'{ba["mean"] * 100:.1f}% '
                f'+/- {ba["std"] * 100:.1f}%'
            )
        else:
            print(f'Balanced Accuracy: {ba * 100:.1f}%')

    if 'cohen_kappa' in metrics:
        ck = metrics['cohen_kappa']
        if isinstance(ck, dict):
            print(f"Cohen's Kappa:     {ck['mean']:.3f} +/- {ck['std']:.3f}")
        else:
            print(f"Cohen's Kappa:     {ck:.3f}")

    if 'per_class' in metrics:
        print('\nPer-class metrics:')
        print(
            f'  {"Class":>8s}  {"Precision":>10s}  '
            f'{"Recall":>10s}  {"F1":>10s}'
        )
        print('  ' + '-' * 44)
        for name in cfg.class_names:
            if name in metrics['per_class']:
                c = metrics['per_class'][name]
                if isinstance(c.get('precision'), dict):
                    print(
                        f'  {name:>8s}  '
                        f'{c["precision"]["mean"]:.3f}'
                        f'+/-{c["precision"]["std"]:.3f}  '
                        f'{c["recall"]["mean"]:.3f}'
                        f'+/-{c["recall"]["std"]:.3f}  '
                        f'{c["f1"]["mean"]:.3f}'
                        f'+/-{c["f1"]["std"]:.3f}'
                    )
                else:
                    print(
                        f'  {name:>8s}  '
                        f'{c["precision"]:10.3f}  '
                        f'{c["recall"]:10.3f}  '
                        f'{c["f1"]:10.3f}'
                    )

    if 'confusion_matrix' in metrics:
        print('\nConfusion Matrix:')
        cm = metrics['confusion_matrix']
        header = '        ' + '  '.join(f'{n:>5s}' for n in cfg.class_names)
        print(header)
        for i, row in enumerate(cm):
            row_str = '  '.join(f'{v:5d}' for v in row)
            print(f'{cfg.class_names[i]:>6s}  {row_str}')

    print()
