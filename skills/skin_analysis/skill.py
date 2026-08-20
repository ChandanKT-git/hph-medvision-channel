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

import cv2
import numpy as np
import torch

from hiperhealth.pipeline import BaseSkill, SkillMetadata
from hiperhealth.pipeline.context import PipelineContext
from hiperhealth.pipeline.session import Inquiry
from torch import nn

from shared.confidence import TemperatureScaling
from shared.explainability import (
    GradCAMExplainer,
    get_explainer,
    overlay_heatmap,
)
from shared.models.registry import ModelRegistry
from shared.preprocessing import (
    ImagePreprocessor,
    ImageQualityError,
)

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


def _determine_requires_review(
    class_probs: dict[str, float],
    top_confidence: float,
) -> bool:
    """Decide if the prediction needs human review.

    Two safety rules trigger clinician review:

    1. **Low confidence** — the model's top prediction
       is below ``_DEFAULT_CONFIDENCE_THRESHOLD`` (0.7),
       meaning the model is uncertain.
    2. **Dangerous class** — melanoma or basal cell
       carcinoma has probability above
       ``_DANGEROUS_THRESHOLD`` (0.1), even if a benign
       class is the top prediction.

    Parameters
    ----------
    class_probs : dict[str, float]
        Mapping of class names to their probabilities.
    top_confidence : float
        The confidence of the highest-scoring class.

    Returns
    -------
    bool
        ``True`` if the prediction should be flagged
        for human clinician review.
    """
    if top_confidence < _DEFAULT_CONFIDENCE_THRESHOLD:
        return True

    for cls_name in _DANGEROUS_CLASSES:
        if class_probs.get(cls_name, 0.0) > _DANGEROUS_THRESHOLD:
            return True

    return False


def _format_prompt_fragment(
    observations: list[dict[str, Any]],
) -> str:
    """Format visual observations as text for the LLM.

    This text gets injected into the DiagnosticsSkill's
    system prompt via
    ``ctx.extras['prompt_fragments']['diagnosis']``.

    Parameters
    ----------
    observations : list[dict[str, Any]]
        Visual observation dicts from the intake stage.

    Returns
    -------
    str
        Markdown-formatted summary of visual findings.
        Empty string if no observations.
    """
    if not observations:
        return ''

    lines: list[str] = [
        '## Visual Observations (MedVision Skin Analysis)',
        '',
        'The following skin lesion analysis was performed '
        'using computer vision (DINOv2 + Grad-CAM):',
        '',
    ]

    for obs in observations:
        finding = obs.get('finding', 'unknown')
        display = obs.get('display_name', finding)
        confidence = obs.get('confidence', 0.0)
        status = obs.get('status', 'preliminary')
        review = obs.get('requires_review', True)

        lines.append(f'- **Primary finding:** {display}')
        cal_label = 'calibrated' if obs.get('calibrated') else 'raw'
        lines.append(f'  - Confidence: {confidence:.0%} ({cal_label})')
        lines.append(f'  - Status: {status}')
        if review:
            lines.append('  - ⚠️ Flagged for clinician review')

        class_probs = obs.get('class_probabilities', {})
        if class_probs:
            sorted_probs = sorted(
                class_probs.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            lines.append('  - Differential:')
            for name, prob in sorted_probs:
                lines.append(f'    - {name}: {prob:.1%}')

        lines.append('')

    lines.append(
        '> Note: This is an AI-assisted preliminary '
        'observation. Clinical correlation is required. '
        'Visual analysis should not be used as the sole '
        'basis for diagnosis.'
    )

    return '\n'.join(lines)


# ── Skill Class ────────────────────────────────────────


class SkinAnalysisSkill(BaseSkill):
    """Skin lesion classification with explainable AI.

    Lifecycle
    ---------
    - **check_requirements("intake")**: Requests a skin
      image if not in ``ctx.patient``.
    - **execute("intake")**: Runs DINOv2 inference +
      Grad-CAM, stores results in
      ``ctx.results["intake"]["visual_observations"]``.
    - **pre("diagnosis")**: Injects a formatted text
      summary into ``ctx.extras["prompt_fragments"]
      ["diagnosis"]`` for DiagnosticsSkill.
    """

    def __init__(self) -> None:
        """Initialise skill metadata and placeholders."""
        super().__init__(
            SkillMetadata(
                name=_SKILL_NAME,
                version='0.1.0',
                stages=('intake', 'diagnosis'),
                description=(
                    'Skin lesion classification using '
                    'fine-tuned DINOv2 with Grad-CAM '
                    'explainability.'
                ),
            )
        )

        self._model: nn.Module | None = None
        self._preprocessor: ImagePreprocessor | None = None
        self._explainer: GradCAMExplainer | None = None
        self._label_map: dict[str, Any] | None = None
        self._scaler: TemperatureScaling | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model, preprocessor, and label map.

        Called once on first inference.  Subsequent calls
        are no-ops (early return).
        """
        if self._model is not None:
            return

        logger.info(
            '[%s] Loading model and resources...',
            _SKILL_NAME,
        )

        self._label_map = _load_label_map()

        registry = ModelRegistry()
        weights_path = registry.get_weights(
            model_id=_MODEL_ID,
            url=_MODEL_URL,
            expected_sha256=_MODEL_SHA256,
        )

        self._model = _load_model(weights_path)

        self._preprocessor = ImagePreprocessor()

        self._explainer = get_explainer(
            self._model,
            backend='dinov2',
        )

        self._scaler = TemperatureScaling(
            initial_temperature=_FITTED_TEMPERATURE,
        )
        self._scaler.mark_fitted()

        logger.info(
            '[%s] Model loaded successfully',
            _SKILL_NAME,
        )

    # ── Hook 1: check_requirements ─────────────────────

    def check_requirements(
        self,
        stage: str,
        ctx: PipelineContext,
    ) -> list[Inquiry]:
        """Request a skin image if not provided.

        Parameters
        ----------
        stage : str
            The current pipeline stage.
        ctx : PipelineContext
            The pipeline context with patient data.

        Returns
        -------
        list[Inquiry]
            An ``Inquiry`` for ``'skin_image'`` if
            missing, or empty list if already provided.
        """
        if stage != 'intake':
            return []

        if ctx.patient.get('skin_image'):
            return []

        return [
            Inquiry(
                skill_name=self.metadata.name,
                stage=stage,
                field='skin_image',
                label='Skin Photo',
                description=(
                    'Please upload a clear, well-lit '
                    'photograph of the skin area of '
                    'concern.  Minimum 224x224 pixels.'
                ),
                priority='required',
                input_type='image',
            ),
        ]

    # ── Hook 2: execute ────────────────────────────────

    def execute(
        self,
        stage: str,
        ctx: PipelineContext,
    ) -> PipelineContext:
        """Run skin lesion classification during intake.

        Parameters
        ----------
        stage : str
            The current pipeline stage.
        ctx : PipelineContext
            Pipeline context containing patient data.

        Returns
        -------
        PipelineContext
            Updated context with visual observations in
            ``ctx.results["intake"]
            ["visual_observations"]``.
        """
        if stage != 'intake':
            return ctx

        image_path = ctx.patient.get('skin_image')
        if not image_path:
            logger.warning(
                '[%s] No skin_image in patient data, skipping',
                _SKILL_NAME,
            )
            return ctx

        self._ensure_loaded()
        assert self._model is not None
        assert self._preprocessor is not None
        assert self._explainer is not None
        assert self._label_map is not None
        assert self._scaler is not None

        # Step 1: Preprocess.
        try:
            input_tensor = self._preprocessor.preprocess(image_path)
        except (
            FileNotFoundError,
            ImageQualityError,
        ) as exc:
            logger.warning(
                '[%s] Image preprocessing failed: %s',
                _SKILL_NAME,
                exc,
            )
            ctx.results.setdefault(stage, {})['visual_observations'] = [
                {
                    'skill': _SKILL_NAME,
                    'status': 'error',
                    'error': str(exc),
                }
            ]
            return ctx

        # Step 2: Inference.
        with torch.no_grad():
            logits = self._model(input_tensor)
            # Step 3: Temperature-scaled probabilities.
            probs = self._scaler(logits)

        probs_np = probs.squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs_np))
        top_confidence = float(probs_np[pred_idx])

        # Step 4: Map to labels.
        classes = self._label_map['classes']
        pred_class: dict[str, Any] = classes[pred_idx]

        class_probs = {
            cls['name']: float(probs_np[cls['index']]) for cls in classes
        }

        # Step 5: Grad-CAM heatmap.
        heatmap_path: str | None = None
        try:
            input_for_cam = input_tensor.clone().requires_grad_(True)
            heatmap = self._explainer.generate(
                input_for_cam,
                target_class=pred_idx,
            )

            output_dir = Path(
                ctx.extras.get(
                    'heatmap_dir',
                    '/tmp/medvision/heatmaps',
                )
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            original = cv2.imread(str(image_path))
            if original is not None:
                original_rgb = cv2.cvtColor(
                    original,
                    cv2.COLOR_BGR2RGB,
                )
                overlay = overlay_heatmap(
                    original_rgb,
                    heatmap,
                )
                session_id = ctx.session_id or 'unknown'
                out_file = output_dir / (f'{session_id}_skin_heatmap.png')
                cv2.imwrite(
                    str(out_file),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
                heatmap_path = str(out_file)
                logger.info(
                    '[%s] Heatmap saved to %s',
                    _SKILL_NAME,
                    heatmap_path,
                )

        except Exception as exc:
            logger.warning(
                '[%s] Grad-CAM failed: %s',
                _SKILL_NAME,
                exc,
            )

        # Step 6: Build result.
        requires_review = _determine_requires_review(
            class_probs,
            top_confidence,
        )

        observation: dict[str, Any] = {
            'skill': _SKILL_NAME,
            'finding': pred_class['name'],
            'display_name': (pred_class['display_name']),
            'confidence': top_confidence,
            'calibrated': True,
            'body_site': 'skin',
            'snomed_code': pred_class['snomed_code'],
            'snomed_display': (pred_class['snomed_display']),
            'heatmap_path': heatmap_path,
            'status': 'preliminary',
            'requires_review': requires_review,
            'class_probabilities': class_probs,
        }

        ctx.results.setdefault(stage, {})['visual_observations'] = [
            observation
        ]

        logger.info(
            '[%s] Classified as %s (%.1f%%)',
            _SKILL_NAME,
            pred_class['display_name'],
            top_confidence * 100,
        )

        return ctx

    # ── Hook 3: pre ────────────────────────────────────

    def pre(
        self,
        stage: str,
        ctx: PipelineContext,
    ) -> PipelineContext:
        """Inject visual findings into diagnosis prompt.

        Parameters
        ----------
        stage : str
            The current pipeline stage.
        ctx : PipelineContext
            Pipeline context with intake results.

        Returns
        -------
        PipelineContext
            Updated context with ``prompt_fragments``
            injected.
        """
        if stage != 'diagnosis':
            return ctx

        intake_results = ctx.results.get('intake', {})
        observations: list[dict[str, Any]] = intake_results.get(
            'visual_observations', []
        )

        if not observations:
            return ctx

        if len(observations) == 1 and observations[0].get('status') == 'error':
            return ctx

        fragment = _format_prompt_fragment(observations)
        if not fragment:
            return ctx

        ctx.extras.setdefault(
            'prompt_fragments',
            {},
        )['diagnosis'] = fragment

        logger.info(
            '[%s] Injected visual findings into diagnosis prompt_fragments',
            _SKILL_NAME,
        )

        return ctx
