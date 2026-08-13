"""Unit tests for SkinAnalysisSkill."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from hiperhealth.pipeline.context import PipelineContext

from skills.skin_analysis.skill import SkinAnalysisSkill


def test_check_requirements_no_image() -> None:
    """Test check_requirements requests skin_image if missing."""
    skill = SkinAnalysisSkill()
    ctx = PipelineContext(patient={})

    inquiries = skill.check_requirements('intake', ctx)

    assert len(inquiries) == 1
    assert inquiries[0].field == 'skin_image'
    assert inquiries[0].priority == 'required'
    assert inquiries[0].input_type == 'image'
    assert inquiries[0].stage == 'intake'


def test_check_requirements_has_image() -> None:
    """Test check_requirements returns empty if image exists."""
    skill = SkinAnalysisSkill()
    ctx = PipelineContext(patient={'skin_image': '/path/to/img.jpg'})

    inquiries = skill.check_requirements('intake', ctx)

    assert len(inquiries) == 0


def test_check_requirements_wrong_stage() -> None:
    """Test check_requirements only runs on intake stage."""
    skill = SkinAnalysisSkill()
    ctx = PipelineContext(patient={})

    inquiries = skill.check_requirements('diagnosis', ctx)

    assert len(inquiries) == 0


@patch('skills.skin_analysis.skill.ImagePreprocessor')
@patch('skills.skin_analysis.skill.ModelRegistry')
@patch('skills.skin_analysis.skill._load_model')
@patch('skills.skin_analysis.skill.get_explainer')
@patch('skills.skin_analysis.skill.cv2.imwrite')
@patch('skills.skin_analysis.skill.cv2.imread')
def test_execute_happy_path(
    mock_imread: MagicMock,
    mock_imwrite: MagicMock,
    mock_get_explainer: MagicMock,
    mock_load_model: MagicMock,
    mock_registry: MagicMock,
    mock_preprocessor: MagicMock,
) -> None:
    """Test execute hook runs inference and saves heatmap."""
    mock_prep_inst = mock_preprocessor.return_value
    mock_prep_inst.preprocess.return_value = torch.randn(1, 3, 224, 224)

    mock_model = mock_load_model.return_value
    mock_model.return_value = torch.randn(1, 7)

    mock_exp = mock_get_explainer.return_value
    mock_exp.generate.return_value = np.zeros((224, 224), dtype=np.float32)

    mock_imread.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

    skill = SkinAnalysisSkill()
    ctx = PipelineContext(
        patient={'skin_image': '/fake/img.jpg'},
        session_id='test1',
    )

    ctx = skill.execute('intake', ctx)

    assert 'visual_observations' in ctx.results['intake']
    obs = ctx.results['intake']['visual_observations'][0]

    assert obs['skill'] == 'medvision.skin_analysis'
    assert obs['status'] == 'preliminary'
    assert 'heatmap_path' in obs
    assert obs['heatmap_path'] is not None


def test_execute_no_image() -> None:
    """Test execute hook skips if skin_image is missing."""
    skill = SkinAnalysisSkill()
    ctx = PipelineContext(patient={})
    ctx = skill.execute('intake', ctx)
    assert 'intake' not in ctx.results


@patch('skills.skin_analysis.skill.ImagePreprocessor')
@patch('skills.skin_analysis.skill.ModelRegistry')
@patch('skills.skin_analysis.skill._load_model')
@patch('skills.skin_analysis.skill.get_explainer')
def test_execute_preprocessing_error(
    mock_get_explainer: MagicMock,
    mock_load_model: MagicMock,
    mock_registry: MagicMock,
    mock_preprocessor: MagicMock,
) -> None:
    """Test execute hook handles preprocessing errors gracefully."""
    mock_prep_inst = mock_preprocessor.return_value
    from shared.preprocessing import ImageQualityError

    mock_prep_inst.preprocess.side_effect = ImageQualityError('Too dark')

    skill = SkinAnalysisSkill()
    ctx = PipelineContext(patient={'skin_image': '/fake/img.jpg'})

    ctx = skill.execute('intake', ctx)

    obs = ctx.results['intake']['visual_observations'][0]
    assert obs['status'] == 'error'
    assert 'Too dark' in obs['error']


@patch('skills.skin_analysis.skill.ImagePreprocessor')
@patch('skills.skin_analysis.skill.ModelRegistry')
@patch('skills.skin_analysis.skill._load_model')
@patch('skills.skin_analysis.skill.get_explainer')
def test_execute_gradcam_failure(
    mock_get_explainer: MagicMock,
    mock_load_model: MagicMock,
    mock_registry: MagicMock,
    mock_preprocessor: MagicMock,
) -> None:
    """Test execute hook continues if Grad-CAM fails."""
    mock_prep_inst = mock_preprocessor.return_value
    mock_prep_inst.preprocess.return_value = torch.randn(1, 3, 224, 224)

    mock_model = mock_load_model.return_value
    mock_model.return_value = torch.randn(1, 7)

    mock_exp = mock_get_explainer.return_value
    mock_exp.generate.side_effect = RuntimeError('GradCAM failed')

    skill = SkinAnalysisSkill()
    ctx = PipelineContext(patient={'skin_image': '/fake/img.jpg'})

    ctx = skill.execute('intake', ctx)

    obs = ctx.results['intake']['visual_observations'][0]
    assert obs['status'] == 'preliminary'
    assert obs['heatmap_path'] is None


def test_pre_injects_prompt_fragments() -> None:
    """Test pre hook injects visual observations into diagnosis prompt."""
    skill = SkinAnalysisSkill()
    ctx = PipelineContext(patient={})
    ctx.results['intake'] = {
        'visual_observations': [
            {
                'finding': 'melanoma',
                'display_name': 'Melanoma',
                'confidence': 0.85,
                'requires_review': True,
            }
        ]
    }

    ctx = skill.pre('diagnosis', ctx)

    assert 'prompt_fragments' in ctx.extras
    assert 'diagnosis' in ctx.extras['prompt_fragments']
    fragment = ctx.extras['prompt_fragments']['diagnosis']
    assert 'Melanoma' in fragment
    assert '85%' in fragment
    assert 'Flagged for clinician review' in fragment


def test_pre_skips_when_no_observations() -> None:
    """Test pre hook skips if no observations exist."""
    skill = SkinAnalysisSkill()
    ctx = PipelineContext(patient={})
    ctx = skill.pre('diagnosis', ctx)
    assert 'prompt_fragments' not in ctx.extras


def test_pre_skips_error_observations() -> None:
    """Test pre hook skips if observation is an error."""
    skill = SkinAnalysisSkill()
    ctx = PipelineContext(patient={})
    ctx.results['intake'] = {
        'visual_observations': [{'status': 'error', 'error': 'Too dark'}]
    }
    ctx = skill.pre('diagnosis', ctx)
    assert 'prompt_fragments' not in ctx.extras
