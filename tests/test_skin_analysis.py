"""Unit tests for SkinAnalysisSkill."""

from __future__ import annotations

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
