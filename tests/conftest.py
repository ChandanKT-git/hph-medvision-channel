"""Shared pytest fixtures for MedVision tests."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from hiperhealth.pipeline.context import PipelineContext
from hiperhealth.pipeline.session import Session

from skills.skin_analysis.skill import SkinAnalysisSkill

# ── Unit-test image fixtures ──────────────────────────


@pytest.fixture()
def valid_image(tmp_path: Path) -> Path:
    """256x256 RGB JPEG with sharp edges (high Laplacian variance)."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.line(img, (0, 0), (255, 255), (128, 128, 128), 2)
    path = tmp_path / 'valid.jpg'
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def small_image(tmp_path: Path) -> Path:
    """50x50 image — below the 224px minimum resolution."""
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    path = tmp_path / 'small.jpg'
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def blurry_image(tmp_path: Path) -> Path:
    """256x256 image with heavy Gaussian blur (low Laplacian)."""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (31, 31), sigmaX=15)
    path = tmp_path / 'blurry.jpg'
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def narrow_image(tmp_path: Path) -> Path:
    """224x800 image — exceeds the 3.0 max aspect ratio."""
    img = np.random.randint(0, 255, (224, 800, 3), dtype=np.uint8)
    path = tmp_path / 'narrow.jpg'
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def rgba_png(tmp_path: Path) -> Path:
    """256x256 RGBA PNG image."""
    img = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
    path = tmp_path / 'rgba.png'
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def non_image_file(tmp_path: Path) -> Path:
    """Plain text file with unsupported extension."""
    path = tmp_path / 'notes.txt'
    path.write_text('not an image')
    return path


@pytest.fixture()
def corrupt_image(tmp_path: Path) -> Path:
    """File with .jpg extension but random bytes inside."""
    path = tmp_path / 'corrupt.jpg'
    path.write_bytes(b'\x00\x01\x02\x03\x04\x05')
    return path


# ── Integration-test fixtures ─────────────────────────


@contextmanager
def _mock_skin_patches() -> Generator[
    dict[str, MagicMock],
    None,
    None,
]:
    """Apply all heavy-dependency patches for SkinAnalysisSkill.

    Patches DINOv2 model loading, ModelRegistry, ImagePreprocessor,
    Grad-CAM explainer, and OpenCV I/O so integration tests run
    without network access or GPU.
    """
    with (
        patch('skills.skin_analysis.skill.ImagePreprocessor') as mock_prep,
        patch('skills.skin_analysis.skill.ModelRegistry') as mock_reg,
        patch('skills.skin_analysis.skill._load_model') as mock_load,
        patch('skills.skin_analysis.skill.get_explainer') as mock_explainer,
        patch('skills.skin_analysis.skill.cv2.imwrite') as mock_imwrite,
        patch('skills.skin_analysis.skill.cv2.imread') as mock_imread,
    ):
        mock_prep.return_value.preprocess.return_value = torch.randn(
            1, 3, 224, 224
        )

        mock_load.return_value.return_value = torch.randn(1, 7)

        mock_explainer.return_value.generate.return_value = np.zeros(
            (224, 224), dtype=np.float32
        )

        mock_imread.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        yield {
            'preprocessor': mock_prep,
            'registry': mock_reg,
            'load_model': mock_load,
            'explainer': mock_explainer,
            'imwrite': mock_imwrite,
            'imread': mock_imread,
        }


@pytest.fixture()
def mock_skin_skill() -> Generator[
    SkinAnalysisSkill,
    None,
    None,
]:
    """SkinAnalysisSkill with all heavy deps mocked."""
    with _mock_skin_patches():
        yield SkinAnalysisSkill()


@pytest.fixture()
def pipeline_ctx_with_image() -> PipelineContext:
    """PipelineContext with realistic patient data including skin_image."""
    return PipelineContext(
        patient={
            'skin_image': '/fake/patient_42/skin_photo.jpg',
            'chief_complaint': 'changing mole on left forearm',
            'age': 45,
        },
        session_id='integration-test-001',
    )


@pytest.fixture()
def tmp_session(tmp_path: Path) -> Session:
    """Parquet-backed Session with clinical data pre-set."""
    session_path = tmp_path / 'test_session.parquet'
    session = Session.create(str(session_path))
    session.set_clinical_data(
        {
            'chief_complaint': 'changing mole on left forearm',
            'age': 45,
        }
    )
    return session
