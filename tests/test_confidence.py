"""Unit tests for shared.confidence module."""

import pytest
import torch

from shared.confidence import (
    CalibrationError,
    TemperatureScaling,
)


class TestCalibrationError:
    """Tests for the CalibrationError exception."""

    def test_reason_stored(self) -> None:
        err = CalibrationError(reason='mismatched shapes')
        assert err.reason == 'mismatched shapes'
        assert 'mismatched shapes' in str(err)


class TestTemperatureScaling:
    """Tests for the TemperatureScaling module."""

    def test_default_initialization(self) -> None:
        scaler = TemperatureScaling()
        assert scaler.temperature == 1.5
        assert not scaler._fitted

    def test_custom_initialization(self) -> None:
        scaler = TemperatureScaling(initial_temperature=2.0)
        assert scaler.temperature == 2.0

    def test_forward_before_fit_raises(self) -> None:
        scaler = TemperatureScaling()
        logits = torch.randn(2, 5)
        with pytest.raises(CalibrationError, match=r'fit.*before forward'):
            scaler(logits)

    def test_fit_invalid_logits_shape(self) -> None:
        scaler = TemperatureScaling()
        bad_logits = torch.randn(5)  # 1D instead of 2D
        labels = torch.zeros(5, dtype=torch.long)
        with pytest.raises(CalibrationError, match='logits must be 2-D'):
            scaler.fit(bad_logits, labels)

    def test_fit_invalid_labels_shape(self) -> None:
        scaler = TemperatureScaling()
        logits = torch.randn(5, 3)
        bad_labels = torch.zeros(5, 1, dtype=torch.long)  # 2D instead of 1D
        with pytest.raises(CalibrationError, match='labels must be 1-D'):
            scaler.fit(logits, bad_labels)

    def test_fit_mismatched_samples(self) -> None:
        scaler = TemperatureScaling()
        logits = torch.randn(5, 3)
        labels = torch.zeros(4, dtype=torch.long)  # 4 labels for 5 samples
        with pytest.raises(
            CalibrationError, match='5 samples but labels has 4'
        ):
            scaler.fit(logits, labels)
