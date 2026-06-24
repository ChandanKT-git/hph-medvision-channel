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

    def test_fit_learns_temperature(self) -> None:
        torch.manual_seed(42)
        # Create "overconfident" logits for 100 samples, 3 classes
        # The correct class (label 0) has a logit of 5.0 (very high)
        logits = torch.randn(100, 3)
        logits[:, 0] = 5.0

        # But we make the labels random! So the model is very confident
        # but completely wrong. To minimize NLL, the temperature MUST
        # increase (> 1.5) to soften those overconfident 5.0 logits.
        labels = torch.randint(0, 3, (100,))

        scaler = TemperatureScaling(initial_temperature=1.0)
        learned_t = scaler.fit(logits, labels)

        assert scaler._fitted
        assert learned_t == scaler.temperature
        assert learned_t > 1.0  # T increased to soften overconfidence

    def test_forward_produces_valid_probabilities(self) -> None:
        scaler = TemperatureScaling(initial_temperature=2.0)
        scaler._fitted = True  # Hack for testing forward without fitting

        logits = torch.tensor([[2.0, 4.0, 6.0]])
        probs = scaler(logits)

        # Shape should match
        assert probs.shape == (1, 3)

        # Probabilities should sum to 1.0
        assert torch.allclose(probs.sum(dim=-1), torch.tensor(1.0))

        # With T=2.0, logits [2, 4, 6] become [1, 2, 3]
        # Softmax of [1, 2, 3] is roughly [0.09, 0.24, 0.66]
        expected = torch.softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
        assert torch.allclose(probs, expected)
