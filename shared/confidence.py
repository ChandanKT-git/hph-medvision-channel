"""Post-hoc confidence calibration for MedVision models."""

import torch

from torch import nn


class CalibrationError(Exception):
    """Raised when confidence calibration fails.

    Parameters
    ----------
    reason : str
        Human-readable description of the failure.

    Attributes
    ----------
    reason : str
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f'Calibration error: {reason}')


class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling for calibrated probabilities.

    Learns a single scalar parameter *T* on a held-out
    validation set.  During inference, logits are divided by
    *T* before softmax, producing calibrated probabilities.

    Parameters
    ----------
    initial_temperature : float
        Starting value for *T* before optimization.  ``1.5``
        is a common starting point because most neural
        networks are overconfident (T > 1 softens outputs).

    References
    ----------
    Guo et al., "On Calibration of Modern Neural Networks",
    ICML 2017.  https://arxiv.org/abs/1706.04599
    """

    def __init__(self, initial_temperature: float = 1.5) -> None:
        super().__init__()
        self._temperature = nn.Parameter(
            torch.tensor(initial_temperature, dtype=torch.float32)
        )
        self._fitted = False

    @property
    def temperature(self) -> float:
        """Current temperature value."""
        return float(self._temperature.item())
