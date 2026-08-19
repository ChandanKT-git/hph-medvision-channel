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
        if initial_temperature <= 0:
            raise ValueError('initial_temperature must be > 0')
        self._temperature = nn.Parameter(
            torch.tensor(initial_temperature, dtype=torch.float32)
        )
        self._fitted = False

    @property
    def temperature(self) -> float:
        """Current temperature value."""
        return float(self._temperature.item())

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Learn the optimal temperature on a validation set.

        Uses L-BFGS to minimise negative log-likelihood (NLL)
        of the validation labels under temperature-scaled
        softmax probabilities.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model outputs **before softmax**, shape
            ``(N, C)`` where *N* is the number of samples
            and *C* is the number of classes.
        labels : torch.Tensor
            Ground-truth class indices, shape ``(N,)`` with
            integer values in ``[0, C)``.
        lr : float
            Learning rate for L-BFGS optimizer.
        max_iter : int
            Maximum L-BFGS iterations.

        Returns
        -------
        float
            The learned temperature value.

        Raises
        ------
        CalibrationError
            If logits/labels shapes are incompatible or if
            optimisation produces an invalid temperature.
        """
        if logits.ndim != 2:
            raise CalibrationError(
                f'logits must be 2-D (N, C), got shape {tuple(logits.shape)}'
            )
        if labels.ndim != 1:
            raise CalibrationError(
                f'labels must be 1-D (N,), got shape {tuple(labels.shape)}'
            )
        if logits.size(0) != labels.size(0):
            raise CalibrationError(
                f'logits has {logits.size(0)} samples but '
                f'labels has {labels.size(0)}'
            )

        if self._temperature.device != logits.device:
            self.to(logits.device)
        if labels.device != logits.device:
            labels = labels.to(logits.device)

        optimizer = torch.optim.LBFGS(
            [self._temperature], lr=lr, max_iter=max_iter
        )
        nll_loss = nn.CrossEntropyLoss()

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits / self._temperature
            loss: torch.Tensor = nll_loss(scaled, labels)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimizer.step(_closure)  # type: ignore[no-untyped-call]

        learned_t = self.temperature
        if (not torch.isfinite(self._temperature).item()) or learned_t <= 0:
            self._fitted = False
            raise CalibrationError(
                f'optimisation produced invalid temperature {learned_t!r}'
            )

        self._fitted = True
        return learned_t

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by the learned temperature.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model outputs, shape ``(N, C)`` or
            ``(1, C)`` for single-sample inference.

        Returns
        -------
        torch.Tensor
            Calibrated probabilities after temperature
            scaling and softmax, same shape as input.

        Raises
        ------
        CalibrationError
            If ``fit()`` has not been called yet.
        """
        if not self._fitted:
            raise CalibrationError('fit() must be called before forward()')
        scaled = logits / self._temperature
        return torch.softmax(scaled, dim=-1)
