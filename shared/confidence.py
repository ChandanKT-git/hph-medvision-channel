"""Post-hoc confidence calibration for MedVision models."""


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
