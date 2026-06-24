"""Unit tests for shared.confidence module."""

from shared.confidence import (
    CalibrationError,
)


class TestCalibrationError:
    """Tests for the CalibrationError exception."""

    def test_reason_stored(self) -> None:
        err = CalibrationError(reason='mismatched shapes')
        assert err.reason == 'mismatched shapes'
        assert 'mismatched shapes' in str(err)
