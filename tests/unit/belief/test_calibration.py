from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wsi_lab.belief.calibration import (  # noqa: E402
    TemperatureScaler,
    fit_temperature,
    save_calibration,
)
from wsi_lab.errors import TestSplitAccessError  # noqa: E402


def test_temperature_fit_improves_overconfident_validation_logits(tmp_path) -> None:
    logits = torch.tensor([8.0, 6.0, -8.0, -6.0, 5.0, -5.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    result = fit_temperature(logits, labels, max_iter=50)
    assert result.temperature > 1.0
    assert result.nll_after < result.nll_before
    assert result.applied
    path = tmp_path / "temperature.json"
    save_calibration(path, result)
    assert '"schema_version": 1' in path.read_text()


def test_temperature_scaler_divides_logits() -> None:
    scaler = TemperatureScaler(2.0)
    assert torch.equal(scaler(torch.tensor([2.0, -4.0])), torch.tensor([1.0, -2.0]))


def test_temperature_calibration_rejects_test_partition() -> None:
    with pytest.raises(TestSplitAccessError, match="validation-only"):
        fit_temperature(
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 1.0]),
            partition="test",
        )
