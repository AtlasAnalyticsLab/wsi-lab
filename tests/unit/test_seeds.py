from __future__ import annotations

import pytest

from wsi_lab.utils.seeds import configure_deterministic_cuda


def test_deterministic_cuda_workspace_default_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert configure_deterministic_cuda() == ":4096:8"

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    assert configure_deterministic_cuda() == ":16:8"

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_deterministic_cuda()
