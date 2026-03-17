from __future__ import annotations

import pytest

from grums.inference import MCEMConfig, MCEMInference


def test_mcem_sigma_must_be_positive_and_finite() -> None:
    with pytest.raises(ValueError, match="sigma must be a finite positive value"):
        _ = MCEMInference(MCEMConfig(sigma=0.0))

    with pytest.raises(ValueError, match="sigma must be a finite positive value"):
        _ = MCEMInference(MCEMConfig(sigma=float("nan")))
