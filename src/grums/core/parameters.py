"""Parameter containers for GRUM model state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GRUMParameters:
    """GRUM parameter bundle.

    Attributes:
        delta: Intrinsic utility vector of shape (m,).
        interaction: Interaction matrix B of shape (K, L).
    """

    delta: FloatArray
    interaction: FloatArray

    def __post_init__(self) -> None:
        if self.delta.ndim != 1:
            raise ValueError("delta must be a 1D vector")
        if self.interaction.ndim != 2:
            raise ValueError("interaction must be a 2D matrix")
        if not np.isfinite(self.delta).all() or not np.isfinite(self.interaction).all():
            raise ValueError("all parameter values must be finite")

    @property
    def n_alternatives(self) -> int:
        return int(self.delta.shape[0])

    @property
    def n_agent_features(self) -> int:
        return int(self.interaction.shape[0])

    @property
    def n_alternative_features(self) -> int:
        return int(self.interaction.shape[1])
