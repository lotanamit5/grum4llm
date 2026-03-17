"""Core mathematical operations for GRUM."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from grums.core.parameters import GRUMParameters

FloatArray = NDArray[np.float64]


def compute_mean_utilities(
    params: GRUMParameters,
    agent_features: FloatArray,
    alternative_features: FloatArray,
) -> FloatArray:
    """Compute deterministic utilities mu_ij = delta_j + x_i B z_j^T.

    Returns an array of shape (n_agents, n_alternatives).
    """

    if agent_features.ndim != 2:
        raise ValueError("agent_features must be a 2D matrix")
    if alternative_features.ndim != 2:
        raise ValueError("alternative_features must be a 2D matrix")

    n_agents, k = agent_features.shape
    n_alternatives, l = alternative_features.shape

    if params.n_agent_features != k:
        raise ValueError("B row dimension must match agent feature dimension")
    if params.n_alternative_features != l:
        raise ValueError("B column dimension must match alternative feature dimension")
    if params.n_alternatives != n_alternatives:
        raise ValueError("delta length must match number of alternatives")

    interaction_term = agent_features @ params.interaction @ alternative_features.T
    return interaction_term + params.delta.reshape(1, n_alternatives).repeat(n_agents, axis=0)


def predict_deterministic_rankings(
    params: GRUMParameters,
    agent_features: FloatArray,
    alternative_features: FloatArray,
) -> list[tuple[int, ...]]:
    """Predict per-agent rankings from deterministic utilities."""

    mu = compute_mean_utilities(params, agent_features, alternative_features)
    return [tuple(np.argsort(-row)) for row in mu]
