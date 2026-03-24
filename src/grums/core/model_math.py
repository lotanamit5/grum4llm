"""Core mathematical operations for GRUM."""

from __future__ import annotations

import torch

from grums.core.parameters import GRUMParameters

Tensor = torch.Tensor


def compute_mean_utilities(
    params: GRUMParameters,
    agent_features: Tensor,
    alternative_features: Tensor,
) -> Tensor:
    """Compute deterministic utilities mu_ij = delta_j + x_i B z_j^T.

    Returns an array of shape (n_agents, n_alternatives).
    """

    if agent_features.dim() != 2:
        raise ValueError("agent_features must be a 2D matrix")
    if alternative_features.dim() != 2:
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
    return interaction_term + params.delta.view(1, n_alternatives).expand(n_agents, -1)


def predict_deterministic_rankings(
    params: GRUMParameters,
    agent_features: Tensor,
    alternative_features: Tensor,
) -> list[tuple[int, ...]]:
    """Predict per-agent rankings from deterministic utilities."""

    mu = compute_mean_utilities(params, agent_features, alternative_features)
    return [tuple(row.tolist()) for row in torch.argsort(mu, dim=1, descending=True)]
