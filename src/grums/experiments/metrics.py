"""Evaluation metrics for GRUM experiments."""

from __future__ import annotations

import torch
import numpy as np
from scipy.stats import kendalltau

from grums.core.model_math import predict_deterministic_rankings
from grums.core.parameters import GRUMParameters

Tensor = torch.Tensor


def social_choice_kendall_tau(delta_true: Tensor, delta_est: Tensor) -> float:
    """Kendall tau between social-choice rankings induced by true and estimated delta."""

    # Using argsort on CPU for kendalltau
    rank_true = torch.argsort(delta_true, descending=True).cpu().numpy()
    rank_est = torch.argsort(delta_est, descending=True).cpu().numpy()
    tau, _ = kendalltau(rank_true, rank_est)
    if np.isnan(tau):
        return 0.0
    return float(tau)


def social_choice_suboptimality(
    params_true: GRUMParameters,
    params_est: GRUMParameters,
    agent_features: Tensor,
    alternative_features: Tensor,
) -> float:
    """Kendall tau between social-choice rankings from full GRUMParameters."""
    _ = agent_features
    _ = alternative_features
    return social_choice_kendall_tau(params_true.delta, params_est.delta)


def personalized_mean_kendall_tau(
    params_true: GRUMParameters,
    params_est: GRUMParameters,
    agent_features: Tensor,
    alternative_features: Tensor,
) -> float:
    """Average per-agent Kendall tau between predicted deterministic rankings."""

    true_rankings = predict_deterministic_rankings(params_true, agent_features, alternative_features)
    est_rankings = predict_deterministic_rankings(params_est, agent_features, alternative_features)

    taus: list[float] = []
    for r_true, r_est in zip(true_rankings, est_rankings, strict=True):
        tau, _ = kendalltau(r_true, r_est)
        taus.append(0.0 if np.isnan(tau) else float(tau))

    return float(np.mean(taus))


def raw_mean_kendall_tau(
    params_est: GRUMParameters,
    agent_features: Tensor,
    alternative_features: Tensor,
    observed_rankings: list[tuple[int, ...]],
) -> float:
    """Average per-agent Kendall tau between true deterministic rankings and raw observed rankings."""

    true_rankings = predict_deterministic_rankings(params_est, agent_features, alternative_features)

    taus: list[float] = []
    # Observed rankings might be a subset of total agents
    for r_true, r_obs in zip(true_rankings[:len(observed_rankings)], observed_rankings, strict=True):
        tau, _ = kendalltau(r_true, r_obs)
        taus.append(0.0 if np.isnan(tau) else float(tau))

    return float(np.mean(taus))


def moving_average(series: Tensor, window: int) -> Tensor:
    """Simple trailing moving average used for smoothed report curves."""

    if window <= 0:
        raise ValueError("window must be positive")
    if series.dim() != 1:
        raise ValueError("series must be 1D")
    if window > series.size(0):
        raise ValueError(f"window {window} cannot exceed series length {series.size(0)}")

    weights = torch.ones(window, dtype=series.dtype, device=series.device) / float(window)
    
    # Use 1D convolution
    # Input needs (Batch, Channel, Length)
    res = torch.nn.functional.conv1d(
        series.view(1, 1, -1),
        weights.view(1, 1, -1),
        padding=0
    )
    return res.view(-1)
