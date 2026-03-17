"""Evaluation metrics for GRUM experiments."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from grums.core.model_math import predict_deterministic_rankings
from grums.core.parameters import GRUMParameters


def social_choice_kendall_tau(delta_true: np.ndarray, delta_est: np.ndarray) -> float:
    """Kendall tau between social-choice rankings induced by true and estimated delta."""

    rank_true = np.argsort(-delta_true)
    rank_est = np.argsort(-delta_est)
    tau, _ = kendalltau(rank_true, rank_est)
    if np.isnan(tau):
        return 0.0
    return float(tau)


def personalized_mean_kendall_tau(
    params_true: GRUMParameters,
    params_est: GRUMParameters,
    agent_features: np.ndarray,
    alternative_features: np.ndarray,
) -> float:
    """Average per-agent Kendall tau between predicted deterministic rankings."""

    true_rankings = predict_deterministic_rankings(params_true, agent_features, alternative_features)
    est_rankings = predict_deterministic_rankings(params_est, agent_features, alternative_features)

    taus: list[float] = []
    for r_true, r_est in zip(true_rankings, est_rankings, strict=True):
        tau, _ = kendalltau(r_true, r_est)
        taus.append(0.0 if np.isnan(tau) else float(tau))

    return float(np.mean(taus))
