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


def social_choice_suboptimality(
    params_true: "GRUMParameters",
    params_est: "GRUMParameters",
    agent_features: np.ndarray,
    alternative_features: np.ndarray,
) -> float:
    """Kendall tau between social-choice rankings from full GRUMParameters.

    Convenience wrapper over ``social_choice_kendall_tau`` for callers (e.g. the
    sushi experiment) that hold full parameter objects rather than bare delta arrays.
    The ``agent_features`` and ``alternative_features`` arguments are accepted for
    API symmetry with ``personalized_mean_kendall_tau`` but are not used, since
    social-choice quality depends only on the marginal ``delta`` vector.
    """
    _ = agent_features
    _ = alternative_features
    return social_choice_kendall_tau(params_true.delta, params_est.delta)


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


def raw_mean_kendall_tau(
    params_est: GRUMParameters,
    agent_features: np.ndarray,
    alternative_features: np.ndarray,
    observed_rankings: list[np.ndarray],
) -> float:
    """Average per-agent Kendall tau between true deterministic rankings and raw observed rankings."""

    true_rankings = predict_deterministic_rankings(params_est, agent_features, alternative_features)

    taus: list[float] = []
    # Observed rankings might be a subset of total agents
    for r_true, r_obs in zip(true_rankings[:len(observed_rankings)], observed_rankings, strict=True):
        tau, _ = kendalltau(r_true, r_obs)
        taus.append(0.0 if np.isnan(tau) else float(tau))

    return float(np.mean(taus))


def moving_average(series: np.ndarray, window: int) -> np.ndarray:
    """Simple trailing moving average used for smoothed report curves."""

    if window <= 0:
        raise ValueError("window must be positive")
    if series.ndim != 1:
        raise ValueError("series must be 1D")
    if window > series.shape[0]:
        raise ValueError("window cannot exceed series length")

    weights = np.ones(window, dtype=float) / float(window)
    return np.convolve(series, weights, mode="valid")
