"""Evaluation metrics for GRUM experiments."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def social_choice_kendall_tau(delta_true: np.ndarray, delta_est: np.ndarray) -> float:
    """Kendall tau between social-choice rankings induced by true and estimated delta."""

    rank_true = np.argsort(-delta_true)
    rank_est = np.argsort(-delta_est)
    tau, _ = kendalltau(rank_true, rank_est)
    if np.isnan(tau):
        return 0.0
    return float(tau)
