"""Observed and candidate Fisher information utilities for Normal-family GRUM."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from grums.core.parameters import GRUMParameters

FloatArray = NDArray[np.float64]


def _param_design_row(agent_x: FloatArray, alt_z: FloatArray, alt_idx: int, m: int) -> FloatArray:
    row = np.zeros(m + (agent_x.shape[0] * alt_z.shape[0]), dtype=float)
    row[alt_idx] = 1.0
    row[m:] = np.kron(alt_z, agent_x)
    return row


def candidate_fisher_information(
    candidate_agent_features: FloatArray,
    alternative_features: FloatArray,
    n_alternatives: int,
    sigma: float,
) -> FloatArray:
    """Approximate Fisher information from one candidate agent query.

    For the Normal-family baseline this uses the linearized utility design matrix:
    I_h ≈ (1/sigma^2) * X_h^T X_h for m utility equations.
    """

    if candidate_agent_features.ndim != 1:
        raise ValueError("candidate_agent_features must be a 1D vector")
    if alternative_features.ndim != 2:
        raise ValueError("alternative_features must be a 2D matrix")
    if n_alternatives != alternative_features.shape[0]:
        raise ValueError("n_alternatives must match alternative_features row count")

    rows = [
        _param_design_row(candidate_agent_features, alternative_features[j], j, n_alternatives)
        for j in range(n_alternatives)
    ]
    design = np.vstack(rows)
    return (design.T @ design) / (sigma**2)


def observed_fisher_information(
    params: GRUMParameters,
    agent_features: FloatArray,
    alternative_features: FloatArray,
    sigma: float,
) -> FloatArray:
    """Approximate observed Fisher information J_D(theta).

    This baseline computes a deterministic approximation from stacked linear rows over all
    observed agents and alternatives in the Normal-family model.
    """

    if agent_features.ndim != 2 or alternative_features.ndim != 2:
        raise ValueError("feature matrices must be 2D")

    m = params.n_alternatives
    rows: list[np.ndarray] = []
    for x in agent_features:
        for j in range(m):
            rows.append(_param_design_row(x, alternative_features[j], j, m))
    design = np.vstack(rows)
    return (design.T @ design) / (sigma**2)


def posterior_precision(
    observed_fisher: FloatArray,
    prior_precision: float,
) -> FloatArray:
    if observed_fisher.ndim != 2 or observed_fisher.shape[0] != observed_fisher.shape[1]:
        raise ValueError("observed_fisher must be a square matrix")
    return observed_fisher + (prior_precision * np.eye(observed_fisher.shape[0]))
