"""Observed and candidate Fisher information utilities for Normal-family GRUM."""

from __future__ import annotations

import torch

from grums.core.parameters import GRUMParameters

Tensor = torch.Tensor


def _param_design_row(agent_x: Tensor, alt_z: Tensor, alt_idx: int, m: int) -> Tensor:
    # agent_x: (K,), alt_z: (L,) -> row: (m + K*L)
    row = torch.zeros(m + (agent_x.size(0) * alt_z.size(0)), dtype=torch.float64, device=agent_x.device)
    row[alt_idx] = 1.0
    row[m:] = torch.kron(agent_x, alt_z)
    return row


def candidate_fisher_information(
    candidate_agent_features: Tensor,
    alternative_features: Tensor,
    n_alternatives: int,
    sigma: float,
) -> Tensor:
    """Approximate Fisher information from one candidate agent query.

    For the Normal-family baseline this uses the linearized utility design matrix:
    I_h ≈ (1/sigma^2) * X_h^T X_h for m utility equations.
    """

    if candidate_agent_features.dim() != 1:
        raise ValueError("candidate_agent_features must be a 1D vector")
    if alternative_features.dim() != 2:
        raise ValueError("alternative_features must be a 2D matrix")
    if n_alternatives != alternative_features.size(0):
        raise ValueError("n_alternatives must match alternative_features row count")

    device = candidate_agent_features.device
    
    # Vectorized design matrix construction
    # delta part: eye(m)
    # interaction part: kron(agent_x, alternative_features)
    # agent_x is (K,), alternative_features is (m, L)
    # result should be (m, m + K*L)
    
    m = n_alternatives
    k = candidate_agent_features.size(0)
    l = alternative_features.size(1)
    
    # Kronecker of a vector with a matrix
    # torch.kron(A, B) where A is (1, K) and B is (m, L) -> (m, K*L)
    interaction_block = torch.kron(candidate_agent_features.unsqueeze(0), alternative_features)
    
    design = torch.cat([torch.eye(m, device=device, dtype=torch.float64), interaction_block], dim=1)
    
    return (design.T @ design) / (sigma**2)


def observed_fisher_information(
    params: GRUMParameters,
    agent_features: Tensor,
    alternative_features: Tensor,
    sigma: float,
) -> Tensor:
    """Approximate observed Fisher information J_D(theta).

    This baseline computes a deterministic approximation from stacked linear rows over all
    observed agents and alternatives in the Normal-family model.
    """

    if agent_features.dim() != 2 or alternative_features.dim() != 2:
        raise ValueError("feature matrices must be 2D")

    device = agent_features.device
    m = params.n_alternatives
    n_agents = agent_features.size(0)
    k = params.n_agent_features
    l = params.n_alternative_features
    
    # We need to construct the full design matrix of size (N*m, m + K*L)
    # N = n_agents, m = n_alts
    
    # delta_block: repeat eye(m) N times vertically
    delta_block = torch.eye(m, device=device, dtype=torch.float64).repeat(n_agents, 1)
    
    # interaction_block: for each agent i, kron(x_i, Z) -> (m, K*L)
    # Stacked: (N*m, K*L)
    # This is exactly kron(agent_features, alternative_features) if we reshape carefully?
    # No, kron(A, B) where A is (N, K) and B is (m, L) results in (N*m, K*L) with the right structure
    interaction_block = torch.kron(agent_features, alternative_features)
    
    design = torch.cat([delta_block, interaction_block], dim=1)
    
    return (design.T @ design) / (sigma**2)


def posterior_precision(
    observed_fisher: Tensor,
    prior_precision: float,
) -> Tensor:
    if observed_fisher.dim() != 2 or observed_fisher.size(0) != observed_fisher.size(1):
        raise ValueError("observed_fisher must be a square matrix")
    return observed_fisher + (prior_precision * torch.eye(observed_fisher.size(0), device=observed_fisher.device, dtype=torch.float64))
