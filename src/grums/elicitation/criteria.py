"""Design criteria for adaptive elicitation in GRUM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

Tensor = torch.Tensor


class DesignCriterion(Protocol):
    def score(self, prior_plus_candidate_info: Tensor, theta_vector: Tensor) -> float:
        """Return scalar score where larger values are better."""


class RandomCriterion:
    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def score(self, prior_plus_candidate_info: Tensor, theta_vector: Tensor) -> float:
        _ = prior_plus_candidate_info
        _ = theta_vector
        return float(torch.rand(1, generator=self.generator).item())


@dataclass(frozen=True)
class DOptimalityCriterion:
    jitter: float = 1e-9

    def score(self, prior_plus_candidate_info: Tensor, theta_vector: Tensor) -> float:
        _ = theta_vector
        matrix = prior_plus_candidate_info + (self.jitter * torch.eye(prior_plus_candidate_info.size(0), device=prior_plus_candidate_info.device))
        logdet = torch.logdet(matrix)
        if not torch.isfinite(logdet):
            return float("-inf")
        return float(logdet.item())


@dataclass(frozen=True)
class EOptimalityCriterion:
    def score(self, prior_plus_candidate_info: Tensor, theta_vector: Tensor) -> float:
        _ = theta_vector
        eigvals = torch.linalg.eigvalsh(prior_plus_candidate_info)
        return float(eigvals.min().item())


@dataclass(frozen=True)
class SocialChoiceCriterion:
    n_alternatives: int
    min_variance: float = 1e-12

    def score(self, prior_plus_candidate_info: Tensor, theta_vector: Tensor) -> float:
        if theta_vector.dim() != 1:
            raise ValueError("theta_vector must be 1D")
        if theta_vector.size(0) < self.n_alternatives:
            raise ValueError("theta_vector shorter than number of alternatives")

        delta = theta_vector[: self.n_alternatives]
        covariance = torch.linalg.inv(prior_plus_candidate_info)

        # Vectorized pairwise certainty calculation
        j1_idx, j2_idx = torch.triu_indices(self.n_alternatives, self.n_alternatives, offset=1)
        
        diff_mean = torch.abs(delta[j1_idx] - delta[j2_idx])
        diff_var = (
            covariance[j1_idx, j1_idx]
            + covariance[j2_idx, j2_idx]
            - 2.0 * covariance[j1_idx, j2_idx]
        )
        diff_std = torch.sqrt(torch.clamp(diff_var, min=self.min_variance))
        certainty = diff_mean / diff_std
        
        return float(certainty.min().item())


@dataclass(frozen=True)
class PersonalizedChoiceCriterion:
    """Approximate Eq. (6) by averaging per-agent pairwise certainty."""

    n_alternatives: int
    n_agent_features: int
    n_alternative_features: int
    alternative_features: Tensor
    population_agents: Tensor
    min_variance: float = 1e-12

    def score(self, prior_plus_candidate_info: Tensor, theta_vector: Tensor) -> float:
        if theta_vector.dim() != 1:
            raise ValueError("theta_vector must be 1D")
        if self.alternative_features.shape != (self.n_alternatives, self.n_alternative_features):
            raise ValueError("alternative_features has incompatible shape")
        if self.population_agents.dim() != 2 or self.population_agents.size(1) != self.n_agent_features:
            raise ValueError("population_agents has incompatible shape")

        expected_len = self.n_alternatives + (self.n_agent_features * self.n_alternative_features)
        if theta_vector.size(0) < expected_len:
            raise ValueError("theta_vector is shorter than expected parameter size")

        delta = theta_vector[: self.n_alternatives].to(torch.float64)
        b_vec = theta_vector[
            self.n_alternatives : self.n_alternatives + (self.n_agent_features * self.n_alternative_features)
        ].to(torch.float64)
        b = b_vec.view(self.n_agent_features, self.n_alternative_features)
        covariance = torch.linalg.inv(prior_plus_candidate_info.to(torch.float64))

        j1_idx, j2_idx = torch.triu_indices(self.n_alternatives, self.n_alternatives, offset=1)
        
        z_diff = self.alternative_features[j1_idx].to(torch.float64) - self.alternative_features[j2_idx].to(torch.float64)
        delta_diff = delta[j1_idx] - delta[j2_idx]
        
        per_agent_scores: list[float] = []
        for x in self.population_agents:
            x_f64 = x.to(torch.float64)
            # x is (K,), b is (K, L), z_diff is (n_pairs, L)
            # mu_diff = delta_diff + x @ b @ z_diff.T
            mu_diff = delta_diff + (x_f64 @ b @ z_diff.T)
            diff_mean = torch.abs(mu_diff)
            
            # kronecker product for gradients
            grad_int = (x_f64.view(1, -1, 1) * z_diff.view(-1, 1, self.n_alternative_features)).reshape(len(j1_idx), -1)
            
            # Full gradient for delta differences
            grad_delta = torch.zeros((len(j1_idx), self.n_alternatives), device=x.device, dtype=torch.float64)
            grad_delta[torch.arange(len(j1_idx)), j1_idx] = 1.0
            grad_delta[torch.arange(len(j1_idx)), j2_idx] = -1.0
            
            grad = torch.cat([grad_delta, grad_int], dim=1) # (n_pairs, total_params)
            
            diff_var = torch.diag(grad @ covariance @ grad.T)
            diff_std = torch.sqrt(torch.clamp(diff_var, min=self.min_variance))
            certainty = diff_mean / diff_std
            per_agent_scores.append(certainty.min().item())

        return float(torch.tensor(per_agent_scores).mean().item())
