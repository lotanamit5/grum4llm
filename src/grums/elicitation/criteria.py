"""Design criteria for adaptive elicitation in GRUM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class DesignCriterion(Protocol):
    def score(self, prior_plus_candidate_info: FloatArray, theta_vector: FloatArray) -> float:
        """Return scalar score where larger values are better."""


@dataclass(frozen=True)
class DOptimalityCriterion:
    jitter: float = 1e-9

    def score(self, prior_plus_candidate_info: FloatArray, theta_vector: FloatArray) -> float:
        _ = theta_vector
        matrix = prior_plus_candidate_info + (self.jitter * np.eye(prior_plus_candidate_info.shape[0]))
        sign, logdet = np.linalg.slogdet(matrix)
        if sign <= 0:
            return float("-inf")
        return float(logdet)


@dataclass(frozen=True)
class EOptimalityCriterion:
    def score(self, prior_plus_candidate_info: FloatArray, theta_vector: FloatArray) -> float:
        _ = theta_vector
        eigvals = np.linalg.eigvalsh(prior_plus_candidate_info)
        return float(eigvals.min())


@dataclass(frozen=True)
class SocialChoiceCriterion:
    n_alternatives: int
    min_variance: float = 1e-12

    def score(self, prior_plus_candidate_info: FloatArray, theta_vector: FloatArray) -> float:
        if theta_vector.ndim != 1:
            raise ValueError("theta_vector must be 1D")
        if theta_vector.shape[0] < self.n_alternatives:
            raise ValueError("theta_vector shorter than number of alternatives")

        delta = theta_vector[: self.n_alternatives]
        covariance = np.linalg.inv(prior_plus_candidate_info)

        min_certainty = float("inf")
        for j1 in range(self.n_alternatives):
            for j2 in range(j1 + 1, self.n_alternatives):
                diff_mean = abs(delta[j1] - delta[j2])
                diff_var = (
                    covariance[j1, j1]
                    + covariance[j2, j2]
                    - 2.0 * covariance[j1, j2]
                )
                diff_std = np.sqrt(max(diff_var, self.min_variance))
                certainty = diff_mean / diff_std
                min_certainty = min(min_certainty, certainty)

        return float(min_certainty)


@dataclass(frozen=True)
class PersonalizedChoiceCriterion:
    """Approximate Eq. (6) by averaging per-agent pairwise certainty."""

    n_alternatives: int
    n_agent_features: int
    n_alternative_features: int
    alternative_features: FloatArray
    population_agents: FloatArray
    min_variance: float = 1e-12

    def score(self, prior_plus_candidate_info: FloatArray, theta_vector: FloatArray) -> float:
        if theta_vector.ndim != 1:
            raise ValueError("theta_vector must be 1D")
        if self.alternative_features.shape != (self.n_alternatives, self.n_alternative_features):
            raise ValueError("alternative_features has incompatible shape")
        if self.population_agents.ndim != 2 or self.population_agents.shape[1] != self.n_agent_features:
            raise ValueError("population_agents has incompatible shape")

        expected_len = self.n_alternatives + (self.n_agent_features * self.n_alternative_features)
        if theta_vector.shape[0] < expected_len:
            raise ValueError("theta_vector is shorter than expected parameter size")

        delta = theta_vector[: self.n_alternatives]
        b_vec = theta_vector[
            self.n_alternatives : self.n_alternatives + (self.n_agent_features * self.n_alternative_features)
        ]
        b = b_vec.reshape(self.n_agent_features, self.n_alternative_features)
        covariance = np.linalg.inv(prior_plus_candidate_info)

        per_agent_scores: list[float] = []
        for x in self.population_agents:
            min_certainty = float("inf")
            for j1 in range(self.n_alternatives):
                for j2 in range(j1 + 1, self.n_alternatives):
                    z1 = self.alternative_features[j1]
                    z2 = self.alternative_features[j2]

                    mu1 = float(delta[j1] + x @ b @ z1)
                    mu2 = float(delta[j2] + x @ b @ z2)
                    diff_mean = abs(mu1 - mu2)

                    grad = np.zeros(expected_len, dtype=float)
                    grad[j1] = 1.0
                    grad[j2] = -1.0
                    grad[self.n_alternatives :] = np.kron(x, z1 - z2)

                    diff_var = float(grad @ covariance @ grad)
                    diff_std = np.sqrt(max(diff_var, self.min_variance))
                    certainty = diff_mean / diff_std
                    min_certainty = min(min_certainty, certainty)
            per_agent_scores.append(min_certainty)

        return float(np.mean(per_agent_scores))
