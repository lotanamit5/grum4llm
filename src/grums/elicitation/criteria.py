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
