"""Monte Carlo EM inference for Normal-family GRUM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import truncnorm

from grums.core.model_math import compute_mean_utilities
from grums.core.parameters import GRUMParameters

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MCEMConfig:
    n_iterations: int = 20
    n_gibbs_samples: int = 100
    n_gibbs_burnin: int = 50
    sigma: float = 1.0
    prior_precision: float = 1e-2
    tolerance: float = 1e-5
    random_seed: int = 0


@dataclass(frozen=True)
class MCEMResult:
    params: GRUMParameters
    objective_trace: tuple[float, ...]
    converged: bool
    n_iterations: int


class MCEMInference:
    """Algorithm-3 style MC-EM for Normal utility noise.

    This implementation prioritizes clarity and reproducibility for the baseline path.
    """

    def __init__(self, config: MCEMConfig | None = None) -> None:
        self.config = config or MCEMConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def fit_map(
        self,
        initial_params: GRUMParameters,
        rankings: list[tuple[int, ...]],
        agent_features: FloatArray,
        alternative_features: FloatArray,
    ) -> MCEMResult:
        params = initial_params
        objective_trace: list[float] = []

        converged = False
        for step in range(1, self.config.n_iterations + 1):
            s = self._e_step(params, rankings, agent_features, alternative_features)
            new_params = self._m_step(s, params, agent_features, alternative_features)

            q_val = self._q_objective(s, new_params, agent_features, alternative_features)
            objective_trace.append(float(q_val))

            param_diff = np.linalg.norm(new_params.delta - params.delta) + np.linalg.norm(
                new_params.interaction - params.interaction
            )
            params = new_params

            if param_diff < self.config.tolerance:
                converged = True
                return MCEMResult(
                    params=params,
                    objective_trace=tuple(objective_trace),
                    converged=converged,
                    n_iterations=step,
                )

        return MCEMResult(
            params=params,
            objective_trace=tuple(objective_trace),
            converged=converged,
            n_iterations=self.config.n_iterations,
        )

    def _e_step(
        self,
        params: GRUMParameters,
        rankings: list[tuple[int, ...]],
        agent_features: FloatArray,
        alternative_features: FloatArray,
    ) -> FloatArray:
        mu = compute_mean_utilities(params, agent_features, alternative_features)
        n_agents, n_alts = mu.shape
        out = np.zeros((n_agents, n_alts), dtype=float)

        for i in range(n_agents):
            ranking = rankings[i]
            if len(ranking) != n_alts:
                raise ValueError("this baseline E-step expects full rankings")
            samples = self._gibbs_samples_for_agent(mu[i], ranking)
            out[i] = samples.mean(axis=0)

        return out

    def _gibbs_samples_for_agent(self, mean_vec: FloatArray, ranking: tuple[int, ...]) -> FloatArray:
        m = len(ranking)
        sigma = self.config.sigma

        ranked_means = [float(mean_vec[a]) for a in ranking]
        for idx in range(1, m):
            if ranked_means[idx] >= ranked_means[idx - 1]:
                ranked_means[idx] = ranked_means[idx - 1] - 1e-3

        current = np.zeros(m, dtype=float)
        for pos, alt in enumerate(ranking):
            current[alt] = ranked_means[pos]

        collected: list[np.ndarray] = []
        total = self.config.n_gibbs_burnin + self.config.n_gibbs_samples

        for t in range(total):
            for pos, alt in enumerate(ranking):
                upper = np.inf if pos == 0 else current[ranking[pos - 1]]
                lower = -np.inf if pos == m - 1 else current[ranking[pos + 1]]
                mean = mean_vec[alt]
                a = (lower - mean) / sigma
                b = (upper - mean) / sigma
                current[alt] = truncnorm.rvs(a, b, loc=mean, scale=sigma, random_state=self._rng)

            if t >= self.config.n_gibbs_burnin:
                collected.append(current.copy())

        return np.vstack(collected)

    def _m_step(
        self,
        s_matrix: FloatArray,
        prev_params: GRUMParameters,
        agent_features: FloatArray,
        alternative_features: FloatArray,
    ) -> GRUMParameters:
        sigma2 = self.config.sigma**2
        lam = self.config.prior_precision

        n_agents, n_alts = s_matrix.shape
        k = agent_features.shape[1]
        l = alternative_features.shape[1]

        xbzt = agent_features @ prev_params.interaction @ alternative_features.T
        delta = (s_matrix - xbzt).sum(axis=0) / (n_agents + lam * sigma2)

        y = (s_matrix - delta.reshape(1, n_alts)).reshape(-1)
        rows: list[np.ndarray] = []
        for x in agent_features:
            for z in alternative_features:
                rows.append(np.kron(z, x))
        design = np.vstack(rows)

        ridge = lam * sigma2 * np.eye(k * l)
        lhs = design.T @ design + ridge
        rhs = design.T @ y
        b_vec = np.linalg.solve(lhs, rhs)
        b_matrix = b_vec.reshape(k, l)

        return GRUMParameters(delta=delta, interaction=b_matrix)

    def _q_objective(
        self,
        s_matrix: FloatArray,
        params: GRUMParameters,
        agent_features: FloatArray,
        alternative_features: FloatArray,
    ) -> float:
        sigma2 = self.config.sigma**2
        lam = self.config.prior_precision

        mu = compute_mean_utilities(params, agent_features, alternative_features)
        residual = s_matrix - mu
        data_term = -0.5 / sigma2 * float(np.sum(residual**2))
        prior_term = -0.5 * lam * float(np.sum(params.delta**2) + np.sum(params.interaction**2))
        return data_term + prior_term
