"""Adaptive elicitation loop (Algorithm 1 style) for the social-choice track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from grums.contracts import AgentRecord, AlternativeRecord, PreferenceProvider, RankingObservation
from grums.core.parameters import GRUMParameters
from grums.elicitation.criteria import DesignCriterion
from grums.inference import (
    MCEMConfig,
    MCEMInference,
    candidate_fisher_information,
    observed_fisher_information,
    posterior_precision,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ElicitationStep:
    iteration: int
    selected_agent_id: str
    criterion_score: float
    n_observations: int


@dataclass(frozen=True)
class AdaptiveElicitationResult:
    final_params: GRUMParameters
    observations: tuple[RankingObservation, ...]
    history: tuple[ElicitationStep, ...]


class AdaptiveElicitationEngine:
    """Provider-agnostic adaptive elicitation loop."""

    def __init__(
        self,
        criterion: DesignCriterion,
        mcem_config: MCEMConfig | None = None,
    ) -> None:
        self.criterion = criterion
        self.mcem_config = mcem_config or MCEMConfig()
        self.inference = MCEMInference(self.mcem_config)

    def run(
        self,
        provider: PreferenceProvider,
        initial_params: GRUMParameters,
        initial_observations: list[RankingObservation],
        observed_agents: list[AgentRecord],
        candidate_agents: list[AgentRecord],
        alternatives: list[AlternativeRecord],
        n_rounds: int,
    ) -> AdaptiveElicitationResult:
        if len(initial_observations) == 0:
            raise ValueError("initial_observations must not be empty")
        if len(observed_agents) != len(initial_observations):
            raise ValueError("observed_agents must align with initial_observations")

        if n_rounds < 0:
            raise ValueError("n_rounds must be non-negative")

        alternatives_sorted = sorted(alternatives, key=lambda a: a.alternative_id)
        alternative_features = np.vstack([a.features for a in alternatives_sorted])

        observations = list(initial_observations)
        queried_ids = {o.agent_id for o in observations}
        history: list[ElicitationStep] = []

        observed_lookup = {a.agent_id: a for a in observed_agents}
        candidate_lookup = {a.agent_id: a for a in candidate_agents}

        params = initial_params

        for t in range(1, n_rounds + 1):
            aligned_agents = [observed_lookup[o.agent_id] for o in observations]
            agent_features = np.vstack([a.features for a in aligned_agents])
            rankings = [o.ranking for o in observations]

            fit = self.inference.fit_map(
                initial_params=params,
                rankings=rankings,
                agent_features=agent_features,
                alternative_features=alternative_features,
            )
            params = fit.params

            obs_fisher = observed_fisher_information(
                params,
                agent_features,
                alternative_features,
                sigma=self.mcem_config.sigma,
            )
            base_precision = posterior_precision(obs_fisher, self.mcem_config.prior_precision)
            theta_vec = np.concatenate([params.delta, params.interaction.reshape(-1)])

            best_agent: AgentRecord | None = None
            best_score = float("-inf")

            for candidate in candidate_agents:
                if candidate.agent_id in queried_ids:
                    continue
                cand_info = candidate_fisher_information(
                    candidate.features,
                    alternative_features,
                    n_alternatives=params.n_alternatives,
                    sigma=self.mcem_config.sigma,
                )
                score = self.criterion.score(base_precision + cand_info, theta_vec)
                if score > best_score:
                    best_score = score
                    best_agent = candidate

            if best_agent is None:
                break

            new_obs = provider.query_full_ranking(best_agent, alternatives_sorted)
            observations.append(new_obs)
            queried_ids.add(best_agent.agent_id)
            observed_lookup[best_agent.agent_id] = candidate_lookup[best_agent.agent_id]

            history.append(
                ElicitationStep(
                    iteration=t,
                    selected_agent_id=best_agent.agent_id,
                    criterion_score=float(best_score),
                    n_observations=len(observations),
                )
            )

        # MAP on full D: in-loop fit runs before each new query, so the last elicited
        # ranking is only incorporated here (and n_rounds=0 still gets a MAP on seed data).
        if observations:
            aligned_agents = [observed_lookup[o.agent_id] for o in observations]
            agent_features = np.vstack([a.features for a in aligned_agents])
            rankings = [o.ranking for o in observations]
            final_fit = self.inference.fit_map(
                initial_params=params,
                rankings=rankings,
                agent_features=agent_features,
                alternative_features=alternative_features,
            )
            params = final_fit.params

        return AdaptiveElicitationResult(
            final_params=params,
            observations=tuple(observations),
            history=tuple(history),
        )
