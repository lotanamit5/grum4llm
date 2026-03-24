"""Adaptive elicitation loop (Algorithm 1 style) for the social-choice track."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from grums.contracts import (
    AgentRecord, 
    AlternativeRecord, 
    Observation,
    PreferenceProvider, 
    RankingObservation,
    PairwiseObservation,
    compile_constraint_graph
)
from grums.core.parameters import GRUMParameters
from grums.elicitation.criteria import DesignCriterion
from grums.inference import (
    MCEMConfig,
    MCEMInference,
    candidate_fisher_information,
    observed_fisher_information,
    posterior_precision,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class ElicitationStep:
    iteration: int
    selected_agent_id: str
    criterion_score: float
    n_observations: int


@dataclass(frozen=True)
class AdaptiveElicitationResult:
    final_params: GRUMParameters
    observations: tuple[Observation, ...]
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(
        self,
        provider: PreferenceProvider,
        initial_params: GRUMParameters,
        initial_observations: list[Observation],
        observed_agents: list[AgentRecord],
        candidate_agents: list[AgentRecord],
        alternatives: list[AlternativeRecord],
        n_rounds: int,
        *,
        on_after_map: Callable[[int, GRUMParameters], None] | None = None,
    ) -> AdaptiveElicitationResult:
        if len(initial_observations) == 0:
            raise ValueError("initial_observations must not be empty")
        if len(observed_agents) != len(initial_observations):
            raise ValueError("observed_agents must align with initial_observations")

        if n_rounds < 0:
            raise ValueError("n_rounds must be non-negative")

        alternatives_sorted = sorted(alternatives, key=lambda a: a.alternative_id)
        alternative_features = torch.vstack([a.features for a in alternatives_sorted]).to(self.device).to(torch.float64)

        observations = list(initial_observations)
        queried_ids = {o.agent_id for o in observations}
        history: list[ElicitationStep] = []

        observed_lookup = {a.agent_id: a for a in observed_agents}
        candidate_lookup = {a.agent_id: a for a in candidate_agents}

        params = GRUMParameters(
            delta=initial_params.delta.to(self.device).to(torch.float64),
            interaction=initial_params.interaction.to(self.device).to(torch.float64)
        )

        for t in range(1, n_rounds + 1):
            aligned_agents = [observed_lookup[o.agent_id] for o in observations]
            agent_features = torch.vstack([a.features for a in aligned_agents]).to(self.device).to(torch.float64)

            fit = self.inference.fit_map(
                initial_params=params,
                observations=observations,
                agent_features=agent_features,
                alternative_features=alternative_features,
            )
            params = fit.params

            if on_after_map is not None:
                on_after_map(len(observations), params)

            obs_fisher = observed_fisher_information(
                params,
                agent_features,
                alternative_features,
                sigma=self.mcem_config.sigma,
            )
            base_precision = posterior_precision(obs_fisher, self.mcem_config.prior_precision)
            theta_vec = torch.cat([params.delta, params.interaction.reshape(-1)])

            best_agent: AgentRecord | None = None
            best_score = float("-inf")

            for candidate in candidate_agents:
                if candidate.agent_id in queried_ids:
                    continue
                
                cand_feat = candidate.features.to(self.device).to(torch.float64)
                cand_info = candidate_fisher_information(
                    cand_feat,
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

            # TODO: Add logic to decide whether to query full or pairwise.
            # For now, default to full ranking to maintain legacy behavior for Figure repro.
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

        # MAP on full D
        if observations:
            aligned_agents = [observed_lookup[o.agent_id] for o in observations]
            agent_features = torch.vstack([a.features for a in aligned_agents]).to(self.device).to(torch.float64)
            
            final_fit = self.inference.fit_map(
                initial_params=params,
                observations=observations,
                agent_features=agent_features,
                alternative_features=alternative_features,
            )
            params = final_fit.params

            if on_after_map is not None:
                on_after_map(len(observations), params)

        return AdaptiveElicitationResult(
            final_params=params,
            observations=tuple(observations),
            history=tuple(history),
        )
