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
)
from grums.elicitation.designs import QueryDesign
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
        candidate_designs: list[QueryDesign],
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
        active_designs = list(candidate_designs)
        history: list[ElicitationStep] = []

        # We maintain a lookup of features for all agents we have queried or might query
        observed_lookup = {a.agent_id: a for a in observed_agents}

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

            best_design: QueryDesign | None = None
            best_score = float("-inf")
            best_idx = -1

            for idx, design in enumerate(active_designs):
                info = design.get_information(params, self.mcem_config.sigma)
                score = self.criterion.score(base_precision + info, theta_vec)
                if score > best_score:
                    best_score = score
                    best_design = design
                    best_idx = idx

            if best_design is None:
                break

            # Execute the best design
            new_obs = best_design.execute(provider)
            observations.append(new_obs)
            
            # Update lookups for future fit_map/fisher calls
            target_agent = best_design.agent
            if target_agent.agent_id not in observed_lookup:
                observed_lookup[target_agent.agent_id] = target_agent
            
            # Remove the used design
            active_designs.pop(best_idx)

            history.append(
                ElicitationStep(
                    iteration=t,
                    selected_agent_id=target_agent.agent_id,
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
