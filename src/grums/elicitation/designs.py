"""Unified query design definitions for adaptive elicitation."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from torch import Tensor
    from grums.contracts import (
        PreferenceProvider, 
        Observation, 
        AgentRecord, 
        AlternativeRecord,
        RankingObservation,
        PairwiseObservation
    )
    from grums.core.parameters import GRUMParameters


class QueryDesign(Protocol):
    """Protocol for an abstract elicitation question (design)."""
    
    @property
    def agent(self) -> AgentRecord:
        """The agent being targeted by this design."""
        ...

    def get_information(self, params: GRUMParameters, sigma: float) -> torch.Tensor:
        """Returns the Fisher Information matrix (m+KL, m+KL) for this specific query."""
        ...

    def execute(self, provider: PreferenceProvider) -> Observation:
        """Performs the actual query against the provider and returns the result."""
        ...


class FullRankingDesign:
    """A design representing a request for a full ranking from a targeted agent."""

    def __init__(self, agent: AgentRecord, alternatives: list[AlternativeRecord]):
        self.agent = agent
        self.alternatives = alternatives

    def get_information(self, params: GRUMParameters, sigma: float) -> torch.Tensor:
        from grums.inference.fisher import candidate_fisher_information
        
        # Sort alternatives by ID to ensure consistent design matrix alignment
        alts_sorted = sorted(self.alternatives, key=lambda a: a.alternative_id)
        alt_features = torch.vstack([a.features for a in alts_sorted])
        
        return candidate_fisher_information(
            self.agent.features, 
            alt_features, 
            params.n_alternatives, 
            sigma
        )

    def execute(self, provider: PreferenceProvider) -> RankingObservation:
        return provider.query_full_ranking(self.agent, self.alternatives)


class PairwiseDesign:
    """A design representing a request for a single pairwise comparison (A vs B)."""

    def __init__(self, agent: AgentRecord, alt_a: AlternativeRecord, alt_b: AlternativeRecord):
        self.agent = agent
        self.alt_a = alt_a
        self.alt_b = alt_b

    def get_information(self, params: GRUMParameters, sigma: float) -> torch.Tensor:
        """Information from difference U_A - U_B. 
        
        I_h = (1 / 2*sigma^2) * (psi_A - psi_B)(psi_A - psi_B)^T
        """
        from grums.inference.fisher import _param_design_row
        
        m = params.n_alternatives
        psi_a = _param_design_row(
            self.agent.features, 
            self.alt_a.features.to(self.agent.features.device), 
            self.alt_a.alternative_id, 
            m
        )
        psi_b = _param_design_row(
            self.agent.features, 
            self.alt_b.features.to(self.agent.features.device), 
            self.alt_b.alternative_id, 
            m
        )
        
        diff = (psi_a - psi_b).unsqueeze(1)
        # Variance of the difference noise is 2 * sigma^2 for i.i.d. Normal noise
        return (diff @ diff.T) / (2.0 * (sigma ** 2))

    def execute(self, provider: PreferenceProvider) -> PairwiseObservation:
        return provider.query_pairwise(self.agent, self.alt_a, self.alt_b)
