"""Shared contracts between GRUM core and external providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union

import torch

ArrayF64 = torch.Tensor


@dataclass(frozen=True)
class RankingObservation:
    """One agent ranking over alternatives.

    Alternatives are identified by integer ids and listed in descending preference.
    """

    agent_id: str
    ranking: tuple[int, ...]


@dataclass(frozen=True)
class PairwiseObservation:
    """A pairwise comparison where winner_id is strictly preferred over loser_id."""

    agent_id: str
    winner_id: int
    loser_id: int


Observation = Union[RankingObservation, PairwiseObservation]


@dataclass(frozen=True)
class AgentRecord:
    """Agent features used by GRUM."""

    agent_id: str
    features: ArrayF64


@dataclass(frozen=True)
class AlternativeRecord:
    """Alternative features used by GRUM."""

    alternative_id: int
    features: ArrayF64


class PreferenceProvider(Protocol):
    """Provider boundary for preference elicitation queries."""

    def query_full_ranking(self, agent: AgentRecord, alternatives: list[AlternativeRecord]) -> RankingObservation:
        """Return one full ranking for the selected agent."""

    def query_pairwise(self, agent: AgentRecord, alt_a: AlternativeRecord, alt_b: AlternativeRecord) -> PairwiseObservation:
        """Return a pairwise comparison for two alternatives, selecting the preferred winner."""


def compile_constraint_graph(observations: list[Observation]) -> dict[str, list[tuple[int, int]]]:
    """Compiles distinct observations into a unified list of (winner, loser) pairs per agent.
    
    Returns:
        dict mapping agent_id -> list of (winner, loser) edges evaluating U_winner > U_loser.
    """
    graph: dict[str, list[tuple[int, int]]] = {}
    
    for obs in observations:
        edges = graph.setdefault(obs.agent_id, [])
        if isinstance(obs, RankingObservation):
            for i in range(len(obs.ranking) - 1):
                edges.append((obs.ranking[i], obs.ranking[i+1]))
        elif isinstance(obs, PairwiseObservation):
            edges.append((obs.winner_id, obs.loser_id))
            
    return graph
