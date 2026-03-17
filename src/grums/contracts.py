"""Shared contracts between GRUM core and external providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

ArrayF64 = np.ndarray


@dataclass(frozen=True)
class RankingObservation:
    """One agent ranking over alternatives.

    Alternatives are identified by integer ids and listed in descending preference.
    """

    agent_id: str
    ranking: tuple[int, ...]


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
