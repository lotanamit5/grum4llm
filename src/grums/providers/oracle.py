"""Deterministic rankings from a fixed lookup (simulation / replay)."""

from __future__ import annotations

from grums.contracts import AgentRecord, AlternativeRecord, PreferenceProvider, RankingObservation


class OracleRankingProvider(PreferenceProvider):
    """Returns precomputed full rankings by agent id (no source metadata exposed)."""

    def __init__(self, ranking_by_agent_id: dict[str, tuple[int, ...]]) -> None:
        self._ranking_by_agent_id = ranking_by_agent_id

    def query_full_ranking(
        self,
        agent: AgentRecord,
        alternatives: list[AlternativeRecord],
    ) -> RankingObservation:
        _ = alternatives
        return RankingObservation(agent_id=agent.agent_id, ranking=self._ranking_by_agent_id[agent.agent_id])
