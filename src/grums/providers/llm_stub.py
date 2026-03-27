"""Placeholder provider for LLM-backed elicitation (wire real API later)."""

from __future__ import annotations

from grums.contracts import AgentRecord, AlternativeRecord, PreferenceProvider, RankingObservation, PairwiseObservation


class StubLLMPreferenceProvider(PreferenceProvider):
    """Deterministic stub: ranks alternatives by ascending alternative_id (for plumbing tests only)."""

    def query_full_ranking(
        self,
        agent: AgentRecord,
        alternatives: list[AlternativeRecord],
    ) -> RankingObservation:
        alts = sorted(alternatives, key=lambda a: a.alternative_id)
        ranking = tuple(a.alternative_id for a in alts)
        return RankingObservation(agent_id=agent.agent_id, ranking=ranking)

    def query_pairwise(
        self,
        agent: AgentRecord,
        alt_a: AlternativeRecord,
        alt_b: AlternativeRecord,
    ) -> PairwiseObservation:
        """Always prefers the alternative with the lower id (deterministic)."""
        if alt_a.alternative_id <= alt_b.alternative_id:
            return PairwiseObservation(agent_id=agent.agent_id, winner_id=alt_a.alternative_id, loser_id=alt_b.alternative_id)
        return PairwiseObservation(agent_id=agent.agent_id, winner_id=alt_b.alternative_id, loser_id=alt_a.alternative_id)
