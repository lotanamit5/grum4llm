from grums.contracts import AgentRecord, AlternativeRecord, PreferenceProvider, RankingObservation, PairwiseObservation


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

    def query_pairwise(
        self,
        agent: AgentRecord,
        alt_a: AlternativeRecord,
        alt_b: AlternativeRecord,
    ) -> PairwiseObservation:
        ranking = self._ranking_by_agent_id[agent.agent_id]
        idx_a = ranking.index(alt_a.alternative_id)
        idx_b = ranking.index(alt_b.alternative_id)
        
        if idx_a < idx_b:
            return PairwiseObservation(agent_id=agent.agent_id, winner_id=alt_a.alternative_id, loser_id=alt_b.alternative_id)
        else:
            return PairwiseObservation(agent_id=agent.agent_id, winner_id=alt_b.alternative_id, loser_id=alt_a.alternative_id)
