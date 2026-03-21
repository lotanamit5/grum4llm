from __future__ import annotations

import numpy as np
import pytest

from grums.contracts import AgentRecord, AlternativeRecord
from grums.providers import (
    OracleRankingProvider,
    StubLLMPreferenceProvider,
    build_preference_provider,
)


def test_oracle_returns_stored_ranking() -> None:
    p = OracleRankingProvider({"u1": (2, 0, 1)})
    alts = [
        AlternativeRecord(alternative_id=0, features=np.zeros(2)),
        AlternativeRecord(alternative_id=1, features=np.zeros(2)),
        AlternativeRecord(alternative_id=2, features=np.zeros(2)),
    ]
    obs = p.query_full_ranking(AgentRecord(agent_id="u1", features=np.zeros(2)), alts)
    assert obs.ranking == (2, 0, 1)


def test_stub_llm_ranks_by_alternative_id() -> None:
    p = StubLLMPreferenceProvider()
    alts = [
        AlternativeRecord(alternative_id=2, features=np.zeros(1)),
        AlternativeRecord(alternative_id=0, features=np.zeros(1)),
    ]
    obs = p.query_full_ranking(AgentRecord(agent_id="x", features=np.zeros(1)), alts)
    assert obs.ranking == (0, 2)


def test_factory_oracle_and_stub() -> None:
    o = build_preference_provider("oracle", ranking_by_agent_id={"a": (1, 0)})
    assert isinstance(o, OracleRankingProvider)
    s = build_preference_provider("llm_stub")
    assert isinstance(s, StubLLMPreferenceProvider)


def test_factory_oracle_requires_lookup() -> None:
    with pytest.raises(ValueError, match="ranking_by_agent_id"):
        build_preference_provider("oracle")
