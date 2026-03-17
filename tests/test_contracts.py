import numpy as np

from grums.contracts import AgentRecord, AlternativeRecord, RankingObservation


def test_contracts_hold_expected_values() -> None:
    agent = AgentRecord(agent_id="a1", features=np.array([1.0, 2.0]))
    alternatives = [
        AlternativeRecord(alternative_id=0, features=np.array([0.1, 0.2])),
        AlternativeRecord(alternative_id=1, features=np.array([0.3, 0.4])),
    ]
    obs = RankingObservation(agent_id=agent.agent_id, ranking=(1, 0))

    assert agent.features.shape == (2,)
    assert len(alternatives) == 2
    assert obs.ranking[0] == 1
