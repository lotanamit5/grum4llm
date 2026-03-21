from __future__ import annotations

import numpy as np

from grums.contracts import AgentRecord, AlternativeRecord, PreferenceProvider, RankingObservation
from grums.core import GRUMParameters
from grums.elicitation import AdaptiveElicitationEngine
from grums.inference import MCEMConfig, MCEMInference


class TraceCriterion:
    def score(self, prior_plus_candidate_info: np.ndarray, theta_vector: np.ndarray) -> float:
        _ = theta_vector
        return float(np.trace(prior_plus_candidate_info))


class MockProvider(PreferenceProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def query_full_ranking(
        self,
        agent: AgentRecord,
        alternatives: list[AlternativeRecord],
    ) -> RankingObservation:
        self.calls.append(agent.agent_id)
        ranking = tuple(a.alternative_id for a in alternatives)
        return RankingObservation(agent_id=agent.agent_id, ranking=ranking)


def _fixture_data():
    alternatives = [
        AlternativeRecord(alternative_id=0, features=np.array([1.0, 0.0], dtype=float)),
        AlternativeRecord(alternative_id=1, features=np.array([0.2, 1.0], dtype=float)),
        AlternativeRecord(alternative_id=2, features=np.array([0.3, -0.4], dtype=float)),
    ]

    seed_agent = AgentRecord(agent_id="seed", features=np.array([0.1, 0.1], dtype=float))
    initial_obs = [RankingObservation(agent_id="seed", ranking=(0, 1, 2))]

    candidates = [
        AgentRecord(agent_id="low", features=np.array([0.1, 0.1], dtype=float)),
        AgentRecord(agent_id="high", features=np.array([3.0, 3.0], dtype=float)),
    ]

    init_params = GRUMParameters(delta=np.zeros(3), interaction=np.zeros((2, 2)))

    return alternatives, seed_agent, initial_obs, candidates, init_params


def test_engine_selects_best_candidate_via_criterion() -> None:
    alternatives, seed_agent, initial_obs, candidates, init_params = _fixture_data()

    provider = MockProvider()
    engine = AdaptiveElicitationEngine(
        criterion=TraceCriterion(),
        mcem_config=MCEMConfig(n_iterations=2, n_gibbs_samples=15, n_gibbs_burnin=10, random_seed=5),
    )

    result = engine.run(
        provider=provider,
        initial_params=init_params,
        initial_observations=initial_obs,
        observed_agents=[seed_agent],
        candidate_agents=candidates,
        alternatives=alternatives,
        n_rounds=1,
    )

    assert len(result.history) == 1
    assert result.history[0].selected_agent_id == "high"
    assert provider.calls == ["high"]


def test_engine_updates_observations_and_history_across_rounds() -> None:
    alternatives, seed_agent, initial_obs, candidates, init_params = _fixture_data()

    provider = MockProvider()
    engine = AdaptiveElicitationEngine(
        criterion=TraceCriterion(),
        mcem_config=MCEMConfig(n_iterations=2, n_gibbs_samples=15, n_gibbs_burnin=10, random_seed=7),
    )

    result = engine.run(
        provider=provider,
        initial_params=init_params,
        initial_observations=initial_obs,
        observed_agents=[seed_agent],
        candidate_agents=candidates,
        alternatives=alternatives,
        n_rounds=2,
    )

    assert len(result.history) == 2
    assert len(result.observations) == 3
    assert result.history[-1].n_observations == 3


def test_final_params_match_standalone_map_on_all_observations() -> None:
    """Returned MAP must use every observation, including the last elicited one."""
    alternatives, seed_agent, initial_obs, candidates, init_params = _fixture_data()
    cfg = MCEMConfig(n_iterations=2, n_gibbs_samples=15, n_gibbs_burnin=10, random_seed=11)

    provider = MockProvider()
    engine = AdaptiveElicitationEngine(criterion=TraceCriterion(), mcem_config=cfg)
    result = engine.run(
        provider=provider,
        initial_params=init_params,
        initial_observations=initial_obs,
        observed_agents=[seed_agent],
        candidate_agents=candidates,
        alternatives=alternatives,
        n_rounds=1,
    )

    assert len(result.observations) == 2
    alt_sorted = sorted(alternatives, key=lambda a: a.alternative_id)
    alt_features = np.vstack([a.features for a in alt_sorted])
    seed_ranking = initial_obs[0].ranking
    seed_feat = np.vstack([seed_agent.features])
    inf = MCEMInference(cfg)
    mid = inf.fit_map(
        initial_params=init_params,
        rankings=[seed_ranking],
        agent_features=seed_feat,
        alternative_features=alt_features,
    )
    agents_aligned = []
    for obs in result.observations:
        if obs.agent_id == "seed":
            agents_aligned.append(seed_agent)
        else:
            agents_aligned.append(next(a for a in candidates if a.agent_id == obs.agent_id))
    agent_features = np.vstack([a.features for a in agents_aligned])
    rankings = [o.ranking for o in result.observations]

    ref = inf.fit_map(
        initial_params=mid.params,
        rankings=rankings,
        agent_features=agent_features,
        alternative_features=alt_features,
    )

    np.testing.assert_allclose(result.final_params.delta, ref.params.delta, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(result.final_params.interaction, ref.params.interaction, rtol=1e-5, atol=1e-5)


def test_engine_rejects_missing_initial_data() -> None:
    alternatives, _, _, candidates, init_params = _fixture_data()

    provider = MockProvider()
    engine = AdaptiveElicitationEngine(criterion=TraceCriterion())

    try:
        _ = engine.run(
            provider=provider,
            initial_params=init_params,
            initial_observations=[],
            observed_agents=[],
            candidate_agents=candidates,
            alternatives=alternatives,
            n_rounds=1,
        )
    except ValueError as err:
        assert "initial_observations" in str(err)
    else:
        raise AssertionError("Expected ValueError for empty initial observations")
