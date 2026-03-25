import torch
import pytest
from unittest.mock import MagicMock
from grums.elicitation.engine import AdaptiveElicitationEngine
from grums.elicitation.criteria import RandomCriterion
from grums.elicitation.designs import FullRankingDesign, PairwiseDesign
from grums.contracts import AgentRecord, AlternativeRecord, RankingObservation, PairwiseObservation
from grums.core.parameters import GRUMParameters
from grums.inference import MCEMConfig

def test_engine_unified_mixed_designs():
    m, k, l = 3, 2, 2
    device = torch.device("cpu")
    
    # Setup
    alternatives = [
        AlternativeRecord(alternative_id=i, features=torch.randn(l)) 
        for i in range(m)
    ]
    agents = [
        AgentRecord(agent_id=f"a{i}", features=torch.randn(k))
        for i in range(2)
    ]
    
    # 1 full ranking design for agent 0, 1 pairwise for agent 1
    candidate_designs = [
        FullRankingDesign(agents[0], alternatives),
        PairwiseDesign(agents[1], alternatives[0], alternatives[1])
    ]
    
    initial_params = GRUMParameters(
        delta=torch.zeros(m),
        interaction=torch.zeros((k, l))
    )
    
    # Dummy provider
    provider = MagicMock()
    provider.query_full_ranking.return_value = RankingObservation(
        agent_id="a0", 
        ranking=(0, 1, 2)
    )
    provider.query_pairwise.return_value = PairwiseObservation(
        agent_id="a1",
        winner_id=0,
        loser_id=1
    )
    
    # Initial observations (required)
    initial_obs = [
        RankingObservation(agent_id="init", ranking=(2, 1, 0))
    ]
    initial_agents = [
        AgentRecord(agent_id="init", features=torch.randn(k))
    ]
    
    engine = AdaptiveElicitationEngine(
        criterion=RandomCriterion(seed=42),
        mcem_config=MCEMConfig(sigma=1.0, n_iterations=2, n_gibbs_samples=10) # Fast for testing
    )
    engine.device = device
    
    result = engine.run(
        provider=provider,
        initial_params=initial_params,
        initial_observations=initial_obs,
        observed_agents=initial_agents,
        candidate_designs=candidate_designs,
        alternatives=alternatives,
        n_rounds=2
    )
    
    # Check that both designs were used (since there were 2 rounds and 2 designs)
    assert len(result.observations) == 3 # 1 initial + 2 rounds
    assert len(result.history) == 2
    
    # Verify both provider methods called
    provider.query_full_ranking.assert_called_once()
    provider.query_pairwise.assert_called_once()
    
    assert result.final_params.delta.shape == (m,)
    assert result.final_params.interaction.shape == (k, l)

def test_engine_empty_designs_graceful_exit():
    m, k, l = 3, 2, 2
    alternatives = [AlternativeRecord(alternative_id=i, features=torch.randn(l)) for i in range(m)]
    
    initial_obs = [RankingObservation(agent_id="init", ranking=(0, 1, 2))]
    initial_agents = [AgentRecord(agent_id="init", features=torch.randn(k))]
    
    engine = AdaptiveElicitationEngine(RandomCriterion(42))
    engine.device = torch.device("cpu")
    
    result = engine.run(
        provider=MagicMock(),
        initial_params=GRUMParameters(delta=torch.zeros(m), interaction=torch.zeros((k, l))),
        initial_observations=initial_obs,
        observed_agents=initial_agents,
        candidate_designs=[], # No designs
        alternatives=alternatives,
        n_rounds=5
    )
    
    assert len(result.observations) == 1
    assert len(result.history) == 0
