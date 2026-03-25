import torch
import pytest
from grums.elicitation.engine import AdaptiveElicitationEngine
from grums.elicitation.criteria import SocialChoiceCriterion
from grums.elicitation.designs import FullRankingDesign, PairwiseDesign
from grums.providers.synthetic import SyntheticProvider
from grums.core.parameters import GRUMParameters
from grums.inference import MCEMConfig
from grums.experiments.metrics import social_choice_kendall_tau
import itertools
import time

def test_elicitation_e2e_comparison():
    # Use a tiny version of ds0
    seed = 42
    m = 5
    provider = SyntheticProvider(ds_name="ds0", seed=seed)
    
    # Take first 10 agents for the test
    all_agents = provider.agents[:10]
    alternatives = provider.alternatives # 5 alts
    
    initial_agents = all_agents[:2]  # 2 initial
    candidate_agents = all_agents[2:] # 8 candidates
    
    initial_obs = [
        provider.query_full_ranking(a, alternatives) for a in initial_agents
    ]
    
    mcem_config = MCEMConfig(
        n_iterations=5, 
        n_gibbs_samples=20, 
        n_gibbs_burnin=10,
        sigma=1.0
    )
    criterion = SocialChoiceCriterion(n_alternatives=m)
    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=mcem_config)
    
    init_params = GRUMParameters(
        delta=torch.zeros(m),
        interaction=torch.zeros((all_agents[0].features.size(0), alternatives[0].features.size(0)))
    )
    
    # 1. Run with Full Ranking Designs
    full_designs = [FullRankingDesign(a, alternatives) for a in candidate_agents]
    t0_full = time.perf_counter()
    res_full = engine.run(
        provider=provider,
        initial_params=init_params,
        initial_observations=initial_obs,
        observed_agents=initial_agents,
        candidate_designs=full_designs,
        alternatives=alternatives,
        n_rounds=3
    )
    t1_full = time.perf_counter()
    dt_full = t1_full - t0_full
    
    tau_full = social_choice_kendall_tau(provider.true_params.delta, res_full.final_params.delta)
    
    # 2. Run with Pairwise Designs
    # We generate all pairs for all candidate agents
    pairwise_designs = []
    for a in candidate_agents:
        for alt1, alt2 in itertools.combinations(alternatives, 2):
            pairwise_designs.append(PairwiseDesign(a, alt1, alt2))
            
    t0_pair = time.perf_counter()
    res_pair = engine.run(
        provider=provider,
        initial_params=init_params,
        initial_observations=initial_obs,
        observed_agents=initial_agents,
        candidate_designs=pairwise_designs,
        alternatives=alternatives,
        # We run 3 rounds.
        n_rounds=3
    )
    t1_pair = time.perf_counter()
    dt_pair = t1_pair - t0_pair
    
    tau_pair = social_choice_kendall_tau(provider.true_params.delta, res_pair.final_params.delta)
    
    print(f"\nResults (3 rounds):")
    print(f"Full Ranking: Tau={tau_full:.4f}, Time={dt_full:.4f}s")
    print(f"Pairwise:     Tau={tau_pair:.4f}, Time={dt_pair:.4f}s")
    
    # Assertions
    assert len(res_full.observations) == 5 # 2 + 3
    assert len(res_pair.observations) == 5 # 2 + 3
    assert torch.all(torch.isfinite(res_full.final_params.delta))
    assert torch.all(torch.isfinite(res_pair.final_params.delta))
    assert -1.0 <= tau_full <= 1.0
    assert -1.0 <= tau_pair <= 1.0
