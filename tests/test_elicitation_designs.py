import torch
import pytest
from grums.elicitation.designs import FullRankingDesign, PairwiseDesign
from grums.contracts import AgentRecord, AlternativeRecord
from grums.core.parameters import GRUMParameters
from grums.inference.fisher import candidate_fisher_information

def test_full_ranking_design_information():
    m, k, l = 5, 2, 2
    device = torch.device("cpu")
    
    agent = AgentRecord(agent_id="a1", features=torch.randn(k))
    alternatives = [
        AlternativeRecord(alternative_id=i, features=torch.randn(l)) 
        for i in range(m)
    ]
    params = GRUMParameters(
        delta=torch.zeros(m),
        interaction=torch.zeros((k, l))
    )
    
    design = FullRankingDesign(agent, alternatives)
    info = design.get_information(params, sigma=1.0)
    
    # Shape should be (m + k*l, m + k*l) -> (5 + 4, 5 + 4) -> (9, 9)
    assert info.shape == (9, 9)
    assert torch.all(torch.isfinite(info))
    
    # Manual check against candidate_fisher_information
    alt_features = torch.vstack([a.features for a in alternatives])
    expected = candidate_fisher_information(agent.features, alt_features, m, sigma=1.0)
    assert torch.allclose(info, expected)

def test_pairwise_design_information():
    m, k, l = 5, 2, 2
    agent = AgentRecord(agent_id="a1", features=torch.randn(k))
    alt_a = AlternativeRecord(alternative_id=1, features=torch.randn(l))
    alt_b = AlternativeRecord(alternative_id=3, features=torch.randn(l))
    
    params = GRUMParameters(
        delta=torch.zeros(m),
        interaction=torch.zeros((k, l))
    )
    
    design = PairwiseDesign(agent, alt_a, alt_b)
    info = design.get_information(params, sigma=1.0)
    
    # Shape should be (9, 9)
    assert info.shape == (9, 9)
    # Pairwise info is rank-1
    eigvals = torch.linalg.eigvalsh(info)
    # Should have only one significant eigenvalue
    assert (eigvals > 1e-10).sum() == 1
    
    # Check that it scale correctly with sigma
    info_half = design.get_information(params, sigma=2.0)
    assert torch.allclose(info / 4.0, info_half)

def test_pairwise_design_diff_direction():
    # Verify that info matrix points in the (psi_a - psi_b) direction
    m, k, l = 3, 1, 1
    agent = AgentRecord(agent_id="a1", features=torch.tensor([1.0]))
    alt_a = AlternativeRecord(alternative_id=0, features=torch.tensor([1.0]))
    alt_b = AlternativeRecord(alternative_id=1, features=torch.tensor([-1.0]))
    
    params = GRUMParameters(delta=torch.zeros(3), interaction=torch.zeros((1, 1)))
    design = PairwiseDesign(agent, alt_a, alt_b)
    info = design.get_information(params, sigma=1.0)
    
    # Psi_a = [1, 0, 0, 1*1] = [1, 0, 0, 1]
    # Psi_b = [0, 1, 0, 1*-1] = [0, 1, 0, -1]
    # diff = [1, -1, 0, 2]
    # info = 1/2 * diff @ diff.T
    expected_diff = torch.tensor([1.0, -1.0, 0.0, 2.0], dtype=torch.float64)
    expected_info = 0.5 * torch.outer(expected_diff, expected_diff)
    
    assert torch.allclose(info, expected_info)
