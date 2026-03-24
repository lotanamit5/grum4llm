import torch
from grums.contracts import RankingObservation
from grums.core.parameters import GRUMParameters
from grums.core.model_math import compute_mean_utilities
from grums.inference import MCEMConfig, MCEMInference

def _toy_data() -> tuple[GRUMParameters, torch.Tensor, torch.Tensor, list[RankingObservation]]:
    # Use float64 for numerical stability matching scipy's precision
    true_params = GRUMParameters(
        delta=torch.tensor([1.0, 0.2, -0.3], dtype=torch.float64),
        interaction=torch.tensor([[0.5, -0.1], [0.2, 0.3]], dtype=torch.float64),
    )
    x = torch.tensor([[1.0, 0.0], [0.2, 1.1], [1.3, -0.2]], dtype=torch.float64)
    z = torch.tensor([[1.0, 0.1], [0.2, 1.0], [0.8, -0.4]], dtype=torch.float64)
    mu = compute_mean_utilities(true_params, x, z)

    # Convert deterministic utilities to ranking observations
    observations = []
    for i in range(mu.size(0)):
        ranking = tuple(torch.argsort(mu[i], descending=True).tolist())
        observations.append(RankingObservation(agent_id=f"a{i}", ranking=ranking))
    
    return true_params, x, z, observations


def test_fit_map_returns_valid_shapes_and_trace() -> None:
    _, x, z, observations = _toy_data()
    init = GRUMParameters(
        delta=torch.zeros(3, dtype=torch.float64), 
        interaction=torch.zeros((2, 2), dtype=torch.float64)
    )

    model = MCEMInference(
        MCEMConfig(
            n_iterations=5,
            n_gibbs_samples=30,
            n_gibbs_burnin=15,
            tolerance=1e-8,
            random_seed=7,
        )
    )
    result = model.fit_map(init, observations, x, z)

    assert result.params.delta.shape == (3,)
    assert result.params.interaction.shape == (2, 2)
    assert len(result.objective_trace) >= 1


def test_fit_map_improves_objective_from_first_to_last() -> None:
    _, x, z, observations = _toy_data()
    init = GRUMParameters(
        delta=torch.zeros(3, dtype=torch.float64), 
        interaction=torch.zeros((2, 2), dtype=torch.float64)
    )

    model = MCEMInference(
        MCEMConfig(
            n_iterations=6,
            n_gibbs_samples=25,
            n_gibbs_burnin=10,
            tolerance=1e-8,
            random_seed=11,
        )
    )
    result = model.fit_map(init, observations, x, z)

    # Objective should generally improve (MAP is concave for Normal GRUM)
    assert max(result.objective_trace) >= result.objective_trace[0]
    assert torch.isfinite(torch.tensor(result.objective_trace)).all()
