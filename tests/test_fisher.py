import torch

from grums.core.parameters import GRUMParameters
from grums.inference import (
    candidate_fisher_information,
    observed_fisher_information,
    posterior_precision,
)


def test_observed_fisher_shape_and_symmetry() -> None:
    params = GRUMParameters(delta=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64), 
                           interaction=torch.zeros((2, 2), dtype=torch.float64))
    x = torch.tensor([[1.0, 0.0], [0.3, 0.7]], dtype=torch.float64)
    z = torch.tensor([[1.0, 0.0], [0.5, 1.2], [0.2, -0.4]], dtype=torch.float64)

    fisher = observed_fisher_information(params, x, z, sigma=1.0)
    expected_dim = 3 + (2 * 2)

    assert fisher.shape == (expected_dim, expected_dim)
    torch.testing.assert_close(fisher, fisher.T)


def test_candidate_fisher_is_psd() -> None:
    x = torch.tensor([0.3, 1.1], dtype=torch.float64)
    z = torch.tensor([[1.0, 0.0], [0.2, 1.0], [0.5, -0.3]], dtype=torch.float64)

    fisher = candidate_fisher_information(x, z, n_alternatives=3, sigma=1.0)
    eigvals = torch.linalg.eigvalsh(fisher)

    assert torch.all(eigvals >= -1e-10)


def test_posterior_precision_adds_prior_diagonal() -> None:
    base = torch.eye(5, dtype=torch.float64)
    precision = posterior_precision(base, prior_precision=0.1)
    torch.testing.assert_close(torch.diag(precision), torch.tensor([1.1] * 5, dtype=torch.float64))
