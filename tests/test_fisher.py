import numpy as np

from grums.core import GRUMParameters
from grums.inference import (
    candidate_fisher_information,
    observed_fisher_information,
    posterior_precision,
)


def test_observed_fisher_shape_and_symmetry() -> None:
    params = GRUMParameters(delta=np.array([0.0, 0.0, 0.0]), interaction=np.zeros((2, 2)))
    x = np.array([[1.0, 0.0], [0.3, 0.7]])
    z = np.array([[1.0, 0.0], [0.5, 1.2], [0.2, -0.4]])

    fisher = observed_fisher_information(params, x, z, sigma=1.0)
    expected_dim = 3 + (2 * 2)

    assert fisher.shape == (expected_dim, expected_dim)
    np.testing.assert_allclose(fisher, fisher.T)


def test_candidate_fisher_is_psd() -> None:
    x = np.array([0.3, 1.1])
    z = np.array([[1.0, 0.0], [0.2, 1.0], [0.5, -0.3]])

    fisher = candidate_fisher_information(x, z, n_alternatives=3, sigma=1.0)
    eigvals = np.linalg.eigvalsh(fisher)

    assert np.all(eigvals >= -1e-10)


def test_posterior_precision_adds_prior_diagonal() -> None:
    base = np.eye(5)
    precision = posterior_precision(base, prior_precision=0.1)
    np.testing.assert_allclose(np.diag(precision), np.array([1.1] * 5))
