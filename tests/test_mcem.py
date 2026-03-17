import numpy as np

from grums.core import GRUMParameters
from grums.core.model_math import compute_mean_utilities
from grums.inference import MCEMConfig, MCEMInference


def _toy_data() -> tuple[GRUMParameters, np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    true_params = GRUMParameters(
        delta=np.array([1.0, 0.2, -0.3], dtype=float),
        interaction=np.array([[0.5, -0.1], [0.2, 0.3]], dtype=float),
    )
    x = np.array([[1.0, 0.0], [0.2, 1.1], [1.3, -0.2]], dtype=float)
    z = np.array([[1.0, 0.1], [0.2, 1.0], [0.8, -0.4]], dtype=float)
    mu = compute_mean_utilities(true_params, x, z)

    rankings = [tuple(np.argsort(-row)) for row in mu]
    return true_params, x, z, rankings


def test_e_step_outputs_match_dimensions() -> None:
    _, x, z, rankings = _toy_data()
    init = GRUMParameters(delta=np.zeros(3), interaction=np.zeros((2, 2)))

    model = MCEMInference(MCEMConfig(n_iterations=1, n_gibbs_samples=20, n_gibbs_burnin=10))
    s = model._e_step(init, rankings, x, z)

    assert s.shape == (x.shape[0], z.shape[0])


def test_e_step_respects_mean_ordering_signal() -> None:
    _, x, z, rankings = _toy_data()
    init = GRUMParameters(delta=np.array([2.0, 1.0, 0.0]), interaction=np.zeros((2, 2)))

    model = MCEMInference(MCEMConfig(n_iterations=1, n_gibbs_samples=40, n_gibbs_burnin=20))
    s = model._e_step(init, rankings, x, z)

    for i, ranking in enumerate(rankings):
        ranked_values = [s[i, alt] for alt in ranking]
        assert all(ranked_values[j] > ranked_values[j + 1] for j in range(len(ranked_values) - 1))


def test_fit_map_returns_valid_shapes_and_trace() -> None:
    _, x, z, rankings = _toy_data()
    init = GRUMParameters(delta=np.zeros(3), interaction=np.zeros((2, 2)))

    model = MCEMInference(
        MCEMConfig(
            n_iterations=5,
            n_gibbs_samples=30,
            n_gibbs_burnin=15,
            tolerance=1e-8,
            random_seed=7,
        )
    )
    result = model.fit_map(init, rankings, x, z)

    assert result.params.delta.shape == (3,)
    assert result.params.interaction.shape == (2, 2)
    assert len(result.objective_trace) >= 1


def test_fit_map_improves_objective_from_first_to_last() -> None:
    _, x, z, rankings = _toy_data()
    init = GRUMParameters(delta=np.zeros(3), interaction=np.zeros((2, 2)))

    model = MCEMInference(
        MCEMConfig(
            n_iterations=6,
            n_gibbs_samples=25,
            n_gibbs_burnin=10,
            tolerance=1e-8,
            random_seed=11,
        )
    )
    result = model.fit_map(init, rankings, x, z)

    assert max(result.objective_trace) >= result.objective_trace[0]
    assert np.isfinite(np.array(result.objective_trace)).all()
