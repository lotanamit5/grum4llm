import numpy as np
import pytest

from grums.core import GRUMParameters, compute_mean_utilities


def test_compute_mean_utilities_matches_formula() -> None:
    params = GRUMParameters(
        delta=np.array([0.5, -0.25], dtype=float),
        interaction=np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float),
    )
    x = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=float)
    z = np.array([[1.0, 1.0], [2.0, 0.0]], dtype=float)

    got = compute_mean_utilities(params, x, z)

    expected = np.array(
        [
            [4.5, 3.75],
            [3.5, -0.25],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(got, expected)


def test_compute_mean_utilities_raises_on_shape_mismatch() -> None:
    params = GRUMParameters(
        delta=np.array([0.0, 0.0], dtype=float),
        interaction=np.array([[1.0, 0.0]], dtype=float),
    )
    x = np.array([[1.0, 2.0]], dtype=float)
    z = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="B row dimension"):
        _ = compute_mean_utilities(params, x, z)
