import torch
import pytest

from grums.core import GRUMParameters, compute_mean_utilities


def test_compute_mean_utilities_matches_formula() -> None:
    params = GRUMParameters(
        delta=torch.tensor([0.5, -0.25], dtype=torch.float64),
        interaction=torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    )
    x = torch.tensor([[1.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
    z = torch.tensor([[1.0, 1.0], [2.0, 0.0]], dtype=torch.float64)

    got = compute_mean_utilities(params, x, z)

    expected = torch.tensor(
        [
            [4.5, 3.75],
            [3.5, -0.25],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, expected)


def test_compute_mean_utilities_raises_on_shape_mismatch() -> None:
    params = GRUMParameters(
        delta=torch.tensor([0.0, 0.0], dtype=torch.float64),
        interaction=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
    )
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    z = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="B row dimension"):
        _ = compute_mean_utilities(params, x, z)
