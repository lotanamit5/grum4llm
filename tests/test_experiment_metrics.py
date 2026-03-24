import torch
import pytest

from grums.experiments.metrics import moving_average, social_choice_kendall_tau


def test_social_choice_kendall_tau_bounds() -> None:
    true_delta = torch.tensor([3.0, 2.0, 1.0])
    est_delta = torch.tensor([1.0, 2.0, 3.0])

    tau = social_choice_kendall_tau(true_delta, est_delta)
    assert -1.0 <= tau <= 1.0


def test_social_choice_kendall_tau_identity_is_one() -> None:
    delta = torch.tensor([0.2, 0.1, -1.2, 3.0])
    tau = social_choice_kendall_tau(delta, delta)
    assert tau == 1.0


def test_moving_average_values() -> None:
    series = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    got = moving_average(series, window=2)
    torch.testing.assert_close(got, torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64))


def test_moving_average_validation() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        _ = moving_average(torch.tensor([1.0, 2.0]), window=0)
    with pytest.raises(ValueError):
        _ = moving_average(torch.tensor([1.0, 2.0]), window=3)
