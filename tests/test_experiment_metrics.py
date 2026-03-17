import numpy as np
import pytest

from grums.experiments.metrics import moving_average, social_choice_kendall_tau


def test_social_choice_kendall_tau_bounds() -> None:
    true_delta = np.array([3.0, 2.0, 1.0])
    est_delta = np.array([1.0, 2.0, 3.0])

    tau = social_choice_kendall_tau(true_delta, est_delta)
    assert -1.0 <= tau <= 1.0


def test_social_choice_kendall_tau_identity_is_one() -> None:
    delta = np.array([0.2, 0.1, -1.2, 3.0])
    tau = social_choice_kendall_tau(delta, delta)
    assert tau == 1.0


def test_moving_average_values() -> None:
    series = np.array([1.0, 2.0, 3.0, 4.0])
    got = moving_average(series, window=2)
    np.testing.assert_allclose(got, np.array([1.5, 2.5, 3.5]))


def test_moving_average_validation() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        _ = moving_average(np.array([1.0, 2.0]), window=0)
    with pytest.raises(ValueError, match="window cannot exceed"):
        _ = moving_average(np.array([1.0, 2.0]), window=3)
