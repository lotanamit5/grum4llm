import numpy as np

from grums.experiments.metrics import social_choice_kendall_tau


def test_social_choice_kendall_tau_bounds() -> None:
    true_delta = np.array([3.0, 2.0, 1.0])
    est_delta = np.array([1.0, 2.0, 3.0])

    tau = social_choice_kendall_tau(true_delta, est_delta)
    assert -1.0 <= tau <= 1.0


def test_social_choice_kendall_tau_identity_is_one() -> None:
    delta = np.array([0.2, 0.1, -1.2, 3.0])
    tau = social_choice_kendall_tau(delta, delta)
    assert tau == 1.0
