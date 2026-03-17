import numpy as np

from grums.elicitation import DOptimalityCriterion, EOptimalityCriterion, SocialChoiceCriterion


def test_d_optimality_prefers_more_information() -> None:
    d_opt = DOptimalityCriterion()
    theta = np.array([1.0, 0.2, -0.1, 0.0])

    weak = np.diag([1.0, 1.0, 1.0, 1.0])
    strong = np.diag([2.0, 2.0, 2.0, 2.0])

    assert d_opt.score(strong, theta) > d_opt.score(weak, theta)


def test_e_optimality_prefers_higher_min_eigenvalue() -> None:
    e_opt = EOptimalityCriterion()
    theta = np.array([0.0, 0.0, 0.0, 0.0])

    weak = np.diag([1.0, 2.0, 2.0, 2.0])
    strong = np.diag([1.5, 2.0, 2.0, 2.0])

    assert e_opt.score(strong, theta) > e_opt.score(weak, theta)


def test_social_criterion_matches_pairwise_certainty_intuition() -> None:
    criterion = SocialChoiceCriterion(n_alternatives=3)

    theta_low_sep = np.array([0.1, 0.0, -0.1, 0.0])
    theta_high_sep = np.array([1.0, 0.0, -1.0, 0.0])

    precision = np.diag([5.0, 5.0, 5.0, 2.0])

    score_low = criterion.score(precision, theta_low_sep)
    score_high = criterion.score(precision, theta_high_sep)

    assert score_high > score_low
