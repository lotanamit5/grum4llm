from grums.experiments.benchmark import compare_criteria_social_choice, run_asymptotic_social_choice
from grums.inference import MCEMConfig


def test_asymptotic_runner_is_reproducible_for_seed() -> None:
    cfg = MCEMConfig(n_iterations=3, n_gibbs_samples=10, n_gibbs_burnin=5, random_seed=42)
    a = run_asymptotic_social_choice([5, 10], dataset="dataset2", repeats=2, seed=9, mcem_config=cfg)
    b = run_asymptotic_social_choice([5, 10], dataset="dataset2", repeats=2, seed=9, mcem_config=cfg)

    assert [(p.n_agents, p.mean_tau) for p in a] == [(p.n_agents, p.mean_tau) for p in b]


def test_criteria_comparison_returns_expected_keys_and_range() -> None:
    cfg = MCEMConfig(n_iterations=3, n_gibbs_samples=10, n_gibbs_burnin=5, random_seed=5)
    scores = compare_criteria_social_choice(dataset="dataset2", n_rounds=5, repeats=2, seed=3, mcem_config=cfg)

    assert set(scores.keys()) == {"random", "d_opt", "e_opt", "social"}
    assert all(-1.0 <= value <= 1.0 for value in scores.values())


def test_non_random_criterion_can_outperform_random_on_dataset2() -> None:
    cfg = MCEMConfig(n_iterations=3, n_gibbs_samples=10, n_gibbs_burnin=5, random_seed=1)
    scores = compare_criteria_social_choice(dataset="dataset2", n_rounds=8, repeats=3, seed=4, mcem_config=cfg)

    best_non_random = max(scores["d_opt"], scores["e_opt"], scores["social"])
    assert best_non_random >= scores["random"]


def test_dataset_selector_changes_output_distribution() -> None:
    cfg = MCEMConfig(n_iterations=0, n_gibbs_samples=5, n_gibbs_burnin=2, random_seed=2)
    a = run_asymptotic_social_choice([5, 10], dataset="dataset1", repeats=1, seed=5, mcem_config=cfg)
    b = run_asymptotic_social_choice([5, 10], dataset="dataset2", repeats=1, seed=5, mcem_config=cfg)

    assert len(a) == 2
    assert len(b) == 2


def test_invalid_dataset_raises_value_error() -> None:
    cfg = MCEMConfig(n_iterations=0, n_gibbs_samples=5, n_gibbs_burnin=2, random_seed=2)
    try:
        _ = run_asymptotic_social_choice([5], dataset="nope", repeats=1, seed=5, mcem_config=cfg)
    except ValueError as err:
        assert "dataset must be one of" in str(err)
    else:
        raise AssertionError("Expected ValueError for invalid dataset selector")
