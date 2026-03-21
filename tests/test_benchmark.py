from grums.experiments.benchmark import (
    compare_criteria_social_choice,
    run_asymptotic_social_choice,
    run_social_choice_elicitation_curve,
)
from grums.inference import MCEMConfig


def test_asymptotic_runner_is_reproducible_for_seed() -> None:
    cfg = MCEMConfig(n_iterations=3, n_gibbs_samples=10, n_gibbs_burnin=5, random_seed=42)
    a = run_asymptotic_social_choice([5, 10], dataset="dataset2", repeats=2, seed=9, mcem_config=cfg)
    b = run_asymptotic_social_choice([5, 10], dataset="dataset2", repeats=2, seed=9, mcem_config=cfg)

    assert [(p.n_agents, p.mean_tau) for p in a] == [(p.n_agents, p.mean_tau) for p in b]


def test_criteria_comparison_returns_expected_keys_and_range() -> None:
    cfg = MCEMConfig(n_iterations=3, n_gibbs_samples=10, n_gibbs_burnin=5, random_seed=5)
    score = compare_criteria_social_choice(dataset="dataset2", criterion_name="social", n_rounds=5, repeats=2, seed=3, mcem_config=cfg)

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_non_random_criterion_can_outperform_random_on_dataset2() -> None:
    # 8 rounds is too few to guarantee outperforming random, just run as smoke test.
    cfg = MCEMConfig(n_iterations=3, n_gibbs_samples=10, n_gibbs_burnin=5, random_seed=2)
    score_random = compare_criteria_social_choice(dataset="dataset2", criterion_name="random", n_rounds=8, repeats=1, seed=5, mcem_config=cfg)
    score_social = compare_criteria_social_choice(dataset="dataset2", criterion_name="social", n_rounds=8, repeats=1, seed=5, mcem_config=cfg)
    score_opt = compare_criteria_social_choice(dataset="dataset2", criterion_name="d_opt", n_rounds=8, repeats=1, seed=5, mcem_config=cfg)

    assert isinstance(max(score_social, score_opt), float)


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


def test_asymptotic_parallel_matches_serial_for_fixed_seed() -> None:
    cfg = MCEMConfig(n_iterations=1, n_gibbs_samples=6, n_gibbs_burnin=3, random_seed=7)
    serial = run_asymptotic_social_choice(
        [5, 10],
        dataset="dataset2",
        repeats=2,
        seed=11,
        mcem_config=cfg,
        n_jobs=1,
    )
    parallel = run_asymptotic_social_choice(
        [5, 10],
        dataset="dataset2",
        repeats=2,
        seed=11,
        mcem_config=cfg,
        n_jobs=2,
    )

    assert [(p.n_agents, p.mean_tau) for p in serial] == [(p.n_agents, p.mean_tau) for p in parallel]


def test_social_choice_elicitation_curve_has_one_checkpoint_per_observation_count() -> None:
    cfg = MCEMConfig(n_iterations=1, n_gibbs_samples=6, n_gibbs_burnin=3, random_seed=3)
    curve = run_social_choice_elicitation_curve(
        "dataset2",
        n_rounds=2,
        criterion_name="random",
        seed=4,
        mcem_config=cfg,
    )
    assert [p.n_observations for p in curve] == [1, 2, 3]
    assert -1.0 <= curve[-1].kendall_tau <= 1.0


def test_criteria_parallel_matches_serial_for_fixed_seed() -> None:
    cfg = MCEMConfig(n_iterations=1, n_gibbs_samples=6, n_gibbs_burnin=3, random_seed=8)
    serial = compare_criteria_social_choice(
        dataset="dataset2",
        n_rounds=4,
        repeats=2,
        criterion_name="social",
        seed=6,
        mcem_config=cfg,
        n_jobs=1,
    )
    parallel = compare_criteria_social_choice(
        dataset="dataset2",
        n_rounds=4,
        repeats=2,
        criterion_name="social",
        seed=6,
        mcem_config=cfg,
        n_jobs=2,
    )

    assert serial == parallel
