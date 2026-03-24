import torch

from grums.core.parameters import GRUMParameters
from grums.core.model_math import predict_deterministic_rankings
from grums.elicitation import PersonalizedChoiceCriterion
from grums.experiments.metrics import personalized_mean_kendall_tau
from grums.experiments.personalized import run_personalized_asymptotic
from grums.inference import MCEMConfig


def test_predict_deterministic_rankings_returns_per_agent_orders() -> None:
    params = GRUMParameters(
        delta=torch.tensor([0.3, 0.1, -0.2], dtype=torch.float64),
        interaction=torch.tensor([[0.4, -0.1], [0.2, 0.5]], dtype=torch.float64),
    )
    x = torch.tensor([[1.0, 0.0], [0.5, 1.2]], dtype=torch.float64)
    z = torch.tensor([[1.0, 0.1], [0.2, 1.0], [0.7, -0.5]], dtype=torch.float64)

    rankings = predict_deterministic_rankings(params, x, z)

    assert len(rankings) == 2
    assert all(len(r) == 3 for r in rankings)


def test_personalized_mean_kendall_tau_identity_is_one() -> None:
    params = GRUMParameters(
        delta=torch.tensor([0.2, 0.0, -0.1], dtype=torch.float64),
        interaction=torch.tensor([[0.3, 0.2], [0.1, -0.4]], dtype=torch.float64),
    )
    x = torch.tensor([[1.0, 0.0], [0.3, 0.8]], dtype=torch.float64)
    z = torch.tensor([[1.0, 0.0], [0.2, 1.0], [0.4, -0.2]], dtype=torch.float64)

    tau = personalized_mean_kendall_tau(params, params, x, z)
    assert tau == 1.0


def test_personalized_criterion_increases_with_clearer_preferences() -> None:
    n_alts, k, l = 3, 2, 2
    z = torch.tensor([[1.0, 0.0], [0.2, 1.0], [0.5, -0.4]], dtype=torch.float64)
    pop = torch.tensor([[1.0, 0.1], [0.2, 0.9]], dtype=torch.float64)

    criterion = PersonalizedChoiceCriterion(
        n_alternatives=n_alts,
        n_agent_features=k,
        n_alternative_features=l,
        alternative_features=z,
        population_agents=pop,
    )

    theta_small = torch.tensor([0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    theta_large = torch.tensor([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    precision = torch.eye(n_alts + k * l)

    assert criterion.score(precision, theta_large) > criterion.score(precision, theta_small)


def test_personalized_asymptotic_is_seed_stable() -> None:
    cfg = MCEMConfig(n_iterations=2, n_gibbs_samples=10, n_gibbs_burnin=5, random_seed=13)

    a = run_personalized_asymptotic([5, 10], repeats=2, seed=4, mcem_config=cfg)
    b = run_personalized_asymptotic([5, 10], repeats=2, seed=4, mcem_config=cfg)

    assert [(p.n_agents, p.mean_person_tau) for p in a] == [(p.n_agents, p.mean_person_tau) for p in b]

def test_personalized_compare_criteria_is_seed_stable() -> None:
    from grums.experiments.personalized import compare_criteria_personalized_choice
    
    cfg = MCEMConfig(n_iterations=2, n_gibbs_samples=10, n_gibbs_burnin=5, random_seed=13)
    
    a = compare_criteria_personalized_choice(
        dataset="dataset1", n_rounds=2, repeats=2, criterion_name="personalized", seed=4, mcem_config=cfg
    )
    b = compare_criteria_personalized_choice(
        dataset="dataset1", n_rounds=2, repeats=2, criterion_name="personalized", seed=4, mcem_config=cfg
    )
    
    assert a == b

