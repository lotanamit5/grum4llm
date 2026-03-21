"""Personalized-choice experiment helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grums.core.parameters import GRUMParameters
from grums.contracts import AgentRecord, AlternativeRecord, RankingObservation
from grums.experiments.benchmark import ElicitationCurvePoint
from grums.experiments.metrics import personalized_mean_kendall_tau, raw_mean_kendall_tau
from grums.experiments.synthetic_data import SyntheticDataset, make_dataset_1, make_dataset_2, make_dataset_consistency
from grums.inference import MCEMConfig, MCEMInference
from grums.elicitation import (
    AdaptiveElicitationEngine,
    DOptimalityCriterion,
    EOptimalityCriterion,
    SocialChoiceCriterion,
    PersonalizedChoiceCriterion,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

class _OracleProvider:
    def __init__(self, ranking_by_agent_id: dict[str, tuple[int, ...]]) -> None:
        self.ranking_by_agent_id = ranking_by_agent_id

    def query_full_ranking(
        self,
        agent: AgentRecord,
        alternatives: list[AlternativeRecord],
    ) -> RankingObservation:
        return RankingObservation(agent_id=agent.agent_id, ranking=self.ranking_by_agent_id[agent.agent_id])

class _RandomCriterion:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def score(self, prior_plus_candidate_info: np.ndarray, theta_vector: np.ndarray) -> float:
        return float(self.rng.random())



@dataclass(frozen=True)
class PersonalizedPoint:
    n_agents: int
    mean_tau: float
    raw_mean_tau: float = 0.0


def _default_initial_params(m: int, k: int, l: int) -> GRUMParameters:
    return GRUMParameters(delta=np.zeros(m, dtype=float), interaction=np.zeros((k, l), dtype=float))


def run_personalized_asymptotic(
    agent_counts: list[int],
    repeats: int = 3,
    seed: int = 0,
    dataset: str = "dataset2",
    mcem_config: MCEMConfig | None = None,
) -> list[PersonalizedPoint]:
    """Personalized-choice analogue of asymptotic social-choice runner."""

    config = mcem_config or MCEMConfig(n_iterations=8, n_gibbs_samples=30, n_gibbs_burnin=15)
    points: list[PersonalizedPoint] = []

    for n in agent_counts:
        taus: list[float] = []
        raw_taus: list[float] = []
        for r in range(repeats):
            if dataset == "consistency":
                data = make_dataset_consistency(n_agents=max(agent_counts), seed=seed + r)
            elif dataset == "dataset1":
                data = make_dataset_1(n_agents=max(agent_counts), seed=seed + r)
            else:
                data = make_dataset_2(n_agents=max(agent_counts), seed=seed + r)
                
            init = _default_initial_params(
                m=data.params_true.n_alternatives,
                k=data.agent_features.shape[1],
                l=data.alternative_features.shape[1],
            )
            inf = MCEMInference(config)
            fit = inf.fit_map(
                initial_params=init,
                rankings=list(data.rankings[:n]),
                agent_features=data.agent_features[:n],
                alternative_features=data.alternative_features,
            )

            tau = personalized_mean_kendall_tau(
                params_true=data.params_true,
                params_est=fit.params,
                agent_features=data.agent_features,
                alternative_features=data.alternative_features,
            )
            taus.append(tau)

            raw_tau = raw_mean_kendall_tau(
                params_true=data.params_true,
                agent_features=data.agent_features,
                alternative_features=data.alternative_features,
                observed_rankings=list(data.rankings[:n]),
            )
            raw_taus.append(raw_tau)

        points.append(PersonalizedPoint(n_agents=n, mean_tau=float(np.mean(taus)), raw_mean_tau=float(np.mean(raw_taus))))

    return points

def _dataset_builder(dataset: str):
    if dataset == "dataset1":
        return make_dataset_1
    if dataset == "dataset2":
        return make_dataset_2
    if dataset == "consistency":
        return make_dataset_consistency
    raise ValueError("dataset must be one of: consistency, dataset1, dataset2")


def _single_criteria_personalized_task(
    dataset: str,
    n_rounds: int,
    repeat_index: int,
    criterion_name: str,
    seed: int,
    config: MCEMConfig,
) -> float:
    build_dataset = _dataset_builder(dataset)
    # Figure 4 compares criteria over 100 agents consistently
    data = build_dataset(n_agents=100, seed=seed + repeat_index)
    m = data.params_true.n_alternatives

    alternatives = [AlternativeRecord(alternative_id=j, features=data.alternative_features[j]) for j in range(m)]
    agents = [AgentRecord(agent_id=f"a{i}", features=data.agent_features[i]) for i in range(data.agent_features.shape[0])]
    ranking_by_agent = {f"a{i}": data.rankings[i] for i in range(len(data.rankings))}
    provider = _OracleProvider(ranking_by_agent)

    seed_obs = RankingObservation(agent_id="a0", ranking=data.rankings[0])
    observed_agents = [agents[0]]
    candidates = agents[1:]

    criteria = {
        "random": _RandomCriterion(seed + 1000 + repeat_index),
        "d_opt": DOptimalityCriterion(),
        "e_opt": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=m),
        "personalized": PersonalizedChoiceCriterion(
            n_alternatives=m,
            n_agent_features=data.agent_features.shape[1],
            n_alternative_features=data.alternative_features.shape[1],
            alternative_features=data.alternative_features,
            population_agents=data.agent_features,
        ),
    }

    criterion = criteria[criterion_name]
    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=config)
    result = engine.run(
        provider=provider,
        initial_params=_default_initial_params(m, data.agent_features.shape[1], data.alternative_features.shape[1]),
        initial_observations=[seed_obs],
        observed_agents=observed_agents,
        candidate_agents=candidates,
        alternatives=alternatives,
        n_rounds=n_rounds,
    )
    return personalized_mean_kendall_tau(
        data.params_true, result.final_params, data.agent_features, data.alternative_features
    )


def run_personalized_elicitation_curve(
    dataset: str,
    n_rounds: int,
    criterion_name: str,
    seed: int,
    mcem_config: MCEMConfig | None = None,
) -> list[ElicitationCurvePoint]:
    """Single-seed personalized Kendall curve (paper Fig. 4 style)."""

    config = mcem_config or MCEMConfig(n_iterations=6, n_gibbs_samples=25, n_gibbs_burnin=12)
    build_dataset = _dataset_builder(dataset)
    n_pool = max(100, n_rounds + 1)
    data = build_dataset(n_agents=n_pool, seed=seed)
    m = data.params_true.n_alternatives
    kf, lf = data.agent_features.shape[1], data.alternative_features.shape[1]

    alternatives = [AlternativeRecord(alternative_id=j, features=data.alternative_features[j]) for j in range(m)]
    agents = [AgentRecord(agent_id=f"a{i}", features=data.agent_features[i]) for i in range(data.agent_features.shape[0])]
    ranking_by_agent = {f"a{i}": data.rankings[i] for i in range(len(data.rankings))}
    provider = _OracleProvider(ranking_by_agent)

    seed_obs = RankingObservation(agent_id="a0", ranking=data.rankings[0])
    observed_agents = [agents[0]]
    candidates = agents[1:]

    criteria = {
        "random": _RandomCriterion(seed + 1000),
        "d_opt": DOptimalityCriterion(),
        "e_opt": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=m),
        "personalized": PersonalizedChoiceCriterion(
            n_alternatives=m,
            n_agent_features=kf,
            n_alternative_features=lf,
            alternative_features=data.alternative_features,
            population_agents=data.agent_features,
        ),
    }

    criterion = criteria[criterion_name]
    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=config)
    init = _default_initial_params(m, kf, lf)

    tau_by_n: dict[int, float] = {}

    def _on_after_map(n_obs: int, params: GRUMParameters) -> None:
        tau_by_n[n_obs] = personalized_mean_kendall_tau(
            data.params_true, params, data.agent_features, data.alternative_features
        )

    _ = engine.run(
        provider=provider,
        initial_params=init,
        initial_observations=[seed_obs],
        observed_agents=observed_agents,
        candidate_agents=candidates,
        alternatives=alternatives,
        n_rounds=n_rounds,
        on_after_map=_on_after_map,
    )

    return [
        ElicitationCurvePoint(n_observations=n, kendall_tau=tau_by_n[n]) for n in sorted(tau_by_n.keys())
    ]


def compare_criteria_personalized_choice(
    dataset: str = "dataset2",
    n_rounds: int = 20,
    repeats: int = 3,
    criterion_name: str = "personalized",
    seed: int = 0,
    mcem_config: MCEMConfig | None = None,
    n_jobs: int = 1,
    progress_update: Callable[[int], None] | None = None,
) -> float:
    """Compare personalized-choice quality after adaptive elicitation for a specific criterion."""
    config = mcem_config or MCEMConfig(n_iterations=6, n_gibbs_samples=25, n_gibbs_burnin=12)
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")

    out: list[float] = []

    if n_jobs == 1:
        for r in range(repeats):
            tau = _single_criteria_personalized_task(dataset, n_rounds, r, criterion_name, seed, config)
            out.append(tau)
            if progress_update is not None:
                progress_update(n_rounds)
        return float(np.mean(out))

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(_single_criteria_personalized_task, dataset, n_rounds, r, criterion_name, seed, config)
            for r in range(repeats)
        ]
        for fut in as_completed(futures):
            out.append(fut.result())
            if progress_update is not None:
                progress_update(n_rounds)

    return float(np.mean(out))
