"""Reproducible runners for social-choice experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np

from grums.contracts import AgentRecord, AlternativeRecord, PreferenceProvider, RankingObservation
from grums.core.parameters import GRUMParameters
from grums.elicitation import (
    AdaptiveElicitationEngine,
    DOptimalityCriterion,
    EOptimalityCriterion,
    SocialChoiceCriterion,
)
from grums.experiments.metrics import social_choice_kendall_tau
from grums.experiments.synthetic_data import SyntheticDataset, make_dataset_1, make_dataset_2
from grums.inference import MCEMConfig, MCEMInference


@dataclass(frozen=True)
class AsymptoticPoint:
    n_agents: int
    mean_tau: float


class _OracleProvider(PreferenceProvider):
    def __init__(self, ranking_by_agent_id: dict[str, tuple[int, ...]]) -> None:
        self.ranking_by_agent_id = ranking_by_agent_id

    def query_full_ranking(
        self,
        agent: AgentRecord,
        alternatives: list[AlternativeRecord],
    ) -> RankingObservation:
        _ = alternatives
        return RankingObservation(agent_id=agent.agent_id, ranking=self.ranking_by_agent_id[agent.agent_id])


class _RandomCriterion:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def score(self, prior_plus_candidate_info: np.ndarray, theta_vector: np.ndarray) -> float:
        _ = prior_plus_candidate_info
        _ = theta_vector
        return float(self.rng.random())


def _default_initial_params(dataset: SyntheticDataset) -> GRUMParameters:
    k = dataset.agent_features.shape[1]
    l = dataset.alternative_features.shape[1]
    m = dataset.params_true.n_alternatives
    return GRUMParameters(delta=np.zeros(m, dtype=float), interaction=np.zeros((k, l), dtype=float))


def _dataset_builder(dataset: str):
    if dataset == "dataset1":
        return make_dataset_1
    if dataset == "dataset2":
        return make_dataset_2
    raise ValueError("dataset must be one of: dataset1, dataset2")


def _single_asymptotic_task(
    dataset: str,
    n: int,
    repeat_index: int,
    max_agents: int,
    seed: int,
    config: MCEMConfig,
) -> tuple[int, float]:
    build_dataset = _dataset_builder(dataset)
    data = build_dataset(n_agents=max_agents, seed=seed + repeat_index)
    init = _default_initial_params(data)
    inf = MCEMInference(config)
    fit = inf.fit_map(
        initial_params=init,
        rankings=list(data.rankings[:n]),
        agent_features=data.agent_features[:n],
        alternative_features=data.alternative_features,
    )
    tau = social_choice_kendall_tau(data.params_true.delta, fit.params.delta)
    return n, tau


def _single_criteria_repeat_task(
    dataset: str,
    n_rounds: int,
    repeat_index: int,
    criterion_name: str,
    seed: int,
    config: MCEMConfig,
) -> float:
    build_dataset = _dataset_builder(dataset)
    data = build_dataset(n_agents=max(n_rounds + 1, 30), seed=seed + repeat_index)
    m = data.params_true.n_alternatives

    alternatives = [
        AlternativeRecord(alternative_id=j, features=data.alternative_features[j]) for j in range(m)
    ]
    agents = [
        AgentRecord(agent_id=f"a{i}", features=data.agent_features[i])
        for i in range(data.agent_features.shape[0])
    ]
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
        "personalized": SocialChoiceCriterion(n_alternatives=m),
    }

    criterion = criteria[criterion_name]
    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=config)
    result = engine.run(
        provider=provider,
        initial_params=_default_initial_params(data),
        initial_observations=[seed_obs],
        observed_agents=observed_agents,
        candidate_agents=candidates,
        alternatives=alternatives,
        n_rounds=n_rounds,
    )
    return social_choice_kendall_tau(data.params_true.delta, result.final_params.delta)


def run_asymptotic_social_choice(
    agent_counts: list[int],
    dataset: str = "dataset2",
    repeats: int = 3,
    seed: int = 0,
    mcem_config: MCEMConfig | None = None,
    n_jobs: int = 1,
    progress_update: Callable[[int], None] | None = None,
) -> list[AsymptoticPoint]:
    """Run social-choice recovery vs number of observed agents.

    Dataset can be `dataset1` or `dataset2`.
    """

    config = mcem_config or MCEMConfig(n_iterations=8, n_gibbs_samples=30, n_gibbs_burnin=15)
    _dataset_builder(dataset)
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")
    points: list[AsymptoticPoint] = []

    if n_jobs == 1:
        build_dataset = _dataset_builder(dataset)
        for n in agent_counts:
            taus: list[float] = []
            for r in range(repeats):
                data = build_dataset(n_agents=max(agent_counts), seed=seed + r)
                init = _default_initial_params(data)
                inf = MCEMInference(config)

                fit = inf.fit_map(
                    initial_params=init,
                    rankings=list(data.rankings[:n]),
                    agent_features=data.agent_features[:n],
                    alternative_features=data.alternative_features,
                )
                taus.append(social_choice_kendall_tau(data.params_true.delta, fit.params.delta))
                if progress_update is not None:
                    progress_update(1)

            points.append(AsymptoticPoint(n_agents=n, mean_tau=float(np.mean(taus))))

        return points

    max_agents = max(agent_counts)
    taus_by_n: dict[int, list[float]] = {n: [] for n in agent_counts}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(_single_asymptotic_task, dataset, n, r, max_agents, seed, config)
            for n in agent_counts
            for r in range(repeats)
        ]
        for fut in as_completed(futures):
            n, tau = fut.result()
            taus_by_n[n].append(tau)
            if progress_update is not None:
                progress_update(1)

    for n in agent_counts:
        points.append(AsymptoticPoint(n_agents=n, mean_tau=float(np.mean(taus_by_n[n]))))

    return points


def compare_criteria_social_choice(
    dataset: str = "dataset2",
    n_rounds: int = 20,
    repeats: int = 3,
    criterion_name: str = "social",
    seed: int = 0,
    mcem_config: MCEMConfig | None = None,
    n_jobs: int = 1,
    progress_update: Callable[[int], None] | None = None,
) -> float:
    """Compare social-choice quality after adaptive elicitation for a specific criterion.

    Returns mean Kendall tau across repeats for the criterion.
    """

    config = mcem_config or MCEMConfig(n_iterations=6, n_gibbs_samples=25, n_gibbs_burnin=12)
    _dataset_builder(dataset)
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")

    out: list[float] = []

    if n_jobs == 1:
        build_dataset = _dataset_builder(dataset)
        for r in range(repeats):
            data = build_dataset(n_agents=max(n_rounds + 1, 30), seed=seed + r)
            m = data.params_true.n_alternatives

            alternatives = [
                AlternativeRecord(alternative_id=j, features=data.alternative_features[j]) for j in range(m)
            ]
            agents = [
                AgentRecord(agent_id=f"a{i}", features=data.agent_features[i])
                for i in range(data.agent_features.shape[0])
            ]
            ranking_by_agent = {f"a{i}": data.rankings[i] for i in range(len(data.rankings))}
            provider = _OracleProvider(ranking_by_agent)

            seed_obs = RankingObservation(agent_id="a0", ranking=data.rankings[0])
            observed_agents = [agents[0]]
            candidates = agents[1:]

            criteria = {
                "random": _RandomCriterion(seed + 1000 + r),
                "d_opt": DOptimalityCriterion(),
                "e_opt": EOptimalityCriterion(),
                "social": SocialChoiceCriterion(n_alternatives=m),
                "personalized": SocialChoiceCriterion(n_alternatives=m),
            }

            criterion = criteria[criterion_name]
            engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=config)
            result = engine.run(
                provider=provider,
                initial_params=_default_initial_params(data),
                initial_observations=[seed_obs],
                observed_agents=observed_agents,
                candidate_agents=candidates,
                alternatives=alternatives,
                n_rounds=n_rounds,
            )
            tau = social_choice_kendall_tau(data.params_true.delta, result.final_params.delta)
            out.append(tau)
            if progress_update is not None:
                progress_update(n_rounds)

        return float(np.mean(out))

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(_single_criteria_repeat_task, dataset, n_rounds, r, criterion_name, seed, config)
            for r in range(repeats)
        ]
        for fut in as_completed(futures):
            rep_score = fut.result()
            out.append(rep_score)
            if progress_update is not None:
                progress_update(n_rounds)

    return float(np.mean(out))
