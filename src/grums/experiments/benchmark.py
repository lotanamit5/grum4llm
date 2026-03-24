"""Reproducible runners for social-choice experiments."""

from __future__ import annotations

import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from grums.contracts import AgentRecord, AlternativeRecord, RankingObservation, Observation
from grums.core.parameters import GRUMParameters
from grums.elicitation import (
    AdaptiveElicitationEngine,
    RandomCriterion,
    DOptimalityCriterion,
    EOptimalityCriterion,
    SocialChoiceCriterion,
)
from grums.experiments.metrics import social_choice_kendall_tau, personalized_mean_kendall_tau, raw_mean_kendall_tau
from grums.experiments.synthetic_data import (
    SyntheticDataset, 
    make_dataset_1,
    make_dataset_2,
    make_dataset_consistency
)
from grums.inference import MCEMConfig, MCEMInference
from grums.providers import OracleRankingProvider

Tensor = torch.Tensor

@dataclass(frozen=True)
class AsymptoticPoint:
    n_agents: int
    social_tau: float
    mean_person_tau: float
    raw_person_tau: float


@dataclass(frozen=True)
class ElicitationCurvePoint:
    """One checkpoint along a single adaptive elicitation trajectory (paper Fig. 3 style)."""

    n_observations: int
    social_tau: float
    mean_person_tau: float
    raw_person_tau: float


def _default_initial_params(dataset: SyntheticDataset, device: torch.device) -> GRUMParameters:
    k = int(dataset.agent_features.size(1))
    l = int(dataset.alternative_features.size(1))
    m = int(dataset.params_true.n_alternatives)
    return GRUMParameters(
        delta=torch.zeros(m, dtype=torch.float64, device=device), 
        interaction=torch.zeros((k, l), dtype=torch.float64, device=device)
    )


def _dataset_builder(dataset: str):
    if dataset == "dataset0":
        return make_dataset_consistency
    if dataset == "dataset1":
        return make_dataset_1
    if dataset == "dataset2":
        return make_dataset_2
    raise ValueError("dataset must be one of: dataset1, dataset2")


def _single_asymptotic_task(
    dataset_name: str,
    n_agents_to_obs: int,
    repeat_index: int,
    max_agents: int,
    seed: int,
    config: MCEMConfig,
) -> tuple[int, float, float, float]:
    build_dataset = _dataset_builder(dataset_name)
    data = build_dataset(n_agents=max_agents, seed=seed + repeat_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move data to Tensors
    agent_features = data.agent_features.to(device)
    alt_features = data.alternative_features.to(device)
    true_params = GRUMParameters(
        delta=data.params_true.delta.to(device),
        interaction=data.params_true.interaction.to(device)
    )
    
    init = _default_initial_params(data, device)
    inf = MCEMInference(config)
    
    obs_list = [
        RankingObservation(agent_id=f"a{i}", ranking=data.rankings[i]) 
        for i in range(n_agents_to_obs)
    ]
    
    fit = inf.fit_map(
        initial_params=init,
        observations=obs_list,
        agent_features=agent_features[:n_agents_to_obs],
        alternative_features=alt_features,
    )
    
    tau = social_choice_kendall_tau(true_params.delta, fit.params.delta)
    mean_person_tau = personalized_mean_kendall_tau(true_params, fit.params, agent_features, alt_features)
    raw_person_tau = raw_mean_kendall_tau(fit.params, agent_features, alt_features, list(data.rankings[:n_agents_to_obs]))
    return n_agents_to_obs, tau, mean_person_tau, raw_person_tau


def _single_criteria_repeat_task(
    dataset_name: str,
    n_rounds: int,
    repeat_index: int,
    criterion_name: str,
    seed: int,
    config: MCEMConfig,
) -> tuple[float, float, float]:
    build_dataset = _dataset_builder(dataset_name)
    data = build_dataset(n_agents=max(n_rounds + 1, 30), seed=seed + repeat_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = data.params_true.n_alternatives

    agent_features = data.agent_features.to(device)
    alt_features = data.alternative_features.to(device)
    true_params = GRUMParameters(
        delta=data.params_true.delta.to(device),
        interaction=data.params_true.interaction.to(device)
    )

    alternatives = [
        AlternativeRecord(alternative_id=j, features=alt_features[j]) for j in range(m)
    ]
    agents = [
        AgentRecord(agent_id=f"a{i}", features=agent_features[i])
        for i in range(agent_features.size(0))
    ]
    ranking_by_agent = {f"a{i}": data.rankings[i] for i in range(len(data.rankings))}
    provider = OracleRankingProvider(ranking_by_agent)

    seed_obs = RankingObservation(agent_id="a0", ranking=data.rankings[0])
    observed_agents = [agents[0]]
    candidates = agents[1:]

    criteria = {
        "random": RandomCriterion(seed + 1000 + repeat_index),
        "d_opt": DOptimalityCriterion(),
        "e_opt": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=m),
    }

    criterion = criteria[criterion_name]
    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=config)
    
    init = _default_initial_params(data, device)
    
    result = engine.run(
        provider=provider,
        initial_params=init,
        initial_observations=[seed_obs],
        observed_agents=observed_agents,
        candidate_agents=candidates,
        alternatives=alternatives,
        n_rounds=n_rounds,
    )
    soc_tau = social_choice_kendall_tau(true_params.delta, result.final_params.delta)
    mean_tau = personalized_mean_kendall_tau(true_params, result.final_params, agent_features, alt_features)
    raw_tau = raw_mean_kendall_tau(result.final_params, agent_features, alt_features, list(data.rankings))
    return soc_tau, mean_tau, raw_tau


def run_social_choice_elicitation_curve(
    dataset_name: str,
    n_rounds: int,
    criterion_name: str,
    seed: int,
    mcem_config: MCEMConfig | None = None,
) -> list[ElicitationCurvePoint]:
    """Single-seed adaptive elicitation with Kendall τ after each MAP state."""

    config = mcem_config or MCEMConfig(n_iterations=6, n_gibbs_samples=25, n_gibbs_burnin=12)
    build_dataset = _dataset_builder(dataset_name)
    data = build_dataset(n_agents=max(n_rounds + 1, 30), seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = data.params_true.n_alternatives
    
    agent_features = data.agent_features.to(device)
    alt_features = data.alternative_features.to(device)
    true_params = GRUMParameters(
        delta=data.params_true.delta.to(device),
        interaction=data.params_true.interaction.to(device)
    )

    alternatives = [
        AlternativeRecord(alternative_id=j, features=alt_features[j]) for j in range(m)
    ]
    agents = [
        AgentRecord(agent_id=f"a{i}", features=agent_features[i]) for i in range(agent_features.size(0))
    ]
    ranking_by_agent = {f"a{i}": data.rankings[i] for i in range(len(data.rankings))}
    provider = OracleRankingProvider(ranking_by_agent)

    seed_obs = RankingObservation(agent_id="a0", ranking=data.rankings[0])
    observed_agents = [agents[0]]
    candidates = agents[1:]

    criteria = {
        "random": RandomCriterion(seed + 1000),
        "d_opt": DOptimalityCriterion(),
        "e_opt": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=m),
    }

    criterion = criteria[criterion_name]
    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=config)
    
    init = _default_initial_params(data, device)

    tau_by_n: dict[int, tuple[float, float, float]] = {}

    def _on_after_map(n_obs: int, params: GRUMParameters) -> None:
        soc_tau = social_choice_kendall_tau(true_params.delta, params.delta)
        mean_tau = personalized_mean_kendall_tau(true_params, params, agent_features, alt_features)
        raw_tau = raw_mean_kendall_tau(params, agent_features, alt_features, list(data.rankings))
        tau_by_n[n_obs] = (soc_tau, mean_tau, raw_tau)

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
        ElicitationCurvePoint(n_observations=n, social_tau=t_soc, mean_person_tau=t_mean, raw_person_tau=t_raw)
        for n, (t_soc, t_mean, t_raw) in sorted(tau_by_n.items())
    ]


def run_asymptotic_social_choice(
    agent_counts: list[int],
    dataset_name: str = "dataset2",
    repeats: int = 3,
    seed: int = 0,
    mcem_config: MCEMConfig | None = None,
    n_jobs: int = 1,
    progress_update: Callable[[int], None] | None = None,
) -> list[AsymptoticPoint]:
    """Run social-choice recovery vs number of observed agents."""

    config = mcem_config or MCEMConfig(n_iterations=8, n_gibbs_samples=30, n_gibbs_burnin=15)
    _dataset_builder(dataset_name)
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")
    points: list[AsymptoticPoint] = []

    if n_jobs == 1:
        for n in agent_counts:
            taus: list[tuple[float, float, float]] = []
            for r in range(repeats):
                n_out, t_soc, t_mean, t_raw = _single_asymptotic_task(dataset_name, n, r, max(agent_counts), seed, config)
                taus.append((t_soc, t_mean, t_raw))
                if progress_update is not None:
                    progress_update(1)

            points.append(
                AsymptoticPoint(
                    n_agents=n, 
                    social_tau=float(np.mean([x[0] for x in taus])),
                    mean_person_tau=float(np.mean([x[1] for x in taus])),
                    raw_person_tau=float(np.mean([x[2] for x in taus])),
                )
            )

        return points

    max_agents = max(agent_counts)
    taus_by_n: dict[int, list[tuple[float, float, float]]] = {n: [] for n in agent_counts}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(_single_asymptotic_task, dataset_name, n, r, max_agents, seed, config)
            for n in agent_counts
            for r in range(repeats)
        ]
        for fut in as_completed(futures):
            n, t_soc, t_mean, t_raw = fut.result()
            taus_by_n[n].append((t_soc, t_mean, t_raw))
            if progress_update is not None:
                progress_update(1)

    for n in agent_counts:
        lst = taus_by_n[n]
        points.append(
            AsymptoticPoint(
                n_agents=n, 
                social_tau=float(np.mean([x[0] for x in lst])),
                mean_person_tau=float(np.mean([x[1] for x in lst])),
                raw_person_tau=float(np.mean([x[2] for x in lst])),
            )
        )

    return points


def compare_criteria_social_choice(
    dataset_name: str = "dataset2",
    n_rounds: int = 20,
    repeats: int = 3,
    criterion_name: str = "social",
    seed: int = 0,
    mcem_config: MCEMConfig | None = None,
    n_jobs: int = 1,
    progress_update: Callable[[int], None] | None = None,
) -> dict[str, float]:
    """Compare social-choice quality after adaptive elicitation for a specific criterion."""

    config = mcem_config or MCEMConfig(n_iterations=6, n_gibbs_samples=25, n_gibbs_burnin=12)
    _dataset_builder(dataset_name)
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")

    out: list[tuple[float, float, float]] = []

    if n_jobs == 1:
        for r in range(repeats):
            val = _single_criteria_repeat_task(dataset_name, n_rounds, r, criterion_name, seed, config)
            out.append(val)
            if progress_update is not None:
                progress_update(n_rounds)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(_single_criteria_repeat_task, dataset_name, n_rounds, r, criterion_name, seed, config)
                for r in range(repeats)
            ]
            for fut in as_completed(futures):
                out.append(fut.result())
                if progress_update is not None:
                    progress_update(n_rounds)

    return {
        "social": float(np.mean([x[0] for x in out])),
        "mean_person": float(np.mean([x[1] for x in out])),
        "raw_person": float(np.mean([x[2] for x in out])),
    }
