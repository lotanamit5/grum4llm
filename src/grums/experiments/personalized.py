"""Personalized-choice experiment helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np

from grums.core.parameters import GRUMParameters
from grums.contracts import AgentRecord, AlternativeRecord, RankingObservation
from grums.providers import OracleRankingProvider
from grums.experiments.benchmark import ElicitationCurvePoint, AsymptoticPoint
from grums.experiments.metrics import personalized_mean_kendall_tau, raw_mean_kendall_tau, social_choice_kendall_tau
from grums.experiments.synthetic_data import SyntheticDataset, make_dataset_1, make_dataset_2, make_dataset_consistency
from grums.inference import MCEMConfig, MCEMInference
from grums.elicitation import (
    AdaptiveElicitationEngine,
    RandomCriterion,
    DOptimalityCriterion,
    EOptimalityCriterion,
    SocialChoiceCriterion,
    PersonalizedChoiceCriterion,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

Tensor = torch.Tensor


def _default_initial_params(m: int, k: int, l: int, device: torch.device) -> GRUMParameters:
    return GRUMParameters(
        delta=torch.zeros(m, dtype=torch.float64, device=device), 
        interaction=torch.zeros((k, l), dtype=torch.float64, device=device)
    )


def run_personalized_asymptotic(
    agent_counts: list[int],
    repeats: int = 3,
    seed: int = 0,
    dataset: str = "dataset2",
    mcem_config: MCEMConfig | None = None,
) -> list[AsymptoticPoint]:
    """Personalized-choice analogue of asymptotic social-choice runner."""

    config = mcem_config or MCEMConfig(n_iterations=8, n_gibbs_samples=30, n_gibbs_burnin=15)
    points: list[AsymptoticPoint] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for n in agent_counts:
        taus: list[tuple[float, float, float]] = []
        for r in range(repeats):
            if dataset == "consistency":
                data = make_dataset_consistency(n_agents=max(agent_counts), seed=seed + r)
            elif dataset == "dataset1":
                data = make_dataset_1(n_agents=max(agent_counts), seed=seed + r)
            else:
                data = make_dataset_2(n_agents=max(agent_counts), seed=seed + r)
                
            # Move data to target device
            agent_features = data.agent_features.to(device)
            alt_features = data.alternative_features.to(device)
            true_params = GRUMParameters(
                delta=data.params_true.delta.to(device),
                interaction=data.params_true.interaction.to(device)
            )

            init = _default_initial_params(
                m=data.params_true.n_alternatives,
                k=int(data.agent_features.size(1)),
                l=int(data.alternative_features.size(1)),
                device=device
            )
            inf = MCEMInference(config)
            
            # Convert rankings to RankingObservation list
            obs_list = [
                RankingObservation(agent_id=f"a{i}", ranking=data.rankings[i]) 
                for i in range(n)
            ]
            
            fit = inf.fit_map(
                initial_params=init,
                observations=obs_list,
                agent_features=agent_features[:n],
                alternative_features=alt_features,
            )

            t_soc = social_choice_kendall_tau(true_params.delta, fit.params.delta)
            t_mean = personalized_mean_kendall_tau(
                params_true=true_params,
                params_est=fit.params,
                agent_features=agent_features,
                alternative_features=alt_features,
            )
            t_raw = raw_mean_kendall_tau(
                params_est=fit.params,
                agent_features=agent_features,
                alternative_features=alt_features,
                observed_rankings=list(data.rankings[:n]),
            )
            taus.append((t_soc, t_mean, t_raw))

        points.append(
            AsymptoticPoint(
                n_agents=n, 
                social_tau=float(np.mean([x[0] for x in taus])),
                mean_person_tau=float(np.mean([x[1] for x in taus])),
                raw_person_tau=float(np.mean([x[2] for x in taus]))
            )
        )

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
) -> tuple[float, float, float]:
    build_dataset = _dataset_builder(dataset)
    # Figure 4 compares criteria over 100 agents consistently
    data = build_dataset(n_agents=100, seed=seed + repeat_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = data.params_true.n_alternatives

    # Move data to Tensors and device
    agent_features = data.agent_features.to(device)
    alt_features = data.alternative_features.to(device)
    true_params = GRUMParameters(
        delta=data.params_true.delta.to(device),
        interaction=data.params_true.interaction.to(device)
    )

    alternatives = [AlternativeRecord(alternative_id=j, features=alt_features[j]) for j in range(m)]
    agents = [AgentRecord(agent_id=f"a{i}", features=agent_features[i]) for i in range(agent_features.size(0))]
    ranking_by_agent = {f"a{i}": data.rankings[i] for i in range(len(data.rankings))}
    provider = OracleRankingProvider(ranking_by_agent)

    seed_obs = RankingObservation(agent_id="a0", ranking=data.rankings[0])
    observed_agents = [agents[0]]
    candidates = agents[1:]

    criteria = {
        "random": RandomCriterion(seed + 1000 + repeat_index),
        "d_opt": DOptimalityCriterion(),
        "EOptimalityCriterion": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=m),
        "personalized": PersonalizedChoiceCriterion(
            n_alternatives=m,
            n_agent_features=int(agent_features.size(1)),
            n_alternative_features=int(alt_features.size(1)),
            alternative_features=alt_features,
            population_agents=agent_features,
        ),
    }

    criterion = criteria[criterion_name]
    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=config)
    
    init = _default_initial_params(m, int(agent_features.size(1)), int(alt_features.size(1)), device)
    
    result = engine.run(
        provider=provider,
        initial_params=init,
        initial_observations=[seed_obs],
        observed_agents=observed_agents,
        candidate_agents=candidates,
        alternatives=alternatives,
        n_rounds=n_rounds,
    )
    t_soc = social_choice_kendall_tau(true_params.delta, result.final_params.delta)
    t_mean = personalized_mean_kendall_tau(
        true_params, result.final_params, agent_features, alt_features
    )
    t_raw = raw_mean_kendall_tau(
        result.final_params, agent_features, alt_features, list(data.rankings)
    )
    return t_soc, t_mean, t_raw


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = data.params_true.n_alternatives
    
    agent_features = data.agent_features.to(device)
    alt_features = data.alternative_features.to(device)
    true_params = GRUMParameters(
        delta=data.params_true.delta.to(device),
        interaction=data.params_true.interaction.to(device)
    )
    
    kf, lf = int(agent_features.size(1)), int(alt_features.size(1))

    alternatives = [AlternativeRecord(alternative_id=j, features=alt_features[j]) for j in range(m)]
    agents = [AgentRecord(agent_id=f"a{i}", features=agent_features[i]) for i in range(agent_features.size(0))]
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
        "personalized": PersonalizedChoiceCriterion(
            n_alternatives=m,
            n_agent_features=kf,
            n_alternative_features=lf,
            alternative_features=alt_features,
            population_agents=agent_features,
        ),
    }

    criterion = criteria[criterion_name]
    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=config)
    init = _default_initial_params(m, kf, lf, device)

    tau_by_n: dict[int, tuple[float, float, float]] = {}

    def _on_after_map(n_obs: int, params: GRUMParameters) -> None:
        t_soc = social_choice_kendall_tau(true_params.delta, params.delta)
        t_mean = personalized_mean_kendall_tau(
            true_params, params, agent_features, alt_features
        )
        t_raw = raw_mean_kendall_tau(
            params, agent_features, alt_features, list(data.rankings)
        )
        tau_by_n[n_obs] = (t_soc, t_mean, t_raw)

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
        ElicitationCurvePoint(n_observations=n, social_tau=soc, mean_person_tau=mean, raw_person_tau=r_tau) 
        for n, (soc, mean, r_tau) in sorted(tau_by_n.items())
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
) -> dict[str, float]:
    """Compare personalized-choice quality after adaptive elicitation for a specific criterion."""
    config = mcem_config or MCEMConfig(n_iterations=6, n_gibbs_samples=25, n_gibbs_burnin=12)
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")

    out: list[tuple[float, float, float]] = []

    if n_jobs == 1:
        for r in range(repeats):
            val = _single_criteria_personalized_task(dataset, n_rounds, r, criterion_name, seed, config)
            out.append(val)
            if progress_update is not None:
                progress_update(n_rounds)
        return {
            "social": float(np.mean([x[0] for x in out])),
            "mean_person": float(np.mean([x[1] for x in out])),
            "raw_person": float(np.mean([x[2] for x in out])),
        }

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(_single_criteria_personalized_task, dataset, n_rounds, r, criterion_name, seed, config)
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
