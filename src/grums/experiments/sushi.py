"""Sushi experiment logic."""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

from grums.contracts import AgentRecord, AlternativeRecord, RankingObservation
from grums.datasets.sushi import load_sushi, SushiDataset
from grums.inference import MCEMConfig, MCEMInference
from grums.elicitation import (
    AdaptiveElicitationEngine,
    RandomCriterion,
    DOptimalityCriterion,
    EOptimalityCriterion,
    SocialChoiceCriterion,
    PersonalizedChoiceCriterion,
)
from grums.experiments.metrics import social_choice_kendall_tau, personalized_mean_kendall_tau, raw_mean_kendall_tau
from grums.providers import OracleRankingProvider

Tensor = torch.Tensor

# Global cache for the ground-truth fit so processes don't redundantly re-fit 1000 agents
_SUSHI_FIT_CACHE = None

def _get_sushi_ground_truth(dataset_path: str, mcem_config: MCEMConfig, seed: int):
    global _SUSHI_FIT_CACHE
    if _SUSHI_FIT_CACHE is not None:
        return _SUSHI_FIT_CACHE

    dataset = load_sushi(dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1000 agents for ground truth
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset.agent_features), size=1000, replace=False)
    
    # Convert dataset components to Tensors
    train_agents = torch.from_numpy(dataset.agent_features[idx]).to(device).to(torch.float64)
    train_rankings = [dataset.rankings[i] for i in idx]
    alternatives = torch.from_numpy(dataset.alternative_features).to(device).to(torch.float64)
    
    from grums.core.parameters import GRUMParameters
    init = GRUMParameters(
        delta=torch.zeros(alternatives.size(0), device=device, dtype=torch.float64),
        interaction=torch.zeros((train_agents.size(1), alternatives.size(1)), device=device, dtype=torch.float64)
    )
    
    inf = MCEMInference(mcem_config)
    fit = inf.fit_map(
        initial_params=init,
        observations=[RankingObservation(agent_id=f"gt_{i}", ranking=r) for i, r in enumerate(train_rankings)],
        agent_features=train_agents,
        alternative_features=alternatives,
    )
    _SUSHI_FIT_CACHE = (dataset, fit.params, set(idx))
    return _SUSHI_FIT_CACHE

def _single_criteria_sushi_task(
    dataset_path: str,
    n_rounds: int,
    repeat_index: int,
    criterion_name: str,
    metric: str,
    seed: int,
    config: MCEMConfig,
) -> dict[str, float]:
    dataset, true_params, train_idx = _get_sushi_ground_truth(dataset_path, config, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = dataset.n_alternatives
    
    # Alternatives features to Tensor
    alt_features_tensor = torch.from_numpy(dataset.alternative_features).to(device).to(torch.float64)
    alternatives = [AlternativeRecord(alternative_id=j, features=alt_features_tensor[j]) for j in range(m)]
    
    all_indices = np.arange(len(dataset.agent_features))
    test_indices = np.array([i for i in all_indices if i not in train_idx])
    
    test_agent_features = dataset.agent_features[test_indices]
    test_rankings = [dataset.rankings[i] for i in test_indices]
    
    # For Figure 5, just pick 100 random agents per repeat from the 4000 test set
    rng = np.random.default_rng(seed + repeat_index)
    idx = rng.choice(len(test_agent_features), size=100, replace=False)
    
    selected_features_np = test_agent_features[idx]
    selected_rankings = [test_rankings[i] for i in idx]
    
    selected_features = torch.from_numpy(selected_features_np).to(device).to(torch.float64)
    
    agents = [AgentRecord(agent_id=f"a_{i}", features=selected_features[i]) for i in range(len(selected_features))]
    ranking_by_agent = {a.agent_id: r for a, r in zip(agents, selected_rankings)}
    
    provider = OracleRankingProvider(ranking_by_agent)
    seed_obs = RankingObservation(agent_id=agents[0].agent_id, ranking=selected_rankings[0])
    
    criteria = {
        "random": RandomCriterion(seed + 1000 + repeat_index),
        "d_opt": DOptimalityCriterion(),
        "e_opt": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=m),
        "personalized": PersonalizedChoiceCriterion(
            n_alternatives=m,
            n_agent_features=int(selected_features.size(1)),
            n_alternative_features=int(alt_features_tensor.size(1)),
            alternative_features=alt_features_tensor,
            population_agents=selected_features,
        ),
    }
    
    criterion = criteria[criterion_name]
    
    from grums.core.parameters import GRUMParameters
    init_params = GRUMParameters(
        delta=torch.zeros(m, device=device, dtype=torch.float64),
        interaction=torch.zeros((selected_features.size(1), alt_features_tensor.size(1)), device=device, dtype=torch.float64)
    )
    
    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=config)
    result = engine.run(
        provider=provider,
        initial_params=init_params,
        initial_observations=[seed_obs],
        observed_agents=[agents[0]],
        candidate_agents=agents[1:],
        alternatives=alternatives,
        n_rounds=n_rounds,
    )
    
    soc_tau = social_choice_kendall_tau(true_params.delta, result.final_params.delta)
    mean_person = personalized_mean_kendall_tau(true_params, result.final_params, selected_features, alt_features_tensor)
    raw_person = raw_mean_kendall_tau(result.final_params, selected_features, alt_features_tensor, selected_rankings)
    return {"social": soc_tau, "mean_person": mean_person, "raw_person": raw_person}

def compare_criteria_sushi_choice(
    dataset_path: str = ".data",
    n_rounds: int = 20,
    repeats: int = 3,
    criterion_name: str = "social",
    metric: str = "social",
    seed: int = 0,
    mcem_config: MCEMConfig | None = None,
    n_jobs: int = 1,
    progress_update: Callable[[int], None] | None = None,
) -> dict[str, float]:
    config = mcem_config or MCEMConfig(n_iterations=6, n_gibbs_samples=25, n_gibbs_burnin=12)
    out: list[dict[str, float]] = []
    
    if n_jobs == 1:
        for r in range(repeats):
            val = _single_criteria_sushi_task(dataset_path, n_rounds, r, criterion_name, metric, seed, config)
            out.append(val)
            if progress_update: progress_update(1)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(_single_criteria_sushi_task, dataset_path, n_rounds, r, criterion_name, metric, seed, config)
                for r in range(repeats)
            ]
            for fut in as_completed(futures):
                out.append(fut.result())
                if progress_update: progress_update(1)
                
    return {
        "social": float(np.mean([x["social"] for x in out])),
        "mean_person": float(np.mean([x["mean_person"] for x in out])),
        "raw_person": float(np.mean([x["raw_person"] for x in out])),
    }
