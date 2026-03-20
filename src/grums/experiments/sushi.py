"""Sushi experiment logic."""

from __future__ import annotations

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
    DOptimalityCriterion,
    EOptimalityCriterion,
    SocialChoiceCriterion,
    PersonalizedChoiceCriterion,
)
from grums.experiments.metrics import social_choice_suboptimality, personalized_mean_kendall_tau

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

# Global cache for the ground-truth fit so processes don't redundantly re-fit 1000 agents
_SUSHI_FIT_CACHE = None

def _get_sushi_ground_truth(dataset_path: str, mcem_config: MCEMConfig, seed: int):
    global _SUSHI_FIT_CACHE
    if _SUSHI_FIT_CACHE is not None:
        return _SUSHI_FIT_CACHE

    dataset = load_sushi(dataset_path)
    
    # 1000 agents for ground truth
    train_agents = dataset.agent_features[:1000]
    train_rankings = dataset.rankings[:1000]
    alternatives = dataset.alternative_features
    
    from grums.core.parameters import GRUMParameters
    init = GRUMParameters(
        delta=np.zeros(alternatives.shape[0]),
        interaction=np.zeros((train_agents.shape[1], alternatives.shape[1]))
    )
    
    inf = MCEMInference(mcem_config)
    fit = inf.fit_map(
        initial_params=init,
        rankings=list(train_rankings),
        agent_features=train_agents,
        alternative_features=alternatives,
    )
    _SUSHI_FIT_CACHE = (dataset, fit.params)
    return _SUSHI_FIT_CACHE

def _single_criteria_sushi_task(
    dataset_path: str,
    n_rounds: int,
    repeat_index: int,
    criterion_name: str,
    metric: str,
    seed: int,
    config: MCEMConfig,
) -> float:
    dataset, true_params = _get_sushi_ground_truth(dataset_path, config, seed)
    m = dataset.n_alternatives
    
    alternatives = [AlternativeRecord(alternative_id=j, features=dataset.alternative_features[j]) for j in range(m)]
    
    # Test agents logic: use a specific pool from the remaining 4000.
    test_agent_features = dataset.agent_features[1000:]
    test_rankings = dataset.rankings[1000:]
    
    # For Figure 5, just pick 100 random agents per repeat from the 4000 test set
    rng = np.random.default_rng(seed + repeat_index)
    idx = rng.choice(len(test_agent_features), size=100, replace=False)
    
    selected_features = test_agent_features[idx]
    selected_rankings = [test_rankings[i] for i in idx]
    
    agents = [AgentRecord(agent_id=f"a_{i}", features=feat) for i, feat in enumerate(selected_features)]
    ranking_by_agent = {a.agent_id: r for a, r in zip(agents, selected_rankings)}
    
    provider = _OracleProvider(ranking_by_agent)
    seed_obs = RankingObservation(agent_id=agents[0].agent_id, ranking=selected_rankings[0])
    
    criteria = {
        "random": _RandomCriterion(seed + 1000 + repeat_index),
        "d_opt": DOptimalityCriterion(),
        "e_opt": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=m),
        "personalized": PersonalizedChoiceCriterion(
            n_alternatives=m,
            n_agent_features=selected_features.shape[1],
            n_alternative_features=dataset.alternative_features.shape[1],
            alternative_features=dataset.alternative_features,
            population_agents=selected_features,
        ),
    }
    
    criterion = criteria[criterion_name]
    
    from grums.core.parameters import GRUMParameters
    init_params = GRUMParameters(
        delta=np.zeros(m, dtype=float),
        interaction=np.zeros((selected_features.shape[1], dataset.alternative_features.shape[1]), dtype=float)
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
    
    if metric == "social":
        return social_choice_suboptimality(true_params, result.final_params, selected_features, dataset.alternative_features)
    else:
        return personalized_mean_kendall_tau(true_params, result.final_params, selected_features, dataset.alternative_features)

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
) -> float:
    config = mcem_config or MCEMConfig(n_iterations=6, n_gibbs_samples=25, n_gibbs_burnin=12)
    out: list[float] = []
    
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
                
    return float(np.mean(out))
