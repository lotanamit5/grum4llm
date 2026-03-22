#!/usr/bin/env python3
"""Run a single active learning experiment on the Color Ranking dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from grums.contracts import AgentRecord, AlternativeRecord, RankingObservation
from grums.core.parameters import GRUMParameters
from grums.elicitation import (
    AdaptiveElicitationEngine,
    DOptimalityCriterion,
    EOptimalityCriterion,
    RandomCriterion,
    SocialChoiceCriterion,
)
from grums.experiments.metrics import social_choice_kendall_tau
from grums.inference import MCEMConfig, MCEMInference
from grums.providers import OracleRankingProvider


def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    alternatives = []
    for cid_str, cdata in data["colors"].items():
        cid = int(cid_str)
        alternatives.append(
            AlternativeRecord(
                alternative_id=cid,
                features=np.array(cdata["features"], dtype=float)
            )
        )
    # Ensure alternatives are sorted by ID for consistency 0..4
    alternatives.sort(key=lambda a: a.alternative_id)
    
    agents = []
    prompts_data = data["prompts"]
    for aid in sorted(prompts_data.keys()):
        agents.append(
            AgentRecord(
                agent_id=aid,
                features=np.array(prompts_data[aid]["features"], dtype=float)
            )
        )
    
    rankings_by_agent = {}
    for aid, ranking in data["rankings"].items():
        rankings_by_agent[aid] = tuple(ranking)
        
    return alternatives, agents, rankings_by_agent


def compute_ground_truth(alternatives, agents, rankings_by_agent, mcem_config):
    """Fit full MAP on the entire 50-agent dataset to establish a ground truth."""
    k = agents[0].features.shape[0]
    l = alternatives[0].features.shape[0]
    m = len(alternatives)
    
    init_params = GRUMParameters(delta=np.zeros(m), interaction=np.zeros((k, l)))
    inf = MCEMInference(mcem_config)
    
    # We fit MAP on all agents
    all_rankings = [rankings_by_agent[a.agent_id] for a in agents]
    agent_feats = np.stack([a.features for a in agents])
    alt_feats = np.stack([a.features for a in alternatives])
    
    fit = inf.fit_map(
        initial_params=init_params,
        rankings=all_rankings,
        agent_features=agent_feats,
        alternative_features=alt_feats,
    )
    return fit.params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to subrun yaml config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 0))
    dataset_path = Path(cfg.get("dataset_path", "data/color_dataset.json"))
    criterion_name = str(cfg.get("criterion", "social"))
    n_rounds = int(cfg.get("n_rounds", 20))
    output_json = Path(cfg["output_json"])

    alternatives, agents, rankings_by_agent = load_dataset(dataset_path)
    if not agents:
        raise ValueError("Dataset has no agents.")

    mcem_config = MCEMConfig(
        n_iterations=8,
        n_gibbs_samples=30,
        n_gibbs_burnin=15,
    )
    
    true_params = compute_ground_truth(alternatives, agents, rankings_by_agent, mcem_config)
    
    provider = OracleRankingProvider(rankings_by_agent)
    
    criteria_map = {
        "random": RandomCriterion(seed),
        "d_opt": DOptimalityCriterion(),
        "e_opt": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=len(alternatives)),
        "personalized": SocialChoiceCriterion(n_alternatives=len(alternatives)), 
    }
    criterion = criteria_map[criterion_name]

    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=mcem_config)

    # Use first agent as seed observation, rest as candidates
    seed_obs = RankingObservation(agent_id=agents[0].agent_id, ranking=rankings_by_agent[agents[0].agent_id])
    
    k = agents[0].features.shape[0]
    l = alternatives[0].features.shape[0]
    m = len(alternatives)
    init_params = GRUMParameters(delta=np.zeros(m), interaction=np.zeros((k, l)))

    tau_by_n = {}
    
    def _on_after_map(n_obs: int, params: GRUMParameters) -> None:
        tau_by_n[n_obs] = social_choice_kendall_tau(true_params.delta, params.delta)

    engine.run(
        provider=provider,
        initial_params=init_params,
        initial_observations=[seed_obs],
        observed_agents=[agents[0]],
        candidate_agents=agents[1:],
        alternatives=alternatives,
        n_rounds=n_rounds,
        on_after_map=_on_after_map,
    )

    curve_data = [
        {"n_observations": n, "kendall_tau": tau_by_n[n]}
        for n in sorted(tau_by_n.keys())
    ]

    payload = {
        "seed": seed,
        "n_rounds": n_rounds,
        "criterion": criterion_name,
        "criteria_curve": curve_data,
        "true_delta": true_params.delta.tolist(),
        "final_tau": curve_data[-1]["kendall_tau"] if curve_data else 0.0,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
