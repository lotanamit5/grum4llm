#!/usr/bin/env python3

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grums.inference import MCEMConfig
from grums.providers.synthetic import SyntheticProvider
from grums.experiments.sushi import _get_sushi_ground_truth
from grums.providers import OracleRankingProvider
from grums.contracts import AgentRecord, AlternativeRecord, RankingObservation
from grums.elicitation import (
    AdaptiveElicitationEngine,
    RandomCriterion,
    DOptimalityCriterion,
    EOptimalityCriterion,
    SocialChoiceCriterion,
    PersonalizedChoiceCriterion,
)
from grums.experiments.metrics import social_choice_kendall_tau, personalized_mean_kendall_tau, raw_mean_kendall_tau

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg.get("dataset", "ds1")
    criterion_name = cfg.get("criterion", "social")
    steps = cfg.get("steps", 75)
    checkpoints = cfg.get("checkpoints", 0)
    seed = cfg.get("seed", 42)
    mcem_cfg = cfg.get("mcem", {})

    mcem_config = MCEMConfig(
        n_iterations=mcem_cfg.get("n_iterations", 8),
        n_gibbs_samples=mcem_cfg.get("n_gibbs_samples", 30),
        n_gibbs_burnin=mcem_cfg.get("n_gibbs_burnin", 15),
    )

    if dataset_name in ("ds0", "ds1", "ds2"):
        provider = SyntheticProvider(ds_name=dataset_name, seed=seed)
        true_params = provider.true_params
        agents = provider.agents
        alternatives = provider.alternatives
        ranking_by_agent = provider._ranking_by_agent_id
        
        test_agent_features = np.array([a.features for a in agents])
        test_alternative_features = np.array([a.features for a in alternatives])
        test_rankings = [ranking_by_agent[a.agent_id] for a in agents]

    elif dataset_name == "sushi":
        dataset_path = cfg.get("dataset_path", ".data")
        dataset, true_params, train_idx = _get_sushi_ground_truth(dataset_path, mcem_config, seed)
        m = dataset.n_alternatives
        alternatives = [AlternativeRecord(alternative_id=j, features=dataset.alternative_features[j]) for j in range(m)]

        all_indices = np.arange(len(dataset.agent_features))
        test_indices = np.array([i for i in all_indices if i not in train_idx])
        test_agent_features = dataset.agent_features[test_indices]
        test_rankings = [dataset.rankings[i] for i in test_indices]

        rng = np.random.default_rng(seed)
        idx = rng.choice(len(test_agent_features), size=100, replace=False)
        selected_features = test_agent_features[idx]
        selected_rankings = [test_rankings[i] for i in idx]

        agents = [AgentRecord(agent_id=f"a_{i}", features=feat) for i, feat in enumerate(selected_features)]
        ranking_by_agent = {a.agent_id: r for a, r in zip(agents, selected_rankings)}
        provider = OracleRankingProvider(ranking_by_agent)
        
        test_agent_features = selected_features
        test_alternative_features = dataset.alternative_features
        test_rankings = selected_rankings
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    m = len(alternatives)
    criteria = {
        "random": RandomCriterion(seed),
        "d_opt": DOptimalityCriterion(),
        "e_opt": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=m),
        "personalized": PersonalizedChoiceCriterion(
            n_alternatives=m,
            n_agent_features=test_agent_features.shape[1],
            n_alternative_features=test_alternative_features.shape[1],
            alternative_features=test_alternative_features,
            population_agents=test_agent_features,
        ),
    }
    criterion = criteria[criterion_name]

    from grums.core.parameters import GRUMParameters
    init_params = GRUMParameters(
        delta=np.zeros(m, dtype=float),
        interaction=np.zeros((test_agent_features.shape[1], test_alternative_features.shape[1]), dtype=float)
    )

    seed_obs = RankingObservation(agent_id=agents[0].agent_id, ranking=ranking_by_agent[agents[0].agent_id])

    tau_by_n = {}
    checkpoints_dict = {}

    def _on_after_map(n_obs: int, params: GRUMParameters) -> None:
        s_tau = social_choice_kendall_tau(true_params.delta, params.delta)
        mp_tau = personalized_mean_kendall_tau(true_params, params, test_agent_features, test_alternative_features)
        rp_tau = raw_mean_kendall_tau(params, test_agent_features, test_alternative_features, test_rankings)
        tau_by_n[n_obs] = {"social_tau": s_tau, "mean_person_tau": mp_tau, "raw_person_tau": rp_tau}
        
        if checkpoints > 0 and (n_obs % checkpoints == 0 or n_obs == steps):
            checkpoints_dict[n_obs] = {
                "delta": params.delta.tolist(),
                "interaction": params.interaction.tolist()
            }

    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=mcem_config)
    
    t0 = time.perf_counter()
    engine.run(
        provider=provider,
        initial_params=init_params,
        initial_observations=[seed_obs],
        observed_agents=[agents[0]],
        candidate_agents=agents[1:],
        alternatives=alternatives,
        n_rounds=steps,
        on_after_map=_on_after_map,
    )
    t1 = time.perf_counter()
    
    total_seconds = t1 - t0
    average_step_seconds = total_seconds / steps if steps > 0 else 0.0

    curve_data = [
        {
            "n_observations": n, 
            "social_tau": tau_by_n[n]["social_tau"], 
            "mean_person_tau": tau_by_n[n]["mean_person_tau"],
            "raw_person_tau": tau_by_n[n]["raw_person_tau"]
        }
        for n in sorted(tau_by_n.keys())
    ]

    payload = {
        "dataset": dataset_name,
        "seed": seed,
        "steps": steps,
        "criterion": criterion_name,
        "criteria_curve": curve_data,
        "true_delta": true_params.delta.tolist(),
        "final_tau": curve_data[-1]["social_tau"] if curve_data else 0.0,
        "checkpoints": checkpoints_dict,
        "total_seconds": total_seconds,
        "average_step_seconds": average_step_seconds,
        "finished_at_utc": _utc_now_iso(),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
    else:
        print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
