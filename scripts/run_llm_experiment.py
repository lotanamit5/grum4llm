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
from grums.experiments.metrics import social_choice_kendall_tau, personalized_mean_kendall_tau
from grums.inference import MCEMConfig, MCEMInference
from grums.providers import OracleRankingProvider, build_preference_provider
from grums.experiments.domains import load_domain, get_agent_features


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


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to subrun yaml config")
    parser.add_argument("--dynamic", action="store_true", help="Launch dynamic HuggingFace LLM inference")
    parser.add_argument("--domain", type=str, default="colors", help="Domain alias or path to JSON")
    parser.add_argument("--checkpoints", type=int, default=0, help="Save GRUM parameter checkpoints every N steps")
    parser.add_argument("--dummy", action="store_true", help="Skip loading real LLM for testing in dynamic mode")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 0))
    dataset_path = Path(cfg.get("dataset_path", "data/color_dataset.json"))
    criterion_name = str(cfg.get("criterion", "social"))
    n_rounds = int(cfg.get("n_rounds", 20))
    output_json = Path(cfg["output_json"])

    mcem_config = MCEMConfig(n_iterations=8, n_gibbs_samples=30, n_gibbs_burnin=15)
    
    if not args.dynamic:
        alternatives, agents, rankings_by_agent = load_dataset(dataset_path)
        if not agents:
            raise ValueError("Dataset has no agents.")
            
        true_params = compute_ground_truth(alternatives, agents, rankings_by_agent, mcem_config)
        provider = OracleRankingProvider(rankings_by_agent)
        seed_obs = RankingObservation(agent_id=agents[0].agent_id, ranking=rankings_by_agent[agents[0].agent_id])
    else:
        domain_data = load_domain(args.domain)
        item_names = domain_data["alternatives"]
        alternative_texts = {i: name for i, name in enumerate(item_names)}
        
        z = np.eye(len(item_names))
        alternatives = [AlternativeRecord(alternative_id=i, features=z[i]) for i in range(len(item_names))]
        
        prompt_templates = domain_data["prompts"]
        prompts_by_agent_id = {}
        for i, p in enumerate(prompt_templates):
            prompts_by_agent_id[f"prompt_{i:02d}"] = f"{p} {{alternative}}"
        agent_ids = sorted(list(prompts_by_agent_id.keys()))

        if args.dummy:
            model = None; tokenizer = None
            provider = build_preference_provider("llm_stub")
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            base_run = cfg.get("base_run", {})
            model_name = base_run.get("model_name", "Qwen/Qwen2.5-0.5B")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
            provider = build_preference_provider(
                "huggingface", 
                model=model, 
                tokenizer=tokenizer, 
                prompts_by_agent_id=prompts_by_agent_id, 
                alternative_texts=alternative_texts
            )

        rng = np.random.default_rng(seed)
        emb_method = cfg.get("base_run", {}).get("embedding_method", "hidden_state_pca")
        x = get_agent_features(
            emb_method, agent_ids, prompts_by_agent_id, model, tokenizer, rng, args.dummy, seed
        )
        agents = [AgentRecord(agent_id=aid, features=x[i]) for i, aid in enumerate(agent_ids)]
        
        # Ground truth unknown during unmapped dynamic extraction. Padding a zero matrix.
        true_params = GRUMParameters(delta=np.zeros(len(alternatives)), interaction=np.zeros((len(agents[0].features), len(alternatives[0].features))))
        
        # Pull the first ranking explicitly for seed_obs
        seed_obs = provider.query_full_ranking(agents[0], alternatives)
    
    criteria_map = {
        "random": RandomCriterion(seed),
        "d_opt": DOptimalityCriterion(),
        "e_opt": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=len(alternatives)),
        "personalized": SocialChoiceCriterion(n_alternatives=len(alternatives)), 
    }
    criterion = criteria_map[criterion_name]

    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=mcem_config)
    # Seed initialization

    k = agents[0].features.shape[0]
    l = alternatives[0].features.shape[0]
    m = len(alternatives)
    init_params = GRUMParameters(delta=np.zeros(m), interaction=np.zeros((k, l)))

    tau_by_n = {}
    checkpoints_dict = {}
    checkpoints_interval = cfg.get("base_run", {}).get("checkpoints", args.checkpoints)
    
    def _on_after_map(n_obs: int, params: GRUMParameters) -> None:
        s_tau = social_choice_kendall_tau(true_params.delta, params.delta)
        mp_tau = personalized_mean_kendall_tau(true_params, params, agents, alternatives)
        tau_by_n[n_obs] = {"social": s_tau, "mean_person": mp_tau}
        if checkpoints_interval > 0 and (n_obs % checkpoints_interval == 0 or n_obs == n_rounds):
            checkpoints_dict[n_obs] = {
                "delta": params.delta.tolist(),
                "interaction": params.interaction.tolist()
            }

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
        {"n_observations": n, "social_tau": tau_by_n[n]["social"], "mean_person_tau": tau_by_n[n]["mean_person"]}
        for n in sorted(tau_by_n.keys())
    ]

    payload = {
        "seed": seed,
        "n_rounds": n_rounds,
        "criterion": criterion_name,
        "criteria_curve": curve_data,
        "true_delta": true_params.delta.tolist(),
        "final_tau": curve_data[-1]["social_tau"] if curve_data else 0.0,
        "checkpoints": checkpoints_dict
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
