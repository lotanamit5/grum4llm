#!/usr/bin/env python3
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import sys
import time
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grums.contracts import AgentRecord, AlternativeRecord, PairwiseObservation, Observation
from grums.core.parameters import GRUMParameters
from grums.elicitation import (
    AdaptiveElicitationEngine,
    PairwiseDesign,
)
from grums.providers.factory import build_preference_provider
import utils
from grums.experiments.domains import load_domain, get_agent_features

def normalize_features(features: np.ndarray, cols: list[int]) -> np.ndarray:
    """Applies Min-Max normalization to specified numeric columns."""
    norm_features = features.copy().astype(float)
    for col in cols:
        if col < norm_features.shape[1]:
            min_val = norm_features[:, col].min()
            max_val = norm_features[:, col].max()
            if max_val > min_val:
                norm_features[:, col] = (norm_features[:, col] - min_val) / (max_val - min_val)
            else:
                norm_features[:, col] = 0.0
    return norm_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--dummy", action="store_true", help="Use dummy model for testing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Configuration
    model_id = cfg.get("model_id", "Qwen/Qwen2.5-0.5B")
    domain_name = cfg.get("domain", "colors_pairwise")
    criterion_name = cfg.get("criterion", "social")
    steps = cfg.get("steps", 20)
    seed = cfg.get("seed", 42)
    pca_dim = cfg.get("pca_dim", 8)
    embedding_method = cfg.get("embedding_method", "hidden_state_pca")
    provider_method = cfg.get("provider", {}).get("method", "perplexity")
    provider_labels = cfg.get("provider", {}).get("labels", ("1", "2"))
    
    mcem_config = utils.get_mcem_config(cfg.get("mcem", {}))
    device = utils.get_torch_device(cfg.get("device", "auto"))

    # 2. Load Domain Data
    domain_data = load_domain(domain_name)
    item_names = domain_data["alternatives"]
    alternative_texts = {i: name for i, name in enumerate(item_names)}
    
    # Alternative Features
    m = len(item_names)
    if "alternative_features" in domain_data:
        alt_features = np.array(domain_data["alternative_features"])
        # Apply normalization if it's the laptops domain (hardcoded for now as per previous logic)
        if "laptops" in domain_name:
            alt_features = normalize_features(alt_features, [5, 6, 7])
        alternatives = [
            AlternativeRecord(i, torch.from_numpy(alt_features[i]).to(device).float())
            for i in range(m)
        ]
        # Inject descriptions for formatting
        for i, alt in enumerate(alternatives):
            object.__setattr__(alt, "description", item_names[i])
    else:
        # Default to one-hot if no features provided
        alt_features = np.eye(m)
        alternatives = [
            AlternativeRecord(i, torch.from_numpy(alt_features[i]).to(device).float())
            for i in range(m)
        ]

    # Agent Processing
    prompt_templates = domain_data["prompts"]
    personas = domain_data.get("personas")
    
    agent_ids = []
    prompts_by_agent_id = {}
    
    if personas:
        # Hybrid agent structure (Persona x Template)
        persona_ids = sorted(list(personas.keys()))
        for pid in persona_ids:
            for t_idx, template in enumerate(prompt_templates):
                aid = f"{pid}_t{t_idx}"
                agent_ids.append(aid)
                # Final prompt combines Persona text and Template
                prompts_by_agent_id[aid] = template.replace("{PERSONA}", personas[pid])
    else:
        # Standard agent structure (Template only)
        for i, t in enumerate(prompt_templates):
            aid = f"p_{i:02d}"
            agent_ids.append(aid)
            prompts_by_agent_id[aid] = t
    
    agent_ids = sorted(agent_ids)

    # 3. Initialize Model and Extract Agent Features
    model = None
    tokenizer = None
    if not args.dummy and (embedding_method != "random" or provider_method != "llm_stub"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if provider_method == "perplexity" or embedding_method == "hidden_state_pca":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto", 
                dtype=torch.float16 if device.type == "cuda" else torch.float32
            )

    rng = np.random.default_rng(seed)
    x = get_agent_features(
        embedding_method, 
        agent_ids, 
        prompts_by_agent_id if not personas else {aid: prompt_templates[int(aid.split("_t")[1])] for aid in agent_ids}, 
        model, 
        tokenizer, 
        rng, 
        dummy=args.dummy, 
        seed=seed,
        pca_dim=pca_dim,
        personas=personas
    )
    agents = [
        AgentRecord(aid, torch.from_numpy(x[i]).to(device).float())
        for i, aid in enumerate(agent_ids)
    ]
    # Inject final prompts for provider
    for agent in agents:
        object.__setattr__(agent, "prompt", prompts_by_agent_id[agent.agent_id])

    # 4. Initialize Provider
    kind = "llm_stub" if args.dummy else "huggingface"
    provider = build_preference_provider(
        kind,
        method=provider_method,
        labels=provider_labels,
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        device=device.type,
        prompts_by_agent_id=prompts_by_agent_id,
        alternative_texts=alternative_texts
    )

    # 5. Elicitation Setup
    k = agents[0].features.shape[0]
    l = alternatives[0].features.shape[0]
    
    print(f"\n[INFO] Starting Generalized LLM Elicitation")
    print(f"       Model:      {model_id}")
    print(f"       Domain:     {domain_name}")
    print(f"       Criterion:  {criterion_name}")
    print(f"       Embedding:  {embedding_method}")
    print(f"       Provider:   {provider_method}")
    print(f"       Steps:      {steps}")

    criteria = utils.get_criteria_map(
        m, k, l, seed, 
        torch.vstack([a.features for a in alternatives]),
        torch.vstack([a.features for a in agents]),
    )
    criterion = criteria[criterion_name]

    init_params = utils.get_init_params(m, k, l, device)

    # Initial seed observation
    seed_obs = provider.query_pairwise(agents[0], alternatives[0], alternatives[1])
    
    # 6. Run Engine
    tau_by_n = {}
    query_log = []  # Records each chosen query and its answer
    pbar = tqdm(total=steps, desc="Elicitation Rounds")

    alt_features_sorted = torch.vstack(
        [a.features for a in sorted(alternatives, key=lambda a: a.alternative_id)]
    ).to(device).to(torch.float64)

    def _compute_nll(params: GRUMParameters, obs_list: list[Observation], lookup: dict[str, AgentRecord]) -> float:
        """Compute mean NLL across all pairwise observations using Gaussian pairwise probability."""
        import math
        sigma = mcem_config.sigma
        total_nll = 0.0
        count = 0
        for obs in obs_list:
            if not hasattr(obs, "winner_id"):
                continue
            agent = lookup.get(obs.agent_id)
            if agent is None:
                continue
            x = agent.features.to(device).to(torch.float64).unsqueeze(0)  # (1, k)
            mu = (x @ params.interaction.to(device).to(torch.float64) @ alt_features_sorted.T + params.delta.to(device).to(torch.float64)).squeeze(0)  # (m,)
            mu_w = mu[obs.winner_id].item()
            mu_l = mu[obs.loser_id].item()
            # P(winner > loser) = Phi((mu_w - mu_l) / (sigma * sqrt(2)))
            p = 0.5 * (1.0 + math.erf((mu_w - mu_l) / (sigma * math.sqrt(2.0))))
            p = max(p, 1e-10)
            total_nll += -math.log(p)
            count += 1
        return total_nll / count if count > 0 else float("nan")

    def _on_after_map(n_obs: int, params: GRUMParameters, obs_list: list[Observation], lookup: dict[str, AgentRecord]) -> None:
        # Record the latest observation if this is a new step (n_obs > 1 means a new obs was added)
        if n_obs > len(query_log) + 1 or (n_obs == 1 and len(query_log) == 0):
            latest_obs = obs_list[-1] if obs_list else None
            if latest_obs is not None and hasattr(latest_obs, "winner_id"):
                agent_id = latest_obs.agent_id
                winner_id = latest_obs.winner_id
                loser_id = latest_obs.loser_id
                query_log.append({
                    "step": n_obs - 1,
                    "agent_id": agent_id,
                    "prompt": prompts_by_agent_id.get(agent_id, ""),
                    "alt_a": alternative_texts.get(winner_id, str(winner_id)),
                    "alt_b": alternative_texts.get(loser_id, str(loser_id)),
                    "winner": alternative_texts.get(winner_id, str(winner_id)),
                })

        tau_by_n[n_obs] = {
            "nll": _compute_nll(params, obs_list, lookup),
            "grum": {
                "delta": params.delta.cpu().tolist(),
                "interaction": params.interaction.cpu().tolist()
            }
        }
        
        aligned_agents = [lookup[o.agent_id] for o in obs_list]
        curr_agent_features = torch.vstack([a.features for a in aligned_agents]).to(device).to(torch.float64)
        
        bt_fit = engine.inference.fit_map(
            initial_params=init_params,
            observations=obs_list,
            agent_features=curr_agent_features,
            alternative_features=alt_features_sorted,
            fit_bt=True
        )
        
        tau_by_n[n_obs]["bt"] = {
            "nll": _compute_nll(bt_fit.params, obs_list, lookup),
            "beta": bt_fit.params.delta.cpu().tolist()
            }
        if n_obs > 1: pbar.update(1)

    import itertools
    candidate_designs = []
    for agent in agents[1:]:
        for alt_a, alt_b in itertools.combinations(alternatives, 2):
            candidate_designs.append(PairwiseDesign(agent, alt_a, alt_b))

    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=mcem_config)
    
    t0 = time.perf_counter()
    engine.run(
        provider=provider,
        initial_params=init_params,
        initial_observations=[seed_obs],
        observed_agents=[agents[0]],
        candidate_designs=candidate_designs,
        alternatives=alternatives,
        n_rounds=steps,
        on_after_map=_on_after_map,
    )
    t1 = time.perf_counter()
    pbar.close()

    # 7. Payload & Output
    payload = {
        "model_id": model_id,
        "domain": domain_name,
        "seed": seed,
        "steps": steps,
        "criterion": criterion_name,
        "embedding_method": embedding_method,
        "pca_dim": pca_dim,
        "prompts": prompts_by_agent_id,         # Full prompt map for reconstruction
        "alternatives": alternative_texts,       # Alternative id -> name map
        "agent_features": {aid: x[i].tolist() for i, aid in enumerate(agent_ids)}, # Captured directly in engine
        "alternative_features": {a.alternative_id: a.features.cpu().tolist() for a in alternatives},
        "query_log": query_log,                  # Per-step: prompt, pair, answer
        "history": tau_by_n,                     # Per-step: NLL, GRUM params, BT params
        "timing": {"total_seconds": t1 - t0},
        "finished_at_utc": utils.get_utc_timestamp(),
    }

    utils.save_experiment_result(payload, args.output_json, cfg)

if __name__ == "__main__":
    main()
