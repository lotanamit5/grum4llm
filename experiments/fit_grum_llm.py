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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grums.contracts import AgentRecord, AlternativeRecord, PairwiseObservation
from grums.core.parameters import GRUMParameters
from grums.elicitation import (
    AdaptiveElicitationEngine,
    PairwiseDesign,
)
from grums.providers import HuggingFaceProvider, MockHuggingFaceProvider
import utils
from grums.experiments.metrics import social_choice_kendall_tau
from grums.experiments.domains import load_domain, get_agent_features

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
    
    mcem_config = utils.get_mcem_config(cfg.get("mcem", {}))
    device = utils.get_torch_device(cfg.get("device", "auto"))

    # 2. Load Domain Data
    domain_data = load_domain(domain_name)
    item_names = domain_data["alternatives"]
    alternative_texts = {i: name for i, name in enumerate(item_names)}
    
    # Simple one-hot features for alternatives
    m = len(item_names)
    alt_z = np.eye(m)
    alternatives = [AlternativeRecord(alternative_id=i, features=torch.from_numpy(alt_z[i]).to(device).to(torch.float64)) for i in range(m)]
    
    prompt_templates = domain_data["prompts"]
    prompts_by_agent_id = {f"p_{i:02d}": t for i, t in enumerate(prompt_templates)}
    agent_ids = sorted(list(prompts_by_agent_id.keys()))

    # 3. Initialize Model and Extract Agent Features
    model = None
    tokenizer = None
    if not args.dummy:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        )

    rng = np.random.default_rng(seed)
    # get_agent_features will handle dummy mode internally for embedding fallback
    x = get_agent_features(
        embedding_method, 
        agent_ids, 
        prompts_by_agent_id, 
        model, 
        tokenizer, 
        rng, 
        dummy=args.dummy, 
        seed=seed,
        pca_dim=pca_dim
    )
    agents = [AgentRecord(agent_id=aid, features=torch.from_numpy(x[i]).to(device).to(torch.float64)) for i, aid in enumerate(agent_ids)]

    # 4. Initialize Provider
    if args.dummy:
        provider = MockHuggingFaceProvider(
            prompts_by_agent_id=prompts_by_agent_id,
            alternative_texts=alternative_texts
        )
    else:
        provider = HuggingFaceProvider(
            model=model,
            tokenizer=tokenizer,
            prompts_by_agent_id=prompts_by_agent_id,
            alternative_texts=alternative_texts
        )

    # 5. Elicitation Setup
    k = pca_dim
    l = m # dimension of alternative features (one-hot)
    
    print(f"\n[INFO] Starting LLM Elicitation Experiment")
    print(f"       Model:      {model_id}")
    print(f"       Domain:     {domain_name}")
    print(f"       Criterion:  {criterion_name}")
    print(f"       Steps:      {steps}")
    print(f"       Seed:       {seed}")
    print(f"       PCA Dim:    {pca_dim}")

    criteria = utils.get_criteria_map(
        m, k, l, seed, 
        torch.vstack([a.features for a in alternatives]),
        torch.vstack([a.features for a in agents]),
    )
    criterion = criteria[criterion_name]

    init_params = utils.get_init_params(m, k, l, device)

    # Initial seed observation (first agent compared to first two alternatives)
    seed_obs = provider.query_pairwise(agents[0], alternatives[0], alternatives[1])
    print(f"[INFO] Seed Obs: Agent {agents[0].agent_id} preferred {alternative_texts[seed_obs.winner_id]} over {alternative_texts[seed_obs.loser_id]}")

    # 6. Run Engine
    tau_by_n = {}
    
    from tqdm import tqdm
    pbar = tqdm(total=steps, desc="Elicitation Rounds")

    def _on_after_map(n_obs: int, params: GRUMParameters) -> None:
        # We don't have true params for real LLMs, but we can track the estimated delta
        tau_by_n[n_obs] = {
            "delta": params.delta.cpu().tolist(),
            "interaction": params.interaction.cpu().tolist()
        }
        if n_obs > 1: # We don't update on the seed init (n_obs=1)
            pbar.update(1)

    # Candidate designs: all pairs for all candidate agents
    import itertools
    candidate_designs = []
    for agent in agents[1:]:
        for alt_a, alt_b in itertools.combinations(alternatives, 2):
            candidate_designs.append(PairwiseDesign(agent, alt_a, alt_b))

    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=mcem_config)
    
    print(f"\n[INFO] Running Adaptive Elicitation Pipeline...")
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
        "pca_dim": pca_dim,
        "history": tau_by_n,
        "timing": {
            "total_seconds": t1 - t0,
            "average_step_seconds": (t1 - t0) / steps if steps > 0 else 0
        },
        "finished_at_utc": utils.get_utc_timestamp(),
    }

    utils.save_experiment_result(payload, args.output_json, cfg)

if __name__ == "__main__":
    main()
