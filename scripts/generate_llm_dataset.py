#!/usr/bin/env python3
"""Generate a static dataset of LLM color rankings using Qwen2.5-0.5B."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grums.contracts import AgentRecord, AlternativeRecord
from grums.providers import build_preference_provider

from grums.experiments.domains import load_domain, get_agent_features





def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to orchestrator config defining dataset params")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain", type=str, default="colors", help="Domain alias or path to JSON")
    parser.add_argument("--dummy", action="store_true", help="Skip loading real LLM for testing")
    args = parser.parse_args()
    
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    base_run = cfg.get("base_run", {})
    model_name = base_run.get("model_name", "Qwen/Qwen2.5-0.5B")
    
    # We dynamically attach these to argparse args since they are passed around
    args.embedding_method = base_run.get("embedding_method", "hidden_state_pca")
    args.model_name = model_name
    output_path = Path(base_run.get("dataset_path", "data/color_dataset.json"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    domain_data = load_domain(args.domain)
    item_names = domain_data["alternatives"]
    alternative_texts = {i: name for i, name in enumerate(item_names)}
    
    # Generate one-hot features for the alternatives
    z = np.eye(len(item_names))
    alternatives = [AlternativeRecord(alternative_id=i, features=z[i]) for i in range(len(item_names))]
    
    prompt_templates = domain_data["prompts"]
    prompts_by_agent_id = {}
    for i, p in enumerate(prompt_templates):
        # Format strictly linearly combining the prompt without suffix
        prompts_by_agent_id[f"prompt_{i:02d}"] = f"{p} {{alternative}}"
        
    agent_ids = sorted(list(prompts_by_agent_id.keys()))

    print(f"Loading model {args.model_name}...")
    model = None
    tokenizer = None
    if args.dummy:
        print("Using dummy LLM provider for testing.")
        provider = build_preference_provider("llm_stub")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            device_map="auto",
            torch_dtype=torch.float16,
        )
        provider = build_preference_provider(
            kind="huggingface",
            model=model,
            tokenizer=tokenizer,
            prompts_by_agent_id=prompts_by_agent_id,
            alternative_texts=alternative_texts,
        )

    # Generate 5D features for agents using PCA on embeddings
    x = get_agent_features(
        embedding_method=args.embedding_method, 
        agent_ids=agent_ids, 
        prompts_by_agent_id=prompts_by_agent_id, 
        model=model, 
        tokenizer=tokenizer, 
        rng=rng, 
        dummy=args.dummy, 
        seed=args.seed
    )
    agents = [AgentRecord(agent_id=aid, features=x[i]) for i, aid in enumerate(agent_ids)]

    print("Generating LLM rankings...")
    rankings_raw = {}
    for agent in agents:
        obs = provider.query_full_ranking(agent, alternatives)
        rankings_raw[agent.agent_id] = [int(v) for v in obs.ranking]
        print(f"[{agent.agent_id}] Ranking: {rankings_raw[agent.agent_id]}")

    payload = {
        "metadata": {
            "model": args.model_name,
            "seed": args.seed,
            "n_prompts": len(agents),
            "n_alternatives": len(item_names),
            "embedding_method": args.embedding_method,
        },
        "alternatives": {
            i: {
                "name": item_names[i],
                "features": z[i].tolist()
            } for i in range(len(item_names))
        },
        "prompts": {
            agent.agent_id: {
                "text": prompts_by_agent_id[agent.agent_id],
                "features": agent.features.tolist()
            } for agent in agents
        },
        "rankings": rankings_raw
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    print(f"Dataset generated and saved to {output_path}")

if __name__ == "__main__":
    main()
