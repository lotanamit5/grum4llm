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


def generate_prompts() -> dict[str, str]:
    """Generate 50 distinct color preference prompts."""
    templates = [
        "My favorite color is ",
        "If I had to paint my room, I'd choose ",
        "The most beautiful color in the world is ",
        "When it comes to clothes, I really love wearing ",
        "The best color for a sports car is ",
        "I find myself always drawn to the color ",
        "The color that brings me the most joy is ",
        "If I was a flower, I'd want to be ",
        "My lucky color has always been ",
        "The most relaxing color to stare at is ",
    ]
    prompts = {}
    idx = 0
    
    # We'll reintroduce suffixes or just duplicate templates to ensure we get exactly 50 prompts
    # since the experiment was designed for 50 optional prompts.
    suffixes = ["", " for sure", " without a doubt", " honestly", " definitely"]
    for t in templates:
        for s in suffixes:
            agent_id = f"prompt_{idx:02d}"
            prompt = f"{t}{{alternative}}{s}".strip()
            
            if "{alternative}" not in prompt and t.endswith(" "):
                pass
                
            prompts[agent_id] = prompt
            idx += 1
            if idx >= 50:
                break
        if idx >= 50:
            break

    return prompts


def get_agent_features(args, agent_ids, prompts_by_agent_id, model, tokenizer, rng):
    N = len(agent_ids)
    if args.embedding_method == "random":
        return rng.normal(0, 1, size=(N, 5))
    
    if args.embedding_method == "hidden_state_pca":
        from sklearn.decomposition import PCA
        import torch
        
        print("Extracting hidden states from Qwen...")
        embeddings = []
        for aid in agent_ids:
            prompt = prompts_by_agent_id[aid]
            clean_prompt = prompt.replace("{alternative}", "").strip()
            
            # Use real model unless dummy is specified
            if args.dummy or model is None:
                # Fallback to random if dummy 
                embeddings.append(rng.normal(0, 1, size=(896,)))
            else:
                inputs = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    # Get last hidden state of the last token
                    last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
                embeddings.append(last_hidden)
                
        embeddings = np.array(embeddings)
        print("Applying PCA to reduce from", embeddings.shape[1], "to 5...")
        pca = PCA(n_components=5, random_state=args.seed)
        return pca.fit_transform(embeddings)

    if args.embedding_method == "sentence_transformer_pca":
        from sklearn.decomposition import PCA
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
            
        print("Extracting embeddings using sentence-transformers...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [prompts_by_agent_id[aid].replace("{alternative}", "").strip() for aid in agent_ids]
        embeddings = st_model.encode(texts)
        
        print("Applying PCA to reduce from", embeddings.shape[1], "to 5...")
        pca = PCA(n_components=5, random_state=args.seed)
        return pca.fit_transform(embeddings)
        
    raise ValueError(f"Unknown embedding method: {args.embedding_method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to orchestrator config defining dataset params")
    parser.add_argument("--seed", type=int, default=42)
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

    colors = ["red", "blue", "green", "yellow", "purple"]
    alternative_texts = {i: color for i, color in enumerate(colors)}
    
    # Generate 5D one-hot features for the 5 colors
    z = np.eye(len(colors))
    alternatives = [AlternativeRecord(alternative_id=i, features=z[i]) for i in range(len(colors))]
    
    prompts_by_agent_id = generate_prompts()
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
    x = get_agent_features(args, agent_ids, prompts_by_agent_id, model, tokenizer, rng)
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
            "n_colors": len(colors),
            "embedding_method": args.embedding_method,
        },
        "colors": {
            i: {
                "name": colors[i],
                "features": z[i].tolist()
            } for i in range(len(colors))
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
