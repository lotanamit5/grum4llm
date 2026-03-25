#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# 1. Add project root to sys.path
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
from grums.providers import HuggingFaceProvider, MockHuggingFaceProvider, HuggingFaceChoiceProvider
import utils

def normalize_features(features: np.ndarray) -> np.ndarray:
    """Applies Min-Max normalization to numeric columns [5, 6, 7] (CPU, RAM, Price)."""
    norm_features = features.copy().astype(float)
    for col in [5, 6, 7]:
        min_val = norm_features[:, col].min()
        max_val = norm_features[:, col].max()
        if max_val > min_val:
            norm_features[:, col] = (norm_features[:, col] - min_val) / (max_val - min_val)
        else:
            norm_features[:, col] = 0.0
    return norm_features

def main():
    parser = argparse.ArgumentParser(description="GRUM Laptop Preference Elicitation")
    parser.add_argument("--config", type=Path, required=True, help="Path to trial YAML config")
    parser.add_argument("--output_json", type=Path, required=True, help="Path to save results")
    parser.add_argument("--dummy", action="store_true", help="Use Mock Provider")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = utils.get_torch_device(cfg.get("device", "auto"))
    
    # 2. Load Domain
    domain_path = ROOT / "configs" / "llm" / "domains" / "laptops.json"
    with open(domain_path, "r") as f:
        domain = json.load(f)

    laptops_text = domain["alternatives"]
    laptops_raw_features = np.array(domain["alternative_features"]) # [M, 8]
    personas = domain["personas"] # dict {persona_id: text}
    templates = domain["prompts"]

    # 3. Process Alternatives
    norm_alt_features = normalize_features(laptops_raw_features)
    alternatives = [
        AlternativeRecord(i, torch.tensor(norm_alt_features[i], device=device).float())
        for i in range(len(laptops_text))
    ]
    # Inject descriptions for the provider (used in format(A=..., B=...))
    for i, alt in enumerate(alternatives):
        object.__setattr__(alt, "description", laptops_text[i])

    # 4. Prepare Agents (Hybrid Features: One-hot Persona + Embedded Template)
    # We create one agent for each unique (persona, template) pair
    persona_list = list(personas.keys())
    agent_records = []
    prompts_by_agent_id = {}
    
    # PCA for prompt embeddings
    from sklearn.decomposition import PCA
    from transformers import AutoTokenizer, AutoModel
    
    pca_dim = cfg.get("pca_dim", 8)
    model_id = cfg.get("model_id", "Qwen/Qwen2.5-0.5B")
    
    print(f"[INFO] Embedding {len(templates)} prompt templates...")
    if args.dummy:
        template_embeddings = np.random.normal(0, 1, (len(templates), 896)) 
    else:
        # Use a small model for embedding if possible, or the same model's tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        emb_model = AutoModel.from_pretrained(model_id).to(device)
        template_embeddings = []
        for t in templates:
            inputs = tokenizer(t, return_tensors="pt").to(device)
            with torch.no_grad():
                out = emb_model(**inputs)
                template_embeddings.append(out.last_hidden_state[0, -1, :].cpu().numpy())
        template_embeddings = np.array(template_embeddings)
    
    pca = PCA(n_components=pca_dim, random_state=seed)
    pca_template_features = pca.fit_transform(template_embeddings)

    # Combine one-hot and PCA
    for p_idx, p_id in enumerate(persona_list):
        for t_idx, template in enumerate(templates):
            agent_id = f"{p_id}_t{t_idx}"
            
            # One-hot persona feature
            one_hot = np.zeros(len(persona_list))
            one_hot[p_idx] = 1.0
            
            # Combined feature vector
            feat = np.concatenate([one_hot, pca_template_features[t_idx]])
            
            # The actual prompt used by provider: Persona text + Template
            # We inject the Persona into the Template placeholder
            final_prompt = template.replace("{PERSONA}", personas[p_id])
            
            agent = AgentRecord(agent_id, torch.tensor(feat, device=device).float())
            # Inject prompt into agent record for the provider
            object.__setattr__(agent, "prompt", final_prompt)
            
            agent_records.append(agent)
            prompts_by_agent_id[agent_id] = final_prompt

    # 5. Initialize GRUM & Provider
    m = len(alternatives)
    k_alt = norm_alt_features.shape[1]
    l_agent = len(agent_records[0].features)
    
    init_params = utils.get_init_params(m, l_agent, k_alt, device)
    mcem_cfg = utils.get_mcem_config(cfg)

    if args.dummy:
        provider = MockHuggingFaceProvider(model_id, device=device)
    else:
        provider = HuggingFaceChoiceProvider(model_id, device=device, labels=("1", "2"))

    # Initial seed observation (first agent compared to first two alternatives)
    seed_obs = provider.query_pairwise(agent_records[0], alternatives[0], alternatives[1])
    print(f"[INFO] Seed Obs: Agent {agent_records[0].agent_id} preferred {laptops_text[seed_obs.winner_id]} over {laptops_text[seed_obs.loser_id]}")

    # 6. Setup Criterion
    criterion_name = cfg.get("criterion", "social")
    criteria = utils.get_criteria_map(
        m, l_agent, k_alt, seed,
        torch.vstack([a.features for a in alternatives]),
        torch.vstack([a.features for a in agent_records]),
    )
    criterion = criteria[criterion_name]

    engine = AdaptiveElicitationEngine(criterion=criterion, mcem_config=mcem_cfg)

    # 7. Run Elicitation Pipeline
    steps = cfg.get("steps", 20)
    
    # Candidate designs: all pairs for all candidate agents
    import itertools
    candidate_designs = []
    # Skip the first agent used for seeding to avoid duplicate entries in some criteria
    for agent in agent_records[1:]:
        for alt_a, alt_b in itertools.combinations(alternatives, 2):
            from grums.elicitation import PairwiseDesign
            candidate_designs.append(PairwiseDesign(agent, alt_a, alt_b))

    print(f"\n[INFO] Starting Laptop Recommendation Experiment")
    print(f"       Model:      {model_id}")
    print(f"       Criterion:  {criterion_name}")
    print(f"       Steps:      {steps}")
    print(f"       Laptops:    {m}")
    print(f"       Agents:     {len(agent_records)} (4 Personas x {len(templates)} Templates)")

    tau_by_n = {}
    pbar = tqdm(total=steps, desc="Elicitation Rounds")

    def _on_after_map(n_obs: int, params: GRUMParameters, obs_list: list[Observation], lookup: dict[str, AgentRecord]) -> None:
        # 1. Store GRUM params
        tau_by_n[n_obs] = {
            "grum": {
                "delta": params.delta.cpu().tolist(),
                "interaction": params.interaction.cpu().tolist()
            }
        }
        
        # 2. Fit Bradley-Terry (B=0) on same observations
        aligned_agents = [lookup[o.agent_id] for o in obs_list]
        curr_agent_features = torch.vstack([a.features for a in aligned_agents]).to(device).to(torch.float64)
        
        # Ensure alt features are correctly stacked/sorted
        alts_sorted = sorted(alternatives, key=lambda a: a.alternative_id)
        curr_alt_features = torch.vstack([a.features for a in alts_sorted]).to(device).to(torch.float64)
        
        bt_fit = engine.inference.fit_map(
            initial_params=init_params,
            observations=obs_list,
            agent_features=curr_agent_features,
            alternative_features=curr_alt_features,
            fit_bt=True
        )
        
        tau_by_n[n_obs]["bt"] = {
            "delta": bt_fit.params.delta.cpu().tolist()
        }
        
        if n_obs > 1: # We don't update on the seed init (n_obs=1)
            pbar.update(1)

    # 6. Run Elicitation Pipeline
    t0 = time.perf_counter()
    engine.run(
        provider=provider,
        initial_params=init_params,
        initial_observations=[seed_obs],
        observed_agents=[agent_records[0]],
        candidate_designs=candidate_designs,
        alternatives=alternatives,
        n_rounds=steps,
        on_after_map=_on_after_map,
    )
    t1 = time.perf_counter()
    pbar.close()

    # 7. Finalize and Save
    payload = {
        "model_id": model_id,
        "domain": "laptops",
        "config": cfg,
        "history": tau_by_n,
        "timing": {
            "total_seconds": t1 - t0,
            "average_step_seconds": (t1 - t0) / steps if steps > 0 else 0
        },
        "finished_at_utc": utils.get_utc_timestamp(),
    }

    utils.save_experiment_result(payload, args.output_json, cfg)
    print(f"\n[SUCCESS] Results saved to: {args.output_json}")

if __name__ == "__main__":
    main()
