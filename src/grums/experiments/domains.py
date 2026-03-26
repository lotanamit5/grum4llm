from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
DOMAINS_DIR = ROOT / "configs" / "llm" / "domains"

DOMAIN_ALIASES = {
    "colors_pairwise": DOMAINS_DIR / "colors_pairwise.json",
    "laptops": DOMAINS_DIR / "laptops.json",
}

def load_domain(domain_name: str) -> dict:
    path = DOMAIN_ALIASES.get(domain_name)
    if path is None:
        # Fallback to see if it's a direct file path
        path = Path(domain_name)
    
    if not path.is_file():
        raise FileNotFoundError(f"Domain config not found for '{domain_name}' at {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_agent_features(
    embedding_method: str, 
    agent_ids: list[str], 
    prompts_by_agent_id: dict[str, str], 
    model, 
    tokenizer, 
    rng, 
    dummy: bool = False, 
    seed: int = 42,
    pca_dim: int = 8,
    personas: dict[str, str] | None = None
):
    import numpy as np
    N = len(agent_ids)
    
    if embedding_method == "one_hot":
        return np.eye(N)

    if embedding_method == "random":
        # Check if we should use fixed PCA dim or matching N
        dim = pca_dim if pca_dim > 0 else N
        return rng.normal(0, 1, size=(N, dim))
    
    if embedding_method in ["hidden_state_pca", "hybrid_onehot_pca"]:
        from sklearn.decomposition import PCA
        import torch
        
        # In hybrid mode, we embed TEMPLATES, not final prompts
        # Extract templates if hybrid, otherwise use prompts
        if embedding_method == "hybrid_onehot_pca":
            if not personas:
                raise ValueError("hybrid_onehot_pca requires personas dictionary")
            # We assume agent_ids are like personaID_tIdx
            # But we actually want to embed unique templates once
            unique_templates = sorted(list(set(prompts_by_agent_id.values())))
            text_to_embed = unique_templates
        else:
            text_to_embed = [prompts_by_agent_id[aid] for aid in agent_ids]

        print(f"Extracting hidden states (Last Token) for {len(text_to_embed)} unique texts...")
        embeddings = []
        for text in text_to_embed:
            if dummy or model is None:
                # Fallback to random if dummy (assuming Qwen hidden dim 896)
                embeddings.append(rng.normal(0, 1, size=(896,)))
            else:
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
                embeddings.append(last_hidden)
                
        embeddings = np.array(embeddings)
        print(f"Applying PCA to reduce from {embeddings.shape[1]} to {pca_dim}...")
        pca = PCA(n_components=pca_dim, random_state=seed)
        reduced_embeddings = pca.fit_transform(embeddings)

        if embedding_method == "hidden_state_pca":
            return reduced_embeddings
        
        # Hybrid logic: concatenate one-hot persona + PCA template
        persona_list = sorted(list(personas.keys()))
        final_features = []
        # Re-derive template vs persona per agent_id
        # We expect agent_id to be {persona_id}_t{template_idx}
        for aid in agent_ids:
            p_id = aid.split("_t")[0]
            t_idx = int(aid.split("_t")[1])
            
            p_one_hot = np.zeros(len(persona_list))
            p_one_hot[persona_list.index(p_id)] = 1.0
            
            # reduced_embeddings corresponds to unique_templates
            # We need to find which unique_template index matches template_idx?
            # Actually, laptop worker passed templates directly. 
            # In domains.py we'll assume the order of unique_templates matches template_idx
            # if the input was created correctly.
            final_features.append(np.concatenate([p_one_hot, reduced_embeddings[t_idx]]))
            
        return np.array(final_features)

    if embedding_method == "sentence_transformer_pca":
        from sklearn.decomposition import PCA
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
            
        print(f"Extracting embeddings using sentence-transformers for {N} agents...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [prompts_by_agent_id[aid] for aid in agent_ids]
        embeddings = st_model.encode(texts)
        
        print(f"Applying PCA to reduce from {embeddings.shape[1]} to {pca_dim}...")
        pca = PCA(n_components=pca_dim, random_state=seed)
        return pca.fit_transform(embeddings)
        
    raise ValueError(f"Unknown embedding method: {embedding_method}")
