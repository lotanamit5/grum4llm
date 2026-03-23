from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
DOMAINS_DIR = ROOT / "configs" / "llm" / "domains"

DOMAIN_ALIASES = {
    "colors": DOMAINS_DIR / "colors.json",
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

def get_agent_features(embedding_method: str, agent_ids: list[str], prompts_by_agent_id: dict[str, str], model, tokenizer, rng, dummy: bool = False, seed: int = 42):
    import numpy as np
    N = len(agent_ids)
    if embedding_method == "random":
        return rng.normal(0, 1, size=(N, 5))
    
    if embedding_method == "hidden_state_pca":
        from sklearn.decomposition import PCA
        import torch
        
        print("Extracting hidden states from Qwen...")
        embeddings = []
        for aid in agent_ids:
            prompt = prompts_by_agent_id[aid]
            clean_prompt = prompt.replace("{alternative}", "").strip()
            
            # Use real model unless dummy is specified
            if dummy or model is None:
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
        pca = PCA(n_components=5, random_state=seed)
        return pca.fit_transform(embeddings)

    if embedding_method == "sentence_transformer_pca":
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
        pca = PCA(n_components=5, random_state=seed)
        return pca.fit_transform(embeddings)
        
    raise ValueError(f"Unknown embedding method: {embedding_method}")
