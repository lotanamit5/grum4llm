import pytest
from grums.datasets.sushi import load_sushi, SushiDataset
from pathlib import Path
import tempfile

def test_load_sushi_dimensions() -> None:
    # We use a temporary directory to force download once or cache it natively
    # but since it's an external URL, the first time it will download 217KB.
    # To avoid re-downloading in regular tests, we can just use the default .data dir
    # so it caches after the first run.
    data_dir = Path(".data")
    
    dataset = load_sushi(data_dir=data_dir)
    
    # Verify dimensions based on Kamishima's spec
    assert dataset.n_agents == 5000, "Should have 5000 agents"
    assert dataset.n_alternatives == 100, "Should have 100 alternatives"
    assert len(dataset.rankings) == 5000, "Should have 5000 rankings"
    
    assert dataset.agent_features.shape == (5000, 10), "10 attributes per agent"
    assert dataset.alternative_features.shape == (100, 7), "7 attributes per alternative"
    
    # Check that rankings actually contain 0-99 item IDs
    first_ranking = dataset.rankings[0]
    assert len(first_ranking) == 10, "Each ranking should have exactly 10 items"
    assert all(0 <= item_id < 100 for item_id in first_ranking), "Item IDs must be valid"
