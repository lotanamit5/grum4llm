import pytest
import torch
from unittest.mock import MagicMock
from grums.providers.huggingface import HuggingFaceProvider
from grums.contracts import AgentRecord, AlternativeRecord

class MockModel:
    def __init__(self):
        self.device = "cpu"
    def __call__(self, input_ids, labels=None, **kwargs):
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(float(input_ids.size(1)))
        return mock_output

class MockTokenizer:
    def __init__(self):
        pass
    def __call__(self, text, return_tensors=None, **kwargs):
        mock_inputs = MagicMock()
        mock_inputs.input_ids = torch.zeros((1, len(text)))
        return mock_inputs

def test_query_pairwise_permutation_averaging():
    model = MockModel()
    tokenizer = MockTokenizer()
    
    # Template with placeholders
    prompts = {"agent_1": "Between {A} and {B}, I prefer "}
    alt_texts = {0: "red", 1: "blue"}
    
    provider = HuggingFaceProvider(model, tokenizer, prompts, alt_texts)
    
    agent = AgentRecord(agent_id="agent_1", features=torch.zeros(1))
    alt_0 = AlternativeRecord(alternative_id=0, features=torch.zeros(1))
    alt_1 = AlternativeRecord(alternative_id=1, features=torch.zeros(1))
    
    # In my MockModel, loss = len(text). 
    # score = -loss.
    
    # Permutation 1: A=red, B=blue. 
    # Text choice red: "Between red and blue, I prefer red" (len 34)
    # Text choice blue: "Between red and blue, I prefer blue" (len 35)
    
    # Permutation 2: A=blue, B=red.
    # Text choice red: "Between blue and red, I prefer red" (len 34)
    # Text choice blue: "Between blue and red, I prefer blue" (len 35)
    
    # Average Loss red: (34 + 34) / 2 = 34. Score red = -34.
    # Average Loss blue: (35 + 35) / 2 = 35. Score blue = -35.
    
    # Red should win.
    
    obs = provider.query_pairwise(agent, alt_0, alt_1)
    
    assert obs.winner_id == 0
    assert obs.loser_id == 1
    assert obs.agent_id == "agent_1"

def test_query_pairwise_no_placeholders():
    model = MockModel()
    tokenizer = MockTokenizer()
    
    # Old school template
    prompts = {"agent_1": "I like "}
    alt_texts = {0: "red", 1: "blueish_color"} # Longer name = worse score
    
    provider = HuggingFaceProvider(model, tokenizer, prompts, alt_texts)
    
    agent = AgentRecord(agent_id="agent_1", features=torch.zeros(1))
    alt_0 = AlternativeRecord(alternative_id=0, features=torch.zeros(1))
    alt_1 = AlternativeRecord(alternative_id=1, features=torch.zeros(1))
    
    obs = provider.query_pairwise(agent, alt_0, alt_1)
    
    assert obs.winner_id == 0 # "red" is shorter
    assert obs.loser_id == 1
