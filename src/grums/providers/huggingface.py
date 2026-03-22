"""Provider leveraging HuggingFace models for preference elicitation."""

from __future__ import annotations

import re
import torch
from typing import Any

from grums.contracts import AgentRecord, AlternativeRecord, PreferenceProvider, RankingObservation


class HuggingFaceProvider(PreferenceProvider):
    """Query HuggingFace LLMs to elicit rankings.
    
    Ranks alternatives by the negative perplexity (log likelihood) of the 
    formulated prompt + alternative text string.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompts_by_agent_id: dict[str, str],
        alternative_texts: dict[int, str] | None = None,
    ) -> None:
        """Initialize the HuggingFace provider.
        
        Args:
            model: A transformers language model (e.g., AutoModelForCausalLM).
            tokenizer: The corresponding tokenizer.
            prompts_by_agent_id: Mapping from agent_id to its specific text prompt.
            alternative_texts: Optional mapping from alternative_id to text string.
                If not provided, the alternative_id will be stringified.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._prompts_by_agent_id = prompts_by_agent_id
        self._alternative_texts = alternative_texts or {}

    def _compute_negative_perplexity(self, text: str) -> float:
        inputs = self._tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self._model.device)
        
        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            
        # CrossEntropyLoss returns the mean loss over the sequence.
        # Perplexity is exp(loss), so higher loss = higher perplexity = worse.
        # We return negative loss to rank higher -> better (lower perplexity).
        return -outputs.loss.item()

    def query_full_ranking(
        self,
        agent: AgentRecord,
        alternatives: list[AlternativeRecord],
    ) -> RankingObservation:
        if agent.agent_id not in self._prompts_by_agent_id:
            raise KeyError(f"No prompt defined for agent_id: {agent.agent_id!r}")
            
        system_prompt = self._prompts_by_agent_id[agent.agent_id]
        
        scored_alts = []
        for alt in alternatives:
            alt_text = self._alternative_texts.get(alt.alternative_id, str(alt.alternative_id))
            
            if "{alternative}" in system_prompt:
                raw_text = system_prompt.format(alternative=alt_text)
            else:
                raw_text = f"{system_prompt}{alt_text}"
                
            if getattr(self._tokenizer, "chat_template", None) is not None:
                messages = [{"role": "user", "content": raw_text}]
                text_to_score = self._tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                text_to_score = raw_text
                
            neg_ppl = self._compute_negative_perplexity(text_to_score)
            scored_alts.append((neg_ppl, alt.alternative_id))
            
        # Sort descending by negative perplexity (highest value = lowest perplexity = top rank)
        scored_alts.sort(key=lambda x: x[0], reverse=True)
        ranking = tuple(alt_id for _, alt_id in scored_alts)
        
        return RankingObservation(
            agent_id=agent.agent_id,
            ranking=ranking,
        )
