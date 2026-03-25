"""Provider leveraging HuggingFace models for preference elicitation."""

from __future__ import annotations

import re
import torch
from typing import Any

from grums.contracts import AgentRecord, AlternativeRecord, PreferenceProvider, RankingObservation, PairwiseObservation


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

    def query_pairwise(
        self,
        agent: AgentRecord,
        alt_a: AlternativeRecord,
        alt_b: AlternativeRecord,
    ) -> PairwiseObservation:
        """Elicit a pairwise comparison between two alternatives.
        
        Ranks by averaging negative perplexity across (A, B) and (B, A) prompt permutations.
        """
        if agent.agent_id not in self._prompts_by_agent_id:
            raise KeyError(f"No prompt defined for agent_id: {agent.agent_id!r}")
            
        template = self._prompts_by_agent_id[agent.agent_id]
        text_a = self._alternative_texts.get(alt_a.alternative_id, str(alt_a.alternative_id))
        text_b = self._alternative_texts.get(alt_b.alternative_id, str(alt_b.alternative_id))
        
        # We compute scores for choice A and choice B across both permutations
        # Permutation 1: A=text_a, B=text_b
        # Permutation 2: A=text_b, B=text_a
        
        def get_score(prompt_a, prompt_b, choice_text):
            # If the template doesn't have {A}/{B}, we just append choice_text as before
            # but usually it will have them now.
            t1 = prompt_a.format(A=text_a, B=text_b) if "{A}" in prompt_a else f"{prompt_a}{choice_text}"
            t2 = prompt_b.format(A=text_b, B=text_a) if "{A}" in prompt_b else f"{prompt_b}{choice_text}"
            
            # Note: For the permutation cases, we need to be careful if choice_text 
            # is one of the placeholders. 
            # In Permutation 1 (A=text_a, B=text_b), if user chooses text_a, 
            # the full string is template.format(A=text_a, B=text_b) + text_a
            
            s1 = self._score_text(t1, choice_text)
            s2 = self._score_text(t2, choice_text)
            return (s1 + s2) / 2.0

        score_a = get_score(template, template, text_a)
        score_b = get_score(template, template, text_b)
        
        winner_id = alt_a.alternative_id if score_a > score_b else alt_b.alternative_id
        loser_id = alt_b.alternative_id if winner_id == alt_a.alternative_id else alt_a.alternative_id
        
        return PairwiseObservation(
            agent_id=agent.agent_id,
            winner_id=winner_id,
            loser_id=loser_id,
        )

    def _score_text(self, base_prompt: str, choice_text: str) -> float:
        """Score a choice by the negative perplexity of the full string."""
        full_text = f"{base_prompt}{choice_text}"
        
        if getattr(self._tokenizer, "chat_template", None) is not None:
            # We assume the base_prompt + choice is the final response
            messages = [{"role": "user", "content": full_text}]
            text_to_score = self._tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text_to_score = full_text
            
        return self._compute_negative_perplexity(text_to_score)

# TODO: move to tests directory
class MockHuggingFaceProvider(PreferenceProvider):
    """A lightweight provider for dry-runs that doesn't require a real model.
    
    Simulates preference by assigning higher scores to shorter strings.
    """
    def __init__(
        self,
        model_id: str = "mock-model",
        device: str = "cpu",
        prompts_by_agent_id: dict[str, str] | None = None,
        alternative_texts: dict[int, str] | None = None
    ):
        self.model_id = model_id
        self.device = device
        self._prompts_by_agent_id = prompts_by_agent_id or {}
        self._alternative_texts = alternative_texts or {}

    def query_pairwise(
        self,
        agent: AgentRecord,
        alt_a: AlternativeRecord,
        alt_b: AlternativeRecord,
    ) -> PairwiseObservation:
        # 1. Get Prompt
        if hasattr(agent, "prompt"):
            template = getattr(agent, "prompt")
        else:
            template = self._prompts_by_agent_id.get(agent.agent_id, "Candidate {A} vs {B}")
            
        # 2. Get Descriptions
        text_a = getattr(alt_a, "description") if hasattr(alt_a, "description") else \
                 self._alternative_texts.get(alt_a.alternative_id, str(alt_a.alternative_id))
        text_b = getattr(alt_b, "description") if hasattr(alt_b, "description") else \
                 self._alternative_texts.get(alt_b.alternative_id, str(alt_b.alternative_id))
        
        def get_dummy_score(prompt, a_text, b_text):
            # Simulate "perplexity" via text length
            # If it's a choice format (no placeholders), use as is
            try:
                t = prompt.format(A=a_text, B=b_text)
            except (KeyError, IndexError):
                t = f"{prompt} {a_text} {b_text}"
            return -len(t) # Shorter is better

        score_a = get_dummy_score(template, text_a, text_b)
        score_b = get_dummy_score(template, text_b, text_a)
        
        winner_id = alt_a.alternative_id if score_a > score_b else alt_b.alternative_id
        loser_id = alt_b.alternative_id if winner_id == alt_a.alternative_id else alt_a.alternative_id
        
        return PairwiseObservation(
            agent_id=agent.agent_id,
            winner_id=winner_id,
            loser_id=loser_id,
        )


class HuggingFaceChoiceProvider(HuggingFaceProvider):
    """
    HuggingFace provider for multiple-choice preference elicitation.
    Compares the log-probabilities of two specific labels (e.g., '1' vs '2') 
    instead of full-sequence perplexity.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        labels: tuple[str, str] = ("1", "2"),
        **kwargs,
    ):
        # We don't call super().__init__ because we want different args
        # But we need model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self._model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.labels = labels
        # Pre-tokenize labels (taking the first token for simplicity)
        self.label_ids = [
            self._tokenizer.encode(label, add_special_tokens=False)[-1] 
            for label in labels
        ]
        print(f"[INFO] HuggingFaceChoiceProvider initialized with labels {labels} (ids: {self.label_ids})")

    def query_pairwise(
        self, agent: AgentRecord, alt_a: AlternativeRecord, alt_b: AlternativeRecord
    ) -> PairwiseObservation:
        """
        Query the model for choice probabilities of (1) vs (2).
        Returns a winner_id and loser_id.
        """
        import torch
        import torch.nn.functional as F

        # 1. Forward Pass (A, B)
        prompt1 = agent.prompt.format(A=alt_a.description, B=alt_b.description)
        inputs1 = self._tokenizer(prompt1, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs1 = self._model(**inputs1)
            next_token_logits1 = outputs1.logits[0, -1, :]
            target_logits1 = next_token_logits1[self.label_ids]
            log_probs1 = F.log_softmax(target_logits1, dim=0)
            # Prob(A) is log_probs1[0]
            prob_a1 = torch.exp(log_probs1[0]).item()

        # 2. Permuted Pass (B, A)
        prompt2 = agent.prompt.format(A=alt_b.description, B=alt_a.description)
        inputs2 = self._tokenizer(prompt2, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs2 = self._model(**inputs2)
            next_token_logits2 = outputs2.logits[0, -1, :]
            target_logits2 = next_token_logits2[self.label_ids]
            log_probs2 = F.log_softmax(target_logits2, dim=0)
            # Prob(A) is log_probs2[1]
            prob_a2 = torch.exp(log_probs2[1]).item()

        # 3. Average Probabilities
        avg_prob_a = 0.5 * (prob_a1 + prob_a2)
        
        winner_id = alt_a.alternative_id if avg_prob_a > 0.5 else alt_b.alternative_id
        loser_id = alt_b.alternative_id if winner_id == alt_a.alternative_id else alt_a.alternative_id
        
        return PairwiseObservation(
            agent_id=agent.agent_id,
            winner_id=winner_id,
            loser_id=loser_id,
        )
