"""Evaluation metrics for LLM inference quality assessment."""

import math
from difflib import SequenceMatcher
from typing import List, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    device: str = "cuda",
    max_length: int = 1024,
) -> Tuple[float, float]:
    """Compute perplexity using standard teacher-forcing.
    
    This is the reference quality metric for language modeling.
    
    Args:
        model: HuggingFace causal language model
        tokenizer: HuggingFace tokenizer
        texts: List of text strings to evaluate
        device: Device to run on
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (average NLL, perplexity)
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc.input_ids.to(device)

            # labels = input_ids shifted internally by HF
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss  # mean NLL per token

            n_tokens = input_ids.numel()
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    return avg_nll, ppl


def compute_sliding_window_nll(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    window_size: int = 256,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Approximate NLL under sliding-window KV-cache.
    
    Measures quality degradation due to context truncation.
    
    Args:
        model: HuggingFace causal language model
        tokenizer: HuggingFace tokenizer
        text: Text string to evaluate
        window_size: Size of sliding window
        device: Device to run on
        
    Returns:
        Tuple of (average NLL, perplexity)
    """
    model.eval()
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    total_nll = 0.0
    total_tokens = 0

    # Start with first token as context
    past_key_values = None
    prev_token = input_ids[:, :1]

    with torch.no_grad():
        for i in range(1, input_ids.size(1)):
            out = model(
                input_ids=prev_token,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values

            # Trim cache to window size
            trimmed = []
            for k, v in past_key_values:
                if k.size(2) > window_size:
                    k = k[:, :, -window_size:, :]
                    v = v[:, :, -window_size:, :]
                trimmed.append((k, v))
            past_key_values = tuple(trimmed)

            target = input_ids[:, i]
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(1, target.unsqueeze(1)).item()

            total_nll += nll
            total_tokens += 1
            prev_token = target.unsqueeze(1)

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return avg_nll, ppl


def text_similarity(a: str, b: str) -> float:
    """Compute text similarity using sequence matcher.
    
    Args:
        a: First text string
        b: Second text string
        
    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    return SequenceMatcher(None, a, b).ratio()


def token_agreement_rate(tok_a: List[int], tok_b: List[int]) -> float:
    """Fraction of matching tokens at the same positions.
    
    Args:
        tok_a: First token ID sequence
        tok_b: Second token ID sequence
        
    Returns:
        Agreement rate between 0.0 and 1.0
    """
    L = min(len(tok_a), len(tok_b))
    if L == 0:
        return 0.0
    return sum(1 for i in range(L) if tok_a[i] == tok_b[i]) / L
