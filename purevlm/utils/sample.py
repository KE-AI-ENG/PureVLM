"""
Sampling utilities for text generation
"""
import torch
import torch.nn.functional as F


def apply_sampling_penalties(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0
) -> torch.Tensor:
    """Apply repetition and presence penalties to logits.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        generated_ids: Previously generated token IDs
        repetition_penalty: Penalty for repeating tokens (> 1.0 discourages repetition)
        presence_penalty: Penalty for tokens that have appeared (subtracts from logits)

    Returns:
        Modified logits tensor
    """
    if generated_ids is None or generated_ids.numel() == 0 or (1.0 == repetition_penalty and 0.0 == presence_penalty):
        return logits

    batch_size = logits.size(0)
    for b in range(batch_size):
        seen_tokens = torch.unique(generated_ids[b])

        # repetition_penalty
        if repetition_penalty != 1.0:
            token_logits = logits[b, seen_tokens]
            neg_mask = token_logits < 0
            token_logits[neg_mask] *= repetition_penalty
            token_logits[~neg_mask] /= repetition_penalty
            logits[b, seen_tokens] = token_logits

        # presence_penalty
        if presence_penalty != 0.0:
            logits[b, seen_tokens] -= presence_penalty

    return logits


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0
) -> torch.Tensor:
    """Filter logits using top-k and/or top-p (nucleus) filtering.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        top_k: Keep only top k tokens with highest probability (k > 0)
        top_p: Keep the top tokens with cumulative probability >= top_p (0.0 < top_p <= 1.0)

    Returns:
        Filtered logits tensor with unlikely tokens set to -inf
    """
    batch_size, vocab_size = logits.size()

    # top_k filtering
    if top_k > 0:
        top_k = min(top_k, vocab_size)
        kth_values = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth_values, float('-inf'))

    # top_p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Restore to original indices
        for b in range(batch_size):
            indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
            logits[b, indices_to_remove] = float('-inf')

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_k: int = 0,
    top_p: float = 1.0,
    generated_ids: torch.Tensor = None,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0
) -> torch.Tensor:
    """Sample next token from logits with various strategies.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        temperature: Sampling temperature (higher = more random)
        do_sample: If True, sample from distribution; if False, use greedy decoding
        top_k: Keep only top k tokens with highest probability
        top_p: Keep the top tokens with cumulative probability >= top_p
        generated_ids: Previously generated tokens for penalty calculation
        repetition_penalty: Penalty for repeating tokens
        presence_penalty: Penalty for tokens that have appeared

    Returns:
        Next token IDs of shape (batch_size, 1)
    """
    # Apply penalties
    logits = apply_sampling_penalties(logits, generated_ids, repetition_penalty, presence_penalty)

    if do_sample:
        # Apply filtering
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        # Sample from distribution
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        # Greedy decoding
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    return next_token
