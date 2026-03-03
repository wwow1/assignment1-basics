import torch
from cs336_basics.transformer.softmax import softmax
from torch import einsum

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    """
    Given the key, query, and value tensors, return the output of scaled dot product attention.

    Args:
        q: The query tensor.
        k: The key tensor.
        v: The value tensor.

    Returns:
        The output of scaled dot product attention.s
    """
    d_k = k.shape[-1]
    score = einsum("... q d, ... k d -> ... q k", q, k) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
    attention = softmax(score, dim=-1)
    return einsum("... q k, ... k d -> ... q d", attention, v)

