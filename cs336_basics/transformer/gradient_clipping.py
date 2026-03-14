
import math
import torch
from typing import Iterable

def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_gradient_norm: float,
    epsilon: float = 1e-6,
):
    """
    Clip the gradient of the parameter to the given maximum norm.
    """
    # 1. Collect all valid gradients into a list (so we can iterate twice)
    #    If params is an iterator, list() consumes it.
    params_list = [p for p in params if p.grad is not None]
    
    if not params_list:
        return

    # 2. Compute total norm by flattening all gradients
    #    We can compute the norm of each grad, stack them, and compute the norm of that vector.
    #    This is mathematically equivalent to flattening everything into one giant vector.
    #    sqrt(sum(g_i^2)) = sqrt(sum(norm(g_tensor_j)^2))
    
    # Using list comprehension to get norms of all gradients
    grads_norms = torch.stack([p.grad.norm(p=2) for p in params_list])
    total_norm = torch.norm(grads_norms, p=2)

    # 3. Clip gradients if needed
    if total_norm > max_gradient_norm:
        clip_coef = max_gradient_norm / (total_norm + epsilon)
        for p in params_list:
            # In-place modification of the gradient
            p.grad.detach().mul_(clip_coef)
