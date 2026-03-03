import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        x: The input tensor to softmax.
        dim: The dimension to softmax.

    Returns:
        The output of softmaxing the given `dim` of the input.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_safe = x - x_max
    exp_x = x_safe.exp()
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x
