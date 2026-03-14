from torch import nn
import torch

class Embedding(torch.nn.Module):
    def __init__(
        self, 
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean = 0, std = 1, a = -3,  b = 3)

    # Transfer x from (batch_size, context_length) to (batch_size, context_length, embedding_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]