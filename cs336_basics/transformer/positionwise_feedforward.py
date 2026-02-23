import torch
import math

class SwiGLU(torch.nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight1 = torch.nn.Parameter(torch.zeros(d_ff, d_model, device=device, dtype=dtype))
        self.weight2 = torch.nn.Parameter(torch.zeros(d_model, d_ff, device=device, dtype=dtype))
        self.weight3 = torch.nn.Parameter(torch.zeros(d_ff, d_model, device=device, dtype=dtype))
        std = math.sqrt(2 / (d_model + d_ff))
        torch.nn.init.trunc_normal_(self.weight1, mean=0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.weight2, mean=0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.weight3, mean=0, std=std, a=-3 * std, b=3 * std)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.silu(x @ self.weight1.T) * (x @ self.weight3.T)) @ self.weight2.T