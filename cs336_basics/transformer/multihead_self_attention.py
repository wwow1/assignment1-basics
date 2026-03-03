import math
import torch
from einops import rearrange
from einops import einsum

from cs336_basics.transformer.rope import RotaryPositionalEmbedding
from cs336_basics.transformer.scaled_dot_product_attention import scaled_dot_product_attention

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj_weight = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.k_proj_weight = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.v_proj_weight = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.out_proj_weight = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
        else:
            self.rope = None

        torch.nn.init.trunc_normal_(self.q_proj_weight, mean=0, std=math.sqrt(2 / d_model), a=-3 * math.sqrt(2 / d_model), b=3 * math.sqrt(2 / d_model))
        torch.nn.init.trunc_normal_(self.k_proj_weight, mean=0, std=math.sqrt(2 / d_model), a=-3 * math.sqrt(2 / d_model), b=3 * math.sqrt(2 / d_model))
        torch.nn.init.trunc_normal_(self.v_proj_weight, mean=0, std=math.sqrt(2 / d_model), a=-3 * math.sqrt(2 / d_model), b=3 * math.sqrt(2 / d_model))
        torch.nn.init.trunc_normal_(self.out_proj_weight, mean=0, std=math.sqrt(2 / d_model), a=-3 * math.sqrt(2 / d_model), b=3 * math.sqrt(2 / d_model))

    def forward(
        self,
        in_features: torch.Tensor, # (..., sequence_length, d_in),
        token_positions: torch.Tensor | None = None, # (..., sequence_length)
    ) -> torch.Tensor:
        """
        Given the query, key, and value tensors, return the output of multihead self attention.
        """
        batch_size, seq_len, _ = in_features.shape

        origin_q = einsum(in_features, self.q_proj_weight, "... seq d_in, d_k d_in   -> ... seq d_k")
        multi_q = rearrange(origin_q, "... seq (h k) -> ... h seq k", h=self.num_heads)
        origin_k = einsum(in_features, self.k_proj_weight, "... seq d_in, d_k d_in   -> ... seq d_k")
        multi_k = rearrange(origin_k, "... seq (h k) -> ... h seq k", h=self.num_heads)
        
        if self.rope is not None and token_positions is not None:
            multi_q = self.rope(multi_q, token_positions)
            multi_k = self.rope(multi_k, token_positions)

        origin_v = einsum(in_features, self.v_proj_weight, "... seq d_in, d_v d_in   -> ... seq d_v")
        multi_v = rearrange(origin_v, "... seq (h v) -> ... h seq v", h=self.num_heads)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=in_features.device, dtype=torch.bool))
        attention = scaled_dot_product_attention(multi_q, multi_k, multi_v, mask)
        merged_attention = rearrange(attention, "... h seq v -> ... seq (h v)")
        return einsum(merged_attention, self.out_proj_weight, "... seq d_v, d_model d_v -> ... seq d_model")
