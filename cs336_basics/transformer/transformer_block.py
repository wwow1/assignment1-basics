import torch
from cs336_basics.transformer.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.transformer.positionwise_feedforward import SwiGLU
from cs336_basics.transformer.rmsnorm import RMSNorm

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.attention = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.feedforward = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        y = x + self.attention(self.norm1(x), token_positions=token_positions)
        out = y + self.feedforward(self.norm2(y))
        return out
