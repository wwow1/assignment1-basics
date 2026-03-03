import torch
from cs336_basics.transformer.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.transformer.positionwise_feedforward import SwiGLU
from cs336_basics.transformer.rmsnorm import RMSNorm

class Transformer_block(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.attention = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.feedforward = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, token_positions=None):
        y = x + self.attention(self.norm1(x), token_positions=token_positions)
        out = y + self.feedforward(self.norm2(y))
        return out
