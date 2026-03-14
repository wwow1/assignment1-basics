import torch
from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.transformer_block import TransformerBlock
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.softmax import softmax

class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_layer = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.last_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_layer = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    # Transfer x from (batch_size, context_length) to (batch_size, context_length, vocab_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        token_positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        
        x = self.embedding(x)
        for layer in self.transformer_layer:
            x = layer(x, token_positions=token_positions)
        x = self.last_norm(x)
        x = self.output_layer(x)
        return x