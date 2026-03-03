import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # to different k, we need to compute different theta_base
        theta_base = 1.0 / theta ** (torch.arange(0, d_k, 2).float() / d_k)
        position = torch.arange(0, max_seq_len, device=device).float()

        # (seq_len, d_k)
        freqs = torch.outer(position, theta_base)
        # Compute cos and sin and cache them
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """
        *batch_dims, seq_len, d_k = x.shape
        
        # 1. Fetch cos/sin for the given positions
        # shape: (seq_len, d_k // 2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        
        # 2. Construct the rotation matrix R
        # R = [[cos, -sin],
        #      [sin,  cos]]
        # shape: (1, ..., 1, seq_len, d_k // 2, 2, 2)
        R = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin,  cos], dim=-1)
        ], dim=-2)
        
        # 3. Reshape input x to pairs
        # shape: (..., seq_len, d_k // 2, 2)
        x_reshaped = x.view(*batch_dims, seq_len, d_k // 2, 2)
        
        # 4. Apply rotation using einsum
        # ...: represents all batch dimensions (broadcasted automatically)
        # s: seq_len
        # h: d_k // 2 (half dimension)
        # i: 2 (output pair index)
        # j: 2 (input pair index)
        #
        # R shape: (1, ..., 1, s, h, 2, 2)  <- broadcasting happens on 1s
        # x shape: (b, ..., h, s, h, 2)
        #
        # Formula: x_out[..., s, h, i] = sum_j (R[..., s, h, i, j] * x[..., s, h, j])
        x_out = torch.einsum("... s h i j, ... s h j -> ... s h i", R, x_reshaped)
        
        # 5. Flatten back to original shape
        # (..., seq_len, d_k // 2, 2) -> (..., seq_len, d_k)
        return x_out.flatten(-2)

