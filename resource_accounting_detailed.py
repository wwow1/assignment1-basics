import argparse

def format_flops(flops):
    return f"{flops/1e9:.2f} GFLOPs"

def calculate_detailed_flops(model_name, num_layers, d_model, num_heads, d_ff, vocab_size=50257, context_length=1024):
    print(f"\n--- {model_name} ---")
    print(f"Config: L={num_layers}, d={d_model}, d_ff={d_ff}, heads={num_heads}, V={vocab_size}, T={context_length}")

    # 1. Attention Projections (W_Q, W_K, W_V, W_O)
    # Per layer: 4 * 2 * T * d^2
    attn_proj_flops = num_layers * 8 * context_length * (d_model**2)
    
    # 2. Attention Scores & Aggregation (QK^T, AV)
    # Per layer: 2 * (2 * T^2 * d) = 4 * T^2 * d
    attn_score_flops = num_layers * 4 * (context_length**2) * d_model
    
    # 3. FFN (SwiGLU: 3 matrices)
    # Per layer: 3 * (2 * T * d * d_ff) = 6 * T * d * d_ff
    ffn_flops = num_layers * 6 * context_length * d_model * d_ff
    
    # 4. Logits (Output Head)
    # 2 * T * d * V
    logits_flops = 2 * context_length * d_model * vocab_size
    
    total_flops = attn_proj_flops + attn_score_flops + ffn_flops + logits_flops
    
    print(f"Total FLOPs: {format_flops(total_flops)}")
    print("Breakdown:")
    print(f"  - Attn Proj: {attn_proj_flops/total_flops*100:.2f}% ({format_flops(attn_proj_flops)})")
    print(f"  - Attn Score: {attn_score_flops/total_flops*100:.2f}% ({format_flops(attn_score_flops)})")
    print(f"  - FFN: {ffn_flops/total_flops*100:.2f}% ({format_flops(ffn_flops)})")
    print(f"  - Logits: {logits_flops/total_flops*100:.2f}% ({format_flops(logits_flops)})")
    
    return {
        "total": total_flops,
        "attn_proj": attn_proj_flops,
        "attn_score": attn_score_flops,
        "ffn": ffn_flops,
        "logits": logits_flops
    }

if __name__ == "__main__":
    # (d) Models
    calculate_detailed_flops("GPT-2 Small", 12, 768, 12, 3072)
    calculate_detailed_flops("GPT-2 Medium", 24, 1024, 16, 4096)
    calculate_detailed_flops("GPT-2 Large", 36, 1280, 20, 5120)
    calculate_detailed_flops("GPT-2 XL", 48, 1600, 25, 6400)
    
    # (e) GPT-2 XL with Context Length 16384
    print("\n--- (e) GPT-2 XL with Context Length 16384 ---")
    calculate_detailed_flops("GPT-2 XL (16k)", 48, 1600, 25, 6400, context_length=16384)
