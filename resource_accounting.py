import argparse

def format_num(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)

def calculate_resources(vocab_size, d_model, num_layers, d_ff, context_length, batch_size=1, dtype_bytes=4):
    print(f"\nConfiguration: d_model={d_model}, layers={num_layers}, d_ff={d_ff}, vocab={vocab_size}, ctx={context_length}")
    
    # --- 1. Parameters ---
    # Embedding
    p_emb = vocab_size * d_model
    
    # Block
    # Attention: 4 projections (q, k, v, o) of size (d, d) -> 4 * d^2
    p_attn = 4 * d_model * d_model
    # FFN (SwiGLU): 3 projections (w1, w2, w3). w1, w3: (d, d_ff), w2: (d_ff, d) -> 3 * d * d_ff
    p_ffn = 3 * d_model * d_ff
    # Norms: 2 norms per layer, each has weight of size d -> 2 * d
    p_norm = 2 * d_model
    
    p_block = p_attn + p_ffn + p_norm
    
    # Output
    p_final_norm = d_model
    p_head = vocab_size * d_model
    
    # Total (Untied embeddings)
    p_total = p_emb + (num_layers * p_block) + p_final_norm + p_head
    
    # Total (Tied embeddings - if applicable, usually subtracts one p_emb)
    p_total_tied = p_total - p_head 
    
    print("\n[Parameters]")
    print(f"Embedding: {format_num(p_emb)} ({p_emb})")
    print(f"Per Block: {format_num(p_block)} ({p_block})")
    print(f"  - Attention: {format_num(p_attn)}")
    print(f"  - FFN: {format_num(p_ffn)}")
    print(f"Total (Untied): {format_num(p_total)} ({p_total})")
    
    # --- 2. FLOPs (Forward Pass per token) ---
    # Approx 2 * N
    flops_fwd_approx = 2 * p_total
    
    # Precise
    # Attn: Projections (4d^2 * 2) + Scores (2 * d * T) + Values (2 * d * T) -> 8d^2 + 4dT
    # FFN: 3 * 2 * d * d_ff -> 6 d d_ff
    # Logits: 2 * V * d
    
    flops_attn = 8 * d_model**2 + 4 * d_model * context_length
    flops_ffn = 6 * d_model * d_ff
    flops_block = flops_attn + flops_ffn
    
    flops_fwd_precise = (num_layers * flops_block) + (2 * vocab_size * d_model)
    
    print("\n[FLOPs per Token (Forward)]")
    print(f"Approx (2N): {format_num(flops_fwd_approx)}")
    print(f"Precise (Ctx={context_length}): {format_num(flops_fwd_precise)}")
    
    # --- 3. Memory (KV Cache) ---
    # KV Cache per layer: 2 (K,V) * batch * T * d
    kv_cache_elements = 2 * num_layers * batch_size * context_length * d_model
    kv_cache_bytes = kv_cache_elements * dtype_bytes
    
    print("\n[Memory - KV Cache (Batch=1, Full Context)]")
    print(f"Elements: {format_num(kv_cache_elements)}")
    print(f"Size (FP32): {kv_cache_bytes / (1024**2):.2f} MB")

    return p_total

if __name__ == "__main__":
    # GPT-2 Small
    print("--- GPT-2 Small ---")
    calculate_resources(vocab_size=50257, d_model=768, num_layers=12, d_ff=3072, context_length=1024)
    
    # GPT-2 XL
    print("\n--- GPT-2 XL ---")
    calculate_resources(vocab_size=50257, d_model=1600, num_layers=48, d_ff=6400, context_length=1024)
