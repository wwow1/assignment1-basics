import torch
import cs336_basics.transformer.transformer_lm as transformer_lm
import cs336_basics.tokenizer as tokenizer

class Decoder:
    def __init__(self, transformer_lm: transformer_lm.TransformerLM, tokenizer: tokenizer.Tokenizer):
        self.module = transformer_lm
        self.tokenizer = tokenizer
    
    def top_p_sampling_simplified(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        # logits shape: [vocab_size] (Assuming batch_size=1)
        
        # 1. Sort: Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        
        # 2. Cumulative probability
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 3. Cutoff
        # Find the first index where cumulative probability > p
        cutoff_index = (cumulative_probs > p).float().argmax()
        k = cutoff_index.item() + 1
        
        # 4. Take top k
        top_k_logits = sorted_logits[:k]
        top_k_indices = sorted_indices[:k]
        
        # 5. Sample
        probs = torch.softmax(top_k_logits, dim=-1)
        sample_idx = torch.multinomial(probs, num_samples=1)
        
        # 6. Map back to original ID
        original_token_id = top_k_indices[sample_idx]
        
        return original_token_id

    def forward(self, prompt: str, temp: float = 1.0,
                p: float = 0.9, max_reply_token: int = 100, end_token: str = "<|endoftext|>",
                device: torch.device | None = None) -> str:
        encoded_prompt = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor(encoded_prompt, dtype=torch.long, device=device)
        prompt_tensor = prompt_tensor.view(1, -1)
        i = 0
        reply_str = ""
        while i < max_reply_token:
            logits = self.module.forward(prompt_tensor)
            next_token_logits = logits[0, -1, :]
            reply_token = self.top_p_sampling_simplified(next_token_logits / temp, p)
            decoded_reply_token = self.tokenizer.decode(reply_token.tolist())
            reply_str += decoded_reply_token
            if decoded_reply_token == end_token:
                break
            prompt_tensor = torch.cat([prompt_tensor, reply_token.view(1, 1)], dim=-1)
            i += 1

        return reply_str
