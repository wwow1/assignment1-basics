from __future__ import annotations
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize_string(text: str, special_tokens: list[str], keep_special_tokens: bool = False) -> list[str]:
    """
    Split the input text into tokens based on the regex pattern and special tokens.
    """
    if not special_tokens:
        return re.findall(PAT, text)
    
    special_tokens_set = set(special_tokens)
    # Sort special tokens by length descending to handle overlapping tokens correctly
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    special_pattern = "(" + "|".join(re.escape(t) for t in special_tokens_sorted) + ")"
    parts = re.split(special_pattern, text)
    
    final_tokens = []
    for part in parts:
        if not part:
            continue
        if part in special_tokens_set:
            if keep_special_tokens:
                final_tokens.append(part)
            continue
        else:
            final_tokens.extend(re.findall(PAT, part))
            
    return final_tokens