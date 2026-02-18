import cProfile
import pstats
import os
from cs336_basics.train_bpe import train_bpe

def main():
    input_path = "TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    print(f"Training BPE on {input_path} with vocab_size={vocab_size} and special_tokens={special_tokens}")
    
    # Run the profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    train_bpe(input_path, vocab_size, special_tokens)
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20) # Print top 20 functions by cumulative time

if __name__ == "__main__":
    main()
