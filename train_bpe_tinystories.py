import os
import numpy as np
from cs336_basics.tokenizer import train_bpe
from cs336_basics.tokenizer import Tokenizer

def encode_and_save(tokenizer, input_path, output_path):
    print(f"Encoding {input_path}...")
    
    # Use a temporary file to store raw bytes to avoid memory issues with large lists
    temp_path = output_path + ".tmp"
    total_tokens = 0
    
    # Buffer for writing to file
    buffer = []
    BUFFER_SIZE = 1024 * 1024  # 1M tokens buffer
    
    with open(input_path, "r", encoding="utf-8") as f, open(temp_path, "wb") as out_f:
        # Use encode_iterable to process the file stream efficiently
        for _id in tokenizer.encode_iterable(f):
            buffer.append(_id)
            if len(buffer) >= BUFFER_SIZE:
                chunk_arr = np.array(buffer, dtype=np.uint16)
                out_f.write(chunk_arr.tobytes())
                total_tokens += len(buffer)
                buffer = []
        
        # Write remaining tokens
        if buffer:
            chunk_arr = np.array(buffer, dtype=np.uint16)
            out_f.write(chunk_arr.tobytes())
            total_tokens += len(buffer)
            buffer = []

    print(f"Saving to {output_path}...")
    
    # Load raw bytes as numpy array and save as .npy
    # This is much more memory efficient than converting a huge list
    arr = np.fromfile(temp_path, dtype=np.uint16)
    np.save(output_path, arr)
    
    # Clean up temp file
    os.remove(temp_path)
    
    print(f"Saved {total_tokens} tokens to {output_path}")

def main():
    train_path = "data/TinyStoriesV2-GPT4-train.txt"
    valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
    
    # 1. Train BPE on validation set
    print(f"Training BPE on {valid_path}...")
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    # train_bpe returns (vocab, merges)
    vocab, merges = train_bpe(valid_path, vocab_size, special_tokens)
    
    # Create tokenizer
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    
    # 2. Encode and save validation set
    encode_and_save(tokenizer, valid_path, "TinyStories-valid.npy")
    
    # 3. Encode and save training set
    encode_and_save(tokenizer, train_path, "TinyStories-train.npy")

if __name__ == "__main__":
    main()
