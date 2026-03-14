import argparse
import json
import os
import numpy as np
from cs336_basics.tokenizer import train_bpe
from cs336_basics.tokenizer import Tokenizer


def encode_and_save(tokenizer, input_path, output_path):
    print(f"Encoding {input_path}...")
    temp_path = output_path + ".tmp"
    total_tokens = 0
    buffer = []
    buffer_size = 1024 * 1024

    with open(input_path, "r", encoding="utf-8") as f, open(temp_path, "wb") as out_f:
        for _id in tokenizer.encode_iterable(f):
            buffer.append(_id)
            if len(buffer) >= buffer_size:
                chunk_arr = np.array(buffer, dtype=np.uint16)
                out_f.write(chunk_arr.tobytes())
                total_tokens += len(buffer)
                buffer = []

        if buffer:
            chunk_arr = np.array(buffer, dtype=np.uint16)
            out_f.write(chunk_arr.tobytes())
            total_tokens += len(buffer)
            buffer = []

    print(f"Saving to {output_path}...")
    arr = np.fromfile(temp_path, dtype=np.uint16)
    np.save(output_path, arr)
    os.remove(temp_path)
    print(f"Saved {total_tokens} tokens to {output_path}")


def save_mapping(vocab, merges, mapping_prefix):
    vocab_path = f"{mapping_prefix}_vocab.json"
    merges_path = f"{mapping_prefix}_merges.json"
    vocab_payload = {str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()}
    merges_payload = [[left.hex(), right.hex()] for left, right in merges]
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_payload, f, ensure_ascii=False)
    print(f"Saved vocab mapping to {vocab_path}")
    print(f"Saved merges mapping to {merges_path}")


def load_mapping(mapping_prefix):
    vocab_path = f"{mapping_prefix}_vocab.json"
    merges_path = f"{mapping_prefix}_merges.json"
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Mapping vocab file not found: {vocab_path}")
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"Mapping merges file not found: {merges_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_payload = json.load(f)
    with open(merges_path, "r", encoding="utf-8") as f:
        merges_payload = json.load(f)
    vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in vocab_payload.items()}
    merges = [(bytes.fromhex(left_hex), bytes.fromhex(right_hex)) for left_hex, right_hex in merges_payload]
    return vocab, merges


def main():
    parser = argparse.ArgumentParser(description="Train BPE for TinyStories and encode datasets")
    parser.add_argument("--mode", type=str, default="train_and_encode_all", choices=["train_and_encode_all", "encode_valid_only"])
    parser.add_argument("--train-path", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid-path", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--train-output", type=str, default="TinyStories-train.npy")
    parser.add_argument("--valid-output", type=str, default="TinyStories-valid.npy")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--mapping-prefix", type=str, default="TinyStories-10000")
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]

    if args.mode == "train_and_encode_all":
        print(f"Training BPE on {args.valid_path} with vocab_size={args.vocab_size}...")
        vocab, merges = train_bpe(args.valid_path, args.vocab_size, special_tokens)
        save_mapping(vocab, merges, args.mapping_prefix)
        tokenizer = Tokenizer(vocab, merges, special_tokens)
        encode_and_save(tokenizer, args.valid_path, args.valid_output)
        encode_and_save(tokenizer, args.train_path, args.train_output)
        return

    vocab, merges = load_mapping(args.mapping_prefix)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    encode_and_save(tokenizer, args.valid_path, args.valid_output)
    print("Finished valid re-encoding with saved mapping.")

if __name__ == "__main__":
    main()
