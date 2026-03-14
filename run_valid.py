import argparse
import json
import math
import os

import numpy as np
import torch

from cs336_basics.transformer.cross_entropy import cross_entropy_loss
from cs336_basics.transformer.transformer_lm import TransformerLM


def run_valid(
    config_path: str = "config.json",
) -> None:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    run_name = config.get("run_name", "default_run")
    checkpoint_path = os.path.join(config["checkpoint_dir_path"], run_name, "final_model.pt")
    valid_dataset_path = config["valid_dataset_path"]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    if not os.path.exists(valid_dataset_path):
        raise FileNotFoundError(f"Validation dataset not found at {valid_dataset_path}")

    device_str = config.get("device", "cuda")
    if device_str != "cuda":
        raise RuntimeError(f"run_valid only supports CUDA, but config.device is '{device_str}'.")
    if not torch.cuda.is_available():
        raise RuntimeError("run_valid requires CUDA, but torch.cuda.is_available() is False.")
    device = torch.device("cuda")
    probe = torch.empty((8, 8), device="cuda")
    torch.nn.init.trunc_normal_(probe, mean=0.0, std=1.0, a=-3.0, b=3.0)
    _ = probe @ probe
    torch.cuda.synchronize()
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation dataset: {valid_dataset_path}")

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
        dtype=None,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = np.lib.format.open_memmap(valid_dataset_path, mode="r")
    context_length = config["context_length"]
    batch_size = config["batch_size"]
    vocab_size = config["vocab_size"]

    dataset_min_id = int(dataset.min())
    dataset_max_id = int(dataset.max())
    if dataset_min_id < 0 or dataset_max_id >= vocab_size:
        raise ValueError(
            f"Validation token id range [{dataset_min_id}, {dataset_max_id}] is out of vocab range [0, {vocab_size - 1}]."
        )

    num_sequences = (dataset.size - 1) // context_length
    if num_sequences <= 0:
        raise ValueError("Validation dataset is too small for the configured context_length.")

    total_loss = 0.0
    total_sequences = 0

    with torch.no_grad():
        for batch_start in range(0, num_sequences, batch_size):
            current_batch_size = min(batch_size, num_sequences - batch_start)
            sequence_indices = np.arange(batch_start, batch_start + current_batch_size, dtype=np.int64)
            starts = sequence_indices * context_length

            src = np.stack([dataset[s : s + context_length] for s in starts])
            dst = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])

            src_t = torch.from_numpy(src).long().to(device)
            dst_t = torch.from_numpy(dst).long().to(device)

            logits = model(src_t)
            loss = cross_entropy_loss(logits, dst_t)

            total_loss += loss.item() * current_batch_size
            total_sequences += current_batch_size

    avg_loss = total_loss / total_sequences
    perplexity = math.exp(avg_loss)

    print(f"Validation average loss: {avg_loss:.6f}")
    print(f"Validation perplexity: {perplexity:.6f}")
    print(f"Evaluated sequences: {total_sequences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate final checkpoint on validation dataset")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    args = parser.parse_args()

    run_valid(config_path=args.config)
