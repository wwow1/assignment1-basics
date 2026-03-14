import json
import torch
import os
from cs336_basics.transformer.training import Trainer

def run_train(config_path: str = "config.json"):
    # 1. Load Configuration
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"Loaded configuration from {config_path}")
    print(json.dumps(config, indent=4))

    # 2. Setup Device
    device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device_str = "cpu"
        else:
            try:
                # Try a simple operation to check if CUDA kernels are available
                torch.zeros(1).to("cuda")
            except RuntimeError as e:
                print(f"Warning: CUDA is available but failed to run a simple operation. Error: {e}")
                print("Falling back to CPU.")
                device_str = "cpu"
    
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # 3. Initialize Trainer
    # Map config keys to Trainer arguments
    # Note: Trainer expects 'batch_per_epoch' for batch size, and 'checkpoint_times' for interval
    
    # Construct run-specific checkpoint directory
    run_name = config.get("run_name", "default_run")
    base_checkpoint_dir = config["checkpoint_dir_path"]
    run_checkpoint_dir = os.path.join(base_checkpoint_dir, run_name)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(run_checkpoint_dir):
        os.makedirs(run_checkpoint_dir)
        print(f"Created checkpoint directory: {run_checkpoint_dir}")
    
    print(f"Checkpoints and logs will be saved to: {run_checkpoint_dir}")
    
    trainer = Trainer(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=tuple(config["betas"]),
        eps=config["eps"],
        dataset_path=config["dataset_path"],
        checkpoint_dir_path=run_checkpoint_dir,
        batch_per_epoch=config["batch_size"],  # Mapping batch_size -> batch_per_epoch
        checkpoint_times=config["checkpoint_interval"], # Mapping checkpoint_interval -> checkpoint_times
        device=device
    )

    # 4. Start Training
    print("Starting training...")
    try:
        trainer.train(
            num_iterations=config["num_iterations"],
            monitor_interval=config.get("monitor_interval", config["monitor_interval"])
        )
        print("Training completed successfully.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Transformer Training")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    args = parser.parse_args()
    
    run_train(args.config)