import os
import torch
import numpy as np
from cs336_basics.transformer.transformer_lm import TransformerLM
from cs336_basics.transformer.adamw import AdamW
from cs336_basics.transformer.data_loader import data_loading
from cs336_basics.transformer.cross_entropy import cross_entropy_loss
from cs336_basics.transformer.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.transformer.monitor import TrainingMonitor

class Trainer:
    def __init__(self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        dataset_path: str = "data/TinyStories-train.npy",
        checkpoint_dir_path: str = "checkpoints/",
        batch_per_epoch: int = 32,
        checkpoint_times: int = 10,
        load_from_checkpoint: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        self.device = device
        self.dataset_path = dataset_path
        self.checkpoint_dir_path = checkpoint_dir_path
        self.batch_per_epoch = batch_per_epoch
        self.checkpoint_times = checkpoint_times
        self.context_length = context_length
        self.dtype = dtype
        
        self.hyperparameters = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "batch_per_epoch": batch_per_epoch,
        }

        if load_from_checkpoint:
            load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                filename=load_from_checkpoint,
            )

    def train(self, num_iterations: int, monitor_interval: int = 10) -> None:
        if not os.path.exists(self.checkpoint_dir_path):
            os.makedirs(self.checkpoint_dir_path)

        monitor = TrainingMonitor(
            log_dir=os.path.join(self.checkpoint_dir_path, "logs"),
            hyperparameters=self.hyperparameters,
            report_interval=monitor_interval
        )
        monitor.start()

        dataset = np.lib.format.open_memmap(self.dataset_path, mode="r")
        for i in range(num_iterations):
            data_batch, target_batch = data_loading(dataset, batch_size=self.batch_per_epoch, context_length=self.context_length, device=self.device)
            self.optimizer.zero_grad()
            logits = self.model(data_batch)
            loss = cross_entropy_loss(logits, target_batch)
            loss.backward()
            self.optimizer.step()
            
            monitor.step(i, loss.item())

            if (i + 1) % self.checkpoint_times == 0:
                cur_checkpoint_file = os.path.join(self.checkpoint_dir_path, f"checkpoint_{i}.pt")
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    iteration=i,
                    filename=cur_checkpoint_file,
                )
        
        monitor.finish()

        # Save the final model state
        final_checkpoint_file = os.path.join(self.checkpoint_dir_path, "final_model.pt")
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=num_iterations,
            filename=final_checkpoint_file,
        )
        print(f"Final model saved to {final_checkpoint_file}")
