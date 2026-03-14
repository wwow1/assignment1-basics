import time
import csv
import os
from typing import Any, Dict

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting will be disabled.")

class TrainingMonitor:
    def __init__(self, log_dir: str, hyperparameters: Dict[str, Any], report_interval: int = 10):
        """
        Initialize the monitor.
        
        Args:
            log_dir: Directory to save logs and plots.
            hyperparameters: Dictionary of hyperparameters to save.
            report_interval: How often (in iterations) to print progress to console.
        """
        self.log_dir = log_dir
        self.report_interval = report_interval
        self.hyperparameters = hyperparameters
        self.history = []
        self.start_time = None
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Save hyperparameters
        self._save_hyperparameters()

    def _save_hyperparameters(self):
        file_path = os.path.join(self.log_dir, "hyperparameters.txt")
        with open(file_path, "w") as f:
            for key, value in self.hyperparameters.items():
                f.write(f"{key}: {value}\n")

    def start(self):
        """Call this when training starts."""
        self.start_time = time.time()
        print(f"Training started. Logs will be saved to {self.log_dir}")

    def step(self, iteration: int, loss: float):
        """
        Record a training step.
        
        Args:
            iteration: Current iteration number.
            loss: Current loss value.
        """
        if self.start_time is None:
            self.start()
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if (iteration + 1) % self.report_interval == 0:
            # Record data
            self.history.append({
                "iteration": iteration,
                "loss": loss,
                "time": elapsed_time
            })
            
            # Report progress
            print(f"[Monitor] Iter: {iteration+1} | Time: {elapsed_time:.2f}s | Loss: {loss:.6f}")

    def finish(self):
        """Call this when training is finished to save CSV and plot."""
        if not self.history:
            print("No training data to save.")
            return

        # Save to CSV
        csv_path = os.path.join(self.log_dir, "metrics.csv")
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["iteration", "loss", "time"])
                writer.writeheader()
                writer.writerows(self.history)
            print(f"Metrics saved to {csv_path}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")

        # Plot
        self._plot_loss()

    def _plot_loss(self):
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping plot: matplotlib not installed")
            return

        try:
            iterations = [x["iteration"] for x in self.history]
            losses = [x["loss"] for x in self.history]
            
            plt.figure(figsize=(12, 6))
            plt.plot(iterations, losses, label="Training Loss", alpha=0.7)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plot_path = os.path.join(self.log_dir, "loss_curve.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Loss plot saved to {plot_path}")
        except Exception as e:
            print(f"Failed to plot loss: {e}")
