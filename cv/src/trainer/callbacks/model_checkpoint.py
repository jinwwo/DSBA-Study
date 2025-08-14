import os
import torch
import json
from trainer.callbacks.base import Callback


class ModelCheckpoint(Callback):
    def __init__(self, save_path, monitor="val_acc", mode="max", filename="best_model.pt"):
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best = float("-inf") if mode == "max" else float("inf")

    def on_epoch_end(self, trainer, epoch, metrics):
        current = metrics.get(self.monitor)
        if current is None:
            return

        improved = (
            current > self.best if self.mode == "max" else current < self.best
        )

        if improved:
            trainer._logger.info(f"[Checkpoint] {self.monitor} improved from {self.best:.4f} to {current:.4f}")
            self.best = current

            # Save model and metrics
            torch.save(trainer.model.state_dict(), os.path.join(self.save_path, self.filename))

            json.dump(
                {"best_epoch": epoch, self.monitor: current},
                open(os.path.join(self.save_path, "best_results.json"), "w"),
                indent=4
            )