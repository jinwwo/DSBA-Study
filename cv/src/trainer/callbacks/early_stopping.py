from trainer.callbacks.base import Callback


class EarlyStopping(Callback):
    def __init__(self, monitor="val_loss", mode="min", patience=5):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, trainer, epoch, metrics):
        current = metrics.get(self.monitor)
        if current is None:
            return

        improved = (
            current < self.best if self.mode == "min" else current > self.best
        )

        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            trainer._logger.info(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")

        if self.counter >= self.patience:
            trainer._logger.info(f"[EarlyStopping] Stopping early at epoch {epoch+1}")
            self.should_stop = True