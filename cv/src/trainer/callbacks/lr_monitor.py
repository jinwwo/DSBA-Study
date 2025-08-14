from trainer.callbacks.base import Callback


class LrMonitor(Callback):
    def on_epoch_end(self, trainer, epoch, metrics):
        lr = trainer.optimizer.param_groups[0]["lr"]
        trainer._logger.info(f"[LrMonitor] Current LR: {lr:.6f}")
        if trainer.wandb:
            trainer.wandb.log({"lr": lr}, step=epoch)