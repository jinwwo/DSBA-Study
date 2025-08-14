class Callback:
    def on_train_start(self, trainer): pass
    def on_epoch_end(self, trainer, epoch, metrics): pass
    def on_validation_end(self, trainer, metrics): pass