import time
import pytorch_lightning as pl


class EpochTimer(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.start_time
        print(f"\nEpoch {trainer.current_epoch} time: {epoch_time:.2f}s")

        # Optional logging to Lightning logger
        pl_module.log("epoch_time_sec", epoch_time)