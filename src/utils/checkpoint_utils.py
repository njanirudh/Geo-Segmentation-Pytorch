# https://github.com/PyTorchLightning/pytorch-lightning/issues/5473#issuecomment-764002682

from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class PeriodicCheckpoint(ModelCheckpoint):
    """
    Callback to create checkpoint every 'n' steps.
    """
    def __init__(self, n: int):
        super().__init__()
        self.every = n

    def on_train_batch_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.global_step % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"latest-{pl_module.global_step}.ckpt"
            prev = (
                    Path(self.dirpath) / f"latest-{pl_module.global_step - self.every}.ckpt"
            )
            trainer.save_checkpoint(current)
            prev.unlink(missing_ok=True)
