import pytorch_lightning as pl
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from src.model.unet import UNET
from src.seg_dataset import SegDatasetLoader

# Setting seed for reproducibility
seed = 666
torch.manual_seed(seed)

Tensor = torch.tensor
Module = torch.nn.Module


class SegmentationModule(pl.LightningModule):
    """
    Pytorch Lightning module for training segmentation.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dataset_path: str, batch_size: int = 10,
                 epochs: int = 50, lr: float = 0.003,
                 gpu: int = 1) -> None:
        super(SegmentationModule, self).__init__()

        self.model = UNET(in_channels, out_channels)

        self.val_loader, self.train_loader = None, None
        self.trainer = None

        self.loss_fn = BCEWithLogitsLoss()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = lr
        self.gpu = gpu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        self.curr_device = inputs.device

        outputs, _ = self.forward(inputs)
        train_loss = self.loss_fn(inputs, outputs)

        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        self.curr_device = inputs.device

        outputs, _ = self.forward(inputs)
        val_loss = self.loss_fn(inputs, outputs)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_dataset = SegDatasetLoader(self.dataset_path)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=4)
        self.num_train_imgs = len(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        val_dataset = SegDatasetLoader(self.dataset_path)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=4)
        self.num_val_imgs = len(self.val_loader)
        return self.val_loader

    def train_model(self):
        self.trainer = pl.Trainer(gpus=self.gpu, max_epochs=self.epochs)
        self.trainer.fit(self,
                         self.train_dataloader(),
                         self.val_dataloader())
