import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

logger = TensorBoardLogger("tb_logs", name="my_model")

from src.model.unet import UNET
from src.seg_dataset import SegDataset
from utils.checkpoint_utils import PeriodicCheckpoint

# Setting seed for reproducibility
seed = 666
torch.manual_seed(seed)

Tensor = torch.tensor
Module = torch.nn.Module


class SegmentationModule(pl.LightningModule):
    """
    Pytorch Lightning module for training the
    UNet segmentation model.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dataset_path: str, use_rgb: bool = False,
                 train_mode: bool = True, batch_size: int = 10,
                 epochs: int = 50, lr: float = 0.003, gpu: int = 1) -> None:
        """
        :param in_channels: Total channels C in the input image.
        :param out_channels: Output channels N (Equal to the total number of labels)
        :param dataset_path: Path to dataset path.
        :param use_rgb: Train using RGB channels ( bands 4,3,2) only.
        :param train_mode: Sets model to training or inference mode.
        :param batch_size: Batch size during training.
        :param epochs: Total epochs for training. (default 50)
        :param lr: Learning rate (default 0.003)
        :param gpu: Set total gpus to use (default 1)
        """
        super(SegmentationModule, self).__init__()

        self.model = UNET(in_channels, out_channels)
        self.model.train(train_mode)  # Set training mode = true

        self.val_loader, self.train_loader = None, None
        self.num_train_imgs, self.num_val_imgs = None, None
        self.trainer, self.curr_device = None, None

        # Model checkpoint saving every 1000 steps
        self.periodic_chkp = PeriodicCheckpoint(500)

        self.loss_fn = CrossEntropyLoss()
        self.dataset_path = dataset_path
        self.use_rgb = use_rgb
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = lr
        self.gpu = gpu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        self.curr_device = inputs.device

        outputs = self.forward(inputs)
        train_loss = self.loss_fn(outputs, labels.long())

        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        self.curr_device = inputs.device

        outputs = self.forward(inputs)
        val_loss = self.loss_fn(outputs, labels.long())

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_dataset = SegDataset(self.dataset_path,
                                   validation=False,
                                   use_rgb=self.use_rgb)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=4)
        self.num_train_imgs = len(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        val_dataset = SegDataset(self.dataset_path,
                                 validation=True,
                                 use_rgb=self.use_rgb)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=4)
        self.num_val_imgs = len(self.val_loader)
        return self.val_loader

    def train_model(self):
        self.trainer = pl.Trainer(gpus=self.gpu, max_epochs=self.epochs,
                                  callbacks=self.periodic_chkp)
        self.trainer.fit(self,
                         self.train_dataloader(),
                         self.val_dataloader())


if __name__ == "__main__":
    DATASET_PATH = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/"

    model_trainer = SegmentationModule(in_channels=12,
                                       out_channels=3,
                                       batch_size=20,
                                       dataset_path=DATASET_PATH)
    model_trainer.train_model()
