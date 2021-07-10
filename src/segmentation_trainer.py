import matplotlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Setting seed for reproducibility
seed = 666
torch.manual_seed(seed)

class SegmentationModule(pl.LightningModule):
    """
    Pytorch Lightning module for training segmentation.
    """

