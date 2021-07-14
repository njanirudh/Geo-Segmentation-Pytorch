import os
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset

from src.utils.tiff_utils import tiff_to_nparray


class SegDataset(Dataset):
    """
    Dataset loader for segmentation dataset.
    """

    def __init__(self, dataset_path: str,
                 augmentation: T = None):
        """
        Generates pytorch dataloader from folder.
        :param dataset_path: path to dataset folder.
        :param augmentation: image augmentations.
        """
        self.batch_size = 1
        self.img_dir = os.path.join(dataset_path, "images")
        self.mask_dir = os.path.join(dataset_path, "labels")

        self.folder_names = [f.stem for f in Path(self.img_dir).iterdir() if f.is_dir()]

    def __getitem__(self, idx) -> [np.array, np.array]:
        """
        Returns a pair of image and mask image.
        :param idx: index of the image/mask pair.
        :return: pair of np.array images.
        """
        current_img_name = self.folder_names[idx]

        img_path = os.path.join(self.img_dir, current_img_name, "07.tif")
        mask_path = os.path.join(self.mask_dir, current_img_name, "dlt.tif")

        image = tiff_to_nparray(img_path).astype(np.float32)
        mask = tiff_to_nparray(mask_path).astype(np.float32).squeeze(0)

        # print(img_path, image.shape)
        # print(mask_path, mask.shape)

        return image, mask

    def __len__(self) -> int:
        """
        Total length of the dataset.
        :return: integer length of the dataset
        """
        return len(self.folder_names)


if __name__ == "__main__":
    dataset_pth = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/"

    seg_dataset = SegDataset(dataset_pth)
    img, mask = seg_dataset[8]
    print(np.unique(mask))