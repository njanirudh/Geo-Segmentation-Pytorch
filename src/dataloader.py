import os
import cv2

from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
from src.utils.tiff_utils import tiff_to_nparray

class SegDatasetLoader(Dataset):
    """
    Dataset loader for segmentation dataset.
    """

    def __init__(self, dataset_path: str, batch_size: int,
                 augmentation: T):
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

        image = tiff_to_nparray(img_path).astype(np.float32)[np.newaxis, ...]
        mask = tiff_to_nparray(mask_path).astype(np.float32)[np.newaxis, ...]

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

    seg_dataset = SegDatasetLoader(dataset_pth, 1, None)
