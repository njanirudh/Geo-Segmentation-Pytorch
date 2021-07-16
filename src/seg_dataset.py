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
                 use_rgb: bool = True,
                 validation: bool = False,
                 augmentation: T = None):
        """
        Generates pytorch dataloader from folder.
        :param dataset_path: path to dataset folder.
        :param augmentation: image augmentations.
        """
        self.batch_size = 1
        self.use_rgb = use_rgb
        self.validation = validation
        self.num_validation_images = 500  # We use 500 images for validation

        self.img_dir = os.path.join(dataset_path, "images")
        self.mask_dir = os.path.join(dataset_path, "labels")

        folder_list = [f.stem for f in Path(self.img_dir).iterdir() if f.is_dir()]
        if not self.validation:
            self.folder_names = folder_list[:-self.num_validation_images]
        else:
            self.folder_names = folder_list[-self.num_validation_images:]

    def __getitem__(self, idx) -> [np.array, np.array]:
        """
        Returns a pair of image and mask image.
        :param idx: index of the image/mask pair.
        :return: pair of np.array images.
        """
        current_img_name = self.folder_names[idx]

        img_path = os.path.join(self.img_dir, current_img_name, "07.tif")
        mask_path = os.path.join(self.mask_dir, current_img_name, "dlt.tif")

        # Use only RGB channels
        if self.use_rgb:
            # print("[INFO] Using only channels (4,3,2)")
            image = tiff_to_nparray(img_path)[2:5]
            mask = tiff_to_nparray(mask_path).squeeze(0)
        else:
            # print("[INFO] Using all channels (1-12)")
            image = tiff_to_nparray(img_path)
            mask = tiff_to_nparray(mask_path).squeeze(0)

        # Remove labels with error
        if np.any(mask > 2):
            np.where(mask > 2, 0, mask)
            print(img_path, mask_path)

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
    print(len(seg_dataset))

    val_dataset = SegDataset(dataset_pth, validation=True)
    print(len(val_dataset))
