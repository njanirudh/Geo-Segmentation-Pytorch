import os
import cv2

from glob import glob
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset

class SegDatasetLoader(Dataset):
    def __init__(self, img_dir:str, mask_dir:str,
                 augmentation:T=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.folder_names = []

    def __getitem__(self, idx):

        current_img_path = self.folder_names[idx]

        img_path =
        mask_path =

        image = cv2.imread(img_path, 1)
        mask = cv2.imread(mask_path, 0)

    def __len__(self):
        pass

if __name__ == "__main__":

    img_path = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/images/"
    mask_path = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/images"

    seg_dataset = SegDatasetLoader(img_path, mask_path)
