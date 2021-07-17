import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg', warn=False, force=True)

from src.segmentation_trainer import SegmentationModule
from src.utils.tiff_utils import tiff_to_nparray

if __name__ == "__main__":
    # ---------------------------------------------------------------
    # --------------- TRAINING CODE (12 bands) ----------------------
    # ---------------------------------------------------------------
    # DATASET_PATH = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/"
    # seg_trainer = SegmentationModule(dataset_path=DATASET_PATH,
    #                                  in_channels=3,
    #                                  out_channels=3,
    #                                  batch_size=20,
    #                                  use_rgb=True,
    #                                  train_mode=True,
    #                                  epochs=500
    #                                  )
    # seg_trainer.train_model()

    # -----------------------------------------------------------------
    # ---------------- INFERENCE CODE (12 bands) -----------------------
    # -----------------------------------------------------------------
    # input_img = torch.rand((1, 12, 64, 64))
    DATASET_PATH = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/"
    np_img = tiff_to_nparray("../data/07.tif") # Test Image
    np_img = np_img[np.newaxis, ...]
    input_img = torch.from_numpy(np_img)

    MODEL_CHKP_PATH = "/home/anirudh/NJ/Interview/Vision-Impulse/Vision-Impulse-Test/src/" \
                      "lightning_logs/version_0/checkpoints/latest-9500.ckpt"
    seg_inference = SegmentationModule(dataset_path=DATASET_PATH,
                                       in_channels=12,
                                       out_channels=3,
                                       batch_size=20,
                                       use_rgb=False,
                                       train_mode=False
                                       )

    # seg_inference = SegmentationModule.load_from_checkpoint(MODEL_CHKP_PATH)
    seg_inference.load_state_dict(torch.load(MODEL_CHKP_PATH), strict=False)

    seg_inference.eval()

    with torch.no_grad():
        output_seg = seg_inference(input_img)
        output_seg = np.argmax(output_seg.numpy(), axis=1)
        output_seg = np.swapaxes(output_seg,0,2)
        output_seg = np.reshape(output_seg, (64, 64))

        print(output_seg.shape)

        plt.imshow(output_seg,  cmap='pink')
        plt.show()

