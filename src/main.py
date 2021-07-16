import numpy as np
import torch

from src.segmentation_trainer import SegmentationModule

if __name__ == "__main__":
    # # -------------------------------------------------------
    # # --------------- TRAINING CODE -------------------------
    # # -------------------------------------------------------
    DATASET_PATH = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/"
    # seg_trainer = SegmentationModule(dataset_path=DATASET_PATH,
    #                                  in_channels=12,
    #                                  out_channels=3,
    #                                  batch_size=20,
    #                                  use_rgb=False,
    #                                  train_mode=True,
    #                                  epochs=500
    #                                  )
    # seg_trainer.train_model()

    # -------------------------------------------------------
    # ---------------- INFERENCE CODE -----------------------
    # -------------------------------------------------------
    input_img = torch.rand((10, 12, 64, 64))
    MODEL_CHKP_PATH = "/home/anirudh/NJ/Interview/Vision-Impulse/Vision-Impulse-Test/src/lightning_logs/version_0/checkpoints/epoch=36-step=9619.ckpt"
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
        print(output_seg.shape)