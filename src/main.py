import numpy as np
import torch

from src.segmentation_trainer import SegmentationModule


def get_segmentation(input_img: np.array,
                     model_path: str,
                     in_channels=12,
                     out_channels=3,
                     batch_size=20
                     ):
    ae_inference = SegmentationModule(in_channels=in_channels,
                                      out_channels=out_channels,
                                      batch_size=batch_size,
                                      dataset_path="")

    # Skips during testing
    if model_path:
        ae_inference = SegmentationModule.load_from_checkpoint(model_path)

    ae_inference.eval()

    with torch.no_grad():
        output_seg = ae_inference(input_img)

    return output_seg


if __name__ == "__main__":
    # -------------------------------------------------------
    # --------------- TRAINING CODE -------------------------
    # -------------------------------------------------------
    DATASET_PATH = "/home/anirudh/NJ/Interview/Vision-Impulse/Dataset/"
    seg_trainer = SegmentationModule(dataset_path=DATASET_PATH,
                                     in_channels=12,
                                     out_channels=3,
                                     batch_size=20,
                                     use_rgb=False,
                                     train_mode=True,
                                     epochs=500
                                     )
    seg_trainer.train_model()

    # # -------------------------------------------------------
    # # ---------------- INFERENCE CODE -----------------------
    # # -------------------------------------------------------
    # input_img = None
    # MODEL_CHKP_PATH = ""
    # seg_inference = SegmentationModule(dataset_path=DATASET_PATH,
    #                                    in_channels=12,
    #                                    out_channels=3,
    #                                    batch_size=20,
    #                                    use_rgb=False,
    #                                    train_mode=False
    #                                    )
    #
    # if MODEL_CHKP_PATH:
    #     seg_inference = SegmentationModule.load_from_checkpoint(MODEL_CHKP_PATH)
    #
    # seg_inference.eval()
    #
    # with torch.no_grad():
    #     output_seg = seg_inference(input_img)
