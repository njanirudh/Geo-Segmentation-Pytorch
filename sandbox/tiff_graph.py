import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg', warn=False, force=True)

import numpy as np
from src.utils.tiff_utils import tiff_to_nparray

if __name__ == "__main__":
    fp = "../data/07.tif"
    fp_gt = "../data/dlt.tif"

    # RGB Img -------------------------------
    tiff_np_img = tiff_to_nparray(fp)
    rgb_img = np.flip(tiff_np_img[2:5, :, :])
    rgb_img = np.swapaxes(rgb_img, 0, 2).astype(np.uint8)
    print(rgb_img.shape)
    #
    # Channel Img -------------------------------
    # tiff_np_img = tiff_to_nparray(fp)
    # rgb_img = tiff_np_img[11, :, :]
    # # rgb_img = np.swapaxes(rgb_img, 0, 2)
    # print(rgb_img.shape)

    tiff_np_img_gt = tiff_to_nparray(fp_gt)
    rgb_image_gt = np.swapaxes(tiff_np_img_gt, 0, 2)
    rgb_image_gt = np.reshape(rgb_image_gt,(64,64)).T

    plt.imshow(rgb_img)
    # plt.imshow(rgb_image, cmap='gray')
    plt.imsave("../assets/rgb.png",rgb_img,cmap='gray')
    plt.show()


