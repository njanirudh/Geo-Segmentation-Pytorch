import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg', warn=False, force=True)


def tiff_to_nparray(path: str) -> np.array:
    multilayer_img = rasterio.open(path)
    return multilayer_img.read()


if __name__ == "__main__":
    fp = "../../data/07.tif"
    fp_gt = "../../data/dlt.tif"

    tiff_np_img = tiff_to_nparray(fp)
    rgb_img = tiff_np_img[2:5, :, :]
    # print(tiff_to_nparray(fp).shape)
    rgb_image = tiff_to_nparray(fp)[2:5, :, :]
    rgb_image = np.swapaxes(rgb_image, 0, 2)

    tiff_np_img_gt = tiff_to_nparray(fp_gt)
    # rgb_image_gt = np.squeeze(tiff_np_img_gt, axis=2)
    rgb_image_gt = np.swapaxes(tiff_np_img_gt, 0, 2)
    rgb_image_gt = np.reshape(rgb_image_gt,(64,64)).T

    # plt.imshow(rgb_image, cmap='pink')
    plt.imshow(rgb_image_gt, cmap='gray')
    plt.show()


