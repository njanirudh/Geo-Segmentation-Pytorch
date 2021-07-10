import rasterio
import numpy as np


def tiff_to_nparray(path:str) -> np.array:
    multilayer_img = rasterio.open(path)
    return multilayer_img.read()


if __name__ == "__main__":

    fp = "../../data/07.tif"
    print(tiff_to_nparray(fp).shape)