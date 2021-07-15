import numpy as np
import rasterio


def tiff_to_nparray(path: str) -> np.array:
    multilayer_img = rasterio.open(path)
    return multilayer_img.read()
