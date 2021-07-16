import numpy as np
import rasterio


def tiff_to_nparray(path: str) -> np.array:
    """
    Convert '.tiff' into a 'numpy' array
    :param path: Path to tiff file.
    :return: Numpy aarray image
    """
    multilayer_img = rasterio.open(path)
    return multilayer_img.read().astype(np.float32)
