import cv2
import numpy as np

import rasterio
import matplotlib.pyplot as plt

fp = r"../data/07.tif"
img = rasterio.open(fp)

print(img.count) # Total image channels
print(img.height, img.width) # Image width & height

print(img.indexes) # Channel index
print(img.crs) # Coordinate reference system
