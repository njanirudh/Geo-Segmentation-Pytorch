import numpy as np

import rasterio
from rasterio.plot import show

import matplotlib
from matplotlib import pyplot
matplotlib.use('TKAgg', warn=False, force=True)

fp = "../data/07.tif"
fp_gt = "../data/dlt.tif"

src = rasterio.open(fp)
# show(src.read()[2:5], transform=src.transform, contour=True)
# show(src.read()[2:5], transform=src.transform)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#
# print(NormalizeData(src.read()[2]))
# print(NormalizeData(src.read()[2]))

x= NormalizeData(src.read()[2])
y= NormalizeData(src.read()[3])
z= NormalizeData(src.read()[4])

stack = []
stack.append(z)
stack.append(y)
stack.append(x)

print(np.array(stack).shape)
rgb_image_gt = np.swapaxes(np.array(stack), 0, 2)
print(rgb_image_gt.shape)
# show(rgb_image_gt, transform=src.transform)
pyplot.imshow(rgb_image_gt)
pyplot.show()
# B -> 93,846
# G -> 1471,5408
# R -> 324,1858