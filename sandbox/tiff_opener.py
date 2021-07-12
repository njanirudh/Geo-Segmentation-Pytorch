import rasterio

fp = r"../data/07.tif"
img = rasterio.open(fp)

print(img.count) # Total image channels
print(img.height, img.width) # Image width & height

print(img.indexes) # Channel index
print(img.crs) # Coordinate reference system
