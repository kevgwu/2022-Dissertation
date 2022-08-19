import fiona
import rasterio.mask
import numpy as np
import os as os
import glob
from PIL import Image
import cv2

#File Paths
tile_path=r'/exports/eddie/scratch/s2225826/original_tiles'
shapefile=r'/home/s2225826/road_boa.geojson'
out_path=r'/exports/eddie/scratch/s2225826/road_tiles/r{}'

#Getting shapefile with road data (file acquired from OpenStreetMap)
with fiona.open(shapefile) as geojson:
    features=[feature['geometry'] for feature in geojson]

#Getting original tiles
tile_images= [f for f in os.listdir(tile_path) if f.endswith('.tif')]
os.chdir(r'/exports/eddie/scratch/s2225826/original_tiles')

#Looping over all tiles to create a binary numpy image for each tile
for tile in tile_images:

    #Creating raster mask
    with rasterio.open(tile) as src:
        #Applying shapefile mask over original tile
        out_image,out_transform=rasterio.mask.mask(src,features,crop=True)
        out_meta=src.meta

    #Transposing into HxWx3.
    out_image=out_image.transpose((1, 2, 0))

    #Extracting only first channel.
    first_channel=out_image[:,:,0]

    #Make all non-zeros into 1 so it's a binary map.
    first_channel[first_channel!=0]=255

    #Saving road binary mask
    im = Image.fromarray(first_channel)
    im.save(out_path.format(tile))
