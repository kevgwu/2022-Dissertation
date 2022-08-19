''' This program stitches image tiles with metadata into a big image '''

import rasterio
from rasterio.merge import merge
import os as os
import glob

def tile_reader(path):
    ''' Using a folder path, will create a list of all images to be stitched '''

    all_imgs=[] #Empty list to add all read images to

    #Identifying .tif files in folder
    text_files = [f for f in os.listdir(path) if f.endswith('.tif')]

    #Reading all .tif files to be stitched
    for file in text_files:
        img = rasterio.open(file)
        all_imgs.append(img)

    return all_imgs, img

def stitcher(all_imgs, img,out_path):
    ''' Provided a .tif images with metadata, images will be stitched into a huge image '''

    ''' Warning: Will create black space in between tiles if tiles are far apart.
    Final stitched image will be much bigger than sum of all image data'''

    #Merging all images into one big image.
    mosaic, meta= merge(all_imgs)

    #Updating metadata for mosaic
    out_meta = img.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": meta,
                     "crs": 4326})

    #Saving stitched image with metadata
    with rasterio.open(out_path.format('mosaic.tif'), "w",**out_meta,) as dest:
        dest.write(mosaic)

def main():

    #Defining file path with all tiles to be stitched.
    tiles_path=r'/exports/csce/eddie/inf/groups/PEP/VIVID_346b8e51-2a9c-4818-9006-80a84c0fde1f-APS/raster_tiles'
    #Defining output path for stitched image
    out_path=r'/exports/eddie/scratch/s2225826/{}'
    os.chdir(tiles_path)

    images, image=tile_reader(tiles_path) #Getting all file names that end with .tif
    stitcher(images, image, out_path) #Stitching tiles and saving stitched image.

if __name__ == "__main__":
    main()
