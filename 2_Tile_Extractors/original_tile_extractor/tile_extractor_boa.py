''' Original tiles were extracted using this python code using the stitched satellite image'''

import rasterio
import pandas as pd
import geopandas as gpd

#File Paths
mosaic_path=r'/exports/eddie/scratch/s2225826/mosaics/boane.tif'
coordinate_path=r'/home/s2225826/boane_data.csv'
output=r'/exports/eddie/scratch/s2225826/original_tiles/tile_b{}.tif'

#Reading Mosaic In (Stitched Image)
mosaic=rasterio.open(mosaic_path)

#Reading coordinates dataframe in
#original geometries
df = gpd.read_file(coordinate_path) #Census df

#convert geometries to match satellite image coordinate system
geometries= gpd.points_from_xy(df.lat, df.long, crs="EPSG:4326")

#Defining tile size in pixels
H = 202 #This should correspond to a 100m x 100mm tile
W = 225

#Setting counter
i=0

#Extracting tiles for each data point in census df.
for geometry in geometries:

    #Getting tile longitude and latitudes from point data type
    lon = geometry.x
    lat = geometry.y

    py, px = mosaic.index(lon, lat) #Define coordinates in pixels

    window = rasterio.windows.Window(px - H//2, py - W//2, W, H) #Define window

    tile = mosaic.read(window=window) #Extract window from array

    #Embedding meta data from original mosaic and updating height,width
    meta=mosaic.meta
    meta['width'], meta['height'] = W, H
    meta['transform'] = rasterio.windows.transform(window, mosaic.transform)

    #Writing tiles
    with rasterio.open(output.format(i), 'w', **meta) as dst:
        dst.write(tile)

    #Updating i for tile names
    i=i+1
