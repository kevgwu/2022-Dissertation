import pandas as pd
import geopandas as gpd
from shapely import wkt
import os
import glob
import rasterio
import rasterio.mask
from PIL import Image
import cv2

#File Paths
footprint_path=r'/home/s2225826/buildings.csv'
tile_path=r'/exports/eddie/scratch/s2225826/original_tiles2'
out_path=r'/exports/eddie/scratch/s2225826/building_tiles/b{}'

#Reading building footprint data
data=pd.read_csv(footprint_path)
data.crs = 'epsg:4326' #Converting csv coordinates to match satellite image coordinates
data['geometry'] = data['geometry'].apply(wkt.loads) #Converting geometries from strings to shapes
data=gpd.GeoDataFrame(data, crs=4326).set_geometry('geometry') #Storing in GeoDF


def bound(gdf,tile):
    ''' This function returns only the building footprints within a tile's range'''

    #Reading bounding coordinates for tile
    x_min=tile.bounds[0]
    y_min=tile.bounds[1]
    x_max=tile.bounds[2]
    y_max=tile.bounds[3]

    #Returns building footprint data within tile's range
    boundedgeometry=gdf[
                 (gdf['minx']>(x_min-0.001))&
                 (gdf['miny']>(y_min-0.001))&
                 (gdf['maxx']<(x_max+0.001))&
                 (gdf['maxy']<(y_max+0.001))]

    return boundedgeometry

#Reading original tiles in
tile_images= [f for f in os.listdir(tile_path) if f.endswith('.tif')]
os.chdir(r'/exports/eddie/scratch/s2225826/original_tiles2')

#Takes each original tile, overlays the building shapefile, and
#creates a binary footprint map for each original tile

for tile in tile_images:

    #Reading individual tile
    src=rasterio.open(tile)

    #Extracting building data within the individual tile
    new_data=bound(data,src)

    #Appending a data point to avoid the building data from being empty
    #and giving errors later.

    new_data=new_data.append(data.iloc[[0]])
    features=new_data['geometry']

    #Applying building footprint data to tile to create footprint map.
    out_image,out_transform=rasterio.mask.mask(src,features,crop=False)
    out_meta=src.meta

    #Updating metadata
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    #Transposing into HxWx3 to match image format.
    out_image=out_image.transpose((1, 2, 0))

    #Extracting only first channel.
    first_channel=out_image[:,:,0]

    #Make all non-zeros into 1 so it's a binary map.
    first_channel[first_channel!=0]=255

    #Resizing matrix from 200x200 to 224x224
    final=cv2.resize(first_channel, dsize=(224,224), interpolation=cv2.INTER_NEAREST)

    #Saving road binary mask
    im = Image.fromarray(final)
    im.save(out_path.format(tile))
