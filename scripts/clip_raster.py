import geopandas as gpd
import rasterio
import rasterio.mask

from utilities import write_tiff

gdf = gpd.read_file(r"D:\Work\Project\P1389\GIS\AE\AW_AE.shp")

with rasterio.open(r"D:\Work\Project\P1389\hydromedah\landgebruik_wss_2019_25m.tif") as src:
    out_image, out_transform = rasterio.mask.mask(src, gdf.geometry, crop=True, filled=True)

write_tiff(
    output_file_path=r"D:\Work\Project\P1389\GIS\HH_raster\raster_25m.tif",
    new_grid_data=out_image[0, :, :],
    transform=out_transform,
    epsg=28992,
)
