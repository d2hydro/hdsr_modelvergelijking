import geopandas as gpd
import rasterio
import rasterio.mask

from utilities import write_tiff

gdf = gpd.read_file(r"D:\Work\Project\P1389\GIS\HH_export\Laterals_HH_AW_Buffered.shp")

with rasterio.open(r"D:\Work\Project\P1389\GIS\SM_raster\AE_25m_ldd.tif") as src:
    # out_image, out_transform = rasterio.mask.mask(src, gdf.geometry, crop=True, filled=True)
    out_image, out_transform = rasterio.mask.mask(
        src, gdf.geometry, all_touched=True, crop=False, filled=True
    )

write_tiff(
    output_file_path=r"D:\Work\Project\P1389\GIS\SM_raster\gauges_ldd.tif",
    new_grid_data=out_image[0, :, :],
    transform=out_transform,
    epsg=28992,
)
