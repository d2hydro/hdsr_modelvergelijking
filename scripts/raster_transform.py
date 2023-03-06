import rasterio
import xarray as xr

from utilities import raster_transform

if __name__ == "__main__":
    # Example script to convert raster
    d_dataset = rasterio.open(r"D:\Work\Project\P1389\GIS\HH_raster\raster_ldd.tif")
    ahn_dataset = rasterio.open(r"D:\Work\Project\P1389\GIS\AE\AE.tif")
    output_path = r"D:\Work\Project\P1389\GIS\SM_raster\AE_25m_ldd.tif"
    _, ahn_data = raster_transform(
        source=ahn_dataset, destination=d_dataset, output_path=output_path, resampling=0
    )
    # _, ahn_data_2 = raster_transform(
    #     source=ahn_dataset, destination=d_dataset, output_path=output_path
    # )
    # ds = xr.Dataset(data_vars={"1": ahn_data, "2": ahn_data_2})
    # print(ds)
    # ds.to_netcdf(r"D:\Work\Project\P1389\GIS\HH_raster\ahn4.nc")
