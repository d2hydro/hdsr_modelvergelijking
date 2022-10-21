import rasterio
import xarray as xr

from utilities import raster_transform

if __name__ == "__main__":
    # Example script to convert raster
    shape_dataset = rasterio.open(r"")
    ahn_dataset = rasterio.open(r"")
    output_path = r""
    ahn_data = raster_transform(source=ahn_dataset, destination=shape_dataset, output_path=output_path)
    ds = xr.Dataset(data_vars={"DEM": ahn_data})