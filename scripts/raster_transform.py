import rasterio

from utilities import raster_transform

if __name__ == "__main__":
    # Example script to convert raster
    shape_dataset = rasterio.open(r"")
    ahn_dataset = rasterio.open(r"")
    output_path = r""
    raster_transform(source=ahn_dataset, destination=shape_dataset, output_path=output_path)
