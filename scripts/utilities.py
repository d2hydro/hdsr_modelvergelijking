from typing import List

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.transform import AffineTransformer
from rasterio.windows import from_bounds
from tqdm import tqdm


def write_tiff(
    output_file_path: str, new_grid_data: np.ndarray, transform: AffineTransformer, epsg: int
) -> None:
    """
    Saves new_grid_data to a tiff file. new_grid_data should be a raster.
    Based on HydroLogic Inundation Toolbox, part of HYDROLIB.

    Args:
        output_file_path (str): location of the output file
        new_grid_data (np.ndarray): data at grid points
        transform (AffineTransformer): transform of the raster
        epsg (int): coordinate reference system (CPS) that is stored in the tiff-file

    Returns:
        None
    """

    # obtain target crs, if not given use EPSG:28992 (Dutch RDS)
    try:
        raster_crs = CRS.from_epsg(epsg)
    except CRSError:
        raster_crs = CRS.from_epsg(28992)

    # Create tif file with relevant options
    t_file = rasterio.open(
        output_file_path,
        "w",
        driver="GTiff",
        height=new_grid_data.shape[0],
        width=new_grid_data.shape[1],
        count=1,
        dtype=new_grid_data.dtype,
        crs=raster_crs,
        transform=transform,
    )

    # Write and close file
    t_file.write(new_grid_data, 1)
    t_file.close()


def raster_transform(
    source: rasterio.io.DatasetReader,
    destination: rasterio.io.DatasetReader,
    output_path: str = None,
    resampling: int = 5,
    epsg: int = 28992,
) -> None:
    """
    Transforms source raster shape to destination raster shape.
    Output is saved at output_path. Resampling options from rasterio are available.

    Args:
        source (rasterio.io.DatasetReader): source raster from which data is extracted.
        destination (asterio.io.DatasetReader): destination raster to which's shape the new raster will conform.
        output_path (str): location to save the output raster. Default None will result in no files being saved to disk
        resampling (int): rasterio resampling options (default: 5 = average). Options are:
                nearest = 0,
                bilinear = 1,
                cubic = 2,
                cubic_spline = 3,
                lanczos = 4,
                average = 5,
                mode = 6,
                gauss = 7,
                max = 8,
                min = 9,
                med = 10,
                q1 = 11,
                q3 = 12,
                sum = 13,
                rms = 14.
        epsg (int): coordinate reference system. Default is Dutch RDS.

    Returns:
        s_data_np (np.ndarray): transformed raster in numpy
        s_data_da (xr.DataArray): transformed raster in Xarray DataArray format
    """
    # Read Rasterio transforms from both source and destination rasters
    s_transform = source.transform
    d_transform = destination.transform

    # Read destination data for its shape, and create a new source data array with the same shape
    d_data = destination.read(1)
    s_data = np.zeros(d_data.shape)

    # Obtain coordinate boundary of the pixel (upper left and lower right bounds)
    ul_x, ul_y = rasterio.transform.xy(transform=d_transform, rows=0, cols=0, offset="ul")
    lr_x, lr_y = rasterio.transform.xy(
        transform=d_transform, rows=d_data.shape[0], cols=d_data.shape[1], offset="lr"
    )

    # Create a Rasterio window from pixel bounds and source_transform
    window = from_bounds(left=ul_x, top=ul_y, right=lr_x, bottom=lr_y, transform=s_transform)

    # Read source data within window and resample
    s_data_np = source.read(
        out_shape=(1, d_data.shape[0], d_data.shape[1]), window=window, resampling=resampling
    )[0, :, :]

    xs, _ = rasterio.transform.xy(
        transform=d_transform, rows=0, cols=np.arange(d_data.shape[1]), offset="center"
    )
    _, ys = rasterio.transform.xy(
        transform=d_transform, rows=np.arange(d_data.shape[0]), cols=0, offset="center"
    )

    # store data in DataArray
    s_data_da = xr.DataArray(
        data=s_data_np,
        dims=["y", "x"],
        coords={"x": xs, "y": ys},
    )

    # If output path is given, save data to disk
    if output_path is not None:
        if (".tif" in output_path.lower()) or (".tiff" in output_path.lower()):
            # write tiff
            write_tiff(
                output_file_path=output_path,
                new_grid_data=s_data_np,
                transform=d_transform,
                epsg=epsg,
            )
        elif ".nc" in output_path.lower():
            s_data_da.to_netcdf(path=output_path, mode="w", format="NETCDF4")
        else:
            raise NameError("File not supported")

    return s_data_np, s_data_da


def stack_rasters(ras_files: List, time_stamps: pd.Index, destination: rasterio.io.DatasetReader, agg_indexer:str=None, resampling: int = 13) -> xr.DataArray:
    """
    Stacks rasters of the same grid together into one Xarray DataArray
    
    Args:
        ras_files (List): file-paths of the rasters that are stacked
        time_stamps (pd.Index): Pandas date-time indexes per file
        destination (rasterio.io.DatasetReader): destination raster to which's shape the new raster will conform.
        agg_indexer (str): Xarray resampler indexer (e.g. "1D")
        resampling (int): rasterio resampling options (default: 5 = average). Options are:
                nearest = 0,
                average = 5,
                sum = 13,

    Returns:
        da (xr.DataArray): DataArray containing the stacked rasters
    """
    # Loop over files in ras_files list and extract and resample raster. Add to list
    ras_data_list = []
    for ix, ras_file in enumerate(tqdm(ras_files)):
        with rasterio.open(ras_file) as src:
            _, ras_data = raster_transform(source=src, destination=destination)
            ras_data_list.append(ras_data)

    # Add the list of DataArrays to a single DataArray with a time dimension and inverse dimensions
    da = xr.concat(objs=ras_data_list, dim=time_stamps).transpose()

    # Aggregate if required, for instance per day
    if agg_indexer is not None:
        resampling = int(resampling)
        if resampling == 0:
            da = da.resample(time=agg_indexer).nearest()
        elif resampling == 5:
            da = da.resample(time=agg_indexer).mean()
        elif resampling == 13:
            da = da.resample(time=agg_indexer).sum()
        else:
            raise NameError("Resampling method {} not implemented".format(resampling))
    
    return da
    