try:
    import rasterio
except ImportError:
    from osgeo import gdal
    import rasterio
import geopandas as gpd
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from rasterio.fill import fillnodata
from rasterio import Affine
import numpy as np


def get_bounds(source, shape):
    bounds = list(shape.bounds)
    res = abs(source.res[0])
    for i in [0, 2]:
        bounds[i] = round(bounds[i] / res - 0.5) * res
    for i in [1, 3]:
        bounds[i] = round(bounds[i] / res + 0.5) * res
    return bounds


def get_transform(source, bounds):
    return Affine(source.transform.a,
                  source.transform.b,
                  bounds[0],
                  source.transform.d,
                  source.transform.e,
                  bounds[3])


def read_window(source, window):
    data = source.read(window=window)[0, :, :]

    if source.scales[0] != 1.:
        data * source.scales[0]
    return data


def read_in_shape_bounds(source, shape, scale=False):
    bounds = get_bounds(source, shape)
    window = from_bounds(*bounds, transform=source.transform)
    return read_window(source, window)


def fill_nodata(data, nodata):
    fill_mask = data != profile["nodata"]
    return fillnodata(data, mask=fill_mask)


def clip_on_shape(data, shape, transform, nodata):
    nodata_mask = np.invert(rasterize((shape, 1),
                            out_shape=data.shape,
                            fill=0,
                            transform=transform
                            ).astype(bool))
    data[nodata_mask] = nodata
    return data

source = rasterio.open(r"d:\projecten\D2203.HDSR-modelvergelijking\06.modelbouw_amerongerwetering\dtm.tif")
gdf = gpd.read_file(r"d:\projecten\D2203.HDSR-modelvergelijking\03.Bronbestanden\Afvoergebieden\Export_Output_5.shp")
shape = gdf.iloc[1]["geometry"]

profile = source.profile
bounds = get_bounds(source, shape)
transform = get_transform(source, bounds)

# %% read raster window on shape bounds
data = read_in_shape_bounds(source, shape)

# %%interpolate raster
data = fill_nodata(data, profile["nodata"])

# %%clip raster on shape
data = clip_on_shape(data, shape, transform, profile["nodata"])

with rasterio.open("data.tif", "w", **profile) as dst:
    dst.write(data, 1)
    dst.scales = (0.01,)

