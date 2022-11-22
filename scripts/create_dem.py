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
from pathlib import Path

from utilities import raster_transform


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

PROJECT_DIR = Path(r"d:\projecten\D2203.HDSR-modelvergelijking")
SCALES = (0.1,) # dit omdat de scales-parameter in de dtm.tif nog niet op 0.01 staat

static_tifs = PROJECT_DIR / "06.modelbouw_amerongerwetering/statictifs"
static_tifs.mkdir(exist_ok=True)
dtm_source = rasterio.open(PROJECT_DIR / "06.modelbouw_amerongerwetering/dtm.tif")
gdf = gpd.read_file(PROJECT_DIR / "03.Bronbestanden/Afvoergebieden/Export_Output_5.shp")
shape = gdf.iloc[1]["geometry"]
grid_source = rasterio.open(PROJECT_DIR / "06.modelbouw_amerongerwetering/ahn_25m.tif")

profile = dtm_source.profile
bounds = get_bounds(dtm_source, shape)
transform = get_transform(dtm_source, bounds)

# %% read raster window on shape bounds
data = read_in_shape_bounds(dtm_source, shape)

# %%interpolate raster
data = fill_nodata(data, profile["nodata"])

# %%clip raster on shape
data = clip_on_shape(data, shape, transform, profile["nodata"])
dem_clipped = PROJECT_DIR / "dem_clipped.tif"


data = np.where(data != profile["nodata"], data * SCALES, profile["nodata"])
profile["dtype"] = data.dtype
profile["width"] = data.shape[1]
profile["height"] = data.shape[0]
with rasterio.open(dem_clipped, "w", **profile) as dst:
    dst.write(data, 1)

dem_clipped_source = rasterio.open(dem_clipped)

dem_data = raster_transform(dem_clipped_source, grid_source, str(static_tifs / "wflow_dem.tif"))
