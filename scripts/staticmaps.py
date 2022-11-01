import geopandas as gpd
import numpy as np
import numbers
from pathlib import Path
import pcraster as pcr
from pcraster._pcraster import Field
import rasterio
from rasterio.features import rasterize
from dataclasses import dataclass
import shutil
import xarray as xr


PROJECT_DIR = Path(r"d:\projecten\D2203.HDSR-modelvergelijking")
WGS84_BBOX = (4.730, 51.924, 5.593, 52.177)
BURN_DEPTH = 10000
RIVER_WIDTH = 5
RIVER_SLOPE = 0.001
NODATA = -9999.


def clip_on_shape(data, shape, transform, nodata):
    nodata_mask = np.invert(rasterize((shape, 1),
                            out_shape=data.shape,
                            fill=0,
                            transform=transform
                            ).astype(bool))
    data[nodata_mask] = nodata
    return data


def array_to_map(data, nodata):
    if issubclass(data.dtype.type, numbers.Integral):
        pcr_type = pcr.Nominal
    elif issubclass(data.dtype.type, numbers.Real):
        pcr_type = pcr.Scalar

    return pcr.numpy2pcr(pcr_type, data, nodata)


def pcr_valuescale(dtype):
    if issubclass(dtype.type, numbers.Real):
        value_scale = "VS_SCALAR"
    elif issubclass(dtype.type, numbers.Integral):
        value_scale = "VS_INTEGER"
    return value_scale


@dataclass
class Reporter:
    tif_dir: Path
    map_dir: Path
    nodata: float
    transform: rasterio.Affine
    width: int
    height: int
    layers: int = 3
    crs: int = 28992
    xs: list = None
    ys: list = None
    ds: xr.Dataset = None

    def __post_init__(self):
        for i in (self.tif_dir, self.map_dir):
            if i.exists():
                shutil.rmtree(i)
            i.mkdir(exist_ok=True, parents=True)

        self.xs, _ = rasterio.transform.xy(
            transform=self.transform, rows=0, cols=np.arange(self.width), offset="center"
        )
        _, self.ys = rasterio.transform.xy(
            transform=self.transform, rows=np.arange(self.height), cols=0, offset="center"
        )
        layer = [i for i in range(self.layers)]
        self.ds = xr.Dataset(
            coords={"x": self.xs, "y": self.ys, "layer": layer},
        )

    def report(self, data, name):
        tif_file = self.tif_dir / f"{name}.tif"
        map_file = self.map_dir / f"{name}.map"
        profile = dict(height=self.height,
                       width=self.width,
                       count=1,
                       dtype=dem_data.dtype,
                       transform=self.transform,
                       nodata=self.nodata,
                       crs=self.crs)
        # write PCRaster map
        if type(data) == Field:
            pcr.report(data, map_file.as_posix())
            data = pcr.pcr2numpy(data, self.nodata)
        else:
            with rasterio.open(map_file,
                               "w",
                               driver="PCRaster",
                               PCRASTER_VALUESCALE=pcr_valuescale(data.dtype),
                               **profile) as dst:
                dst.write(data, 1)
    
        # write GTiff
        with rasterio.open(tif_file, "w", driver="GTiff", **profile) as dst:
            dst.write(data, 1)

        # add to xr dataset
        self.add_to_ds(['y', 'x'], data, name)

    def add_to_ds(self, dims, data, name):
        self.ds[name]=(dims, data)

    def write_netcdf(self, nc_file):
        if self.ds is not None:
            encoding={i:{'_FillValue':self.nodata}  for i in list(self.ds.keys())}
            self.ds.to_netcdf(path=nc_file,
                              mode="w",
                              encoding=encoding,
                              format="NETCDF4")
    
# %% read catchment shape
print("read catchment shape")
gdf = gpd.read_file(PROJECT_DIR / "03.Bronbestanden/Afvoergebieden/Export_Output_5.shp")
catchment_shape = gdf.iloc[1]["geometry"]

# %% get relevant properties from model DEM
print("read model_dem")
dem_source = rasterio.open(PROJECT_DIR / "06.modelbouw_amerongerwetering/ahn_25m.tif")
profile = dem_source.profile

# %% set reporter
print("set_reporter")
static_tifs = PROJECT_DIR / "06.modelbouw_amerongerwetering/statictifs"
static_maps = PROJECT_DIR / "06.modelbouw_amerongerwetering/staticmaps"

reporter = Reporter(tif_dir=static_tifs,
                    map_dir=static_maps,
                    width=profile["width"],
                    height=profile["height"],
                    transform=profile["transform"],
                    nodata=NODATA)

# %% write model_dem as WFlow_dem
print("prepare wflow_dem")
dem_data = dem_source.read(1)
clipped_dem_data = clip_on_shape(dem_source.read(1), catchment_shape, dem_source.transform, NODATA)
reporter.report(dem_data, "wflow_dem")
pcr.setclone(str(static_maps / "wflow_dem.map"))

# %% write land_slope
print("prepare land_slope")
pcr.setglobaloption('unittrue')
dem_map = array_to_map(dem_data, NODATA)
slope_map = pcr.slope(dem_map)
slope_data = pcr.pcr2numpy(slope_map, NODATA)
reporter.report(slope_data, "Slope")

# %% generate wflow_river
print("prepare wflow_river")
gdf = gpd.read_file(PROJECT_DIR / "03.Bronbestanden/Watersysteem/Hydro_Objecten.geojson", bbox=(5.350, 51.952, 5.47, 52.028))
gdf.to_crs("28992", inplace=True)
gdf = gdf.loc[gdf.geometry.intersects(catchment_shape)]
gdf = gdf.loc[gdf.CATEGORIEOPPWATERLICHAAM == 1]
river_data = rasterize(((i, 1) for i in gdf.geometry),
                        out_shape=dem_data.shape,
                        fill=0,
                        transform=profile["transform"]
                        ).astype(int)
reporter.report(river_data, "wflow_river")

# %% prepare river width and length
print("prepare wflow_riverlength and wflow_riverwidth RiverSlope")
river_width = np.where(river_data == 1, RIVER_WIDTH, 0)
river_length = np.where(river_data == 1, abs(dem_source.res[0]), 0)
river_slope = np.where(river_data == 1, RIVER_SLOPE, 0)

reporter.report(river_width, "wflow_riverwidth")
reporter.report(river_length, "wflow_riverlength")
reporter.report(river_slope, "RiverSlope")

# %% generate gauges
print("prepare wflow_gauges")
gdf = gpd.read_file(PROJECT_DIR / "03.Bronbestanden/Watersysteem/Stuwen.geojson")
gdf.to_crs("28992", inplace=True)
outlet_shape = gdf.set_index("CODE").at["ST3011", "geometry"]
outlet_data = rasterize(((outlet_shape, 1)),
                        out_shape=dem_data.shape,
                        fill=0,
                        transform=profile["transform"]
                        ).astype(int)
with rasterio.open(static_tifs / "wflow_gauges.tif", "w", **profile) as dst:
    dst.write(outlet_data, 1)

reporter.report(outlet_data, "outlet")

# %% prepare ldd
print("prepare wflow_ldd")
ldd_dem_data = np.where(river_data == 1, clipped_dem_data - BURN_DEPTH, clipped_dem_data)
ldd_dem_data = np.where(outlet_data == 1, ldd_dem_data - BURN_DEPTH, ldd_dem_data)
profile["dtype"] = ldd_dem_data.dtype
with rasterio.open(static_tifs / "wflow_dem_burned.tif", "w", **profile) as dst:
    dst.write(ldd_dem_data, 1)

reporter.report(ldd_dem_data, "dem_burned")
ldd_dem_map = array_to_map(ldd_dem_data, NODATA)
pcr.setglobaloption('lddin')
ldd_map = pcr.lddcreate(ldd_dem_map, 1e35, 1e35, 1e35, 1e35)
ldd_data = pcr.pcr2numpy(ldd_map, NODATA)

reporter.report(ldd_map, "wflow_ldd")

streamorder_map = pcr.streamorder(ldd_map)
reporter.report(streamorder_map, "streamorder")

# %% prepare wflow catchment
print("prepare wflow_catchment")
outlet_map = array_to_map(outlet_data, NODATA)
catchment_map = pcr.catchment(ldd_map, outlet_map)
reporter.report(catchment_map, "wflow_subcatch")


# %% add missing layers in manual with defaults
print("write defaults")
reporter.report(np.full(dem_data.shape, 0.25)  , "k")
reporter.report(np.full(dem_data.shape, 0.2)  , "specific_yield")

conductance_data = river_data * 200
reporter.report(conductance_data, "infiltration_conductance")
reporter.report(conductance_data, "exfiltration_conductance")
reporter.report(dem_data - 0.5, "river_bottom")

# %% write netcdf
print("write NetCDF")
nc_file = PROJECT_DIR / "06.modelbouw_amerongerwetering/staticmaps.nc"
reporter.write_netcdf(nc_file)
