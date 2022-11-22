from datetime import datetime
from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

from affine import Affine
from dataclasses import dataclass, field
from rasterio.windows import from_bounds, Window
from rasterio.features import rasterize
from shapely.geometry import mapping
import numpy as np
import xarray as xr
import rioxarray as rio
from xarray.core.dataset import Dataset
import pcraster as pcr
#from staticmaps import clip_on_shape
PROJECT_DIR = Path(r"d:\projecten\D2203.HDSR-modelvergelijking")
AREA = "amerongerwetering"
EXTRACT_DATA = False
EXTRACT_WFLOW = False

def round_bounds_to_res(bounds, res):
    for i in [0, 2]:
        bounds[i] = round(bounds[i] / res - 0.5) * res
    for i in [1, 3]:
        bounds[i] = round(bounds[i] / res - 0.5) * res
    return bounds


def get_transform(bounds, res):
    bounds = round_bounds_to_res(bounds, res)
    return Affine(res, 0., bounds[0], 0., -res, bounds[-1])


def snaptomap(points, mmap):
    """
    Snap the points in _points_ to nearest non missing
    values in _mmap_. Can be used to move gauge locations
    to the nearest rivers.

    Input:
        - points - map with points to move
        - mmap - map with points to move to

    Return:
        - map with shifted points
    """
    points = pcr.cover(points, 0)
    # Create unique id map of mmap cells
    unq = pcr.nominal(pcr.cover(pcr.uniqueid(pcr.defined(mmap)), pcr.scalar(0.0)))
    # Now fill holes in mmap map with lues indicating the closes mmap cell.
    dist_cellid = pcr.scalar(pcr.spreadzone(unq, 0, 1))
    # Get map with values at location in points with closes mmap cell
    dist_cellid = pcr.ifthenelse(points > 0, dist_cellid, 0)
    # Spread this out
    dist_fill = pcr.spreadzone(pcr.nominal(dist_cellid), 0, 1)
    # Find the new (moved) locations
    npt = pcr.uniqueid(pcr.boolean(pcr.ifthen(dist_fill == unq, unq)))
    # Now recreate the original value in the points maps
    ptcover = pcr.spreadzone(pcr.cover(points, 0), 0, 1)
    # Now get the org point value in the pt map
    nptorg = pcr.ifthen(npt > 0, ptcover)

    return nptorg


def get_indices(data):
    result = {}
    for i in [i+1 for i in range(int(np.nanmax(data)))]:
        indices = np.where(data == i)
        if not len(indices[0]) == 0:
            result[i] = {"x": indices[0][0], "y": indices[1][0]}
    return result

# %%
@dataclass
class AreasReader:
    data: np.ndarray
    window: Window
    classes: list = field(default_factory=list)
    codes: list = field(default_factory=list)

    @classmethod
    def from_gdf(cls, gdf, res):
        bounds = round_bounds_to_res(gdf.total_bounds, res)
        transform = get_transform(gdf.total_bounds, res)
        window = from_bounds(*bounds, transform=transform)
        shape_iter = [(i, idx) for idx, i in enumerate(gdf.geometry)]
        data = rasterize(shape_iter,
                         out_shape=(round(window.height), round(window.width)),
                         fill=-999,
                         transform=transform).astype(int)
        classes = [i[1] for i in shape_iter]
        codes = gdf.index.to_list()
        return cls(data=data, classes=classes, codes=codes, window=window)

    @property
    def classes_iter(self):
        return zip(self.classes, self.codes)


@dataclass
class DatetimeSpecs:
    start: int
    end: int
    pattern: str

    def from_file(self, file, test=False):
        datetime_str = file.stem[self.start:self.end]
        if test:
            print(datetime_str)
        return datetime.strptime(
            file.stem[self.start:self.end],
            self.pattern
            )


@dataclass
class RasterTimeSeries:
    path: Path
    datetime_specs: DatetimeSpecs
    timestamps: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    areas_reader: AreasReader = None
    files: list = field(default_factory=list)
    suffix: str = ".tif"
    cache_file: Path = None
    df: pd.DataFrame = None

    def __post_init__(self):
        self.files = [i for i in self.path.glob("*") if i.suffix.lower() == self.suffix.lower()]
        with rasterio.open(self.files[0]) as src:
            self.profile = src.profile
        self.from_cache()

    @property
    def res(self):
        if "transform" in self.profile.keys():
            return abs(self.profile["transform"][0])

    def set_reader(self, gdf):
        self.areas_reader = AreasReader.from_gdf(gdf, self.res)

    def extract_areas(self, gdf):
        self.set_reader(gdf)

        results = {}
        for i in self.files:
            ts = self.datetime_specs.from_file(i)
            results[ts] = {}
            with rasterio.open(i) as src:
                data = src.read(window=self.areas_reader.window)[0, :, :] * src.scales[0]
                for idx, code in self.areas_reader.classes_iter:
                    results[ts][code] = data[self.areas_reader.data == idx].mean()
        self.df = pd.DataFrame.from_dict(results, orient="index").sort_index()
        return self.df

    def to_cache(self, cache_file):
        if self.df is not None:
            self.cache_file = Path(cache_file)
            self.df.to_pickle(self.cache_file)

    def from_cache(self):
        if self.cache_file is not None:
            self.df = pd.read_pickle(self.cache_file)
            return self.df

@dataclass
class NetCDFTimeSeries:
    path: Path
    crs: int
    ds: Dataset = None

    def __post_init__(self):
        self.ds = rio.open_rasterio(self.path)

    @property
    def time(self):
        return list(self.ds.indexes["time"].to_datetimeindex().to_pydatetime())

    def extract_areas(self, gdf, variable):

        def _extract_geometry(da, index):
            _gdf = gdf.loc[[index]]
            return da.rio.clip(_gdf.geometry.apply(mapping)).mean(dim=["x","y"], skipna=True).data

        da = getattr(self.ds, variable)
        da.rio.write_crs(self.crs, inplace=True)
        data = {i: _extract_geometry(da, i) for i in gdf.index}
        setattr(self, variable, pd.DataFrame(
            data=data,
            index=self.time
            )
            )
        return getattr(self, variable)

wb_gdf = gpd.read_file(PROJECT_DIR.joinpath("04.Modelvergelijking", AREA, "waterbalans.gpkg"))
wb_gdf.set_index("code", inplace=True)

# %%
print("read E-GLEAM")
cache_file = Path("e_gleam.pickle")
e_gleam_dir = PROJECT_DIR / "03.Bronbestanden/Gleam-HR/E-GLEAM"
datetime_specs = DatetimeSpecs(start=25, end=35, pattern="%Y-%m-%d")
if (not cache_file.exists()) or EXTRACT_DATA:
    gleam_eta = RasterTimeSeries(e_gleam_dir, datetime_specs)
    gleam_eta.extract_areas(wb_gdf)
    gleam_eta.to_cache(cache_file)
else:
    gleam_eta = RasterTimeSeries(e_gleam_dir, datetime_specs, cache_file=cache_file)
    

# %%
print("read WiWB rainfall")
cache_file = Path("wiwb_rain.pickle")
wiwb_rain_dir = PROJECT_DIR / "03.Bronbestanden/Meteo/Amerongerwetering"
datetime_specs = DatetimeSpecs(start=4, end=15, pattern="%Y%m%d_%H")
if (not cache_file.exists()) or EXTRACT_DATA:
    wiwb_rain = RasterTimeSeries(wiwb_rain_dir, datetime_specs, suffix=".asc")
    wiwb_rain.extract_areas(wb_gdf.to_crs(28992))
    wiwb_rain.to_cache(cache_file)
else:
    wiwb_rain = RasterTimeSeries(wiwb_rain_dir, datetime_specs, suffix=".asc", cache_file=cache_file)

# %%
print("read Q metingen")
q_dir = PROJECT_DIR / "03.Bronbestanden/Metingen/Amerongerwetering"
gdf = wb_gdf.to_crs(28992)
data_col_idx = 0
validation_col_idx = 1
valid_values = ["original reliable d0u0", "original reliable d0u1"]
file_mapping = {"ST6067": "1058_Kolland_stuw_debiet_2014_2016_uur.csv",
                "ST3010": "1059_Nooitgedacht_stuw_debiet_2014_2016_uur.csv",
                "ST3011": "1020_Amerongerwetering_stuw_debiet_2014_2016_uur.csv"}
series = []
for k,v in file_mapping.items():
    df = pd.read_csv(q_dir / v, header=[0,1], index_col=0)
    serie = df.loc[df[df.columns[validation_col_idx]].isin(valid_values)][df.columns[data_col_idx]]
    serie.name = k
    series += [serie]

q_meting_df = pd.concat(series, axis=1, join="inner")
meting_q = q_meting_df.copy()
meting_q.loc[:,("ST3011")] = (meting_q["ST3011"] - meting_q["ST3010"]) / gdf.at["ST3011", "geometry"].area * 3600 * 1000
meting_q.loc[:,("ST3010")] = meting_q["ST3010"] - meting_q["ST6067"] / gdf.at["ST3010", "geometry"].area * 3600 * 1000
meting_q.loc[:,("ST6067")] = meting_q["ST6067"] / gdf.to_crs(28992).at["ST6067", "geometry"].area * 3600 * 1000

# %%
print("read WFlow ETA")
netcdf_file = PROJECT_DIR / "06.modelbouw_amerongerwetering/output/test.nc"
wflow_results = NetCDFTimeSeries(netcdf_file, crs=28992)
wflow_results.extract_areas(wb_gdf.to_crs(28992), "eta")

# %%
print("read WFlow Q")
netcdf_file = PROJECT_DIR / "06.modelbouw_amerongerwetering/input/staticmaps.nc"
wflow_static = xr.open_dataset(netcdf_file)
gauge_indices = get_indices(wflow_static.gauges)
wflow_q = pd.DataFrame(index=wflow_results.time)
wflow_q["ST6067"] = (wflow_results.ds.q_river[:, gauge_indices[3]["x"], gauge_indices[3]["y"]].values / gdf.to_crs(28992).at["ST6067", "geometry"].area) * 86400 * 1000
wflow_q["ST3010"] = (wflow_results.ds.q_river[:, gauge_indices[2]["x"], gauge_indices[2]["y"]].values / gdf.to_crs(28992).at["ST3010", "geometry"].area) * 86400 * 1000
wflow_q["ST3011"] = (wflow_results.ds.q_river[:, gauge_indices[1]["x"], gauge_indices[1]["y"]].values / gdf.to_crs(28992).at["ST3011", "geometry"].area) * 86400 * 1000



#%%
print("read WFlow groundwater level")
wflow_results.ds["groundwaterlevel"] = wflow_static.wflow_dem - (
    wflow_static.soilthickness - wflow_results.ds.swd
    ) / 1000