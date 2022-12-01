from affine import Affine
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from rasterio.windows import from_bounds, Window
from rasterio.features import rasterize
from pathlib import Path
import pandas as pd
import rasterio
from xarray.core.dataset import Dataset
import rioxarray as rio
from shapely.geometry import mapping

def round_bounds_to_res(bounds, res):
    for i in [0, 2]:
        bounds[i] = round(bounds[i] / res - 0.5) * res
    for i in [1, 3]:
        bounds[i] = round(bounds[i] / res - 0.5) * res
    return bounds

def get_transform(bounds, res):
    bounds = round_bounds_to_res(bounds, res)
    return Affine(res, 0., bounds[0], 0., -res, bounds[-1])

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


    def to_observations_list(self, parameter, prefix="" ,decimals=2):
        dates = [f"{pd.to_datetime(i).isoformat()}.000Z" for i in self.df.index.values]
        observations_list = []
        for i in self.df.columns:
            observations_list.append({
                "ID": f"{prefix}{i}",
                "Alias": i,
                parameter: {
                "dates":dates,
                "values": self.df[i].round(decimals).to_list()}
                }
                )
        return observations_list

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