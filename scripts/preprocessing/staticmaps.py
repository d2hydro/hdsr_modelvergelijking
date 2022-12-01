import geopandas as gpd
import numpy as np
import numbers
from pathlib import Path
import pcraster as pcr
from pcraster._pcraster import Field
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from rasterio.coords import BoundingBox
from dataclasses import dataclass
import shutil
import xarray as xr
import tempfile
import imod

# %%
PROJECT_DIR = Path(r"d:\projecten\D2203.HDSR-modelvergelijking")
WGS84_BBOX = (4.730, 51.924, 5.593, 52.177)
BURN_DEPTH = 10000
RIVER_WIDTH = 5
RIVER_SLOPE = 0.001
NODATA = -9999.
GAUGES = ["ST3011", "ST3010", "ST6067", "ST2002", "ST2003"]
TEMP_DIR = tempfile.TemporaryDirectory()
MODEL_LAYERS = [1,2,3]

# input files for topology
staticmaps_gpkg = PROJECT_DIR.joinpath(r"06.modelbouw_amerongerwetering/staticmaps.gpkg")
input_dem = PROJECT_DIR / "06.modelbouw_amerongerwetering/ahn_25m.tif"

# input files for parameters
SIPS_DIR = PROJECT_DIR / r"03.Bronbestanden\Sips"
wortelzone_idf =  SIPS_DIR / r"metaswap/wortelzonedikte.idf"
opp_stedelijk_idf = SIPS_DIR / r"metaswap/oppervlakte_stedelijkgebied.idf"
modellagen_dir = SIPS_DIR / "modellagen_25x25m"
kd_waarden_dir = SIPS_DIR / "kd-waarden"
conductance_dir = SIPS_DIR / r"oppervlaktewater\winter"
leakage_dir = SIPS_DIR / r"kwel_wegzijging"
leakage_pattern = "bdgflf_20010328_l{layer}.idf"

# output dirs
staticmaps_nc = PROJECT_DIR / "06.modelbouw_amerongerwetering/input/staticmaps.nc"
static_tifs = PROJECT_DIR / "06.modelbouw_amerongerwetering/statictifs"
static_maps = PROJECT_DIR / "06.modelbouw_amerongerwetering/staticmaps"

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


def from_raster(raster_file, bounds):
    with rasterio.open(raster_file) as temp_src:
        window = from_bounds(*bounds, transform=temp_src.transform)
        data = temp_src.read(window=window)[0, :, :]
    return data

# %%
def from_idf(idf_file, bounds):
    temp_file = Path(TEMP_DIR.name) / "temp_file.tif"
    da = imod.idf.open(idf_file)
    imod.rasterio.save(temp_file, da,pattern="{name}.tif")
    return from_raster(temp_file, bounds)

# %%
def get_thickness(model_layers, bounds):
    for idx, i in enumerate(MODEL_LAYERS):
        top = from_raster(model_layers / f"top_l{i}.asc", bounds)
        bot = from_raster(model_layers / f"bot_l{i}.asc", bounds)
        layer = top - bot
        layer = np.where(layer < 0, 0, layer)
        if idx == 0:
            layer_ident = np.full(layer.shape, 0)
            thickness = np.full(layer.shape, 0)
        layer_ident = np.where(
            layer_ident == 0,
            np.where(layer > 0, i, 0),
            layer_ident
            )
        thickness = np.where(
            layer_ident == i,
            layer * 1000,
            thickness)
    return layer_ident, thickness


def get_conductance(kd_layers, layer_ident, bounds):
    conductance = np.full(layer_ident.shape, 0)
    for i in MODEL_LAYERS:
        kd = from_idf(kd_layers / f"kd-waarde_laag{i}.idf", bounds)
        conductance = np.where(
            layer_ident == i,
            kd,
            conductance
            )
    return conductance

def get_leakage(leakage_layers, layer_ident, bounds, leakage_pattern=leakage_pattern):
    leakage = np.full(layer_ident.shape, 0)
    for i in MODEL_LAYERS:
        leakage_file = leakage_layers / leakage_pattern.format(layer=i)
        print(leakage_file)
        layer_leakage = from_idf(leakage_file, bounds)
        leakage = np.where(
            layer_ident == i,
            layer_leakage,
            -leakage
            )
    return leakage

def get_river_conductance(conductance_layers, shape, bounds):
    exf_conductance = np.full(shape, 0)
    inf_conductance = np.full(shape, 0)
    for i in MODEL_LAYERS:
        exf_files = sorted(
            list(conductance_layers.glob(f"CONDUCTANCE_LAAG{i}*")),
            reverse=True)
        for exf_file in exf_files:
            print(exf_file.name)
            inf_file = conductance_layers / f"INFFACTOR_LAAG{exf_file.stem[16:]}.IDF"
            print(inf_file.name)
            exf = from_idf(exf_file, bounds)
            inf = exf * from_idf(inf_file, bounds)
            exf_conductance = np.where(
                ~np.isnan(exf),
                np.where(
                    exf_conductance == 0,
                    exf,
                    exf_conductance
                    ),
                exf_conductance
                )
            inf_conductance = np.where(
                ~np.isnan(inf),
                np.where(
                    inf_conductance == 0,
                    exf * inf,
                    inf_conductance
                    ),
                inf_conductance
                )
    return inf_conductance, exf_conductance


def get_river_bottom(bottom_layers, dem, bounds):
    river_bottom = np.full(dem.shape, -999)
    for i in MODEL_LAYERS:
        bottom_files = sorted(
            list(bottom_layers.glob(f"BODEMHOOGTE_LAAG{i}*")),
            reverse=True)
        for bottom_file in bottom_files:
            print(bottom_file.name)
            bottom = from_idf(bottom_file, bounds)
            river_bottom = np.where(
                ~np.isnan(bottom),
                np.where(
                    river_bottom == -999,
                    bottom,
                    river_bottom
                    ),
                river_bottom
                )
    river_bottom = np.where(
        river_bottom == -999,
        dem,
        river_bottom)
    return river_bottom

#%%


@dataclass
class Reporter:
    tif_dir: Path
    map_dir: Path
    nodata: float
    transform: rasterio.Affine
    bounds: BoundingBox
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
    
# %% read subcatchment layer
print("prepare catchment shapes form subcatchments")
gdf = gpd.read_file(staticmaps_gpkg, layer="subcatch")
catchment_shape = gdf.geometry.unary_union

"""
----------------------------------------------------------------------------------------
DEM RELATED MODEL TOPOLOGY
----------------------------------------------------------------------------------------
"""

# %% get relevant properties from model DEM
print("read model_dem")
with rasterio.open(input_dem) as dem_source:
    profile = dem_source.profile
    cell_size = dem_source.res[0]
    dem_data = dem_source.read(1)

# %% set reporter
print("set_reporter")
reporter = Reporter(tif_dir=static_tifs,
                    map_dir=static_maps,
                    width=profile["width"],
                    height=profile["height"],
                    transform=profile["transform"],
                    bounds=dem_source.bounds,
                    nodata=NODATA)

# %% write model_dem as WFlow_dem
print("prepare wflow_dem")
clipped_dem_data = clip_on_shape(dem_data, catchment_shape, profile["transform"], NODATA)
reporter.report(dem_data, "wflow_dem")
pcr.setclone(str(static_maps / "wflow_dem.map"))

# %%
print("prepare initial subcatchments map")
shapes_iter = ((i.geometry, i.WFLOW) for i in gdf.itertuples())
subcatchment_data = rasterize(shapes_iter,
                              out_shape=dem_data.shape,
                              fill=0,
                              transform=profile["transform"]
                              ).astype(int)

subcatchment_map = array_to_map(subcatchment_data, NODATA)

# %% write land_slope
print("prepare land_slope")
pcr.setglobaloption('unittrue')
dem_map = array_to_map(dem_data, NODATA)
slope_map = pcr.slope(dem_map)
slope_data = pcr.pcr2numpy(slope_map, NODATA)
reporter.report(slope_data, "Slope")

# %% generate wflow_river
print("prepare wflow_river")
gdf = gpd.read_file(staticmaps_gpkg, layer="wflow_river")
river_data = rasterize(
    ((i.geometry, 1) for i in gdf.itertuples()),
    out_shape=dem_data.shape,
    fill=0,
    transform=profile["transform"]
    ).astype(int)
river_map = array_to_map(river_data, NODATA)
reporter.report(river_data, "wflow_river")

# %% prepare river width and length
print("prepare wflow_riverlength and wflow_riverwidth RiverSlope")
river_width = np.where(river_data == 1, RIVER_WIDTH, 0)
river_length = np.where(river_data == 1, abs(dem_source.res[0]), 0)
river_slope = np.where(river_data == 1, RIVER_SLOPE, 0)

reporter.report(river_width, "wflow_riverwidth")
reporter.report(river_length, "wflow_riverlength")
reporter.report(river_slope, "RiverSlope")

# %%
print("prepare peilbuizen")
gdf = gpd.read_file(
    PROJECT_DIR / r"05.onderzoek_bodemvocht_grondwaterstanden\peilbuizen.gpkg"
    )
shapes_iter = ((i, idx+1) for idx, i in enumerate(gdf.geometry))
peilbuizen_data = rasterize(shapes_iter,
                            out_shape=dem_data.shape,
                            fill=0,
                            transform=profile["transform"]
                            ).astype(int)
reporter.report(peilbuizen_data, "peilbuizen")

# %% generate gauges
print("prepare initial wflow_gauges")
gdf = gpd.read_file(staticmaps_gpkg, layer="gauges")
shapes_iter = ((i.geometry, i.WFLOW) for i in gdf.itertuples())
gauges_data = rasterize(shapes_iter,
                        out_shape=dem_data.shape,
                        fill=0,
                        transform=profile["transform"]
                        ).astype(int)
gauges_map = array_to_map(gauges_data, NODATA)
outlet_map = pcr.ifthenelse(gauges_map == 1, pcr.nominal(1), pcr.nominal(0))
reporter.report(outlet_map , "outlet")

# %% prepare ldd
print("prepare initial ldd: to burned rivers and outlet")
ldd_dem_data = np.where(river_data == 1, clipped_dem_data - BURN_DEPTH, clipped_dem_data)
ldd_dem_data = np.where(gauges_data == 1, ldd_dem_data - BURN_DEPTH, ldd_dem_data)
reporter.report(ldd_dem_data, "wflow_dem_burned")
profile["dtype"] = ldd_dem_data.dtype

reporter.report(ldd_dem_data, "dem_burned")
ldd_dem_map = array_to_map(ldd_dem_data, NODATA)
pcr.setglobaloption('lddin')
ldd_map = pcr.lddcreate(ldd_dem_map, 1e35, 1e35, 1e35, 1e35)
ldd_data = pcr.pcr2numpy(ldd_map, NODATA)

#reporter.report(ldd_map, "wflow_ldd_initial")

streamorder_map = pcr.streamorder(ldd_map)
reporter.report(streamorder_map, "streamorder")

# %% snap and report gauges
print("snap gauges to ldd-rivers (stream-oder >= 4))")
ldd_river_map = pcr.ifthen(streamorder_map >= 4, pcr.nominal(1))
gauges_map = pcr.cover(snaptomap(pcr.ordinal(gauges_map),
                       ldd_river_map), 0)
reporter.report(gauges_map, "gauges")

# %% ldd map
print("prepare final ldd: forced to river and subcatchments")

ldd_path = pcr.path(
    ldd_map,
    pcr.ifthenelse(gauges_map > 0, pcr.boolean(1), pcr.boolean(0))
    )
river_ldd_map = pcr.ifthen(ldd_path == 1, ldd_map)
#reporter.report(ldd_path, "ldd_path")
#reporter.report(river_ldd_map, "wflow_ldd_river")

ldd_forced_map = pcr.ldd(pcr.ifthen(pcr.scalar(river_ldd_map) > 8, pcr.scalar(1)))

# create a covered ldd. Note: we work from downstream to upstream!
for catch in range(1,int(subcatchment_data.max())+1):
    # create subcatch and gauge boolean maps
    subcatch =  pcr.ifthen(subcatchment_map == catch, pcr.boolean(1))
    gauge = pcr.ifthenelse(gauges_map == catch, pcr.boolean(1), pcr.boolean(0))
    subcatch = pcr.ifthenelse(gauge, pcr.boolean(1), subcatch)

    # select dem
    dem_selec = pcr.ifthen(subcatch, ldd_dem_map)

    # burn selected dem at gauge (so it gets forced to the river)
    dem_selec = pcr.ifthenelse(gauge, dem_selec - 10000, dem_selec)

    # create ldd for subcatchment with pit at gauge
    ldd_selec = pcr.lddcreate(dem_selec, 1e35, 1e35, 1e35, 1e35)

    # cover with existing ldd
    ldd_forced_map = pcr.cover(ldd_forced_map,ldd_selec)

# cover and report river ldd
ldd_forced_map = pcr.cover(river_ldd_map, ldd_forced_map)
reporter.report(ldd_forced_map, "wflow_ldd")

# report laterals DEM
ldd_laterals_map = pcr.ifthenelse(gauges_map > 0, pcr.ldd(5), ldd_forced_map)
reporter.report(ldd_laterals_map, "wflow_ldd_laterals")


# %% prepare wflow catchment
print("prepare final subcatchments: from final ldd")
subcatchment_map = pcr.subcatchment(ldd_forced_map, gauges_map)
reporter.report(subcatchment_map, "wflow_subcatch")

"""
----------------------------------------------------------------------------------------
SBM model parameters (vertical) from MetaSwap/IMOD
----------------------------------------------------------------------------------------
"""

# %% add rooting depth
print("prepare rooting depth (from Metaswap)")
rootingdepth_data = from_idf(wortelzone_idf,
    reporter.bounds
    )
rootingdepth_data = rootingdepth_data * 10
reporter.report(rootingdepth_data, "rootingdepth")


# %% add rooting depth
print("prepare pathfrac")
pathfrac_data = from_idf(opp_stedelijk_idf,
    reporter.bounds
    ) 
pathfrac_data = pathfrac_data / cell_size **2
reporter.report(pathfrac_data, "pathfrac")

# %% add rooting depth
print("prepare soilthickness (from IMOD)")
layer_ident, soilthickness = get_thickness(
    modellagen_dir,
    reporter.bounds
    )

reporter.report(layer_ident, "layer_ident")
reporter.report(soilthickness, "soilthickness")

"""
----------------------------------------------------------------------------------------
Groundwater parameters (horizontal) from MetaSwap/IMOD
----------------------------------------------------------------------------------------
"""

# %%
print("prepare conductance (a.k.a. kd) (from IMOD)")
conductance_data = get_conductance(kd_waarden_dir,
                                   layer_ident,
                                   reporter.bounds)
reporter.report(conductance_data, "conductance")

conductivity_data = conductance_data / (soilthickness / 1000)

reporter.report(conductivity_data, "ksat")

# %% 
print("prepare conductance (from IMOD)")
inf_conductance_data, exf_conductance_data = get_river_conductance(conductance_dir,
                                                                   layer_ident.shape,
                                                                   reporter.bounds)

reporter.report(inf_conductance_data, "infiltration_conductance")
reporter.report(exf_conductance_data, "exfiltration_conductance")

# %%
print("river bottom (from IMOD)")
river_bottom_data = get_river_bottom(conductance_dir,
                                     dem_data,
                                     reporter.bounds)

reporter.report(river_bottom_data, "river_bottom")

# %%
print("leakage (from IMOD)")
leakage_data = get_leakage(leakage_dir, layer_ident, reporter.bounds)
leakage_data = (leakage_data / (cell_size ** 2)) * 1000 # m3/day to mm/day
leakage_data = np.where(leakage_data > 5, 5, leakage_data) # limit max to 5 mm/day
leakage_data = np.where(leakage_data < -5, -5, leakage_data) # limit min to -5 mm/day
reporter.report(leakage_data, "deeptransfer")

# %% add missing layers in manual with defaults
print("write defaults")

reporter.report(np.full(dem_data.shape, 0.2)  , "specific_yield")
reporter.report(np.full(dem_data.shape, 5), "maxleakage")

# %% write netcdf
print("write NetCDF")
reporter.write_netcdf(staticmaps_nc)
