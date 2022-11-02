from datetime import datetime as dt
from pathlib import Path

import pandas as pd
import rasterio
import xarray as xr

from utilities import stack_rasters

## WARNING no data on dec 31 2015

# rain files
rf_path = r"D:\Work\Project\P1389\meteo\amerongerwetering\NEERSLAG"
rf_files = list(Path(rf_path).glob("*.ASC"))

start_time = dt.fromisoformat("2014-01-01 00:00:00")
end_time = dt.fromisoformat("2016-12-30 23:00:00")
time_stamps = pd.date_range(start=start_time, end=end_time, freq="1H", name="time")

d_dataset = rasterio.open(r"D:\Work\Project\P1389\GIS\HH_raster\raster_25m.tif")
rf_da = stack_rasters(
    ras_files=rf_files,
    time_stamps=time_stamps,
    destination=d_dataset,
    agg_indexer="1D",
    resampling=13,
)

# evaporation files
evp_path = r"D:\Work\Project\P1389\meteo\amerongerwetering\MAKKINK"
evp_files = list(Path(evp_path).glob("*.ASC"))

start_time = dt.fromisoformat("2014-01-01")
end_time = dt.fromisoformat("2016-12-30")
time_stamps = pd.date_range(start=start_time, end=end_time, freq="1D", name="time")

ev_da = stack_rasters(
    ras_files=evp_files,
    time_stamps=time_stamps,
    destination=d_dataset,
    agg_indexer="1D",
    resampling=5,
)

# save dataset
ds = xr.Dataset({"prec": rf_da, "evp": ev_da})

ds.to_netcdf(
    path=r"D:\Work\Project\P1389\meteo\amerongerwetering\meteo.nc", mode="w", format="NETCDF4"
)
