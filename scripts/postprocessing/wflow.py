from pathlib import Path
import pandas as pd
import geopandas as gpd
from datetime import datetime
import numpy as np
import xarray as xr
import json

from raster_sets import NetCDFTimeSeries
#from staticmaps import clip_on_shape
PROJECT_DIR = Path(r"d:\projecten\D2203.HDSR-modelvergelijking")
AREA = "amerongerwetering"
EXTRACT_DATA = False
EXTRACT_WFLOW = False
# input files for topology
staticmaps_gpkg = PROJECT_DIR.joinpath(r"06.modelbouw_amerongerwetering/staticmaps.gpkg")
peilbuizen_gpkg = PROJECT_DIR / r"05.onderzoek_bodemvocht_grondwaterstanden\peilbuizen.gpkg"

subcatchment_gdf = gpd.read_file(staticmaps_gpkg, layer="subcatch")
subcatchment_gdf.loc[:, "WFLOW"] = subcatchment_gdf["WFLOW"].apply(lambda x:f"subcatch_{x}")
subcatchment_gdf.set_index("WFLOW", inplace=True)

def get_indices(data):
    result = {}
    for i in [i+1 for i in range(int(np.nanmax(data)))]:
        indices = np.where(data == i)
        if not len(indices[0]) == 0:
            result[i] = {"x": indices[1][0], "y": indices[0][0]}
    return result

# %%
print("read WFlow ETA")
netcdf_file = PROJECT_DIR / "06.modelbouw_amerongerwetering/output/results.nc"
wflow_results = NetCDFTimeSeries(netcdf_file, crs=28992)
eta_df = wflow_results.extract_areas(subcatchment_gdf, "eta")

scenario = {
    "scenario": "test",
    "t0": datetime.fromisoformat(wflow_results.ds.time.data[0].isoformat()).strftime("%Y-%m-%dT%H:%M:%S0Z"),
    "timesteps_second": [(i- wflow_results.ds.time.data[0]).total_seconds() for i in wflow_results.ds.time.data],
    "features": []
        }

# for i in eta_df.columns:
#     scenario["features"].append({"id":i,
#                                  "ETA":eta_df[i].to_list()})


# %%
print("read WFlow grondwaterstand")
netcdf_file = PROJECT_DIR / "06.modelbouw_amerongerwetering/input/staticmaps.nc"
wflow_static = xr.open_dataset(netcdf_file)
wflow_results.ds["groundwaterlevel"] = wflow_static.wflow_dem - (wflow_results.ds.zi / 1000)

peilbuis_indices = get_indices(wflow_static.peilbuizen)
wflow_gwl_df = pd.DataFrame(index=wflow_results.time)

for k,v in peilbuis_indices.items():
    scenario["features"].append({"id": f"peilbuis_{k}",
                                 "groundwater_level": [float(i) for i in wflow_results.ds.groundwaterlevel[v["y"], v["x"], :].data]})


#%%
print("read WFlow discharge")
gauge_indices = get_indices(wflow_static.gauges)
df = pd.read_csv(
    PROJECT_DIR / "06.modelbouw_amerongerwetering/output/results.csv",
    index_col=0)

for i in gauge_indices:
    col = f"Q_{i}"
    scenario["features"].append(
        {"id": f"gauge_{i}",
         "Q":df[col].to_list()}
        )

#%%
print("read WFlow eta")
subcatch_indices = get_indices(wflow_static.wflow_subcatch)
for i in subcatch_indices:
    col = f"eta_{i}"
    scenario["features"].append(
        {"id": i,
         "ETA":df[col].to_list()}
        )

# %% Write JS

js_dir = Path(__file__).parent.joinpath("js")
wflow_js = js_dir / "WFLOW.js"

wflow_scenarios = wflow_js.read_text().format(
    scenarios=json.dumps({"scenarios":[scenario]}, indent=1)
    )
results_dir = PROJECT_DIR / r"90.Postprocessing/Resultaten"
wflow_js = results_dir / "WFLOW.js"
wflow_js.write_text(wflow_scenarios)