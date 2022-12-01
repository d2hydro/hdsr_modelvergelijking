from pathlib import Path
import geopandas as gpd
import pandas as pd
import json
import hydropandas as hpd
from readers.dino import grondwaterstanden
from readers.metingen import debiet
from raster_sets import DatetimeSpecs, RasterTimeSeries

PROJECT_DIR = Path(r"d:\projecten\D2203.HDSR-modelvergelijking")
CRS_DICT = { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" }}
RESULT_POINTS_COLUMNS = ["ID", "Object_Type", "SOBEK","DHYRO","WFLOW","HYDROMEDAH", "geometry"]

# input files for topology
staticmaps_gpkg = PROJECT_DIR.joinpath(r"06.modelbouw_amerongerwetering/staticmaps.gpkg")
peilbuizen_gpkg = PROJECT_DIR / r"05.onderzoek_bodemvocht_grondwaterstanden\peilbuizen.gpkg"

js_dir = Path(__file__).parent.joinpath("js")
results_dir = PROJECT_DIR / r"90.Postprocessing/Resultaten"

catchments_gdf = gpd.read_file(
    PROJECT_DIR.joinpath(r"04.Modelvergelijking\amerongerwetering\waterbalans.gpkg")
    )
catchment = catchments_gdf.unary_union
observations_list = [] 


def to_obsevations_js(observations_list):
    observations_js = js_dir / "observations.js"
    observations_dict = {"locations":observations_list}
    observations = observations_js.read_text().format(
        observations=json.dumps(observations_dict, indent=1)
        )
    resultsponits_js = results_dir / "observations.js"
    resultsponits_js.write_text(observations)


#%%
file_mapping = {"ST6067": "1058_Kolland_stuw_debiet_2014_2016_dag.csv",
                "ST3010": "1059_Nooitgedacht_stuw_debiet_2014_2016_dag.csv",
                "ST3011": "1020_Amerongerwetering_stuw_debiet_2014_2016_dag.csv"}

q_observations_list = debiet(
    q_dir=PROJECT_DIR / "03.Bronbestanden\Metingen\Amerongerwetering",
    file_mapping=file_mapping)


# %% dino meetpunten naar results_gdf
print("grondwater")
_, gwl_observations_list = grondwaterstanden(
    PROJECT_DIR.joinpath(
        r"05.onderzoek_bodemvocht_grondwaterstanden\dino\amerongerwetering\Grondwaterstanden_Put"
        ),
    catchment,
    tmin="2014-01-01",
    tmax="2016-12-31"
    )

# %% wiwb rainfall
subcatchment_gdf = gpd.read_file(staticmaps_gpkg, layer="subcatch").set_index("code")
wiwb_rain_dir = PROJECT_DIR / "03.Bronbestanden/Meteo/Amerongerwetering"
datetime_specs = DatetimeSpecs(start=4, end=15, pattern="%Y%m%d_%H")
wiwb_rain = RasterTimeSeries(wiwb_rain_dir, datetime_specs, suffix=".asc")
wiwb_rain.extract_areas(subcatchment_gdf)

# %% gleam eta
e_gleam_dir = PROJECT_DIR / "03.Bronbestanden/Gleam-HR/E-GLEAM"
datetime_specs = DatetimeSpecs(start=25, end=35, pattern="%Y-%m-%d")
gleam_eta = RasterTimeSeries(e_gleam_dir, datetime_specs)
gleam_eta.extract_areas(subcatchment_gdf.to_crs(4326))

# %% write observations.js
observations_list = wiwb_rain.to_observations_list("P") + gleam_eta.to_observations_list("ETA") + q_observations_list + gwl_observations_list
to_obsevations_js(observations_list)