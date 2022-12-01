from pathlib import Path
import geopandas as gpd
import pandas as pd
import json

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

def to_results_gdf(gdf, object_type, id_col="ID", results_gdf=None):
    gdf.to_crs(4326, inplace=True)
    gdf.rename(columns={id_col: "ID"}, inplace=True)
    gdf["Object_Type"] = object_type
    for i in RESULT_POINTS_COLUMNS:
        if i not in gdf.columns:
            gdf[i] = None
    gdf = gdf[RESULT_POINTS_COLUMNS]
    if results_gdf is not None:
        gdf = pd.concat([results_gdf, gdf])
        gdf.reset_index(inplace=True, drop=True)
    return gdf


def to_results_js(results_gdf):
    resultpoints_dict = {i: None for i in ["type", "name", "crs", "features"]}
    gdf_dict = json.loads(results_gdf.to_json())
    resultpoints_dict["features"] = gdf_dict["features"]
    resultpoints_dict["type"] = gdf_dict["type"]
    resultpoints_dict["crs"] = CRS_DICT
    resultpoints_dict["name"] = "resultspoints"

    resultspoints_js = js_dir / "resultspoints.js"
    resultspoints = resultspoints_js.read_text().format(
        geojson=json.dumps(resultpoints_dict, indent=1)
        )
    resultsponits_js = results_dir / "resultspoints.js"
    resultsponits_js.write_text(resultspoints)


# %% debietmeetpunten to results_gdf
print("resultpoints")

# gauges
gdf = gpd.read_file(staticmaps_gpkg, layer="gauges", driver="GPKG")
gdf.loc[:, "WFLOW"] = gdf["WFLOW"].apply(lambda x: f"gauge_{x}")
results_gdf = to_results_gdf(gdf, id_col="CODE", object_type="stuw")

# catchments
gdf = gpd.read_file(staticmaps_gpkg, layer="subcatch", driver="GPKG")
gdf.geometry = gdf.geometry.centroid
gdf.loc[:, "WFLOW"] = gdf["WFLOW"].apply(lambda x: f"subcatch_{x}")
results_gdf = to_results_gdf(gdf, id_col="code", object_type="subcatchment", results_gdf=results_gdf)

# peilbuizen
peilbuizen_gdf = gpd.read_file(peilbuizen_gpkg)
peilbuizen_gdf.loc[:, "WFLOW"] = peilbuizen_gdf["WFLOW"].apply(lambda x: f"peilbuis_{x}")
results_gdf = to_results_gdf(peilbuizen_gdf, id_col="locatie", object_type="peilbuis", results_gdf=results_gdf)

to_results_js(results_gdf)
results_gdf.to_file("results.gpkg", driver="GPKG")