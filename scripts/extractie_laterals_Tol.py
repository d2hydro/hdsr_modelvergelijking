# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:31:26 2021

@author: deRoover
"""
import datetime as dt
from copy import deepcopy as dc

## IMPORTS
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

#%% FUNCTIONS

#%% SETUP
# OUTPUT_WFLOW_PATH = r'D:\work\Project\P1126\05_Analysis\01_WFLOW\WFLOW_julia\temp\output_example-sbm-gwfv5_15.nc'
# OUTPUT_WFLOW_PATH = r'D:\work\Project\P1126\05_Analysis\01_WFLOW\WFLOW_julia\test_model_deTol_v2\data\output_example-sbm-gwfv4_10_buiKockengen_v2.nc'
# BC_BASE_PATH = r"D:\work\Project\P1126\05_Analysis\01_WFLOW\WFLOW_julia\temp\boundaries_base.bc"
BC_OUTPUT = r"D:\Work\Project\P1389\models\test_model_deTol_v2\laterals\laterals_MC.bc"
fn_qlat1 = (
    r"D:\Work\Project\P1389\models\test_model_deTol_v2\rekentijden_test\output_2014_2016_MC.csv"
)
fn_laterals = r"D:\Work\Project\P1389\models\test_model_deTol_v2\laterals\laterals_de_tolv2.shp"

#%% SCRIPT

## LOAD FILES
laterals = gpd.read_file(fn_laterals)
parser = lambda date: dt.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
WFLOW_lats = pd.read_csv(
    fn_qlat1, index_col="DateTime", parse_dates={"DateTime": [0]}, date_parser=parser
)
#%%
lateral_discharge = pd.DataFrame(index=WFLOW_lats.index)
for index, row in laterals.iterrows():
    try:
        subcatchment = f'Q_av_{int(row["subcatchID"])}'
    except:
        lateral_discharge.loc[:, row["NAAM"]] = 0
        continue
    try:
        upstream_subcatchments = [f"Q_av_{int(i)}" for i in row["sup_subcat"].split(",")]
    except:
        upstream_subcatchments = []
    temp = WFLOW_lats.loc[:, subcatchment].values.sum()
    for upstream_subcatchment in upstream_subcatchments:
        temp -= WFLOW_lats.loc[:, upstream_subcatchment].values.sum()
    lateral_discharge.loc[:, f"lat_{row['NAAM']}"] = (
        temp / WFLOW_lats.loc[:, subcatchment].values.sum()
    ) * WFLOW_lats.loc[:, subcatchment].values
lateral_discharge.index = lateral_discharge.index

#%% WRITE TO .BC FILES
setup_bc = f"""
[Forcing]
name       = <name>
function   = timeseries
timeInterpolation = linear
quantity   = time
unit       = minutes since {lateral_discharge.index[0]}
quantity   = lateral_discharge
unit       = m3/s
0.0 0.0
"""

# with open(BC_BASE_PATH, 'rt') as bc_file:
#     bc = bc_file.read()

lateral_discharge_copy = lateral_discharge.fillna(0)
lateral_discharge_copy.index = [
    (lateral_discharge_copy.index[i] - lateral_discharge_copy.index[0]).total_seconds()
    for i in range(len(lateral_discharge_copy.index))
]

bc = ""
for column_id, column_name in enumerate(lateral_discharge_copy.columns):
    bc += setup_bc.replace("<name>", column_name) + lateral_discharge_copy.iloc[
        :, column_id
    ].to_csv(header=False, sep="\t").replace("\r", "")

with open(BC_OUTPUT, "w+") as bc_file:
    bc_file.write(bc)
