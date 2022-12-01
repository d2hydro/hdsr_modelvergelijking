import hydropandas as hpd
import pandas as pd


def grondwaterstanden(put_dir, catchment, tmin="2014-01-01", tmax="2016-12-31"):
    observations_list = [] 
    fnames = put_dir.glob("*1_1.csv")
    gws = [hpd.GroundwaterObs.from_dino(fname=i) for i in fnames]
    oc = hpd.ObsCollection.from_list(gws, name='Dino groundwater')
    gdf = oc.to_gdf().set_crs(28992).to_crs(4326)
    gdf = gdf.loc[gdf.within(catchment)]
    gwl_obs = oc.get_series(tmin="2014-01-01", tmax="2016-12-31")
    gwl_obs = gwl_obs.loc[gdf.index]

    ids = []
    for row in gdf.itertuples():
        ts_df = gwl_obs.loc[row.Index]
        ts_df = ts_df[ts_df.notna()]
        if not ts_df.empty:
            ids.append(row.locatie)
            observations_list.append(
                {
                "ID": row.locatie,
                "Alias": row.Index,
                "groundwater_level": {
                    "dates":[f"{pd.to_datetime(i).isoformat()}.000Z" for i in ts_df.index.values],
                    "values": ts_df.to_list()}
                }
                )
    gdf = gdf.loc[gdf["locatie"].isin(ids)]

    return gdf, observations_list
