import pandas as pd


Q_MAPPING = {"ST6067": "1058_Kolland_stuw_debiet_2014_2016_uur.csv",
             "ST3010": "1059_Nooitgedacht_stuw_debiet_2014_2016_uur.csv",
             "ST3011": "1020_Amerongerwetering_stuw_debiet_2014_2016_uur.csv"}

ALIAS_MAPPING = {"ST6067": "Stuw Kolland",
                 "ST3010": "Stuw Nooitgedacht",
                 "ST3011": "Stuw Amerongerwetering"}

def debiet(q_dir, file_mapping = Q_MAPPING):
    data_col_idx = 0
    validation_col_idx = 1
    valid_values = ["original reliable d0u0", "original reliable d0u1"]

    observations_list = []
    for k,v in file_mapping.items():
        df = pd.read_csv(q_dir / v, header=[0,1], index_col=0)
        serie = df.loc[df[df.columns[validation_col_idx]].isin(valid_values)][df.columns[data_col_idx]]
        observations_list.append({
            "ID": k,
            "Alias": ALIAS_MAPPING[k],
            "Q": {
            "dates":[f"{pd.to_datetime(i).isoformat()}.000Z" for i in serie.index.values],
            "values": serie.to_list()}
            }
            )

    return observations_list