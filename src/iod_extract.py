import pandas as pd

RAW = "/home/ibm-user/shared/zcold/data/raw/iod/"
OUT = "/home/ibm-user/shared/zcold/data/processed/iod25_gm.csv"
GM = ["Bolton","Bury","Manchester","Oldham","Rochdale","Salford",
      "Stockport","Tameside","Trafford","Wigan"]

def load_data_sheet(path):
    """Pick the sheet with the most rows (the data, not Notes)."""
    sheets = pd.read_excel(path, sheet_name=None)
    name, df = max(sheets.items(), key=lambda kv: len(kv[1]))
    print(f"{path.split('/')[-1]} -> sheet '{name}' ({len(df)} rows)")
    return df

def pick(df, wanted):
    """wanted: {new_name: [keywords that must all appear, lowercase]}"""
    out = {}
    for new, keys in wanted.items():
        hits = [c for c in df.columns if all(k in str(c).lower() for k in keys)]
        if hits:
            out[hits[0]] = new
            print(f"  {new:<18} <- {hits[0]}")
        else:
            print(f"  {new:<18} <- NOT FOUND")
    return df[list(out)].rename(columns=out)

ids = {"lsoa21cd": ["lsoa code"], "lsoa21nm": ["lsoa name"],
       "lad_name": ["local authority district name"]}

f5 = pick(load_data_sheet(RAW + "iod25_file5_scores.xlsx"), {**ids,
    "imd_score":        ["index of multiple deprivation", "score"],
    "income_score":     ["income score"],
    "employment_score": ["employment score"],
    "health_score":     ["health deprivation", "score"],
    "living_env_score": ["living environment score"],
    "indoors_score":    ["indoors", "score"],
    "barriers_score":   ["barriers to housing", "score"],
    "idaopi_score":     ["older people", "score"],
})

f1 = pick(load_data_sheet(RAW + "iod25_file1_imd.xlsx"), {
    "lsoa21cd":   ["lsoa code"],
    "imd_rank":   ["index of multiple deprivation", "rank"],
    "imd_decile": ["index of multiple deprivation", "decile"],
})

df = f5.merge(f1, on="lsoa21cd", how="left")
gm = df[df["lad_name"].isin(GM)].copy()
print(f"\nGM LSOAs: {len(gm)}  |  per LAD:\n{gm['lad_name'].value_counts()}")
gm.to_csv(OUT, index=False)
print(f"Saved {OUT}")
