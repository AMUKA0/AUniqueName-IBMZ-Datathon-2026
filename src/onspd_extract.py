import pandas as pd, zipfile

RAW = "/home/ibm-user/shared/zcold/data/raw/onspd/onspd.zip"
PROC = "/home/ibm-user/shared/zcold/data/processed/"
gm = set(pd.read_csv(PROC + "iod25_gm.csv", dtype=str)["lsoa21cd"])

with zipfile.ZipFile(RAW) as z:
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    print("CSVs in zip:", len(csvs))
    # the multi-CSV zip has one file per postcode area; GM covers these areas
    areas = ("_m.", "_bl.", "_ol.", "_sk.", "_wn.", "_wa.")
    pick = [n for n in csvs if any(a in n.lower() for a in areas)] or \
           [n for n in csvs if "data" in n.lower() and "multi" not in n.lower()]
    print("Reading:", pick)
    parts = [pd.read_csv(z.open(n), dtype=str) for n in pick]
df = pd.concat(parts)
print("Columns:", df.columns.tolist())

pc  = next(c for c in ["pcds", "pcd", "pcd7", "pcd8"] if c in df.columns)
ls  = next(c for c in df.columns if c.lower() in ("lsoa21cd", "lsoa21"))
term = "doterm" if "doterm" in df.columns else None
if term: df = df[df[term].isna()]          # live postcodes only

df = df[df[ls].isin(gm)].copy()
df["postcode"] = df[pc].str.upper().str.replace(" ", "")
df["postcode_district"] = df[pc].str.upper().str.split().str[0]
out = df[["postcode", "postcode_district", ls]].rename(columns={ls: "lsoa21cd"})
out.to_csv(PROC + "postcode_lsoa_gm.csv", index=False)

dist = (out.groupby(["lsoa21cd", "postcode_district"]).size().rename("n_postcodes")
          .reset_index().sort_values("n_postcodes", ascending=False)
          .drop_duplicates("lsoa21cd"))
dist.to_csv(PROC + "lsoa_to_district.csv", index=False)
print(f"{len(out)} GM postcodes | {dist.lsoa21cd.nunique()} LSOAs mapped "
      f"| districts: {sorted(dist.postcode_district.unique())[:15]}...")
