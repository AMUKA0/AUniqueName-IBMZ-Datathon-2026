import pandas as pd, glob

RAW = "/home/ibm-user/shared/zcold/data/raw/census/"
PROC = "/home/ibm-user/shared/zcold/data/processed/"
gm_lsoas = set(pd.read_csv(PROC + "iod25_gm.csv", dtype=str)["lsoa21cd"])

def load(table):
    path = glob.glob(f"{RAW}census2021-{table}-lsoa.csv")[0]
    df = pd.read_csv(path)
    code = [c for c in df.columns if "geography code" in c.lower()][0]
    df = df[df[code].isin(gm_lsoas)].set_index(code)
    print(f"\n{table}: {len(df)} GM rows. Columns:")
    for c in df.columns: print("   ", c)
    return df

def cols(df, *keys):
    return [c for c in df.columns if all(k in c.lower() for k in keys)]

out = pd.DataFrame(index=sorted(gm_lsoas)); out.index.name = "lsoa21cd"

age = load("ts007a")
total = age[cols(age, "total")[0]]
old = cols(age, "75 to") + cols(age, "80 to") + cols(age, "85 years and over")
print("75+ cols:", old)
out["pct_75plus"] = age[old].sum(axis=1) / total * 100

hh = load("ts003")
alone = cols(hh, "one person household", "66")
print("alone 66+ col:", alone)
out["pct_alone_66plus"] = hh[alone].sum(axis=1) / hh[cols(hh, "total")[0]] * 100

dis = load("ts038")
dis_cols = cols(dis, "disabled under the equality act")
dis_cols = [c for c in dis_cols if "not disabled" not in c.lower()][:1]
print("disabled col:", dis_cols)
out["pct_disabled"] = dis[dis_cols].sum(axis=1) / dis[cols(dis, "total")[0]] * 100

car = load("ts045")
nocar = cols(car, "no cars")
print("no car col:", nocar)
out["pct_no_car"] = car[nocar].sum(axis=1) / car[cols(car, "total")[0]] * 100

print(out.describe().round(1))
out.round(2).to_csv(PROC + "census_gm.csv")
print("Saved census_gm.csv")
