import pandas as pd

RAW  = "/home/ibm-user/shared/zcold/data/raw/fuelpov/fuelpov_2024.xlsx"
PROC = "/home/ibm-user/shared/zcold/data/processed/"

# --- Fuel poverty (validation only, never a model feature) ---
fp = pd.read_excel(RAW, sheet_name="Table 4", header=2, dtype={"LSOA Code": str})
fp = fp.rename(columns={
    "LSOA Code": "lsoa21cd",
    "Number of households": "n_households",
    "Number of households in fuel poverty": "n_fuel_poor",
    "Proportion of households fuel poor (%)": "fuel_poor_pct",
})[["lsoa21cd", "n_households", "n_fuel_poor", "fuel_poor_pct"]]

# --- Merge everything on lsoa21cd ---
iod    = pd.read_csv(PROC + "iod25_gm.csv",  dtype={"lsoa21cd": str})
census = pd.read_csv(PROC + "census_gm.csv", dtype={"lsoa21cd": str})

df = (iod.merge(census, on="lsoa21cd", how="left")
         .merge(fp,     on="lsoa21cd", how="left"))

print(f"Rows: {len(df)} (expect 1702)")
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])
print(df[["fuel_poor_pct", "n_households"]].describe().round(1))

df.to_csv(PROC + "lsoa_features.csv", index=False)
print("Saved lsoa_features.csv  | columns:", df.columns.tolist())
