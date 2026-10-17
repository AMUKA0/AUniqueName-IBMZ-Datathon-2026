from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"


def main():
    source = RAW / (
        "weather-stations-and-postcodes-"
        "cold-weather-payment-scheme-2025-2026.ods"
    )
    sheets = pd.read_excel(
        source, sheet_name=None, header=None, engine="odf"
    )

    # Find the official table listing individual postcode districts.
    tables = []
    for name, frame in sheets.items():
        for position in range(min(50, len(frame))):
            headers = frame.iloc[position].fillna("").astype(str)
            headers = headers.str.strip().str.lower()
            if (
                (headers == "postcode district").sum() == 1
                and (headers == "weather station").sum() == 1
            ):
                pc = list(headers).index("postcode district")
                st = list(headers).index("weather station")
                table = frame.iloc[position + 1:, [pc, st]].copy()
                table.columns = ["postcode_district", "station_name"]
                tables.append(table)
                print(f"Official table: sheet={name}, header row={position + 1}")
                break

    if not tables:
        raise ValueError("Official postcode table not found")

    official = pd.concat(tables, ignore_index=True).dropna()
    for column in official.columns:
        official[column] = official[column].astype(str).str.strip()
    official["postcode_district"] = official["postcode_district"].str.upper()
    official = official[
        official["postcode_district"].str.fullmatch(r"[A-Z]{1,2}\d{1,2}[A-Z]?")
    ].drop_duplicates()
    if official["postcode_district"].duplicated().any():
        raise ValueError("Conflicting official station assignments")

    primary = pd.read_csv(OUT / "lsoa_to_district.csv", dtype=str)
    postcodes = pd.read_csv(OUT / "postcode_lsoa_gm.csv", dtype=str)
    coords = pd.read_csv(RAW / "station_coordinates.csv")
    if primary["lsoa21cd"].isna().any() or primary["lsoa21cd"].duplicated().any():
        raise ValueError("Missing or duplicate LSOA IDs")
    for frame in [primary, postcodes]:
        frame["postcode_district"] = (
            frame["postcode_district"].str.upper()
            .str.replace(r"\s+", "", regex=True)
        )

    district = official[
        official["postcode_district"].isin(primary["postcode_district"])
    ].merge(coords, on="station_name", how="left", validate="many_to_one")

    result = primary[["lsoa21cd", "postcode_district"]].merge(
        official, on="postcode_district", how="left", validate="many_to_one"
    )
    result["mapping_method"] = "official_primary_district"
    result["evidence_districts"] = result["postcode_district"]

    for index in result.index[result["station_name"].isna()]:
        code = result.at[index, "lsoa21cd"]
        candidates = postcodes.loc[
            postcodes["lsoa21cd"] == code, ["postcode_district"]
        ].drop_duplicates().merge(
            official, on="postcode_district", how="inner",
            validate="one_to_one"
        )
        stations = candidates["station_name"].unique()
        if len(stations) != 1:
            raise ValueError(f"{code}: no unambiguous fallback: {stations}")

        result.at[index, "station_name"] = stations[0]
        result.at[index, "mapping_method"] = "inferred_same_lsoa"
        evidence = ", ".join(sorted(candidates["postcode_district"]))
        result.at[index, "evidence_districts"] = evidence
        print(f"Inferred LSOA {code} -> {stations[0]}, based on {evidence}")

    result = result.rename(
        columns={"postcode_district": "primary_postcode_district"}
    ).merge(coords, on="station_name", how="left", validate="many_to_one")

    if result.isna().any().any() or district.isna().any().any():
        raise ValueError("Missing values; outputs not replaced")
    for column in ["lat", "lon"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not (
        result["lat"].between(-90, 90).all()
        and result["lon"].between(-180, 180).all()
    ):
        raise ValueError("Invalid coordinates")

    district.to_csv(OUT / "district_to_station.csv", index=False)
    result.to_csv(OUT / "lsoa_to_station.csv", index=False)

    notes = RAW / "station_mapping_notes.txt"
    notes.write_text(
        "Coordinate source:\n"
        "https://www.metoffice.gov.uk/research/climate/maps-and-data/"
        "uk-synoptic-and-climate-stations\n"
        "DWP Rostherne uses Met Office Rostherne No 2 coordinates.\n"
        "district_to_station.csv contains official assignments only.\n"
        "lsoa_to_station.csv records LSOA-level inference separately.\n"
        "For unmatched primary districts, use other districts in the same "
        "LSOA only when all mapped candidates agree on one station.\n"
        "No official station assignment is asserted for M60 or M61.\n"
    )

    print(f"\nOfficial district mappings: {len(district)}")
    print(f"LSOA mappings: {len(result)}")
    print(result["mapping_method"].value_counts().to_string())
    print("Saved both CSV files; temperature data unchanged.")


if __name__ == "__main__":
    main()
