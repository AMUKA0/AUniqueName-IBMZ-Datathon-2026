from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"
VAL_START = pd.Timestamp("2020-11-01")
TEST_START = pd.Timestamp("2022-11-01")
TEST_END = pd.Timestamp("2026-03-31")


def main():
    temperatures = pd.read_csv(
        OUT / "station_daily_temp.csv", parse_dates=["date"]
    )
    labels = pd.read_csv(
        OUT / "cwp_labels.csv", parse_dates=["date"]
    )
    outputs = []

    for station, group in temperatures.groupby("station_name"):
        group = group.sort_values("date").reset_index(drop=True)
        expected = pd.date_range(group["date"].min(), group["date"].max())
        if not pd.DatetimeIndex(group["date"]).equals(expected):
            raise ValueError(f"{station}: dates are not continuous")
        if group["tmean"].isna().any():
            raise ValueError(f"{station}: missing temperatures")

        samples = pd.DataFrame({
            "station_name": station,
            "date": group["date"],
            "input_start": group["date"].shift(29),
            "target_start": group["date"].shift(-1),
            "target_end": group["date"].shift(-7),
        })
        targets = []
        for day in range(1, 8):
            column = f"target_day_{day}"
            samples[column] = group["tmean"].shift(-day)
            targets.append(column)

        samples = samples.dropna().copy()
        samples["target_7d_mean"] = samples[targets].mean(axis=1)
        samples["split"] = "unused"

        train = samples["target_end"] < VAL_START
        validation = (
            (samples["date"] >= VAL_START)
            & (samples["target_end"] < TEST_START)
        )
        test = (
            (samples["date"] >= TEST_START)
            & (samples["target_end"] <= TEST_END)
            & samples["date"].dt.month.isin([11, 12, 1, 2, 3])
        )
        samples.loc[train, "split"] = "train"
        samples.loc[validation, "split"] = "validation"
        samples.loc[test, "split"] = "test"
        samples = samples[samples["split"] != "unused"]

        samples = samples.merge(
            labels[["station_name", "date", "cwp_trigger_next_7d"]],
            on=["station_name", "date"],
            how="left",
            validate="one_to_one",
        )
        if samples.isna().any().any():
            raise ValueError(f"{station}: incomplete samples or labels")
        outputs.append(samples)

    result = pd.concat(outputs, ignore_index=True)
    destination = OUT / "forecast_samples.csv"

    print(result.groupby(["split", "station_name"]).agg(
        samples=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        positive_labels=("cwp_trigger_next_7d", "sum"),
    ).to_string())

    result.to_csv(destination, index=False, date_format="%Y-%m-%d")
    print(f"\nSaved: {destination}")
    print("Input temperatures remain in station_daily_temp.csv.")
    print("No model has been trained by this script.")


if __name__ == "__main__":
    main()
