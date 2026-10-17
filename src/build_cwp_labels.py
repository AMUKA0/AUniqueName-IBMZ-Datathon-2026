from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"


def main():
    data = pd.read_csv(
        OUT / "station_daily_temp.csv",
        parse_dates=["date"],
    )
    outputs = []

    for station, group in data.groupby("station_name"):
        group = group.sort_values("date").reset_index(drop=True)
        dates = group["date"]

        expected = pd.date_range(dates.min(), dates.max())
        if not pd.DatetimeIndex(dates).equals(expected):
            raise ValueError(f"{station}: missing or duplicate dates")
        if group["tmean"].isna().any():
            raise ValueError(f"{station}: missing temperatures")

        roll7 = group["tmean"].rolling(7, min_periods=7).mean()

        # Require every day of the seven-day window to be in Nov-Mar.
        in_season = dates.dt.month.isin([11, 12, 1, 2, 3])
        eligible_window = (
            in_season.astype(int).rolling(7, min_periods=7).sum() == 7
        )
        event = ((roll7 <= 0) & eligible_window).astype(float)
        event = event.where(roll7.notna())

        # At date t, inspect windows ending on t+1 through t+7.
        future = pd.concat(
            [event.shift(-days) for days in range(1, 8)],
            axis=1,
        )
        valid = roll7.notna() & future.notna().all(axis=1)

        result = pd.DataFrame({
            "station_name": station,
            "date": dates.dt.strftime("%Y-%m-%d"),
            "roll7_mean": roll7,
            "cwp_trigger_next_7d": future.max(axis=1),
        }).loc[valid].copy()
        result["cwp_trigger_next_7d"] = (
            result["cwp_trigger_next_7d"].astype(int)
        )
        outputs.append(result)

        winter_rows = pd.to_datetime(result["date"]).dt.month.isin(
            [11, 12, 1, 2, 3]
        )
        positives = result.loc[
            winter_rows, "cwp_trigger_next_7d"
        ].sum()
        print(
            f"{station}: rows={len(result)}, "
            f"winter positive-label days={positives}, "
            f"winter rows={winter_rows.sum()}"
        )

    labels = pd.concat(outputs, ignore_index=True)
    if labels.isna().any().any():
        raise ValueError("Missing labels")

    destination = OUT / "cwp_labels.csv"
    labels.to_csv(destination, index=False)
    print(f"\nSaved: {destination}; rows={len(labels)}")
    print("Dropped first 6 and last 7 days per station.")
    print("Positive-label days are NOT payment counts.")


if __name__ == "__main__":
    main()
