from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
import json
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/open_meteo"
OUT = ROOT / "data/processed"
RAW.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("1991-01-01")
END = pd.Timestamp("2026-10-10")
API = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = {
    "temperature_2m_mean": "tmean",
    "temperature_2m_min": "tmin",
    "temperature_2m_max": "tmax",
}


def download(url):
    for attempt in range(4):
        try:
            with urlopen(url, timeout=120) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise RuntimeError(error.read().decode()) from error
            if attempt == 3:
                raise
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
        print("Retrying...", flush=True)
        time.sleep(10 * (attempt + 1))


def main():
    stations = pd.read_csv(
        OUT / "district_to_station.csv"
    )[["station_name", "lat", "lon"]].drop_duplicates()

    if stations.isna().any().any():
        raise ValueError("Missing station coordinates")
    if stations["station_name"].duplicated().any():
        raise ValueError("Conflicting station coordinates")

    frames = []
    for station in stations.itertuples(index=False):
        for year in range(START.year, END.year + 1):
            first = max(START, pd.Timestamp(year, 1, 1))
            last = min(END, pd.Timestamp(year, 12, 31))
            params = {
                "latitude": station.lat,
                "longitude": station.lon,
                "start_date": first.strftime("%Y-%m-%d"),
                "end_date": last.strftime("%Y-%m-%d"),
                "daily": ",".join(VARIABLES),
                "models": "era5",
                "timezone": "Europe/London",
                "temperature_unit": "celsius",
            }
            url = API + "?" + urlencode(params)
            cache = RAW / (
                f"{station.station_name.replace(' ', '_')}"
                f"_{first:%Y%m%d}_{last:%Y%m%d}_era5.json"
            )
            if cache.exists():
                payload = json.loads(cache.read_text())
            else:
                print(f"Downloading {station.station_name} {year}", flush=True)
                payload = download(url)
                if payload.get("error"):
                    raise RuntimeError(payload)
                cache.write_text(json.dumps(payload))
                cache.with_suffix(".url.txt").write_text(url + "\n")
                time.sleep(0.5)

            frame = pd.DataFrame(payload["daily"]).rename(
                columns={"time": "date", **VARIABLES}
            )
            frame = frame[["date", "tmean", "tmin", "tmax"]]
            dates = pd.to_datetime(frame["date"])
            expected = pd.date_range(first, last)
            if not pd.DatetimeIndex(dates).equals(expected):
                raise ValueError(f"Missing or unordered dates: {cache}")
            if frame.isna().any().any():
                raise ValueError(f"Missing temperatures: {cache}")
            if not (
                (frame["tmin"] <= frame["tmean"])
                & (frame["tmean"] <= frame["tmax"])
            ).all():
                raise ValueError(f"Inconsistent temperatures: {cache}")

            frame.insert(0, "station_name", station.station_name)
            frames.append(frame)
            print(f"OK {station.station_name} {year}: {len(frame)} days",
                  flush=True)

    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["station_name", "date"]).any():
        raise ValueError("Duplicate station/date rows")

    print("\nSummary:")
    print(result.groupby("station_name").agg(
        days=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        lowest_tmin=("tmin", "min"),
        highest_tmax=("tmax", "max"),
    ).to_string())
    print("\nMissing values:")
    print(result.isna().sum().to_string())

    destination = OUT / "station_daily_temp.csv"
    temporary = destination.with_suffix(".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(destination)
    print(f"\nSaved: {destination}; rows={len(result)}")


if __name__ == "__main__":
    main()
