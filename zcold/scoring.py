"""[4] Risk scoring: vulnerability x cold-weather hazard -> ranked triage list.

Join: lsoa21cd -> station_name (lsoa_to_station.csv, Daniel) -> P(trigger next 7 days).

Hazard modes:
  replay (as_of='YYYY-MM-DD' in the test winters) -> hazard_test_predictions.csv (model never saw these)
  live   (as_of=None, station_forecast.csv exists) -> hazard.forecast_now() output
  mock   (fallback, clearly labelled)

CLI:  python -m zcold.scoring                 # live or mock
      python -m zcold.scoring 2023-01-10      # replay
"""
import sys
import numpy as np, pandas as pd
from .paths import PROC

FORECAST = PROC / "station_forecast.csv"
REPLAY = PROC / "hazard_test_predictions.csv"
ACT_P, PREP_P = 0.50, 0.20                 # hazard thresholds
HIGH_BANDS = {"High", "Very high"}         # vulnerability bands that qualify


def load_vulnerability():
    v = pd.read_csv(PROC / "lsoa_vulnerability.csv", dtype={"lsoa21cd": str})
    s = pd.read_csv(PROC / "lsoa_to_station.csv", dtype={"lsoa21cd": str})
    return v.merge(s[["lsoa21cd", "station_name", "mapping_method"]], on="lsoa21cd", how="left")


def load_hazard(stations, as_of=None):
    if as_of is not None:
        h = pd.read_csv(REPLAY, parse_dates=["date"])
        h = h[h.date == pd.Timestamp(as_of)]
        if h.empty:
            raise ValueError(f"No replay predictions for {as_of}; use a date in the test winters.")
        return (h.rename(columns={"p_trigger": "p_trigger_7d"})
                 [["station_name", "p_trigger_7d", "pred_tmean_7d", "label"]], f"REPLAY as of {as_of}")
    if FORECAST.exists():
        return pd.read_csv(FORECAST), "LIVE forecast"
    rng = np.random.default_rng(7)
    return (pd.DataFrame({"station_name": stations,
                          "p_trigger_7d": rng.uniform(0.3, 0.8, len(stations)).round(2),
                          "pred_tmean_7d": np.nan}), "MOCK - not real")


def score(vuln, hazard):
    df = vuln.merge(hazard, on="station_name", how="left")
    df["risk_score"] = (df["vuln_index"] * df["p_trigger_7d"]).round(1)
    high_v = df["vuln_band"].isin(HIGH_BANDS)
    df["priority"] = np.select(
        [high_v & (df.p_trigger_7d >= ACT_P), high_v & (df.p_trigger_7d >= PREP_P)],
        ["1 - Act now", "2 - Prepare"], default="3 - Monitor")
    df["alone_66plus_hh"] = (df["pct_alone_66plus"] / 100 * df["n_households"]).round(0)
    df["expected_at_risk"] = (df["p_trigger_7d"] * df["alone_66plus_hh"]).round(1)
    return df.sort_values(["priority", "expected_at_risk"], ascending=[True, False]).reset_index(drop=True)


def summary(df):
    return (df.groupby("priority").agg(lsoas=("lsoa21cd", "size"),
                                       households=("n_households", "sum"),
                                       mean_risk=("risk_score", "mean")).round(1))


def run(as_of=None):
    vuln = load_vulnerability()
    hazard, source = load_hazard(sorted(vuln.station_name.dropna().unique()), as_of)
    print(f"Hazard source: {source}\n{hazard.to_string(index=False)}\n")
    df = score(vuln, hazard)
    print("Triage summary:\n", summary(df).to_string())
    cols = ["lsoa21nm", "station_name", "risk_score", "vuln_index", "p_trigger_7d",
            "archetype", "driver_1", "n_households", "priority"]
    print("\nTop 10 priority areas:\n", df[cols].head(10).to_string(index=False))
    out = PROC / ("lsoa_risk.csv" if as_of is None else f"lsoa_risk_replay_{as_of}.csv")
    df.to_csv(out, index=False)
    print("\nSaved", out)
    return df


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
