"""Scalability proof: run ColdWatch's vulnerability model for ALL 33,755 English neighbourhoods
and an LSTM forecast for every one of them, timed on IBM Z.

python -m zcold.scale      -> data/processed/national_vulnerability.csv, printed timing table
"""
import os, glob, time, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .paths import RAW, PROC, ROOT
from . import structural

TIMES = {}


def _timed(label):
    class T:
        def __enter__(s): s.t = time.perf_counter()
        def __exit__(s, *a): TIMES[label] = time.perf_counter() - s.t
    return T()


def _pick(df, wanted):
    out = {}
    for new, keys in wanted.items():
        hit = next(c for c in df.columns if all(k in str(c).lower() for k in keys))
        out[hit] = new
    return df[list(out)].rename(columns=out)


def load_national():
    """IoD 2025 (files 1 + 5) and Census 2021 (4 tables) for every English LSOA."""
    def data_sheet(path):
        sheets = pd.read_excel(path, sheet_name=None)
        return max(sheets.values(), key=len)

    ids = {"lsoa21cd": ["lsoa code"], "lsoa21nm": ["lsoa name"], "lad_name": ["local authority district name"]}
    f5 = _pick(data_sheet(RAW / "iod/iod25_file5_scores.xlsx"), {**ids,
        "income_score": ["income score"], "health_score": ["health deprivation", "score"],
        "indoors_score": ["indoors", "score"], "idaopi_score": ["older people", "score"],
        "imd_score": ["index of multiple deprivation", "score"]})
    df = f5[f5.lsoa21cd.astype(str).str.startswith("E01")].copy()

    def census(table):
        c = pd.read_csv(glob.glob(str(RAW / f"census/census2021-{table}-lsoa.csv"))[0])
        code = next(x for x in c.columns if "geography code" in x.lower())
        return c[c[code].astype(str).str.startswith("E01")].set_index(code)

    def cols(c, *k):
        return [x for x in c.columns if all(s in x.lower() for s in k)]

    age, hh, dis, car = census("ts007a"), census("ts003"), census("ts038"), census("ts045")
    out = pd.DataFrame(index=df.lsoa21cd)
    out["pct_75plus"] = age[cols(age, "75 to") + cols(age, "80 to") + cols(age, "85 years and over")].sum(axis=1) / age[cols(age, "total")[0]] * 100
    out["pct_alone_66plus"] = hh[cols(hh, "one person household", "66")].sum(axis=1) / hh[cols(hh, "total")[0]] * 100
    d = [x for x in cols(dis, "disabled under the equality act") if "not disabled" not in x.lower()][:1]
    out["pct_disabled"] = dis[d].sum(axis=1) / dis[cols(dis, "total")[0]] * 100
    out["pct_no_car"] = car[cols(car, "no cars")].sum(axis=1) / car[cols(car, "total")[0]] * 100
    out["n_households"] = hh[cols(hh, "total")[0]]
    return df.merge(out.reset_index(), on="lsoa21cd", how="left").dropna()


def run():
    print("Loading national IoD 2025 + Census 2021 (I/O, not part of the model timing)...")
    with _timed("load data (I/O)"):
        nat = load_national()
    print(f"  {len(nat):,} English neighbourhoods in {nat.lad_name.nunique()} local authorities")

    with _timed("vulnerability index (33,755 areas)"):
        nat = structural.vulnerability_index(nat)

    with _timed("KMeans archetypes (k=3)"):
        X = StandardScaler().fit_transform(nat[structural.ARCH_FEATURES])
        nat["archetype_id"] = KMeans(3, n_init=10, random_state=42).fit_predict(X)

    import tensorflow as tf
    model = tf.keras.models.load_model(ROOT / "models/lstm_tmean7.keras")
    x = tf.constant(np.random.default_rng(0).normal(size=(len(nat), 30, 5)).astype("float32"))
    model(x[:10], training=False)                                   # warm-up
    with _timed("LSTM 7-day forecast, every neighbourhood (Telum AI accelerator)"):
        model(x, training=False).numpy()

    nat.to_csv(PROC / "national_vulnerability.csv", index=False)
    total_model = sum(v for k, v in TIMES.items() if "I/O" not in k)
    print("\nTiming on IBM LinuxONE:")
    print(pd.DataFrame({"stage": list(TIMES), "seconds": [round(v, 2) for v in TIMES.values()]}).to_string(index=False))
    print(f"\nModel pipeline for all of England: {total_model:.1f} s "
          f"({len(nat) / total_model:,.0f} neighbourhoods per second)")
    top = (nat.groupby("lad_name").vuln_index.mean().sort_values(ascending=False).head(10).round(1))
    print("\nLocal authorities with the highest mean cold-weather vulnerability:\n", top.to_string())
    json.dump({k: round(v, 3) for k, v in TIMES.items()} | {"neighbourhoods": len(nat)},
              open(ROOT / "models/scale_timing.json", "w"), indent=2)


if __name__ == "__main__":
    run()
