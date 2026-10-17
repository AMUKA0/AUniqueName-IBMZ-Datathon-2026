"""[2]+[3] Cold-hazard model on IBM Z NNPA (TensorFlow + ibm-zdnn-plugin).

The LSTM forecasts the daily mean temperature for EACH of the next 7 days from the previous
30 days. Monte-Carlo sampling of the model's own (correlated) forecast errors then turns that
into P(trigger), using EXACTLY the label definition in cwp_labels.csv (Daniel):
  - window ending d qualifies if mean(tmean[d-6..d]) <= 0 C and all 7 days fall in Nov-Mar
  - label(t) = 1 if a qualifying window ends on any day t+1 .. t+7
  - windows may include days already observed at t (the last 6 observed days are used)

Splits (chronological, targets never cross a boundary):
  train: t < 2019-08-01 - 7d | validation: 2019-08-01 .. 2022-08-01 - 7d | test: t >= 2022-08-01
"""
import os, json, time, glob, importlib.metadata, urllib.request, urllib.parse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, pandas as pd, tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import roc_auc_score, brier_score_loss
from .paths import PROC, ROOT

LOOKBACK, H = 30, 7
TEMP_COLS = ["tmean", "tmin", "tmax"]
FEATURES = TEMP_COLS + ["doy_sin", "doy_cos"]
VAL_START = pd.Timestamp("2019-08-01")
TEST_START = pd.Timestamp("2022-08-01")
WINTER = [11, 12, 1, 2, 3]
MODEL_DIR = ROOT / "models"
N_SAMPLES = 500


# ------------------------------------------------------------------ data
def add_doy(df):
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def load_data():
    t = pd.read_csv(PROC / "station_daily_temp.csv", parse_dates=["date"])
    lab = pd.read_csv(PROC / "cwp_labels.csv", parse_dates=["date"])
    df = t.merge(lab[["station_name", "date", "cwp_trigger_next_7d"]],
                 on=["station_name", "date"], how="left")
    df = add_doy(df.sort_values(["station_name", "date"]).reset_index(drop=True))
    print(f"Stations: {sorted(df.station_name.unique())} | rows {len(df):,} "
          f"| {df.date.min().date()} .. {df.date.max().date()}")
    return df


def _scaled(g, stats):
    return np.column_stack([((g[f] - stats[f][0]) / stats[f][1]).to_numpy() if f in stats
                            else g[f].to_numpy() for f in FEATURES]).astype("float32")


def make_sequences(df, stats):
    """For each day t with 30 days of history and 7 days of future:
    X = scaled features t-29..t, Y = tmean t+1..t+7, obs6 = tmean t-5..t."""
    out = {k: [] for k in ["X", "Y", "obs6", "date", "station", "label"]}
    for st, g in df.groupby("station_name"):
        A, tm, n = _scaled(g, stats), g["tmean"].to_numpy("float32"), len(g)
        idx = np.arange(LOOKBACK - 1, n - H)
        W = sliding_window_view(A, (LOOKBACK, A.shape[1]))[:, 0]      # W[j] ends at j+LOOKBACK-1
        out["X"].append(W[idx - (LOOKBACK - 1)])
        out["Y"].append(np.stack([tm[idx + k] for k in range(1, H + 1)], axis=1))
        out["obs6"].append(np.stack([tm[idx - 5 + j] for j in range(6)], axis=1))
        out["date"].append(g["date"].to_numpy()[idx])
        out["station"].append(np.full(len(idx), st, dtype=object))
        out["label"].append(g["cwp_trigger_next_7d"].to_numpy(dtype=float)[idx])
    s = {k: np.concatenate(v) for k, v in out.items()}
    s["date"] = pd.DatetimeIndex(s["date"])
    return s


# ------------------------------------------------------------------ trigger probability
def window_eligible(dates):
    """(N, 7) bool: the 7-day window ending t+k (k=1..7) lies wholly in Nov-Mar."""
    dates = pd.DatetimeIndex(dates)
    el = np.ones((len(dates), H), bool)
    for k in range(1, H + 1):
        for o in range(k - 6, k + 1):
            el[:, k - 1] &= (dates + pd.Timedelta(days=o)).month.isin(WINTER)
    return el


def _qualifies(full, elig):
    """full: (..., N, 13) tmean t-5..t+7 -> (..., N) bool any qualifying window ending t+1..t+7."""
    pad = np.zeros(full.shape[:-1] + (1,))
    cs = np.concatenate([pad, np.cumsum(full, axis=-1)], axis=-1)
    means = (cs[..., 7:14] - cs[..., 0:7]) / 7.0
    return ((means <= 0) & elig).any(axis=-1)


def p_trigger(pred, obs6, elig, cov, n_samples=N_SAMPLES, seed=0):
    """Monte-Carlo P(trigger) from the LSTM forecast + its correlated error distribution."""
    rng = np.random.default_rng(seed)
    N = len(pred)
    eps = rng.multivariate_normal(np.zeros(H), cov, size=(n_samples, N))     # (S, N, 7)
    full = np.concatenate([np.broadcast_to(obs6, (n_samples, N, 6)), pred[None] + eps], axis=2)
    return _qualifies(full, elig[None]).mean(axis=0)


def p_persistence(obs6, elig):
    """Baseline: assume the next 7 days equal today's mean temperature (0/1)."""
    path = np.repeat(obs6[:, -1:], H, axis=1)
    return _qualifies(np.concatenate([obs6, path], axis=1), elig).astype(float)


# ------------------------------------------------------------------ model
def build_model():
    """Small + regularised: predicts the CHANGE from today's mean temperature (residual target),
    so persistence is the starting point and the LSTM only has to learn the departure from it."""
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(LOOKBACK, len(FEATURES))),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(H),
    ])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def _pr(y, p, thr=0.5):
    f = p >= thr
    tp, fp, fn = int((f & (y == 1)).sum()), int((f & (y == 0)).sum()), int((~f & (y == 1)).sum())
    return {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None, "TP": tp, "FP": fp, "FN": fn}


def evaluate(s, m, pred, cov):
    d, obs6, Y, lab = s["date"][m], s["obs6"][m], s["Y"][m], s["label"][m]
    winter = np.asarray(d.month.isin(WINTER))
    elig = window_eligible(d)
    p_l, p_p = p_trigger(pred, obs6, elig, cov), p_persistence(obs6, elig)

    mae = pd.DataFrame({"horizon_day": range(1, H + 1),
                        "MAE_lstm_C": np.abs(pred - Y)[winter].mean(axis=0).round(2),
                        "MAE_persistence_C": np.abs(obs6[:, -1:] - Y)[winter].mean(axis=0).round(2)})
    ok = winter & ~np.isnan(lab)
    y = lab[ok].astype(int)
    metrics = {"winter_test_days": int(ok.sum()), "positive_days": int(y.sum()),
               "brier_lstm": round(brier_score_loss(y, p_l[ok]), 4),
               "brier_persistence": round(brier_score_loss(y, p_p[ok]), 4),
               "auc_lstm": round(roc_auc_score(y, p_l[ok]), 3) if 0 < y.sum() < len(y) else None,
               "lstm_at_0.5": _pr(y, p_l[ok]), "persistence": _pr(y, p_p[ok])}
    preds = pd.DataFrame({"station_name": s["station"][m], "date": d, "p_trigger": p_l.round(3),
                          "p_persistence": p_p, "label": lab,
                          "pred_tmean_7d": pred.mean(axis=1).round(2),
                          "actual_tmean_7d": Y.mean(axis=1).round(2)})
    return metrics, mae, preds


def replay_candidates(preds, n=8):
    """For each real cold event in the test winters: the EARLIEST day the model's P(trigger)
    reached 0.5 while a qualifying window was indeed coming (label = 1). Best demo dates."""
    p = preds.sort_values(["station_name", "date"]).copy()
    p["event"] = ((p.label == 1) & (p.groupby("station_name")["label"].shift(1) != 1)).cumsum()
    hits = p[(p.label == 1) & (p.p_trigger >= 0.5)]
    first = hits.groupby(["station_name", "event"]).head(1)
    return first[["station_name", "date", "p_trigger", "p_persistence",
                  "pred_tmean_7d", "actual_tmean_7d"]].head(n)


# ------------------------------------------------------------------ IBM Z evidence
def accelerator_report():
    so = glob.glob("/opt/venv/lib*/python3.12/site-packages/tensorflow/core/kernels/_z_ops.so")
    try:
        plugin = importlib.metadata.version("ibm-zdnn-plugin")
    except Exception:
        plugin = None
    return {"nnpa_in_cpuinfo": "nnpa" in open("/proc/cpuinfo").read(),
            "ibm_zdnn_plugin": plugin, "zdnn_kernel_lib": so[0] if so else None,
            "tensorflow": tf.__version__,
            "tf_physical_devices": [d.device_type for d in tf.config.list_physical_devices()]}


def placement_check(model):
    """Logs which device each op runs on for one inference (verbose; for the evidence screenshot)."""
    tf.debugging.set_log_device_placement(True)
    model(tf.zeros((1, LOOKBACK, len(FEATURES))), training=False)
    tf.debugging.set_log_device_placement(False)


def benchmark(model, sizes=(1, 100, 1000, 10000), repeats=10):
    rows = []
    for n in sizes:
        x = tf.constant(np.random.randn(n, LOOKBACK, len(FEATURES)).astype("float32"))
        model(x, training=False)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter(); model(x, training=False).numpy()
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        rows.append({"batch": n, "median_ms": round(med * 1000, 2), "sequences_per_s": int(n / med)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ live forecast
def forecast_now(ignore_season=False):
    """Score today for every station using Open-Meteo (last 30 days + today) -> station_forecast.csv.
    Outside Nov-Mar the payment rule cannot trigger, so P is 0 unless ignore_season=True."""
    model = tf.keras.models.load_model(MODEL_DIR / "lstm_tmean7.keras")
    meta = json.load(open(MODEL_DIR / "lstm_meta.json"))
    stats, cov = meta["stats"], np.array(meta["cov"])
    st = pd.read_csv(PROC / "lsoa_to_station.csv").drop_duplicates("station_name")
    rows = []
    for _, r in st.iterrows():
        q = urllib.parse.urlencode({"latitude": r.lat, "longitude": r.lon, "past_days": 31,
                                    "forecast_days": 1, "timezone": "Europe/London",
                                    "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max"})
        dd = json.load(urllib.request.urlopen("https://api.open-meteo.com/v1/forecast?" + q, timeout=60))["daily"]
        g = pd.DataFrame({"date": pd.to_datetime(dd["time"]), "tmin": dd["temperature_2m_min"],
                          "tmax": dd["temperature_2m_max"], "tmean": dd.get("temperature_2m_mean")})
        g["tmean"] = g["tmean"].fillna((g.tmin + g.tmax) / 2)
        g = add_doy(g.dropna().tail(LOOKBACK))
        x = _scaled(g, stats)[None]
        pred = model.predict(x, verbose=0) + float(g.tmean.iloc[-1])     # residual -> absolute
        today = g.date.iloc[-1]
        elig = np.ones((1, H), bool) if ignore_season else window_eligible([today])
        p = p_trigger(pred, g.tmean.to_numpy()[-6:][None], elig, cov)[0]
        rows.append({"station_name": r.station_name, "forecast_date": today.date(),
                     "p_trigger_7d": round(float(p), 3), "pred_tmean_7d": round(float(pred.mean()), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "station_forecast.csv", index=False)
    return out


# ------------------------------------------------------------------ pipeline
def run(epochs=20, seed=42):
    tf.keras.utils.set_random_seed(seed)          # reproducible weights + Monte-Carlo
    print("IBM Z accelerator:", json.dumps(accelerator_report(), indent=2))
    df = load_data()
    tr = df[df.date < VAL_START]
    stats = {f: (float(tr[f].mean()), float(tr[f].std())) for f in TEMP_COLS}
    s = make_sequences(df, stats)
    d = s["date"]
    m_tr = np.asarray(d < VAL_START - pd.Timedelta(days=H))
    m_va = np.asarray((d >= VAL_START) & (d < TEST_START - pd.Timedelta(days=H)))
    m_te = np.asarray(d >= TEST_START)
    print(f"Sequences: train {m_tr.sum():,} | val {m_va.sum():,} | test {m_te.sum():,}")

    base = lambda m: s["obs6"][m][:, -1:]          # today's mean temp (persistence)
    model = build_model()
    t0 = time.perf_counter()
    model.fit(s["X"][m_tr], s["Y"][m_tr] - base(m_tr),
              validation_data=(s["X"][m_va], s["Y"][m_va] - base(m_va)),
              epochs=epochs, batch_size=256, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    print(f"Training time: {time.perf_counter() - t0:.0f}s")

    resid = model.predict(s["X"][m_va], batch_size=4096, verbose=0) + base(m_va) - s["Y"][m_va]
    cov = np.cov(resid.T)
    pred = model.predict(s["X"][m_te], batch_size=4096, verbose=0) + base(m_te)
    metrics, mae, preds = evaluate(s, m_te, pred, cov)

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "lstm_tmean7.keras")
    json.dump({"stats": stats, "cov": cov.tolist(), "metrics": metrics},
              open(MODEL_DIR / "lstm_meta.json", "w"), indent=2, default=str)
    preds.to_csv(PROC / "hazard_test_predictions.csv", index=False)

    print("\nForecast error by horizon (winter test days):\n", mae.to_string(index=False))
    print("\nTrigger metrics (winters 2022-23 onwards):", json.dumps(metrics, indent=2))
    print("\nReplay candidates (first day of each real cold event):\n",
          replay_candidates(preds).to_string(index=False))
    print("\nNNPA inference benchmark:\n", benchmark(model).to_string(index=False))
    print("\nSaved models/lstm_tmean7.keras, models/lstm_meta.json, processed/hazard_test_predictions.csv")


if __name__ == "__main__":
    run()
