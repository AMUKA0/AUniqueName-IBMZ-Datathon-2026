"""Train a first ERA5 temperature forecaster for zcold (not a DWP decision model).

Run inside the notebook container after uploading this file to zcold/src/:
    /opt/venv/bin/python -u /home/ibm-user/shared/zcold/src/train_lstm.py

No dependencies are installed. Existing data files are read, never rewritten.
NNPA acceleration and ONNX export require separate verification.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import platform
import time

import numpy as np
import pandas as pd

FEATURES = ["tmean", "tmin", "tmax", "sin_doy", "cos_doy"]
TARGETS = [f"target_day_{day}" for day in range(1, 8)]
VAL_START = pd.Timestamp("2020-11-01")
TEST_START = pd.Timestamp("2022-11-01")
TEST_END = pd.Timestamp("2026-03-31")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trigger_from_daily(history6, future7, origins):
    """Reproduce ERA5 labels, requiring all seven window days to be in Nov-Mar."""
    history6 = np.asarray(history6, dtype=np.float64)
    future7 = np.asarray(future7, dtype=np.float64)
    temperatures = np.concatenate([history6, future7], axis=1)
    origins = np.asarray(origins, dtype="datetime64[D]")
    dates = origins[:, None] + np.arange(-5, 8).astype("timedelta64[D]")
    months = dates.astype("datetime64[M]").astype(np.int64) % 12 + 1
    winter = (months >= 11) | (months <= 3)
    trigger = np.zeros(len(origins), dtype=bool)
    for offset in range(7):
        mean = np.round(temperatures[:, offset:offset + 7].mean(axis=1), 12)
        eligible = winter[:, offset:offset + 7].all(axis=1)
        trigger |= eligible & (mean <= 0)
    return trigger.astype(int)


def load_samples(root):
    processed = root / "data/processed"
    temperatures = pd.read_csv(
        processed / "station_daily_temp.csv", parse_dates=["date"]
    )
    manifest = pd.read_csv(
        processed / "forecast_samples.csv",
        parse_dates=["date", "input_start", "target_start", "target_end"],
    )
    required = {
        "station_name", "date", "input_start", "target_start", "target_end",
        "split", "cwp_trigger_next_7d", *TARGETS,
    }
    require(required.issubset(manifest.columns), "Incomplete forecast_samples.csv")
    require(not manifest[list(required)].isna().any().any(), "Missing sample values")
    require(not manifest.duplicated(["station_name", "date"]).any(), "Duplicate samples")
    require(set(manifest["split"]) == {"train", "validation", "test"}, "Unexpected splits")
    require(set(manifest["station_name"]) == set(temperatures["station_name"]),
            "Temperature and sample station names differ")

    manifest = manifest.sort_values(["date", "station_name"]).reset_index(drop=True)
    require((manifest["input_start"] == manifest["date"] - pd.Timedelta(days=29)).all(),
            "Input windows must end at the forecast origin and contain 30 days")
    require((manifest["target_start"] == manifest["date"] + pd.Timedelta(days=1)).all(),
            "Targets must start one day after the forecast origin")
    require((manifest["target_end"] == manifest["date"] + pd.Timedelta(days=7)).all(),
            "Targets must end seven days after the forecast origin")

    masks = {name: manifest["split"].eq(name).to_numpy()
             for name in ["train", "validation", "test"]}
    train = manifest.loc[masks["train"]]
    val = manifest.loc[masks["validation"]]
    test = manifest.loc[masks["test"]]
    require((train["target_end"] < VAL_START).all(), "Training targets cross validation")
    require(((val["date"] >= VAL_START) & (val["target_end"] < TEST_START)).all(),
            "Validation dates cross split boundaries")
    require(((test["date"] >= TEST_START) & (test["target_end"] <= TEST_END)
             & test["date"].dt.month.isin([11, 12, 1, 2, 3])).all(),
            "Unexpected test dates")

    # Float64 reconstruction avoids label changes from premature rounding.
    X = np.empty((len(manifest), 30, 5), dtype=np.float64)
    y = np.empty((len(manifest), 7), dtype=np.float64)
    for station, samples in manifest.groupby("station_name"):
        daily = temperatures[temperatures["station_name"] == station].sort_values("date")
        daily = daily.reset_index(drop=True)
        dates = pd.DatetimeIndex(daily["date"])
        require(dates.equals(pd.date_range(dates.min(), dates.max())),
                f"{station}: missing or duplicate temperature dates")
        values = daily[["tmean", "tmin", "tmax"]].to_numpy(dtype=np.float64)
        require(np.isfinite(values).all(), f"{station}: non-finite temperatures")
        require(((values[:, 1] <= values[:, 0]) & (values[:, 0] <= values[:, 2])).all(),
                f"{station}: inconsistent temperature ordering")
        doy = dates.dayofyear.to_numpy()
        features = np.column_stack([
            values, np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ])
        positions = dates.get_indexer(samples["date"])
        require(((positions >= 29) & (positions + 7 < len(daily))).all(),
                f"{station}: incomplete input or target windows")
        X[samples.index] = features[positions[:, None] + np.arange(-29, 1)]
        y[samples.index] = values[positions[:, None] + np.arange(1, 8), 0]
        for split in masks:
            require((samples["split"] == split).any(), f"{station}: missing {split}")

    require(np.allclose(y, manifest[TARGETS].to_numpy(dtype=float), atol=1e-6, rtol=0),
            "Manifest targets disagree with station_daily_temp.csv")
    reconstructed = trigger_from_daily(X[:, -6:, 0], y, manifest["date"])
    labels = manifest["cwp_trigger_next_7d"].to_numpy(dtype=int)
    require(np.array_equal(reconstructed, labels),
            "Stored labels disagree with reconstructed ERA5 rolling-window labels")
    return manifest, X, y, masks


def metrics(truth, prediction, actual_event, predicted_event):
    error = prediction - truth
    tp = int(((actual_event == 1) & (predicted_event == 1)).sum())
    fp = int(((actual_event == 0) & (predicted_event == 1)).sum())
    fn = int(((actual_event == 1) & (predicted_event == 0)).sum())
    tn = int(((actual_event == 0) & (predicted_event == 0)).sum())
    return {
        "samples": len(truth),
        "daily_mae_c": float(np.abs(error).mean()),
        "daily_rmse_c": float(np.sqrt(np.square(error).mean())),
        "next7_mean_mae_c": float(np.abs(error.mean(axis=1)).mean()),
        "bias_c": float(error.mean()),
        "actual_positive_days": int(actual_event.sum()),
        "predicted_positive_days": int(predicted_event.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "proxy_precision": tp / (tp + fp) if tp + fp else np.nan,
        "proxy_recall": tp / (tp + fn) if tp + fn else np.nan,
        "proxy_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require(args.epochs > 0 and args.batch_size > 0, "Epochs and batch size must be positive")

    manifest, X, y, masks = load_samples(args.root)
    print(manifest.groupby(["split", "station_name"]).agg(
        samples=("date", "size"), first_date=("date", "min"),
        last_date=("date", "max"), positive_days=("cwp_trigger_next_7d", "sum"),
    ).to_string(), flush=True)
    print("PASS: input windows, targets, chronology and proxy labels verified.", flush=True)
    if args.check_only:
        print("Check only: no model trained and no output files created.")
        return

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(args.seed)
    run = args.root / "models/lstm" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {run}", flush=True)

    # Scalers see training inputs only; validation/test data never fit preprocessing.
    train_values = X[masks["train"]].reshape(-1, len(FEATURES))
    centre = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    pd.DataFrame({"feature": FEATURES, "mean": centre, "std": scale}).to_csv(
        run / "scaler.csv", index=False
    )
    normal_X = ((X - centre) / scale).astype(np.float32)
    normal_y = ((y - centre[0]) / scale[0]).astype(np.float32)
    winter = manifest["date"].dt.month.isin([11, 12, 1, 2, 3]).to_numpy()
    val_winter = masks["validation"] & winter
    require(val_winter.any(), "No winter validation samples")

    def dataset(mask, training=False):
        ds = tf.data.Dataset.from_tensor_slices((normal_X[mask], normal_y[mask]))
        if training:
            # Only batch order is shuffled; split membership remains chronological.
            ds = ds.shuffle(int(mask.sum()), seed=args.seed, reshuffle_each_iteration=True)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        return ds.batch(args.batch_size).with_options(options).prefetch(1)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(30, len(FEATURES)), name="weather_history"),
        tf.keras.layers.LSTM(32, name="temperature_lstm"),
        tf.keras.layers.Dense(7, name="next_seven_daily_means"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(run / "best.keras"), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(run / "history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    started = time.perf_counter()
    history = model.fit(
        dataset(masks["train"], training=True),
        validation_data=dataset(val_winter),
        epochs=args.epochs, callbacks=callbacks, verbose=2,
    )
    elapsed = time.perf_counter() - started
    require(np.isfinite(history.history["loss"]).all()
            and np.isfinite(history.history["val_loss"]).all(),
            "Non-finite training/validation losses; do not use this run")
    # Load the checkpoint selected only by winter validation loss.
    model = tf.keras.models.load_model(str(run / "best.keras"), compile=False)

    @tf.function(input_signature=[tf.TensorSpec([None, 30, 5], tf.float32)])
    def infer(batch):
        return model(batch, training=False)

    metric_rows, prediction_frames = [], []
    labels = manifest["cwp_trigger_next_7d"].to_numpy(dtype=int)
    for split in ["validation", "test"]:
        mask = masks[split]
        positions = np.flatnonzero(mask)
        batches = [infer(normal_X[positions[i:i + args.batch_size]]).numpy()
                   for i in range(0, len(positions), args.batch_size)]
        prediction = np.concatenate(batches).astype(np.float64) * scale[0] + centre[0]
        require(np.isfinite(prediction).all(), "Non-finite predictions")
        baseline = np.repeat(X[mask, -1, 0, None], 7, axis=1)
        metadata = manifest.loc[mask].reset_index(drop=True)
        histories = X[mask, -6:, 0]
        record = metadata.copy()
        record["actual_next7_mean"] = y[mask].mean(axis=1)
        model_outputs = {"lstm": prediction, "persistence": baseline}
        for name, forecast in model_outputs.items():
            event = trigger_from_daily(histories, forecast, metadata["date"])
            record[f"{name}_proxy_trigger"] = event
            for day in range(1, 8):
                record[f"{name}_day_{day}"] = forecast[:, day - 1]
            record[f"{name}_next7_mean"] = forecast.mean(axis=1)
            scopes = {"all": np.ones(len(metadata), dtype=bool)}
            if split == "validation":
                scopes["winter"] = winter[mask]
            for scope, scope_mask in scopes.items():
                for station in ["ALL", *sorted(metadata["station_name"].unique())]:
                    select = scope_mask.copy()
                    if station != "ALL":
                        select &= metadata["station_name"].eq(station).to_numpy()
                    if select.any():
                        metric_rows.append({
                            "split": split, "scope": scope, "station_name": station,
                            "model": name,
                            **metrics(y[mask][select], forecast[select],
                                      labels[mask][select], event[select]),
                        })
        prediction_frames.append(record)
    report = pd.DataFrame(metric_rows)
    report.to_csv(run / "metrics.csv", index=False)
    pd.concat(prediction_frames).to_csv(
        run / "predictions.csv", index=False, date_format="%Y-%m-%d"
    )
    manifest.to_csv(run / "sample_manifest.csv", index=False, date_format="%Y-%m-%d")

    package_versions = {}
    for name in ["tensorflow", "numpy", "pandas", "ibm-zdnn-plugin", "tf2onnx"]:
        try:
            package_versions[name] = version(name)
        except PackageNotFoundError:
            package_versions[name] = "not installed"
    notes = [
        "Task: forecast ERA5 daily mean temperature for t+1 through t+7.",
        "Input: t-29 through t, features in scaler.csv order.",
        "Season features: sin/cos(2*pi*dayofyear/365.25).",
        "Output: 7 normalized daily means. Celsius = output*scaler[tmean].std + mean.",
        "Baseline: repeat the last observed tmean for every future day.",
        "Proxy trigger: use predicted ERA5 tmean and six observed past days; full windows in Nov-Mar.",
        "No probability calibration, official DWP decision prediction or entitlement assessment.",
        "DWP diagnostics found ERA5 and station-observation disagreements; labels remain ERA5 proxies.",
        "Chronological splits; preprocessing fitted on training inputs only.",
        "Checkpoint/early stopping selected on winter validation MSE only; no test tuning.",
        "Test data have overlapping windows; samples are not independent events.",
        "NNPA presence/plugin installation are not proof that this model uses NNPA.",
        "Acceleration benchmarking and ONNX export have not been performed by this script.",
        f"Seed: {args.seed}; batch size: {args.batch_size}; maximum epochs: {args.epochs}",
        f"Training seconds (not inference benchmark): {elapsed:.3f}",
        f"Architecture: {platform.machine()}; TensorFlow devices: {tf.config.list_physical_devices()}",
        f"Package versions: {package_versions}",
        "Sources: https://keras.io/api/layers/recurrent_layers/lstm/",
        "https://keras.io/api/callbacks/early_stopping/",
    ]
    for path in [args.root / "data/processed/station_daily_temp.csv",
                 args.root / "data/processed/forecast_samples.csv", Path(__file__)]:
        notes.append(f"SHA256 {path.name}: {hashlib.sha256(path.read_bytes()).hexdigest()}")
    (run / "run_notes.txt").write_text("\n".join(notes) + "\n")
    print("\nTEST RESULTS (ERA5 targets, not official DWP triggers):", flush=True)
    print(report[report["split"] == "test"][[
        "station_name", "model", "samples", "daily_mae_c", "next7_mean_mae_c",
        "actual_positive_days", "predicted_positive_days", "proxy_precision", "proxy_recall",
    ]].to_string(index=False), flush=True)
    print(f"\nSaved model, scaler, history, metrics and predictions to: {run}", flush=True)


if __name__ == "__main__":
    main()
