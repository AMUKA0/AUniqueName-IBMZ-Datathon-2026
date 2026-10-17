"""Diagnose and benchmark the saved zcold LSTM in fresh TensorFlow processes.

Configuration labels describe requested execution paths, NOT proof of NNPA use.
The diagnostic log must show calls from this model to zDNN before claiming
hardware acceleration. Diagnostic/precheck logging is disabled for timed runs.
IBM reference: https://github.com/IBM/ibmz-accelerated-for-tensorflow
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
import hashlib
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

FEATURES = ["tmean", "tmin", "tmax", "sin_doy", "cos_doy"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inputs(root, run, count):
    scaler = pd.read_csv(run / "scaler.csv").set_index("feature").loc[FEATURES]
    mean = scaler["mean"].to_numpy(dtype=float)
    std = scaler["std"].to_numpy(dtype=float)
    require(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(),
            "Invalid scaler")
    # Reuse the original run manifest so a later regenerated split cannot change the benchmark.
    manifest = pd.read_csv(run / "sample_manifest.csv", parse_dates=["date"])
    samples = manifest[manifest["split"] == "test"].sort_values(
        ["date", "station_name"]
    ).head(count).reset_index(drop=True)
    require(len(samples) == count, "Not enough held-out samples for requested batch sizes")
    source = pd.read_csv(root / "data/processed/station_daily_temp.csv", parse_dates=["date"])
    result = np.empty((count, 30, 5), dtype=np.float64)
    for station, subset in samples.groupby("station_name"):
        daily = source[source["station_name"] == station].sort_values("date")
        dates = pd.DatetimeIndex(daily["date"])
        require(dates.equals(pd.date_range(dates.min(), dates.max())),
                f"{station}: temperature dates are not continuous")
        doy = dates.dayofyear.to_numpy()
        features = np.column_stack([
            daily[["tmean", "tmin", "tmax"]].to_numpy(dtype=float),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        ])
        positions = dates.get_indexer(subset["date"])
        require((positions >= 29).all(), "Missing forecast origin or input history")
        result[subset.index] = features[positions[:, None] + np.arange(-29, 1)]
    result = ((result - mean) / std).astype(np.float32)
    require(np.isfinite(result).all(), "Non-finite normalized inputs")
    return result, samples, float(mean[0]), float(std[0])


def configure(device, diagnostic):
    # Must run before importing TensorFlow or loading any model.
    if device == "cpu_requested":
        os.environ["NNPA_DEVICES"] = "0"
    else:
        os.environ.pop("NNPA_DEVICES", None)
    os.environ["ZDNN_ENABLE_PRECHECK"] = "true" if diagnostic else "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" if diagnostic else "2"
    os.environ.pop("TF_CPP_MAX_VLOG_LEVEL", None)
    if diagnostic:
        os.environ["TF_CPP_VMODULE"] = "remapper=1"
    else:
        os.environ.pop("TF_CPP_VMODULE", None)


def worker(args, root, run, out):
    diagnostic = args.phase == "diagnose"
    configure(args.device, diagnostic)
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    print(f"Requested configuration: {args.device}", flush=True)
    print("NNPA_DEVICES:", os.environ.get("NNPA_DEVICES", "unset (plugin default)"), flush=True)
    print("ZDNN_ENABLE_PRECHECK:", os.environ["ZDNN_ENABLE_PRECHECK"], flush=True)
    print("TensorFlow:", tf.__version__, flush=True)
    try:
        print("ibm-zdnn-plugin:", version("ibm-zdnn-plugin"), flush=True)
        native = [str(distribution("ibm-zdnn-plugin").locate_file(path))
                  for path in distribution("ibm-zdnn-plugin").files or []
                  if "_z_ops" in str(path) and str(path).endswith(".so")]
        print("Plugin native libraries:", native, flush=True)
    except PackageNotFoundError:
        print("ibm-zdnn-plugin: NOT INSTALLED", flush=True)

    X, metadata, target_mean, target_std = inputs(root, run, max(args.batches))
    model = tf.keras.models.load_model(str(run / "best.keras"), compile=False)
    require(tuple(model.input_shape[1:]) == (30, 5), "Unexpected model input shape")
    require(model.output_shape[-1] == 7, "Unexpected model output shape")

    @tf.function(input_signature=[tf.TensorSpec([None, 30, 5], tf.float32)])
    def infer(batch):
        return model(batch, training=False)

    if diagnostic:
        print("BEGIN SAVED LSTM MODEL DIAGNOSTIC; no unrelated control model", flush=True)
        for size in sorted(set([1, min(32, len(X))])):
            prediction = infer(tf.convert_to_tensor(X[:size])).numpy()
            require(prediction.shape == (size, 7) and np.isfinite(prediction).all(),
                    "Invalid diagnostic model output")
            print(f"Model inference completed: batch={size}, shape={prediction.shape}", flush=True)
        print("END SAVED LSTM MODEL DIAGNOSTIC", flush=True)
        print("Inspect runtime calls in this log; package presence alone is not NNPA proof.", flush=True)
        return

    rows, raw_rows = [], []
    for size in args.batches:
        tensor = tf.convert_to_tensor(X[:size])
        # Tracing, optimization and warmup are excluded from the measurement.
        for _ in range(args.warmup):
            infer(tensor).numpy()
        durations = []
        for iteration in range(args.iterations):
            started = time.perf_counter_ns()
            prediction = infer(tensor).numpy()  # Materialization synchronizes completion.
            elapsed_ns = time.perf_counter_ns() - started
            durations.append(elapsed_ns / 1_000_000)
            raw_rows.append({"configuration": args.device, "batch_size": size,
                             "iteration": iteration + 1, "latency_ms": durations[-1]})
        require(np.isfinite(prediction).all(), "Non-finite benchmark prediction")
        latencies = np.asarray(durations)
        rows.append({
            "configuration": args.device, "batch_size": size,
            "iterations": args.iterations, "warmup_calls": args.warmup,
            "best_ms": float(latencies.min()), "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "mean_ms": float(latencies.mean()),
            "samples_per_second": float(size * args.iterations * 1000 / latencies.sum()),
            "nnpa_execution_verified": "requires diagnostic log review",
        })
        print(pd.DataFrame(rows[-1:]).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(out / f"{args.device}_timings.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(out / f"{args.device}_raw_latencies.csv", index=False)

    prediction = infer(tf.convert_to_tensor(X)).numpy().astype(float)
    reference = metadata[["station_name", "date"]].copy()
    for day in range(1, 8):
        reference[f"prediction_day_{day}_c"] = prediction[:, day - 1] * target_std + target_mean
    reference.to_csv(out / f"{args.device}_predictions.csv", index=False,
                     date_format="%Y-%m-%d")


def suite(args, root, run):
    out = run / "benchmarks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    print(f"Benchmark output directory: {out}", flush=True)
    base = [sys.executable, "-u", str(Path(__file__).resolve()),
            "--root", str(root), "--run", str(run), "--output", str(out),
            "--iterations", str(args.iterations), "--warmup", str(args.warmup),
            "--batches", *map(str, args.batches)]

    for phase, device, filename in [
        ("diagnose", "plugin_default", "nnpa_diagnostic.log"),
        ("measure", "cpu_requested", "cpu_requested.log"),
        ("measure", "plugin_default", "plugin_default.log"),
    ]:
        print(f"Running {phase}: {device}", flush=True)
        with (out / filename).open("w") as log:
            process = subprocess.run(base + ["--phase", phase, "--device", device],
                                     stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            print("\n".join((out / filename).read_text(errors="replace").splitlines()[-35:]))
            raise RuntimeError(f"{phase}/{device} failed; full log: {out / filename}")

    diagnostics = (out / "nnpa_diagnostic.log").read_text(errors="replace").splitlines()
    relevant = [line for line in diagnostics if re.search(
        r"zdnn_|remapp|fallback|accelerat|nnpa|_z_ops|BEGIN SAVED|END SAVED",
        line, re.IGNORECASE,
    )]
    print("\nDIAGNOSTIC EXCERPT (not an automatic pass/fail):", flush=True)
    print("\n".join(relevant[:45]) if relevant else "No matching diagnostic lines found.")
    cpu = pd.read_csv(out / "cpu_requested_timings.csv")
    candidate = pd.read_csv(out / "plugin_default_timings.csv")
    report = pd.concat([cpu, candidate], ignore_index=True)
    report.to_csv(out / "timings.csv", index=False)
    comparison = cpu[["batch_size", "median_ms"]].rename(columns={"median_ms": "cpu_requested_median_ms"})
    comparison = comparison.merge(candidate[["batch_size", "median_ms"]].rename(
        columns={"median_ms": "plugin_default_median_ms"}), on="batch_size", validate="one_to_one")
    comparison["cpu_to_default_median_ratio"] = (
        comparison["cpu_requested_median_ms"] / comparison["plugin_default_median_ms"]
    )
    comparison.to_csv(out / "configuration_comparison.csv", index=False)
    a = pd.read_csv(out / "cpu_requested_predictions.csv")
    b = pd.read_csv(out / "plugin_default_predictions.csv")
    require(a[["station_name", "date"]].equals(b[["station_name", "date"]]),
            "Execution configurations used different benchmark inputs")
    prediction_cols = [f"prediction_day_{day}_c" for day in range(1, 8)]
    delta = np.abs(a[prediction_cols].to_numpy() - b[prediction_cols].to_numpy())
    drift = pd.DataFrame([{"samples_compared": len(a),
                           "max_abs_prediction_difference_c": float(delta.max()),
                           "mean_abs_prediction_difference_c": float(delta.mean())}])
    drift.to_csv(out / "prediction_difference.csv", index=False)
    notes = [
        "Configuration comparison, not automatically verified CPU-versus-NNPA speedup.",
        "CPU requested: NNPA_DEVICES=0. Plugin default: NNPA_DEVICES unset.",
        "Every configuration runs in a separate fresh Python/TensorFlow process.",
        "NNPA diagnostic: ZDNN_ENABLE_PRECHECK=true; TF_CPP_VMODULE=remapper=1.",
        "Timed runs disable prechecks and remapper logging; 2 intra/inter-op threads.",
        "Times include tf.function dispatch and output materialization (.numpy).",
        "Model loading, input preparation, tracing and warmup are excluded.",
        "Latency is per batch; throughput is samples per second across repeated calls.",
        "Same first held-out samples are reused; this is not end-to-end live-request timing.",
        "Prediction differences cover benchmark inputs only, not full-test accuracy.",
        "Precheck logs identify calls; inspect operation names, return status and fallback.",
        "Accelerating Dense/MatMul alone does not establish fused LSTM acceleration.",
        "Timings are an initial sequential comparison; host contention can affect them.",
        "Reference: https://github.com/IBM/ibmz-accelerated-for-tensorflow",
    ]
    for file in [run / "best.keras", run / "scaler.csv", run / "sample_manifest.csv",
                 root / "data/processed/station_daily_temp.csv", Path(__file__)]:
        notes.append(f"SHA256 {file.name}: {hashlib.sha256(file.read_bytes()).hexdigest()}")
    (out / "benchmark_notes.txt").write_text("\n".join(notes) + "\n")
    print("\nTIMINGS:", flush=True)
    print(report[["configuration", "batch_size", "best_ms", "median_ms", "p95_ms",
                  "samples_per_second"]].to_string(index=False))
    print("\nPREDICTION DIFFERENCE (Celsius):")
    print(drift.to_string(index=False))
    print(f"\nSaved all logs and CSVs to: {out}", flush=True)
    print("NNPA use remains unverified until the diagnostic log is reviewed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path, required=True,
                        help="Saved training run directory, absolute or relative to project root")
    parser.add_argument("--phase", choices=["suite", "diagnose", "measure"], default="suite")
    parser.add_argument("--device", choices=["cpu_requested", "plugin_default"], default="plugin_default")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128, 256])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(args.iterations > 0 and args.warmup > 0 and min(args.batches) > 0,
            "Iteration, warmup and batch counts must be positive")
    args.batches = sorted(set(args.batches))
    root = args.root.resolve()
    run = (args.run if args.run.is_absolute() else root / args.run).resolve()
    require((run / "best.keras").is_file(), f"Saved model not found: {run / 'best.keras'}")
    if args.phase == "suite":
        suite(args, root, run)
    else:
        require(args.output is not None, "Worker requires --output")
        args.output.mkdir(parents=True, exist_ok=True)
        worker(args, root, run, args.output)


if __name__ == "__main__":
    main()
