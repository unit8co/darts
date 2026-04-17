"""
Debug script: historical_forecasts / backtest with MLflow autologging.

Issue (from PR #3022): historical_forecasts(retrain=True) and backtest() call
fit() internally on each iteration. Because _patched_fit runs with
manage_run=True, every internal fit() spawns its own MLflow run when no
active run exists - i.e. the typical autolog-only usage (no explicit
mlflow.start_run() wrapper).

Expected: 1 run per historical_forecasts / backtest call.
Actual:   1 run per stride iteration.

Run with:
    python examples/repl/debug_hf_mlflow.py

Then inspect in the UI:
    mlflow ui --backend-store-uri sqlite:////tmp/mlflow_debug.db
"""

# %%
import os

import mlflow

import darts.utils.mlflow as mlflow_darts
from darts.datasets import AirPassengersDataset
from darts.models import LinearRegressionModel

# ---- setup -------------------------------------------------------------------
series = AirPassengersDataset().load()

STRIDE = 4  # keep iteration count low for fast runs

DB_PATH = "/tmp/mlflow_debug.db"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("debug-hf-mlflow")
client = mlflow.tracking.MlflowClient()
exp_id = mlflow.get_experiment_by_name("debug-hf-mlflow").experiment_id


def run_count(name=None):
    runs = client.search_runs(experiment_ids=[exp_id])
    if name:
        return sum(1 for r in runs if r.info.run_name == name)
    return len(runs)


# ---- 1. normal fit WITH explicit start_run (baseline) -----------------------
print("=" * 60)
print("1. Normal fit() inside mlflow.start_run()")
mlflow_darts.autolog(disable=True)
mlflow_darts.autolog()

before = run_count()
with mlflow.start_run(run_name="fit-with-run"):
    LinearRegressionModel(lags=12).fit(series[:100])
mlflow_darts.autolog(disable=True)

print(f"   Runs created: {run_count() - before}  (expected: 1)")

# ---- 2. historical_forecasts WITH explicit start_run ------------------------
print()
print("=" * 60)
print("2. historical_forecasts(retrain=True) inside mlflow.start_run()")
mlflow_darts.autolog(disable=True)
mlflow_darts.autolog()

before = run_count()
with mlflow.start_run(run_name="hf-with-run"):
    LinearRegressionModel(lags=12).historical_forecasts(
        series,
        start=0.75,
        forecast_horizon=1,
        stride=STRIDE,
        retrain=True,
        last_points_only=True,
    )
mlflow_darts.autolog(disable=True)

print(f"   Runs created: {run_count() - before}  (expected: 1)")

# ---- 3. historical_forecasts WITHOUT start_run (autolog-only usage) ---------
print()
print("=" * 60)
print("3. historical_forecasts(retrain=True) with autolog only (no start_run)")
mlflow_darts.autolog(disable=True)
mlflow_darts.autolog()

before = run_count()
LinearRegressionModel(lags=12).historical_forecasts(
    series,
    start=0.75,
    forecast_horizon=1,
    stride=STRIDE,
    retrain=True,
    last_points_only=True,
)
mlflow_darts.autolog(disable=True)

created = run_count() - before
print(f"   Runs created: {created}  (expected: 1)")
print(f"   -> Each stride iteration spawned its own run: {created > 1}")

# ---- 4. backtest WITHOUT start_run ------------------------------------------
print()
print("=" * 60)
print("4. backtest() with autolog only (no start_run)")
mlflow_darts.autolog(disable=True)
mlflow_darts.autolog()

before = run_count()
LinearRegressionModel(lags=12).backtest(
    series,
    start=0.75,
    forecast_horizon=1,
    stride=STRIDE,
    retrain=True,
)
mlflow_darts.autolog(disable=True)

created = run_count() - before
print(f"   Runs created: {created}  (expected: 1)")
print(f"   -> Each stride iteration spawned its own run: {created > 1}")

# Assert backtest_mape was logged
all_runs = client.search_runs(experiment_ids=[exp_id])
bt_run = sorted(all_runs, key=lambda x: x.info.start_time)[-1]
metrics = bt_run.data.metrics
assert "backtest_mape" in metrics, f"Expected backtest_mape in metrics, got: {metrics}"
print(f"   -> backtest_mape = {metrics['backtest_mape']:.4f}  (PASS)")

# ---- 5. backtest with reduction=None → per-window metrics -------------------
print()
print("=" * 60)
print("5. backtest(reduction=None) — per-window MAPE logged as separate metrics")
mlflow_darts.autolog(disable=True)
mlflow_darts.autolog()

before = run_count()
per_window = LinearRegressionModel(lags=12).backtest(
    series,
    start=0.75,
    forecast_horizon=1,
    stride=STRIDE,
    retrain=True,
    reduction=None,
)
mlflow_darts.autolog(disable=True)

created = run_count() - before
print(f"   Runs created: {created}  (expected: 1)")

all_runs = client.search_runs(experiment_ids=[exp_id])
pw_run = sorted(all_runs, key=lambda x: x.info.start_time)[-1]
history = client.get_metric_history(pw_run.info.run_id, "backtest_mape")
assert history, f"Expected backtest_mape steps in metric history, got nothing"
print(f"   -> backtest_mape logged across {len(history)} steps (windows)")

# Plot per-window MAPE
import matplotlib.pyplot as plt
import numpy as np

window_nums = [p.step for p in sorted(history, key=lambda p: p.step)]
values = [p.value for p in sorted(history, key=lambda p: p.step)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(window_nums, values, marker="o", linewidth=1.5, label="MAPE per window")
ax.axhline(np.mean(values), color="red", linestyle="--", linewidth=1, label=f"mean = {np.mean(values):.2f}%")
ax.set_xlabel("Backtest window index")
ax.set_ylabel("MAPE (%)")
ax.set_title("Per-window MAPE (backtest, reduction=None)")
ax.legend()
fig.tight_layout()
plt.savefig("/tmp/backtest_per_window_mape.png", dpi=150)
print("   -> Plot saved to /tmp/backtest_per_window_mape.png")
plt.show()

# ---- summary -----------------------------------------------------------------
print()
print("=" * 60)
print("Summary of ALL runs in experiment:")
all_runs = client.search_runs(experiment_ids=[exp_id])
for r in sorted(all_runs, key=lambda x: x.info.start_time):
    m = r.data.metrics
    if "backtest_mape" in m:
        h = client.get_metric_history(r.info.run_id, "backtest_mape")
        if len(h) > 1:
            metric_str = f"  backtest_mape ({len(h)} steps, mean={sum(p.value for p in h)/len(h):.4f})"
        else:
            metric_str = f"  backtest_mape={m['backtest_mape']:.4f}"
    else:
        metric_str = ""
    print(f"   [{r.info.status}] {r.info.run_name!r:30s}  id={r.info.run_id[:8]}{metric_str}")
print(f"   Total: {len(all_runs)} runs")
print()
print(f"Inspect in UI: mlflow ui --backend-store-uri sqlite:///{DB_PATH}")
