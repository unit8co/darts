"""
MLflow (auto)-logging:  Multi-model comparison with backtest

Demonstrates:
  * One MLflow run per model — each ``model.fit()`` auto-creates its own run
    (via ``manage_run=True`` in autolog).  The run is named after the model
    class (e.g. ``LinearRegressionModel``, ``ExponentialSmoothing``).
  * Metric keys are unprefixed (``val_mape``, ``backtest_mape``, etc.) so
    the experiment-level comparison view can overlay them directly.
  * Automatic parameter and tag logging via darts autolog.
  * Inline metric logging: ``mape`` and ``rmse`` called inside an active run
    are automatically captured by the patched metric functions.
  * Backtest with pre-computed historical forecasts, multiple metric functions,
    and ``reduction=None`` so each window is logged as a step of
    ``backtest_mape`` / ``backtest_rmse`` in the MLflow UI.

Notes:
  * ``mlflow_darts.autolog()`` is required — it patches ``fit()``, the darts
    metric functions, and ``backtest()`` to make logging automatic.
  * ``manage_run=False`` is passed to ``autolog()`` so that we control the run
    lifecycle ourselves with ``mlflow.start_run()``.  This lets us keep the
    run open for predict + metrics + backtest after ``fit()`` returns.
  * ``last_points_only=False`` is used for historical forecasts so that
    ``backtest()`` receives a list of per-window forecasts and can compute one
    metric value per window (enabling the per-step chart in the UI).
    ``last_points_only=True`` would collapse all windows into a single
    TimeSeries, making the metric treat them as one window.

Run with:
    python mlflow_test_v2.py

Inspect in the UI:
    mlflow ui --backend-store-uri sqlite:////tmp/mlflow_v2.db
"""

import logging

import coolname
import mlflow
from loguru import logger

logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)

import numpy as np

import darts.metrics as darts_metrics
import darts.utils.mlflow as mlflow_darts
from darts.datasets import AirPassengersDataset
from darts.models import (
    ExponentialSmoothing,
    LinearRegressionModel,
    NBEATSModel,
    NHiTSModel,
    TCNModel,
)


def _magic(model, train, val, series, log) -> None:
    log(f"[{model_name}] Run created: run_id={run.info.run_id}")
    # ── fit ───────────────────────────────────────────────────────────
    log(f"[{model_name}] Fitting model on {len(train)} training samples")
    model.fit(train)
    # ── predict + inline metric logging ───────────────────────────────
    log(f"[{model_name}] Predicting {len(val)} steps")
    pred = model.predict(n=len(val))
    # Calls inside active MLflow run are intercepted by patched metric
    # functions and auto-logged -> val_mape, val_rmse
    val_mape = darts_metrics.mape(val, pred)
    val_rmse = darts_metrics.rmse(val, pred)
    log(f"[{model_name}] val_mape={val_mape:.4f}  val_rmse={val_rmse:.4f}")

    # ── backtest ──────────────────────────────────────────────────────
    log(
        f"[{model_name}] Computing historical forecasts "
        f"(start={BT_START}, horizon={FORECAST_HORIZON}, stride={STRIDE})"
    )
    hfc = model.historical_forecasts(
        series,
        start=BT_START,
        forecast_horizon=FORECAST_HORIZON,
        stride=STRIDE,
        retrain=True,
        last_points_only=False,
    )
    log(f"[{model_name}] {len(hfc)} backtest windows computed")

    # reduction=None → one value per window per metric, logged as
    # consecutive steps of backtest_mape / backtest_rmse
    log(f"[{model_name}] Running backtest (logging per-window metrics as steps)")
    model.backtest(
        series=series,
        historical_forecasts=hfc,
        last_points_only=False,
        metric=[darts_metrics.mape, darts_metrics.rmse, darts_metrics.ape],
        reduction=None,
    )
    log(f"[{model_name}] Run complete: {run.info.run_id}")

# ── Data setup ----------------------------------------------------------------
# Cast to float32: MPS doesn't support float64 tensors
series = AirPassengersDataset().load().astype(np.float32)
train, val = series.split_after(0.75)
FORECAST_HORIZON, STRIDE, BT_START = 1, 2, 0.75

# ── Logging setup ─────────────────────────────────────────────────────────────
VERBOSE = False
log = logger.info if VERBOSE else lambda *a, **kw: None
# ── MLflow setup ──────────────────────────────────────────────────────────────
DB_PATH = "/tmp/mlflow_v2.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
# ── Model setup ---------------------------------------------------------------
_torch_kwargs = dict(
    input_chunk_length=12,
    output_chunk_length=FORECAST_HORIZON,
    n_epochs=10,
    pl_trainer_kwargs={"accelerator": "mps", "precision": "32-true", "enable_progress_bar": False},
)
models = [
    LinearRegressionModel(lags=12, output_chunk_length=FORECAST_HORIZON),
    # LinearRegressionModel(lags=24, output_chunk_length=FORECAST_HORIZON),  # same model, more lags
    ExponentialSmoothing(),
    # NBEATSModel(**_torch_kwargs),
    # NBEATSModel(**_torch_kwargs, num_stacks=4, num_blocks=2),  # same model, deeper architecture
    # NHiTSModel(**_torch_kwargs),
    # TCNModel(**_torch_kwargs),
]

#--- MLflow experiment with manage_run=False -------------------------
exp_name: str = coolname.generate_slug(2)
mlflow.set_experiment(exp_name)
mlflow_darts.autolog(manage_run=False) # NOTE: manage_run = False
logger.info(f"Starting experiment: {exp_name}")
for model in models:
    model_name = type(model).__name__
    log(f"[{model_name}] Starting run")
    with mlflow.start_run() as run:
        _magic(model, train, val, series, log)
