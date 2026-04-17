"""
MLflow (auto)-logging:  Multi-model comparison with backtest

Demonstrates:
  * Single MLflow run containing all models — no nested runs needed.
    Metric keys are automatically prefixed with the model class name
    (e.g. ``LinearRegressionModel_val_mape``, ``ExponentialSmoothing_val_mape``),
    so all results live in one run without key collisions.
  * Automatic parameter and tag logging via darts autolog
  * Inline metric logging: ``mape`` and ``rmse`` called inside an active run
    are automatically captured by the patched metric functions
  * Backtest with pre-computed historical forecasts, multiple metric functions,
    and ``reduction=None`` so each window is logged as a step of
    ``{Model}_backtest_mape`` / ``{Model}_backtest_rmse`` in the MLflow UI

Notes:
  * ``mlflow_darts.autolog()`` is required — it patches ``fit()``, the darts
    metric functions, and ``backtest()`` to make logging automatic.
    ``mlflow.start_run()`` only manages the run context; without ``autolog()``
    nothing would be logged.
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

import mlflow
import coolname
from loguru import logger

logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)

import darts.metrics as darts_metrics
import darts.utils.mlflow as mlflow_darts
from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing, LinearRegressionModel

# ── data ──────────────────────────────────────────────────────────────────────
series = AirPassengersDataset().load()
train, val = series.split_after(0.75)

FORECAST_HORIZON = 1
STRIDE = 2
BT_START = 0.5  # more windows: backtest from 50 % of the full series

# ── MLflow setup ──────────────────────────────────────────────────────────────
# autolog() is required becausse it patches fit() / metric functions / backtest()
# start_run() separate only controls the run context
DB_PATH = "/tmp/mlflow_v2.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")


exp_name: str = coolname.generate_slug(2)
run_name: str = coolname.generate_slug(2)
logger.info(f"Starting experiment: {exp_name}")
mlflow.set_experiment(exp_name)
mlflow_darts.autolog()

models = [
    LinearRegressionModel(lags=12, output_chunk_length=FORECAST_HORIZON),
    ExponentialSmoothing(),
]

# NOTE: would this make more sense actually?
# for model in models:
#     with mlflow.start_run(run_name=run_name):
#         ...

with mlflow.start_run(run_name=run_name):
    for model in models:
        # ── fit ───────────────────────────────────────────────────────────
        # NOTE: add val_series=val here for TorchForecastingModels to enable
        #       validation-based early stopping during training.
        model.fit(train)

        # ── predict + inline metric logging ───────────────────────────────
        pred = model.predict(n=len(val))

        # Calls inside an active MLflow run are intercepted by the patched
        # metric functions and logged automatically.
        # Keys are prefixed with the model class name, e.g.:
        #   LinearRegressionModel_val_mape, ExponentialSmoothing_val_mape
        darts_metrics.mape(val, pred)
        darts_metrics.rmse(val, pred)

        # ── backtest ──────────────────────────────────────────────────────
        # Pre-compute historical forecasts once so they can be reused
        # without paying the refitting cost again.
        # last_points_only=False → list[TimeSeries], one per window, so
        # backtest() computes the metric separately for each window.
        hfc = model.historical_forecasts(
            series,
            start=BT_START,
            forecast_horizon=FORECAST_HORIZON,
            stride=STRIDE,
            retrain=True,
            last_points_only=False,
        )

        # reduction=None → one value per window per metric, logged as
        # consecutive steps of {Model}_backtest_mape / {Model}_backtest_rmse
        # so the MLflow UI renders a chart over backtest windows.
        model.backtest(
            series=series,
            historical_forecasts=hfc,
            last_points_only=False,
            metric=[darts_metrics.mape, darts_metrics.rmse],
            reduction=None,
        )