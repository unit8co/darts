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


def _managed_run_scenarios(model, train, val, log) -> None:
    """Explores manage_run=True behaviour across four scenarios.

    Metrics always use manage_run=False internally — they only log into an
    already-active run and never create one themselves.

    Scenario A — bare fit():
        fit() uses with_managed_run which creates a run when none is active,
        then closes it on return.
        Result: one run per model, params logged, no metrics.

    Scenario B — metric with no active run:
        Metric patches have manage_run=False so they never open a run.
        With nothing active there is nowhere to log.
        Result: nothing logged (silent no-op).

    Scenario C — metric inside an explicit start_run():
        Caller opens a run; metric patch sees an active run and logs into it.
        No fit() → no params.
        Result: one run with only the metric logged.

    Scenario D — fit() + metric inside an explicit start_run():
        with_managed_run creates a run "if necessary". Because the caller's
        run is already active, fit() reuses it rather than nesting a child.
        Params and metric all land in the same single run.
        Result: one run per model with both params and metric logged.
    """
    model_name = type(model).__name__

    # ── Scenario A: bare fit ──────────────────────────────────────────
    # Expected: one run with params logged; no metrics.
    log(f"[{model_name}] Scenario A — bare fit")
    model.fit(train)
    pred = model.predict(n=len(val))

    # ── Scenario B: metric outside any run ───────────────────────────
    # Expected: nothing logged (metrics have manage_run=False).
    log(f"[{model_name}] Scenario B — metric with no active run (expect: not logged)")
    darts_metrics.mape(val, pred)

    # ── Scenario C: metric inside explicit start_run ──────────────────
    # Expected: one run with rmse logged, no params.
    log(f"[{model_name}] Scenario C — metric inside explicit start_run")
    with mlflow.start_run():
        darts_metrics.rmse(val, pred)

    # ── Scenario D: fit + metric inside explicit start_run ───────────
    # Expected: one run with both params (from fit) and mape logged.
    # fit() reuses the caller's run — no nesting.
    log(f"[{model_name}] Scenario D — fit + metric inside explicit start_run")
    with mlflow.start_run():
        model.fit(train)
        pred = model.predict(n=len(val))
        darts_metrics.mape(val, pred)


def _magic(model, train, val, series, log) -> None:
    """Advanced use case: caller-managed run with predict, val metrics, and backtest.

    Requires autolog(manage_run=False) and an active mlflow.start_run() in the
    caller. Because manage_run=False, fit() never opens or closes a run on its
    own — the caller's run stays open for the entire workflow.

    What gets logged into the single run:
      - Model params (from fit via the autolog patch).
      - val_mape, val_rmse (from patched darts_metrics calls while run is active).
      - backtest_mape, backtest_rmse, backtest_ape as consecutive steps
        (one value per window, because reduction=None).
    """
    model_name = type(model).__name__
    log(f"[{model_name}] Run created: run_id={mlflow.active_run().info.run_id}")
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
    log(f"[{model_name}] Run complete: {mlflow.active_run().info.run_id}")


def _backtest_reduction_scenarios(model, train, series, log) -> None:
    """Explores how different ``reduction`` values affect MLflow logging.

    Fits once and computes historical forecasts once, then calls backtest with
    four different reductions.  Each variant runs inside its own nested
    child run so the MLflow UI shows them separately without key collisions.

    Reduction variants and their logged shape:
      None      → 1-D array per metric → logged as consecutive steps (chart).
      np.mean   → scalar per metric    → single value (mean over windows).
      np.median → scalar per metric    → single value (median over windows).
      custom    → scalar per metric    → single value (90th-percentile).

    All variants use the same metric list (mape, rmse) and the same
    pre-computed historical forecasts so results are directly comparable.
    """
    model_name = type(model).__name__
    metrics = [darts_metrics.mape, darts_metrics.rmse]

    log(f"[{model_name}] Fitting")
    model.fit(train)
    log(f"[{model_name}] Computing historical forecasts")
    hfc = model.historical_forecasts(
        series,
        start=BT_START,
        forecast_horizon=FORECAST_HORIZON,
        stride=STRIDE,
        retrain=True,
        last_points_only=False,
    )
    log(f"[{model_name}] {len(hfc)} windows")

    reductions = [
        (None, "reduction=None (per-window steps)"),
        (np.mean, "reduction=np.mean"),
        (np.median, "reduction=np.median"),
        (lambda x, axis=None: np.percentile(x, 90, axis=axis), "reduction=p90"),
    ]

    for reduction_fn, label in reductions:
        log(f"[{model_name}] Backtest — {label}")
        with mlflow.start_run(nested=True, run_name=label):
            model.backtest(
                series=series,
                historical_forecasts=hfc,
                last_points_only=False,
                metric=metrics,
                reduction=reduction_fn,
            )


# ── Data setup ----------------------------------------------------------------
# Cast to float32: MPS doesn't support float64 tensors
series = AirPassengersDataset().load().astype(np.float32)
train, val = series.split_after(0.75)
FORECAST_HORIZON, STRIDE, BT_START = 1, 2, 0.75

# ── Logging setup ─────────────────────────────────────────────────────────────
VERBOSE = True
log = logger.info if VERBOSE else lambda *a, **kw: None
# ── MLflow setup ──────────────────────────────────────────────────────────────
DB_PATH = "/tmp/mlflow_v2.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
# ── Model setup ---------------------------------------------------------------
_torch_kwargs = dict(
    input_chunk_length=12,
    output_chunk_length=FORECAST_HORIZON,
    n_epochs=10,
    pl_trainer_kwargs={
        "accelerator": "mps",
        "precision": "32-true",
        "enable_progress_bar": False,
    },
)
models = [
    LinearRegressionModel(lags=12, output_chunk_length=FORECAST_HORIZON),
    # LinearRegressionModel(lags=24, output_chunk_length=FORECAST_HORIZON),  # same model, more lags
    ExponentialSmoothing(),
    NBEATSModel(**_torch_kwargs),
    # NBEATSModel(**_torch_kwargs, num_stacks=4, num_blocks=2),  # same model, deeper architecture
    # NHiTSModel(**_torch_kwargs),
    # TCNModel(**_torch_kwargs),
]

# ── Use case 1: manage_run=True (default) ─────────────────────────────────────
# fit() auto-creates and closes a run per model — no start_run() needed.
exp_name: str = coolname.generate_slug(2)
mlflow.set_experiment(exp_name)
mlflow_darts.autolog(manage_run=True)
logger.info(f"[Use case 1] Experiment: {exp_name}")
for model in models:
    _managed_run_scenarios(model, train, val, log)

# ── Use case 2: manage_run=False + backtest ───────────────────────────────────
# Caller opens the run so predict, val metrics, and backtest all land in it.
# NOTE: this is the recommended way to use MLFlow in Darts. It avoids
#       nesting runs and ensures all metrics land in the same run.
exp_name = coolname.generate_slug(2)
mlflow.set_experiment(exp_name)
mlflow_darts.autolog(manage_run=False)
logger.info(f"[Use case 2] Experiment: {exp_name}")
for i, model in enumerate(models):
    model_name = type(model).__name__
    log(f"[{model_name}] Starting run")
    with mlflow.start_run(
        run_name=model_name,
        description=f"Small exp. run for {model_name}",
    ) as run:
        _magic(model, train, val, series, log)

# ── Use case 3: backtest reduction variants ───────────────────────────────────
# One parent run per model; one nested child run per reduction variant.
exp_name = coolname.generate_slug(2)
mlflow.set_experiment(exp_name)
mlflow_darts.autolog(manage_run=False)
logger.info(f"[Use case 3] Experiment: {exp_name}")
for model in models:
    model_name = type(model).__name__
    with mlflow.start_run(run_name=model_name):
        _backtest_reduction_scenarios(model, train, series, log)
