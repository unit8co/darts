"""
Metrics
-------

For deterministic forecasts (point predictions with `num_samples == 1`):
    - :func:`MAE <darts.metrics.metrics.mae>`: Mean Absolute Error
    - :func:`MSE <darts.metrics.metrics.mse>`: Mean Squared Error
    - :func:`RMSE <darts.metrics.metrics.rmse>`: Root Mean Squared Error
    - :func:`RMSLE <darts.metrics.metrics.rmsle>`: Root Mean Squared Log Error
    - :func:`MAPE <darts.metrics.metrics.mape>`: Mean Absolute Percentage Error
    - :func:`sMAPE <darts.metrics.metrics.smape>`: symmetric Mean Absolute Percentage Error
    - :func:`OPE <darts.metrics.metrics.ope>`: Overall Percentage Error
    - :func:`MASE <darts.metrics.metrics.mase>`: Mean Absolute Scaled Error
    - :func:`MARRE <darts.metrics.metrics.marre>`: Mean Absolute Ranged Relative Error
    - :func:`R2 <darts.metrics.metrics.r2_score>`: Coefficient of Determination
    - :func:`CV <darts.metrics.metrics.coefficient_of_variation>`: Coefficient of Variation

For probabilistic forecasts (storchastic predictions with `num_samples >> 1`):
    - :func:`Rho risk <darts.metrics.metrics.rho_risk>`: Rho/quantile-Risk
    - :func:`Quantile Loss <darts.metrics.metrics.quantile_loss>`: Quantile Loss

For Dynamic Time Warping (DTW):
    - :func:`DTW <darts.metrics.metrics.dtw_metric>`: Dynamic Time Warping Metric
"""

from .metrics import (
    coefficient_of_variation,
    dtw_metric,
    mae,
    mape,
    marre,
    mase,
    mse,
    ope,
    quantile_loss,
    r2_score,
    rho_risk,
    rmse,
    rmsle,
    smape,
)
