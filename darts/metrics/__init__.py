"""
Metrics
-------

For deterministic forecasts (point predictions with `num_samples == 1`):
    - Per time step:
        - :func:`RES <darts.metrics.metrics.res>`: Residuals
        - :func:`AE <darts.metrics.metrics.ae>`: Absolute Error
        - :func:`SE <darts.metrics.metrics.se>`: Squared Error
        - :func:`SLE <darts.metrics.metrics.rmsle>`: Squared Log Error

        Relative Errors:
        - :func:`ASE <darts.metrics.metrics.ae>`: Absolute Scaled Error
        - :func:`SSE <darts.metrics.metrics.msse>`: Squared Scaled Error
        - :func:`APE <darts.metrics.metrics.mape>`: Absolute Percentage Error
        - :func:`sAPE <darts.metrics.metrics.smape>`: symmetric Absolute Percentage Error
        - :func:`ARRE <darts.metrics.metrics.marre>`: Absolute Ranged Relative Error

    - Aggregated over time:
        - :func:`MRES <darts.metrics.metrics.mres>`: Mean Residuals
        - :func:`MAE <darts.metrics.metrics.mae>`: Mean Absolute Error
        - :func:`MSE <darts.metrics.metrics.mse>`: Mean Squared Error
        - :func:`RMSE <darts.metrics.metrics.rmse>`: Root Mean Squared Error
        - :func:`RMSLE <darts.metrics.metrics.rmsle>`: Root Mean Squared Log Error

        Relative Errors:
        - :func:`MASE <darts.metrics.metrics.mase>`: Mean Absolute Scaled Error
        - :func:`MSSE <darts.metrics.metrics.msse>`: Mean Squared Scaled Error
        - :func:`RMSSE <darts.metrics.metrics.rmsse>`: Root Mean Squared Scaled Error
        - :func:`MAPE <darts.metrics.metrics.mape>`: Mean Absolute Percentage Error
        - :func:`sMAPE <darts.metrics.metrics.smape>`: symmetric Mean Absolute Percentage Error
        - :func:`OPE <darts.metrics.metrics.ope>`: Overall Percentage Error
        - :func:`MARRE <darts.metrics.metrics.marre>`: Mean Absolute Ranged Relative Error

        Other metrics:
        - :func:`R2 <darts.metrics.metrics.r2_score>`: Coefficient of Determination
        - :func:`CV <darts.metrics.metrics.coefficient_of_variation>`: Coefficient of Variation

For probabilistic forecasts (storchastic predictions with `num_samples >> 1`):
    - Per time step:
        - :func:`QL <darts.metrics.metrics.ql>`: Quantile Loss
    - Aggregated over time:
        - :func:`MQL <darts.metrics.metrics.mql>`: Mean Quantile Loss
        - :func:`Rho risk <darts.metrics.metrics.rho_risk>`: Rho/quantile-Risk

For Dynamic Time Warping (DTW) (aggregated over time):
    - :func:`DTW <darts.metrics.metrics.dtw_metric>`: Dynamic Time Warping Metric
"""

from .metrics import (
    ae,
    ape,
    arre,
    ase,
    coefficient_of_variation,
    dtw_metric,
    mae,
    mape,
    marre,
    mase,
    mql,
    mres,
    mse,
    msse,
    ope,
    ql,
    r2_score,
    res,
    rho_risk,
    rmse,
    rmsle,
    rmsse,
    sape,
    se,
    sle,
    smape,
    sse,
)
