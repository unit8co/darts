"""
Metrics
-------

For deterministic forecasts (point predictions with `num_samples == 1`), probabilistic forecasts (`num_samples > 1`),
and quantile forecasts. For probabilistic and quantile forecasts, use parameter `q` to define the quantile(s) to
compute the deterministic metrics on:

- Aggregated over time:
    Absolute metrics:
        - :func:`MERR <darts.metrics.metrics.merr>`: Mean Error
        - :func:`MAE <darts.metrics.metrics.mae>`: Mean Absolute Error
        - :func:`MSE <darts.metrics.metrics.mse>`: Mean Squared Error
        - :func:`RMSE <darts.metrics.metrics.rmse>`: Root Mean Squared Error
        - :func:`RMSLE <darts.metrics.metrics.rmsle>`: Root Mean Squared Log Error

    Relative metrics:
        - :func:`MASE <darts.metrics.metrics.mase>`: Mean Absolute Scaled Error
        - :func:`MSSE <darts.metrics.metrics.msse>`: Mean Squared Scaled Error
        - :func:`RMSSE <darts.metrics.metrics.rmsse>`: Root Mean Squared Scaled Error
        - :func:`MAPE <darts.metrics.metrics.mape>`: Mean Absolute Percentage Error
        - :func:`wMAPE <darts.metrics.metrics.wmape>`: weighted Mean Absolute Percentage Error
        - :func:`sMAPE <darts.metrics.metrics.smape>`: symmetric Mean Absolute Percentage Error
        - :func:`OPE <darts.metrics.metrics.ope>`: Overall Percentage Error
        - :func:`MARRE <darts.metrics.metrics.marre>`: Mean Absolute Ranged Relative Error

    Other metrics:
        - :func:`R2 <darts.metrics.metrics.r2_score>`: Coefficient of Determination
        - :func:`CV <darts.metrics.metrics.coefficient_of_variation>`: Coefficient of Variation

- Per time step:
    Absolute metrics:
        - :func:`ERR <darts.metrics.metrics.err>`: Error
        - :func:`AE <darts.metrics.metrics.ae>`: Absolute Error
        - :func:`SE <darts.metrics.metrics.se>`: Squared Error
        - :func:`SLE <darts.metrics.metrics.sle>`: Squared Log Error

    Relative metrics:
        - :func:`ASE <darts.metrics.metrics.ase>`: Absolute Scaled Error
        - :func:`SSE <darts.metrics.metrics.sse>`: Squared Scaled Error
        - :func:`APE <darts.metrics.metrics.ape>`: Absolute Percentage Error
        - :func:`sAPE <darts.metrics.metrics.sape>`: symmetric Absolute Percentage Error
        - :func:`ARRE <darts.metrics.metrics.arre>`: Absolute Ranged Relative Error

For probabilistic forecasts (storchastic predictions with `num_samples >> 1`) and quantile forecasts:

- Aggregated over time:
    Quantile metrics:
        - :func:`MQL <darts.metrics.metrics.mql>`: Mean Quantile Loss
        - :func:`QR <darts.metrics.metrics.qr>`: Quantile Risk

    Quantile interval metrics:
        - :func:`MIW <darts.metrics.metrics.miw>`: Mean Interval Width
        - :func:`MWS <darts.metrics.metrics.miws>`: Mean Interval Winkler Score
        - :func:`MIC <darts.metrics.metrics.mic>`: Mean Interval Coverage
        - :func:`MINCS_QR <darts.metrics.metrics.mincs_qr>`: Mean Interval Non-Conformity Score for Quantile Regression

- Per time step:
    Quantile metrics:
        - :func:`QL <darts.metrics.metrics.ql>`: Quantile Loss

    Quantile interval metrics:
        - :func:`IW <darts.metrics.metrics.iw>`: Interval Width
        - :func:`WS <darts.metrics.metrics.iws>`: Interval Winkler Score
        - :func:`IC <darts.metrics.metrics.ic>`: Interval Coverage
        - :func:`INCS_QR <darts.metrics.metrics.incs_qr>`: Interval Non-Conformity Score for Quantile Regression

For Dynamic Time Warping (DTW) (aggregated over time):

- :func:`DTW <darts.metrics.metrics.dtw_metric>`: Dynamic Time Warping Metric
"""

from darts.metrics.metrics import (
    ae,
    ape,
    arre,
    ase,
    coefficient_of_variation,
    dtw_metric,
    err,
    ic,
    incs_qr,
    iw,
    iws,
    mae,
    mape,
    marre,
    mase,
    merr,
    mic,
    mincs_qr,
    miw,
    miws,
    mql,
    mse,
    msse,
    ope,
    ql,
    qr,
    r2_score,
    rmse,
    rmsle,
    rmsse,
    sape,
    se,
    sle,
    smape,
    sse,
    wmape,
)

ALL_METRICS = {
    ae,
    ape,
    arre,
    ase,
    coefficient_of_variation,
    dtw_metric,
    err,
    iw,
    iws,
    mae,
    mape,
    wmape,
    marre,
    mase,
    merr,
    miw,
    miws,
    mql,
    mse,
    msse,
    ope,
    ql,
    qr,
    r2_score,
    rmse,
    rmsle,
    rmsse,
    sape,
    se,
    sle,
    smape,
    sse,
    ic,
    mic,
    incs_qr,
    mincs_qr,
}

TIME_DEPENDENT_METRICS = {
    ae,
    ape,
    arre,
    ase,
    err,
    ql,
    sape,
    se,
    sle,
    sse,
    iw,
    iws,
    ic,
    incs_qr,
}

Q_INTERVAL_METRICS = {
    iw,
    iws,
    miw,
    miws,
    ic,
    mic,
    incs_qr,
}

NON_Q_METRICS = {dtw_metric}

__all__ = [
    "ae",
    "ape",
    "arre",
    "ase",
    "coefficient_of_variation",
    "dtw_metric",
    "err",
    "mae",
    "mape",
    "wmape",
    "marre",
    "mase",
    "merr",
    "mql",
    "mse",
    "msse",
    "ope",
    "ql",
    "qr",
    "r2_score",
    "rmse",
    "rmsle",
    "rmsse",
    "sape",
    "se",
    "sle",
    "smape",
    "sse",
    "iw",
    "miw",
    "iws",
    "miws",
    "ic",
    "mic",
    "incs_qr",
    "mincs_qr",
]
