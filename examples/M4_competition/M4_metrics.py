from typing import List
import numpy as np
from darts import TimeSeries
from darts.metrics.metrics import get_logger, _get_values_or_raise, raise_if_not


baseline = {
        'Hourly': [18.383, 2.395],
        'Daily': [3.045, 3.278],
        'Weekly': [9.161, 2.777],
        'Monthly': [14.427, 1.063],
        'Quarterly': [11.012, 1.371],
        'Yearly': [16.342, 3.974],
        'Total': [13.564, 1.912],
}

logger = get_logger(__name__)


def smape_m4(actual_series: TimeSeries, pred_series: TimeSeries, intersect: bool = True) -> float:
    if isinstance(actual_series, np.ndarray):
        y_true, y_hat = actual_series, pred_series
    else:
        y_true, y_hat = _get_values_or_raise(actual_series, pred_series, intersect)
    return 200. * np.abs((y_true - y_hat) / (np.abs(y_true)+np.abs(y_hat)))


def mase_m4(train_series: TimeSeries, actual_series: TimeSeries, pred_series: TimeSeries,
            m: int = 1, intersect: bool = True) -> float:
    if isinstance(actual_series, np.ndarray):
        y_true, y_hat = actual_series, pred_series
    else:
        y_true, y_hat = _get_values_or_raise(actual_series, pred_series, intersect)
    insample = train_series.values()
    errors = np.abs(y_true - y_hat)
    scale = np.mean(np.abs(insample[m:] - insample[:-m]))
    raise_if_not(not np.isclose(scale, 0), "cannot use MASE with periodical signals", logger)
    return errors / scale


def owa_m4(frequency: str, smape: np.ndarray, mase: np.ndarray) -> float:
    bl = baseline[frequency]
    return (smape/bl[0] + mase/bl[1]) / 2
