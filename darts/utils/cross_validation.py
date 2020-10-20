"""
Cross-validation functions
--------------------------
"""
import numpy as np
import pandas as pd

from typing import Union, Callable, Optional

from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_if_not, raise_log, raise_if
from darts.models.forecasting_model import ForecastingModel
import darts.metrics as metrics
from darts.utils import get_index_at_point

logger = get_logger(__name__)


def generalized_rolling_origin_evaluation(series: TimeSeries,
                                          model: ForecastingModel,
                                          metric: Union[Callable[[TimeSeries, TimeSeries], float], str] = 'mase',
                                          first_origin: Optional[Union[pd.Timestamp, float, int]] = None,
                                          forecast_horizon: Optional[int] = None,
                                          stride: Optional[int] = None,
                                          n_prediction_steps: Optional[int] = None) -> float:
    """
    This function implements the Generalized Rolling Origin Evaluation from
    `Fiorruci et al (2015) <https://arxiv.org/ftp/arxiv/papers/1503/1503.03529.pdf>`_

    Cross-Validation function to evaluate a forecasting model on a specific TimeSeries,
    using a given metric.

    If `stride = 1`, the execution is similar to a Rolling Origin Evaluation.
    If `stride >= len(series) - first_origin` the execution is similar to a Fixed Origin Evaluation.

    At least one parameter from `stride` and `n_prediction_steps` must be given.

    If ValueErrors occur, the function will return `np.inf`.

    Parameters
    ----------
    series
        A TimeSeries object to use for cross-validation.
    model
        The instance of ForecastingModel to cross-validate.
    metric
        The metric to use.
        Can be either any function that takes two `TimeSeries` and computes an score value (float),
        or a string of the name of the function from darts.metrics.
    first_origin
        Optional. The first point at which a prediction is computed for a future time.
        This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
        In the case of `float`, the parameter will be treated as the proportion of the time series
        that should lie before the first prediction point.
        In the case of `int`, the parameter will be treated as an integer index to the time index of
        `series`.
        In case of a `pandas.Timestamp`, the parameter will be converted to the index corresponding
        to the timestamp in the series provided that the timestamp is present in the series time index
        Defaults to the minimum between (len(series) - 10) and 5.
        Can also be the value of the DateTimeIndex.
    forecast_horizon
        Optional. The forecast horizon for the point predictions.
        Default value is the size of the tail: len(series) - first_origin.
    stride
        Optional. The stride used for rolling the origin.
        Default value is `(len(series) - first_origin) / n_prediction_steps`
    n_prediction_steps
        Optional. The number of prediction steps (ie. max number of calls to predict()).
        By default the maximum value possible if stride is provided.
    Returns
    -------
    Float
        The sum of the predictions errors over the different origins.
    """
    raise_if((stride is None) and (n_prediction_steps is None),
             "At least 1 parameter between stride and n_prediction_steps must be given",
             logger)
    raise_if(stride is not None and stride <= 0, "stride must be strictly positive", logger)

    if isinstance(metric, str):
        raise_if_not(hasattr(metrics, metric),
                     "The provided string for metric doesn't match any metric from darts.metrics",
                     logger)
        metric = getattr(metrics, metric)

    len_series = len(series)

    if first_origin is None:
        first_origin = min(5, len_series - 10)
    first_origin = get_index_at_point(first_origin, series)

    if forecast_horizon is None:
        forecast_horizon = len_series - first_origin

    if n_prediction_steps is None:
        n_prediction_steps = int(np.floor((len_series - first_origin) / stride))
    elif stride is None:
        stride = int(np.floor((len_series - first_origin) / n_prediction_steps))

    max_origin = first_origin + (n_prediction_steps-1) * stride
    raise_if(max_origin >= len_series,
             "The combination formed by the first_origin, n_prediction_steps and stride parameters is invalid"
             " (it will result in setting the `origin` outside of `series`). Try setting smaller values,"
             " or let n_prediction_steps be set to the max possible value by default",
             logger)

    errors = []
    for origin in range(first_origin, max_origin, stride):
        n_pred = min(len_series - origin, forecast_horizon)
        train = series[:origin]
        test = series[origin:]

        try:
            model.fit(train)
            forecast = model.predict(n_pred)
        except ValueError:
            # If cannot forecast with a specific timeseries, return np.inf
            errors.append(np.inf)
            continue
        try:
            if metric == metrics.mase:
                error = metric(test, forecast, train) * n_pred
            else:
                error = metric(test, forecast) * n_pred
            errors.append(error)
        except ValueError:
            # if cannot use the given metric, return np.inf
            errors.append(np.inf)
    errors = np.sum(errors)
    return errors
