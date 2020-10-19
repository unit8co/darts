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

logger = get_logger(__name__)

def generalized_rolling_origin_evaluation(series: TimeSeries,
                                          model: ForecastingModel,
                                          metric: Union[Callable[[TimeSeries, TimeSeries], float], str] = 'mase',
                                          first_origin: Optional[Union[int, pd.Timestamp]] = None,
                                          stride: Optional[int] = None,
                                          n_evaluations: Optional[int] = None,
                                          n_predictions: Optional[int] = None) -> float:
    """
    This function implements the Generalized Rolling Origin Evaluation from
    `Fiorruci et al (2015) <https://arxiv.org/ftp/arxiv/papers/1503/1503.03529.pdf>`_

    Cross-Validation function to evaluate a forecasting model on a specific TimeSeries,
    using a given metric.

    If `stride = 1`, the execution is similar to a Rolling Origin Evaluation.
    If `stride >= len(series) - first_origin` the execution is similar to a Fixed Origin Evaluation.

    At least one parameter from `stride` and `n_evaluations` must be given.

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
        Optional. The index of the first origin. Defaults to the minimum between (len(series) - 10) and 5.
        Can also be the value of the DateTimeIndex.
    stride
        Optional. The stride used for rolling the origin.
        Default value is `n_predictions / n_evaluations`
    n_evaluations
        Optional. Number of evaluation. By default the maximum value possible if stride is provided.
    n_predictions
        Optional. Number of predictions for each evaluation. Defaults is the size of the tail: len(series) - first_origin.
    Returns
    -------
    Float
        The sum of the predictions errors over the different origins.
    """
    raise_if((stride is None) and (n_evaluations is None),
             "At least 1 parameter between stride and n_evaluations must be given",
             logger)
    raise_if(stride is not None and stride <= 0,
             "stride must be strictly positive",
             logger)

    if isinstance(metric, str):
        raise_if_not(hasattr(metrics, metric),
                     "The provided string for metric doesn't match any metric from darts.metrics",
                     logger)
        metric = getattr(metrics, metric)

    len_series = len(series)
    
    if first_origin is None:
        first_origin = min(5, len_series - 10)
    elif isinstance(first_origin, pd.Timestamp):
        raise_if_not(series.is_within_range(first_origin), "first_origin must be inside the TimeSeries")
        first_origin = series.time_index().get_loc(first_origin)
    elif isinstance(first_origin, int):
        raise_if(first_origin >= len_series or first_origin <= 0,
                 "first_origin must be inside the TimeSeries",
                 logger)
    else:
        raise_log(ValueError("first_origin should be either a pd.Timestamp or an int"))

    if n_predictions is None:
        n_predictions = len_series - first_origin

    if n_evaluations is None:
        n_evaluations = int(1 + np.floor((len_series - first_origin) / stride))
    elif stride is None:
        stride = int(np.floor(n_predictions / n_evaluations))

    errors = []
    for i in range(n_evaluations):
        origin = first_origin + i * stride

        if origin >= len_series:
            break

        n_pred = min(len_series - origin, n_predictions)
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
