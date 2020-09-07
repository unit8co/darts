"""
Cross-validation functions
--------------------------
"""

from ..timeseries import TimeSeries
from ..metrics import metrics as mfunc
from ..logging import get_logger, raise_if_not, raise_log
from typing import Union, Callable, Optional
import numpy as np
import pandas as pd

from ..models.forecasting_model import ForecastingModel

logger = get_logger(__name__)


def generalized_rolling_origin_evaluation(ts: TimeSeries, model: ForecastingModel,
                                          metrics: Union[Callable[[TimeSeries, TimeSeries], float], str] = 'mase',
                                          origin1: Optional[Union[int, pd.Timestamp]] = None,
                                          stride: Optional[int] = None, n_evaluations: Optional[int] = None,
                                          n_prediction: Optional[int] = None) -> float:
    """
    This function implements the Generalized Rolling origin Evaluation from
    `Fiorruci et al (2015) <https://arxiv.org/ftp/arxiv/papers/1503/1503.03529.pdf>`_

    Cross-Validation function to evaluate a forecasting model over a specific TimeSeries,
    and using a specific metrics.

    If `stride = 1`, the execution is similar to a Rolling Origin Evaluation.
    If `stride >= len(ts) - origin1` the execution is similar to a Fixed Origin Evaluation.

    At least one parameter from `stride` and `n_evaluations` must be given.

    If ValueErrors occur, the function will return `np.inf`.

    Parameters
    ----------
    ts
        A TimeSeres object to use for cross-validation.
    model
        The instance of ForecastingModel to cross-validate.
    metrics
        The metrics to use. Either a function from taking 2 TimeSeries as parameters,
        or a string of the name of the function from darts.metrics.
    origin1
        Optional. The index of the first origin. Defaults is the minimum between len(ts) - 10 and 5.
        Can also be the value of the DateTimeIndex.
    stride
        Optional. The stride used for rolling the origin. Defaults is n_prediction / n_evaluations if provided.
    n_evaluations
        Optional. Number of evaluation. Defaults is the maximum number possible if stride is provided.
    n_prediction
        Optional. Number of predictions for each evaluation. Defaults is the size of the tail: len(ts) - origin1.
    Returns
    -------
    Float
        The sum of the predictions errors over the different origins.
    """
    raise_if_not((stride is not None) or (n_evaluations is not None),
                 "At least 1 parameter between stride and n_evaluations must be given")
    raise_if_not(callable(metrics) or hasattr(mfunc, metrics),
                 "The metrics should be a function that takes TimeSeries as inputs,"
                 " or a string of a function name from darts.metrics")
    raise_if_not(stride is None or stride > 0,
                 "The stride parameter must be strictly positive")
    if type(metrics) is str and metrics != 'mase':
        metrics = getattr(mfunc, metrics)
    len_ts = len(ts)
    if origin1 is None:
        origin1 = max(5, len_ts - 10)
    elif isinstance(origin1, pd.Timestamp):
        raise_if_not(ts.is_within_range(origin1), "origin1 must be inside the TimeSeries")
        origin1 = ts.time_index().get_loc(origin1)
    elif origin1 >= len_ts or origin1 <= 0:
        raise_log(ValueError("origin1 must be inside the TimeSeries"), logger)
    if n_prediction is None:
        n_prediction = len_ts - origin1
    if n_evaluations is None:
        n_evaluations = int(1 + np.floor((len_ts - origin1) / stride))
    elif stride is None:
        stride = int(np.floor(n_prediction / n_evaluations))
    errors = []
    for i in range(n_evaluations):
        # if origin is further than end timestamp, end function
        if origin1 + i * stride >= len_ts:
            break
        # rolling origin
        origini = origin1 + i * stride
        n_pred = min(len_ts - origini, n_prediction)
        train = ts[:origini]
        test = ts[origini:]

        try:
            model.fit(train)
            forecast = model.predict(n_pred)
        except ValueError:
            # If cannot forecast with a specific timeseries, return np.inf
            errors.append(np.inf)
            continue
        try:
            if metrics == 'mase':
                error = getattr(mfunc, metrics)(test, forecast, train) * n_pred
            else:
                error = metrics(test, forecast) * n_pred
            errors.append(error)
        except ValueError:
            # if cannot use metrics, return np.inf
            errors.append(np.inf)
    errors = np.sum(errors)
    return errors
