"""
Cross-validation functions
--------------------------
"""

from ..timeseries import TimeSeries
from ..metrics import metrics as mfunc
from ..logging import get_logger, raise_if_not
from typing import Union, Callable, Optional
import numpy as np

from ..models.forecasting_model import ForecastingModel

logger = get_logger(__name__)


def generalized_rolling_origin_evaluation(ts: TimeSeries, model: ForecastingModel,
                                          metrics: Union[Callable[[TimeSeries, TimeSeries], float], str] = 'mase',
                                          origin1: Optional[int] = None,
                                          stride: Optional[int] = None, n_evaluation: Optional[int] = None,
                                          n_prediction: Optional[int] = None) -> float:
    """
    This function implements the Generalized Rolling origin Evaluation from
    `Fiorruci et al (2015) <https://arxiv.org/ftp/arxiv/papers/1503/1503.03529.pdf>`_

    Cross-Validation function to evaluate a forecasting model over a specific TimeSeries,
    and using a specific metrics.

    If `stride = 1`, the execution is similar to a Rolling Origin Evaluation.
    If `stride >= len(ts) - origin1` the execution is similar to a Fixed Origin Evaluation.

    At least two parameters from `stride`, `n_evaluation` and `n_prediction` must be given.

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
        The index of the first origin. Defaults is the minimum between len(ts) - 10 and 5.
    stride
        The stride used for rolling the origin. Defaults is n_prediction / n_evaluation if provided.
    n_evaluation
        Number of evaluation. Defaults is the maximum number possible.
    n_prediction
        Number of predictions for each evaluation. Defaults is len(ts) - origin1.
    Returns
    -------
    Float
        The sum of the predictions errors over the different origins.
    """
    raise_if_not(((stride is None) + (n_prediction is None) + (n_evaluation is None)) > 1,
                 "At least 2 parameters between stride, n_prediction and n_evaluation must be given")
    raise_if_not(callable(metrics) or hasattr(mfunc, metrics),
                 "The metrics should be a function from darts.metrics or a string of its name")
    if type(metrics) is str and metrics != 'mase':
        metrics = getattr(mfunc, metrics)
    len_ts = len(ts)
    if origin1 is None:
        origin1 = max(5, len_ts - 10)
    if stride is None:
        stride = int(np.floor(n_prediction / n_evaluation))
    if n_prediction is None:
        n_prediction = len_ts - origin1
    if n_evaluation is None:
        n_evaluation = int(1 + np.floor(n_prediction / stride))
    errors = []
    for i in range(n_evaluation):
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
                error = getattr(mfunc, metrics)(test, forecast, train) * n_evaluation
            else:
                error = metrics(test, forecast) * n_evaluation
            errors.append(error)
        except ValueError:
            # if cannot use metrics, return np.inf
            errors.append(np.inf)
    errors = np.sum(errors)
    return errors
