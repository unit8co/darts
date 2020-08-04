"""
Standard Regression model
-------------------------
"""

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not
from typing import List, Callable, Optional
import numpy as np
import pandas as pd

from ..models.forecasting_model import ForecastingModel

logger = get_logger(__name__)


def generalized_rolling_origin_evaluation(ts: TimeSeries, model: ForecastingModel,
                                          metrics: Callable[[TimeSeries, TimeSeries], float],
                                          fq: Optional[int] = None, n1: Optional[int] = None,
                                          m: Optional[int] = None, p: Optional[int] = None,
                                          H: Optional[int] = None, **model_args) -> float:
    """
    This function implements the Generalized Rolling origin Evaluation from Fiorrucci et al (2015)
    [https://arxiv.org/ftp/arxiv/papers/1503/1503.03529.pdf
    Cross-Validation function to evaluate the Forecastin model `model`.

    If m=1 is computed the Rolling Origin Evaluation.
    If m>=length(y)-n1 is computed the Fixed Origin Evaluation

    if all parameters are give, H will be ignored.

    Parameters
    ----------
    ts
        A TimeSeres object to use for cross-validation.
    model
        The ForecastingModel to cross-validate.
    metrics
        The metrics to use.
    fq
        The seasonality period of ts.
    n1
        The index of the first origin. Defaults is len(ts) - H if provided, otherwise 5.
    m
        The stride used for rolling the origin. Defaults is H/p if provided
    p
        Number of evaluation. Defaults is the maximum
    H
        Number of predictions for each evaluation. Defaults is m*p if provided
    Returns
    -------
    Float
        The sum of the predictions errors over the different origins.
    """
    raise_if_not(((m is None) + (H is None) + (p is None)) > 1,
                 "At least 2 parameters between m, p and H must be given")
    n = len(ts)
    if n1 is None:
        n1 = max(5, n - 10)
    if m is None:
        m = int(np.floor(H/p))
    if H is None:
        H = n - n1
    if p is None:
        p = int(1 + np.floor(H / m))
    if fq is None:
        pass
        # found automatically frequency
    # todo: if model object or class. to chose
    errors = []
    for i in range(p):
        # if origin is further than end timestamp, end function
        if n1 + i * m >= n:
            break
        ni = n1 + i * m
        npred = n - ni
        train = ts[:ni]
        test = ts[ni:]

        try:
            model.fit(train)
            forecast = model.predict(npred)
        except ValueError:
            errors.append(0)
            continue
        try:
            error = metrics(test, forecast)
            errors.append(error)
        except ValueError:
            errors.append(0)
    errors = np.sum(errors)
    return errors
