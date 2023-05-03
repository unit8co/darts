import logging
import random
from typing import Callable, Dict

import numpy as np
import torch

from darts import TimeSeries
from darts.metrics import mae
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    LocalForecastingModel,
)

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def randommness(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def evaluate_model(
    model_class: ForecastingModel,
    series: TimeSeries,
    model_params: Dict = dict(),
    metric: Callable[[TimeSeries, TimeSeries], float] = mae,
    split: float = 0.85,
    past_covariates: TimeSeries = None,
    future_covariates: TimeSeries = None,
    num_test_points: int = 20,
    forecast_horizon=1,
    repeat=3,
    **kwargs
):
    """
    _description_
    This function is used to evaluate a given model on a dataset.

    Parameters
    ----------
    model_class
        The class of the model to be evaluated.
    series
        The dataset on which the model will be evaluated.
    model_params
        The parameters for instantiating the model class.
    metric
        A darts metric function.
    split
        The split ratio between training and validation.
    past_covariates
        The past covariates for the model.
    future_covariates
        The future covariates for the model.
    num_test_points
        The number of test points to be used for the evaluation.
    """

    # some checks and conversions
    model_uses_cov = (
        model_params.get("lags_future_covariates", False)
        or model_params.get("lags_past_covariates", False)
        or model_params.get("add_encoders", False)
    )

    stride = int((1 - split) * len(series) / num_test_points)
    stride = max(stride, 1)
    forecast_horizon = min(stride, forecast_horizon)

    # now we performe the evaluation
    model = model_class(**model_params)

    retrain = True
    if not isinstance(model, LocalForecastingModel):
        retrain = retrain_func

    result = []
    for _ in range(repeat):
        if model_uses_cov:
            result += [
                model.backtest(
                    series,
                    retrain=retrain,
                    metric=metric,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    start=split,
                    stride=stride,
                    forecast_horizon=forecast_horizon,
                    last_points_only=True,
                )
            ]
        else:
            result += [
                model.backtest(
                    series,
                    retrain=retrain,
                    metric=metric,
                    start=split,
                    stride=stride,
                    forecast_horizon=forecast_horizon,
                    last_points_only=True,
                )
            ]
    return np.mean(result)


def retrain_func(counter, pred_time, train_series, past_covariates, future_covariates):
    return counter == 0
