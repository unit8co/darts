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
    target_dataset: TimeSeries,
    model_params: Dict = dict(),
    metric: Callable[[TimeSeries, TimeSeries], float] = mae,
    split: float = 0.8,
    past_cov: TimeSeries = None,
    future_cov: TimeSeries = None,
    num_test_points: int = 20,
    **kwargs
):
    """
    _description_
    This function is used to evaluate a given model on a dataset.

    Parameters
    ----------
    model_class
        The class of the model to be evaluated.
    target_dataset
        The dataset on which the model will be evaluated.
    model_params
        The parameters for instantiating the model class.
    metric
        A darts metric function.
    split
        The split ratio between training and validation.
    past_cov
        The past covariates for the model.
    future_cov
        The future covariates for the model.
    num_test_points
        The number of test points to be used for the evaluation.
    """

    # some checks and conversions
    model_uses_cov = model_params.get(
        "lags_future_covariates", False
    ) or model_params.get("lags_past_covariates", False)

    if model_uses_cov and not (past_cov or future_cov):
        raise ValueError("model uses covariates, but none were provided")

    stride = int((1 - split) * len(target_dataset) / num_test_points)
    stride = max(stride, 1)

    # now we performe the evaluation
    model = model_class(**model_params)

    retrain = True
    if not isinstance(model, LocalForecastingModel):
        retrain = False

        train = target_dataset.split_after(split)[0]
        train_past_cov = past_cov.split_after(split)[0] if past_cov else None
        train_future_cov = future_cov.split_after(split)[0] if future_cov else None

        if model_uses_cov:
            model.fit(
                train,
                past_covariates=train_past_cov,
                future_covariates=train_future_cov,
            )
        else:
            model.fit(train)

    if model_uses_cov:
        return model.backtest(
            target_dataset,
            retrain=retrain,
            metric=metric,
            past_covariates=past_cov,
            future_covariates=future_cov,
            start=split,
            stride=stride,
        )
    else:
        return model.backtest(
            target_dataset, retrain=retrain, metric=metric, start=split, stride=stride
        )
