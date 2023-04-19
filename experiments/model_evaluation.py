import logging
import random

import numpy as np
import torch

from darts.metrics import mae
from darts.models.forecasting.forecasting_model import LocalForecastingModel

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def randommness(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def evaluate_model(
    model_class,
    dataset,
    model_params=dict(),
    metric=mae,
    split=0.7,
    past_cov=None,
    future_cov=None,
    **kwargs
):

    model_uses_cov = model_params.get(
        "lags_future_covariates", False
    ) or model_params.get("lags_past_covariates", False)
    if model_uses_cov and not (past_cov or future_cov):
        raise ValueError("model uses covariates, but none were provided")
    if past_cov and len(dataset) != len(past_cov):
        raise ValueError("past_cov and dataset must have the same length")
    if future_cov and len(dataset) != len(future_cov):
        raise ValueError("future_cov and dataset must have the same length")

    model = model_class(**model_params)
    num_test_points = 20
    stride = int((1 - split) * len(dataset) / num_test_points)
    stride = max(stride, 1)
    retrain = True
    if not isinstance(model, LocalForecastingModel):
        retrain = False

        train = dataset.split_after(split)[0]
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
            dataset,
            retrain=retrain,
            metric=metric,
            past_covariates=past_cov,
            future_covariates=future_cov,
            start=split,
            stride=stride,
        )
    else:
        return model.backtest(
            dataset, retrain=retrain, metric=metric, start=split, stride=stride
        )
