"""
This file implements wrappers around backtest to have the same input structure for all models/all datasets
"""
import logging
import random
from typing import Callable, Dict

import numpy as np
import torch
from pytorch_lightning.callbacks import EarlyStopping

from darts import TimeSeries
from darts.metrics import mae
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    LocalForecastingModel,
)
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

# Silencing pytorch as having multiple training in parallel makes the output unreadable
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
# Add some options for deep learning models
early_stopper = EarlyStopping("train_loss", min_delta=0.001, patience=3, verbose=True)
PL_TRAINER_KWARGS = {
    "enable_progress_bar": False,
    "accelerator": "cpu",
    "callbacks": [early_stopper],
}


def set_randommness(seed=0):
    # set randomness for reproducibility
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
    repeat=1,
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
    set_randommness()

    model_instance, function_args = prepare_data(
        model_class,
        series,
        model_params,
        metric,
        split,
        past_covariates,
        future_covariates,
        num_test_points,
        forecast_horizon,
    )

    losses = [model_instance.backtest(**function_args) for _ in range(repeat)]

    return np.mean(losses)


def illustrate_model(
    model_class: ForecastingModel,
    series: TimeSeries,
    model_params: Dict = dict(),
    split: float = 0.85,
    past_covariates: TimeSeries = None,
    future_covariates: TimeSeries = None,
    num_test_points: int = 20,
    forecast_horizon=1,
    **kwargs
):
    """
    This function performs the same operation as evaluate_model
    but prodive the output instead for visual exploration
    """
    model_instance, function_args = prepare_data(
        model_class,
        series,
        model_params,
        split=split,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        num_test_points=num_test_points,
        forecast_horizon=forecast_horizon,
    )

    return model_instance.historical_forecasts(**function_args)


def prepare_data(
    model_class: ForecastingModel,
    series: TimeSeries,
    model_params: Dict = dict(),
    metric: Callable[[TimeSeries, TimeSeries], float] = None,
    split: float = 0.85,
    past_covariates: TimeSeries = None,
    future_covariates: TimeSeries = None,
    num_test_points: int = 20,
    forecast_horizon=1,
    **kwargs
):
    """
    Standardizes inputs to have the same entry point for all models and datasets
    """

    if issubclass(model_class, TorchForecastingModel):
        model_params["pl_trainer_kwargs"] = PL_TRAINER_KWARGS
    # now we performe the evaluation
    model_instance = model_class(**model_params)

    stride = int((1 - split) * len(series) / num_test_points)
    stride = max(stride, 1)
    forecast_horizon = min(stride, forecast_horizon)

    is_local_model = isinstance(model_instance, LocalForecastingModel)
    retrain = True if is_local_model else retrain_func

    function_args = {
        "series": series,
        "retrain": retrain,
        "start": split,
        "stride": stride,
        "forecast_horizon": forecast_horizon,
        "last_points_only": True,
    }
    if metric is not None:
        function_args["metric"] = metric
    if model_instance.supports_past_covariates and not model_params.get(
        "shared_weights"
    ):
        function_args["past_covariates"] = past_covariates
    if model_instance.supports_future_covariates and not model_params.get(
        "shared_weights"
    ):
        function_args["future_covariates"] = future_covariates

    return model_instance, function_args


def retrain_func(counter, pred_time, train_series, past_covariates, future_covariates):
    """A retrain function telling the model to retrain
    only once at the start of the historical forecast prediction"""
    return counter == 0
