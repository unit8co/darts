# flake8: noqa
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from optuna_params import FIXED_PARAMS

from darts import TimeSeries
from darts.datasets import (
    ETTh1Dataset,
    ExchangeRateDataset,
    GasRateCO2Dataset,
    SunspotsDataset,
    WeatherDataset,
)
from darts.metrics import mae
from darts.models import (
    ARIMA,
    FFT,
    CatBoostModel,
    DLinearModel,
    LightGBMModel,
    LinearRegressionModel,
    NaiveSeasonal,
    NBEATSModel,
    NHiTSModel,
    NLinearModel,
    Prophet,
    TCNModel,
    XGBModel,
)
from darts.models.forecasting.forecasting_model import LocalForecastingModel

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def randommness(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def convert_to_ts(ds):
    return TimeSeries.from_times_and_values(
        pd.to_datetime(ds.time_index), ds.all_values(), columns=ds.components
    )


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


metric = mae
models = [
    NaiveSeasonal,
    FFT,
    Prophet,
    ARIMA,
    TCNModel,
    NHiTSModel,
    NBEATSModel,
    LinearRegressionModel,
    CatBoostModel,
]

datasets = []
ds = ETTh1Dataset().load()
datasets += [
    {
        "dataset_name": "ETTh1",
        "dataset": ds["OT"],
        "future_cov": ds[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]],
        "has_future_cov": True,
    }
]
ds = WeatherDataset().load().resample("1h")
datasets += [
    {
        "dataset_name": "Weather",
        "dataset": ds["T (degC)"],
        "future_cov": ds[
            [
                "p (mbar)",
                "rh (%)",
                "VPmax (mbar)",
                "VPact (mbar)",
                "VPdef (mbar)",
                "sh (g/kg)",
                "H2OC (mmol/mol)",
                "rho (g/m**3)",
                "wv (m/s)",
                "max. wv (m/s)",
                "wd (deg)",
                "rain (mm)",
                "raining (s)",
                "SWDR (W/m²)",
                "PAR (µmol/m²/s)",
                "max. PAR (µmol/m²/s)",
                "CO2 (ppm)",
            ]
        ],
        "has_future_cov": True,
    }
]

ds = convert_to_ts(ExchangeRateDataset().load()["0"])
datasets += [{"dataset": ds, "dataset_name": "ExchangeRate"}]
ds = SunspotsDataset().load()["Sunspots"]
datasets += [{"dataset": ds, "dataset_name": "Sunspots"}]
ds = convert_to_ts(GasRateCO2Dataset().load()["CO2%"])
datasets += [{"dataset": ds, "dataset_name": "GasRateCO2"}]

results = []
for dataset in datasets:
    print("\n\n", dataset["dataset_name"])
    for model_class in models:
        fixed_params = FIXED_PARAMS[model_class.__name__](**dataset)
        output = evaluate_model(
            model_class, **dataset, model_params=fixed_params, split=0.8
        )
        print(model_class.__name__, output)
        results.append((dataset["dataset_name"], model_class.__name__, output))
print(results)
