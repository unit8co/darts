"""
This is the main file for the benchmarking experiment.
"""
import json
import logging
import os
import warnings
from tempfile import TemporaryDirectory
from typing import Dict

import pandas as pd
from model_evaluation import evaluate_model
from optuna_search import optuna_search
from param_space import FIXED_PARAMS

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
    LightGBMModel,
    LinearRegressionModel,
    NaiveSeasonal,
    NBEATSModel,
    NHiTSModel,
    NLinearModel,
    Prophet,
    TCNModel,
)
from darts.utils import missing_values

warnings.filterwarnings("ignore")


def convert_to_ts(ds: TimeSeries):
    return TimeSeries.from_times_and_values(
        pd.to_datetime(ds.time_index), ds.all_values(), columns=ds.components
    )


def experiment(datasets, models, grid_search=False, experiment_dir=None):

    # experiment_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if experiment_dir:
        if not os.path.isdir(experiment_dir):
            os.mkdir(experiment_dir)
    else:
        temp_dir = TemporaryDirectory()
        experiment_dir = str(temp_dir.name)

    # we use dict to handle multiple entries of the same dataset/model
    path_results = os.path.join(experiment_dir, "results.json")
    if os.path.isfile(path_results):
        results = json.load(open(path_results))
    else:
        results = dict()

    for dataset in datasets:
        for model_class in models:

            if results.get(dataset["dataset_name"]) and results[
                dataset["dataset_name"]
            ].get(model_class.__name__):
                continue

            model_params = FIXED_PARAMS[model_class.__name__](**dataset)
            if grid_search:
                model_params = optuna_search(
                    model_class,
                    fixed_params=model_params,
                    **dataset,
                    time_budget=180,
                    optuna_dir=os.path.join(experiment_dir, "optuna")
                )

            output = evaluate_model(
                model_class, **dataset, model_params=model_params, split=0.8
            )

            print("#####################################################")
            print(dataset["dataset_name"], model_class.__name__, output)
            if not results.get(dataset["dataset_name"]):
                results[dataset["dataset_name"]] = dict()
            results[dataset["dataset_name"]][model_class.__name__] = output

            # save results
            json.dump(results, open(path_results, "w"))

    print(format_output(results))


def format_output(results: Dict[str, Dict[str, float]]):
    """Turns a nested dictionnary with (column name, model name, metric value) into a pandas dataframe"""
    results = [
        (dataset, model, metric)
        for dataset, model_dict in results.items()
        for model, metric in model_dict.items()
    ]
    df = pd.DataFrame(results, columns=["dataset", "model", "metric"])
    df = (
        df.groupby(["dataset", "model"], as_index=False)
        .first()
        .pivot(index="dataset", columns="model", values="metric")
    )
    return df


use_optuna = True
metric = mae
models = [
    NaiveSeasonal,
    FFT,
    Prophet,
    NLinearModel,
    LightGBMModel,  # some warnings for boosting and num_iterations overriding some parameters but it works
    TCNModel,
    NHiTSModel,
    NBEATSModel,
    LinearRegressionModel,
    ARIMA,  # Raytune gets stuck on this one
]

datasets = []

ds = convert_to_ts(GasRateCO2Dataset().load()["CO2%"])
datasets += [{"series": ds, "dataset_name": "GasRateCO2"}]

ds = missing_values.fill_missing_values(WeatherDataset().load().resample("1h"))
datasets += [
    {
        "dataset_name": "Weather",
        "series": ds["T (degC)"],
        "past_covariates": ds[
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
        "has_past_cov": True,
    }
]
ds = missing_values.fill_missing_values(ETTh1Dataset().load())
datasets += [
    {
        "dataset_name": "ETTh1",
        "series": ds["OT"],
        "future_covariates": ds[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]],
        "has_future_cov": True,
    }
]

ds = missing_values.fill_missing_values(
    convert_to_ts(ExchangeRateDataset().load()["0"])
)
datasets += [{"series": ds, "dataset_name": "ExchangeRate"}]
ds = SunspotsDataset().load()["Sunspots"]
datasets += [{"series": ds, "dataset_name": "Sunspots"}]

# ray, optuna and pytorch: all are very verbose, so we need to silence them
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.ERROR)

experiment(
    datasets=datasets[:1],
    models=models,
    grid_search=True,
    experiment_dir=os.path.join(os.getcwd(), "results_2"),
)
