import json
import logging
import os
import warnings
from collections import namedtuple
from tempfile import TemporaryDirectory
from typing import Dict, List

import pandas as pd
from model_evaluation import evaluate_model
from optuna_search import optuna_search
from param_space import FIXED_PARAMS

from darts import TimeSeries
from darts.metrics import mae

Dataset = namedtuple(
    "Dataset",
    ["name", "series", "future_covariates", "past_covariates"],
    defaults=(None, None, None),
)


def convert_to_ts(ds: TimeSeries):
    """
    Converts enventual numerical index to datetime index
    (necessary for compatibility down the line)
    """
    return TimeSeries.from_times_and_values(
        pd.to_datetime(ds.time_index), ds.all_values(), columns=ds.components
    )


def create_dataset_dict_entry(
    name: str,
    series: TimeSeries,
    past_covariates=None,
    future_covariates=None,
    max_length=-1,
):
    output = {"dataset_name": name, "series": series[:max_length]}
    if past_covariates:
        output["has_past_cov"] = True
        output["past_covariates"] = past_covariates[:-1]
    if future_covariates:
        output["has_future_cov"] = True
        output["future_covariates"] = future_covariates[:-1]
    return output


def experiment(
    list_datasets: List[Dataset],
    models,
    grid_search=False,
    time_budget=120,
    experiment_dir=None,
    forecast_horizon=1,
    repeat=3,
    metric=mae,
    num_test_points=20,
    silent_search=False,
):
    """
    Takes a list of datasets
    """

    split = 0.8
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

    for dataset in list_datasets:
        for model_class in models:

            stride = int((1 - split) * len(dataset.series) / num_test_points)
            stride = max(stride, 1)

            fh_corrected = forecast_horizon
            if forecast_horizon < 1:
                fh_corrected = int(forecast_horizon * (1 - split) * len(dataset.series))
            print(f"Using forecast horizon of length {fh_corrected}")

            if (
                dataset.name in results
                and model_class.__name__ in results[dataset.name]
            ):
                continue

            model_params = FIXED_PARAMS[model_class.__name__](
                **dataset._asdict(), forecast_horizon=fh_corrected
            )
            if grid_search and time_budget:
                if silent_search:
                    silence_prompt()
                model_params = optuna_search(
                    model_class,
                    fixed_params=model_params,
                    **dataset._asdict(),
                    time_budget=time_budget,
                    optuna_dir=os.path.join(experiment_dir, "optuna"),
                    forecast_horizon=fh_corrected,
                    metric=metric,
                    stride=stride,
                    split=split,
                )
            if silent_search:
                silence_prompt()
            output = evaluate_model(
                model_class,
                **dataset._asdict(),
                model_params=model_params,
                split=split,
                forecast_horizon=fh_corrected,
                repeat=repeat,
                metric=metric,
                stride=stride,
            )

            print("#####################################################")
            print(dataset.name, model_class.__name__, output)

            if dataset.name not in results:
                results[dataset.name] = dict()
            results[dataset.name][model_class.__name__] = output

            # save results
            json.dump(results, open(path_results, "w"))
    results = format_output(results)
    return results


def format_output(results: Dict[str, Dict[str, float]]):
    """Turns a nested dictionnary with (column name, model name, metric value) into a pandas dataframe"""
    results = [  # type: ignore
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


def illustrate(models, ds, split=0.8, forecast_horizon=1, time_budget=0, metric=mae):
    limit = int(len(ds["series"]) * split * 0.95)
    target = ds["series"][limit:]
    target.plot()
    for model_class in models:
        model_params = FIXED_PARAMS[model_class.__name__](**ds)
        if time_budget:
            model_params = optuna_search(
                model_class,
                fixed_params=model_params,
                **ds,
                time_budget=time_budget,
                forecast_horizon=forecast_horizon,
                metric=metric,
            )
        _, output = evaluate_model(  # type: ignore
            model_class,
            **ds,
            model_params=model_params,
            split=0.8,
            forecast_horizon=forecast_horizon,
            get_output_sample=True,
        )
        output.plot(label=f"{model_class.__name__} - {metric(output, target):.2f}")


def silence_prompt():
    # ray, optuna and pytorch: all are very verbose, and ray tune
    # forwards the output for multiple processes in parallel
    # so we need to silence warnings to make the prompt readable
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    warnings.filterwarnings("ignore")
    for logger in loggers:
        logger.setLevel(logging.ERROR)
