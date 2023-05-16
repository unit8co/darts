import tempfile
import warnings
from typing import Callable, Dict

import ray
from model_evaluation import evaluate_model
from param_space import FIXED_PARAMS, OPTUNA_SEARCH_SPACE, optuna2params
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from sklearn.preprocessing import MaxAbsScaler

from darts.dataprocessing.transformers import Scaler

# data and models
from darts.metrics import mae
from darts.models.forecasting.torch_forecasting_model import (
    ForecastingModel,
    TorchForecastingModel,
)
from darts.timeseries import TimeSeries
from darts.utils import missing_values


def evaluation_step(
    config,
    model_class: ForecastingModel,
    series: TimeSeries,
    fixed_params: Dict = dict(),
    metric: Callable[[TimeSeries, TimeSeries], float] = mae,
    split: float = 0.85,
    past_covariates: TimeSeries = None,
    future_covariates: TimeSeries = None,
    forecast_horizon=1,
    stride: int = 1,
):
    import warnings

    warnings.filterwarnings("ignore")

    # convert optuna config to accept more complex params
    config = optuna2params(config)
    model_params = {**fixed_params, **config}

    result = evaluate_model(
        model_class=model_class,
        model_params=model_params,
        series=series,
        metric=metric,
        split=split,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        stride=stride,
        forecast_horizon=forecast_horizon,
    )
    session.report({"metric": result})


def optuna_search(
    model_class: ForecastingModel,
    series: TimeSeries,
    past_covariates: TimeSeries = None,
    future_covariates: TimeSeries = None,
    fixed_params: Dict = None,
    optuna_search_space: Callable = None,
    metric: Callable[[TimeSeries, TimeSeries], float] = mae,
    split: float = 0.85,
    stride: int = 1,
    time_budget=60,
    optuna_dir=None,
    forecast_horizon=1,
    **kwargs,
):

    if not optuna_dir:
        temp_dir = tempfile.TemporaryDirectory()
        optuna_dir = temp_dir.name

    dataset_data = {
        "series": series.split_after(split)[0],
    }

    fixed_params = (
        fixed_params.copy()
        if fixed_params
        else FIXED_PARAMS[model_class.__name__](**dataset_data)
    )

    if not optuna_search_space and model_class.__name__ not in OPTUNA_SEARCH_SPACE:
        warnings.warn(
            "Optuna search space not found. skipping optuna search and returning fixed params instead."
        )
        return fixed_params

    optuna_search_space = optuna_search_space or (
        lambda trial: OPTUNA_SEARCH_SPACE[model_class.__name__](
            trial, **dataset_data, forecast_horizon=forecast_horizon
        )
    )

    # building optuna objects
    trainable_object = ray.tune.with_parameters(
        evaluation_step,
        series=series,
        model_class=model_class,
        metric=metric,
        fixed_params=fixed_params,
        split=split,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        stride=stride,
        forecast_horizon=forecast_horizon,
    )

    search_alg = OptunaSearch(
        space=optuna_search_space,
        metric="metric",
        mode="min",
    )

    tuner = ray.tune.Tuner(
        trainable=trainable_object,
        tune_config=ray.tune.TuneConfig(
            search_alg=search_alg,
            num_samples=-1,
            time_budget_s=time_budget,
            max_concurrent_trials=4,
        ),
        run_config=ray.air.RunConfig(
            local_dir=str(optuna_dir),
            name=f"{model_class.__name__}_tuner_{metric.__name__}",
            verbose=1,
        ),
    )

    tuner_results = tuner.fit()

    # get best results
    best_params = tuner_results.get_best_result(metric="metric", mode="min").config
    best_params = optuna2params(best_params)
    best_params = {**fixed_params, **best_params}
    return best_params


def data_cleaning(data, split, model_class):

    # split : train, validation , test (validation and test have same length)

    listed_splits = [list(s.split_after(split)) for s in data]
    train_original, val_original = map(list, zip(*listed_splits))

    train_len = len(train_original[0])
    val_len = len(val_original[0])
    num_series = len(train_original)
    n_components = train_original[0].n_components

    print("number of series:", num_series)
    print("number of components:", n_components)
    print("training series length:", train_len)
    print("validation series length:", val_len)

    # check if missing values and fill
    for i in range(num_series):
        missing_ratio = missing_values.missing_values_ratio(train_original[i])
        print(f"missing values ratio in training series {i} = {missing_ratio}")
        print("filling training missing values by interpolation")
        if missing_ratio > 0.0:
            missing_values.fill_missing_values(train_original[i])

        missing_ratio = missing_values.missing_values_ratio(val_original[i])
        print(f"missing values ratio in validation series {i} = {missing_ratio}")
        print("filling validation missing values by interpolation")
        if missing_ratio > 0.0:
            missing_values.fill_missing_values(val_original[i])

    # scale data
    if issubclass(model_class, TorchForecastingModel):
        scaler = Scaler(scaler=MaxAbsScaler())
        train = scaler.fit_transform(train_original)
        val = scaler.transform(val_original)
    else:
        train, val = train_original, val_original

    return train, val
