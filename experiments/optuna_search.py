# flake8: noqa
import os
import tempfile
from statistics import mean, stdev

import numpy as np
from model_builders import MODEL_BUILDERS
from optuna_params import PARAMS_GENERATORS
from ray import air, tune
from ray.air import session
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.optuna import OptunaSearch
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm

from darts.dataprocessing.transformers import Scaler

# data and models
from darts.datasets import ETTh1Dataset
from darts.metrics import mae, mape, mase, mse, rmse, smape
from darts.models.forecasting.regression_model import RegressionModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.utils import missing_values
from darts.utils.utils import series2seq


def evaluation_step(config, model_cl, metric, encoders, fixed_params, train, val):

    len_train, len_val = len(train[0]), len(val[0])

    model = MODEL_BUILDERS[model_cl.__name__](
        config,
        len_train,
        len_val,
        encoders=encoders,
        fixed_params=fixed_params,
        out_len=1,
    )  # TODO: replace out_len fixed value

    model.fit(series=train, max_samples_per_ts=fixed_params["MAX_SAMPLES_PER_TS"])

    preds = model.predict(series=train, n=len_val)

    if metric.__name__ == "mase":
        metric_evals = metric(val, preds, train, n_jobs=-1, verbose=True)
    else:
        metric_evals = metric(val, preds, n_jobs=-1, verbose=True)

    metric_evals_reduced = (
        np.mean(metric_evals) if metric_evals != np.nan else float("inf")
    )

    session.report({"metric": metric_evals_reduced})


def optuna_search(
    data, model_map, model_name: str, metric, split: float = 0.8, optuna_dir=None
):

    model_class = model_map[model_name]

    data = series2seq(data)
    train, val = data_cleaning(data, split)

    if not optuna_dir:
        optuna_dir = tempfile.TemporaryDirectory()

    # building optuna objects
    trainable_object = tune.with_parameters(
        evaluation_step,
        model_cl=model_class,
        metric=metric,
        encoders=encoders,
        fixed_params=fixed_params,
        train=train,
        val=val,
    )

    search_alg = OptunaSearch(
        space=PARAMS_GENERATORS[model_cl.__name__],
        metric="metric",
        mode="min",
    )

    tuner = tune.Tuner(
        trainable=trainable_object,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            num_samples=-1,
            time_budget_s=time_budget,
        ),
        run_config=air.RunConfig(
            local_dir=str(optuna_dir),
            name=f"{model_cl.__name__}_tuner_{eval_metric.__name__}",
        ),
    )

    tuner.fit()
    tuner_results = tuner.fit()

    # get best results
    best_params = tuner_results.get_best_result(metric="metric", mode="min").config
    print("best parameters:", best_params)

    best_model = MODEL_BUILDERS[model_cl.__name__](
        **best_params,
        encoders=encoders,
        fixed_params=fixed_params,
        work_dir=optuna_dir,
    )

    return best_model


def data_cleaning(data, split):

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
    if issubclass(model_cl, TorchForecastingModel):
        scaler = Scaler(scaler=MaxAbsScaler())
        train = scaler.fit_transform(train_original)
        val = scaler.transform(val_original)
    else:
        train, val = train_original, val_original

    return train, val


experiment_dir = os.path.join(os.getcwd(), f"test_0312")


dataset = ETTh1Dataset
data = dataset().load()


optuna_search(data, "LinearRegression", mase, optuna_dir=experiment_dir)
