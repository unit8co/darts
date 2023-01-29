import argparse
import json
import os
import pickle
import random
from csv import DictWriter
from datetime import datetime
from statistics import mean, stdev

import numpy as np
import torch
from builders import MODEL_BUILDERS
from ray import air, tune
from ray.air import session
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.optuna import OptunaSearch
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm

from darts.dataprocessing.transformers import Scaler

# data and models
from darts.datasets import (
    ElectricityDataset,
    ETTh1Dataset,
    ETTh2Dataset,
    ETTm1Dataset,
    ETTm2Dataset,
    ILINetDataset,
)
from darts.metrics import mae, mape, mase, mse, rmse, smape
from darts.models import (
    CatBoostModel,
    DLinearModel,
    LightGBMModel,
    LinearRegressionModel,
    NBEATSModel,
    NHiTSModel,
    NLinearModel,
    TCNModel,
    XGBModel,
)
from darts.models.forecasting.regression_model import RegressionModel
from darts.models.forecasting.torch_forecasting_model import (
    FutureCovariatesTorchModel,
    MixedCovariatesTorchModel,
    PastCovariatesTorchModel,
    TorchForecastingModel,
)
from darts.utils import missing_values
from darts.utils.utils import series2seq

# experiment configuration

dataset_map = {
    "ETTh1": ETTh1Dataset,
    "ETTh2": ETTh2Dataset,
    "ETTm1": ETTm1Dataset,
    "ETTm2": ETTm2Dataset,
    "ILINet": ILINetDataset,
    "Electricity": ElectricityDataset,
}
model_map = {
    "TCN": TCNModel,
    "DLinear": DLinearModel,
    "NLinear": NLinearModel,
    "NHiTS": NHiTSModel,
    "LinearRegression": LinearRegressionModel,
    "lgbm": LightGBMModel,
    "xgb": XGBModel,
}
metric_map = {
    "smape": smape,
    "mae": mae,
    "mase": mase,
    "mse": mse,
    "rmse": rmse,
    "mape": mape,
}

encoders_past = {
    "datetime_attribute": {"past": ["month", "week", "hour", "dayofweek"]},
    "cyclic": {"past": ["month", "week", "hour", "dayofweek"]},
}

encoders_future = {
    "datetime_attribute": {"future": ["month", "week", "hour", "dayofweek"]},
    "cyclic": {"future": ["month", "week", "hour", "dayofweek"]},
}

argParser = argparse.ArgumentParser()
argParser.add_argument(
    "--dataset",
    help=f"dataset name, choose one of {list(dataset_map.keys())}",
    type=str,
)
argParser.add_argument(
    "--model", help=f"model name, choose one of {list(model_map.keys())}", type=str
)
argParser.add_argument("--random_seed", help="random seed", type=int, default=42)
argParser.add_argument(
    "--period_unit",
    help="a period to consider as unit in multiple of the frequency to the timeseries"
    ", e.g. a period unit of a day for hourly data would be 24.",
    type=int,
    default=24,
)
argParser.add_argument(
    "--subset_size",
    help="subset size as number of timesteps to keep from the dataset",
    type=int,
    default=int(365 * 1.5),
)
argParser.add_argument(
    "--split", help="split ratio for train and validation/test", type=float, default=0.7
)

argParser.add_argument(
    "--load_as_multivariate",
    help="whether to load the dataset as multivariate",
    type=bool,
    default=False,
)
argParser.add_argument("--encoders", help="encoders to use", type=str)
argParser.add_argument(
    "--optimize_with_metric",
    help="whether to optimize based on a metric evaluation on validation set "
    "or based on val_loss",
    type=bool,
    default=True,
)
argParser.add_argument(
    "--eval_metric", help="evaluation metric to use", default="smape"
)
argParser.add_argument(
    "--time_budget", help="time budget in seconds", type=int, default=900
)

argParser.add_argument(
    "--batch_size", help="batch size where applicable", type=int, default=1024
)
argParser.add_argument(
    "--max_n_epochs", help="maximum number of epochs", type=int, default=100
)
argParser.add_argument(
    "--nr_epochs_val_period",
    help="number of epochs between validation",
    type=int,
    default=1,
)
argParser.add_argument(
    "--max_samples_per_ts",
    help="maximum number of samples per timeseries",
    type=int,
    default=1000,
)

args = argParser.parse_args()
print(f"all arguments : {args}")

dataset = dataset_map[args.dataset]
model_cl = model_map[args.model]
random_seed = args.random_seed
# data
PERIOD_UNIT = args.period_unit
subset_size = args.subset_size * PERIOD_UNIT
split = args.split
load_as_multivariate = args.load_as_multivariate


# model training
fixed_params = {
    "BATCH_SIZE": args.batch_size,
    "MAX_N_EPOCHS": args.max_n_epochs,
    "NR_EPOCHS_VAL_PERIOD": args.nr_epochs_val_period,
    "MAX_SAMPLES_PER_TS": args.max_samples_per_ts,
    "RANDOM_STATE": random_seed,
}

optimize_with_metric = args.optimize_with_metric
eval_metric = metric_map[args.eval_metric]
time_budget = args.time_budget
encoders = (
    json.loads(args.encoders)
    if args.encoders
    else (
        encoders_future
        if issubclass(
            model_cl,
            (MixedCovariatesTorchModel, FutureCovariatesTorchModel, RegressionModel),
        )
        else encoders_past
    )
)

RUNS = 5

IN_MIN = 5  # make argument?
IN_MAX = 30


def _params_NHITS(trial):
    in_len = trial.suggest_int("in_len", IN_MIN * PERIOD_UNIT, IN_MAX * PERIOD_UNIT)

    out_len = trial.suggest_int("out_len", 1, (in_len-1) * PERIOD_UNIT)

    num_stacks = trial.suggest_int("num_stacks", 2, 5)
    num_blocks = trial.suggest_int("num_blocks", 1, 3)
    num_layers = trial.suggest_int("num_layers", 2, 5)
    activation = trial.suggest_categorical(
        "activation",
        ["ReLU", "RReLU", "PReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid"],
    )

    MaxPool1d = trial.suggest_categorical("MaxPool1d", [False, True])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])

    constants = {
        "layer_widths": 512,
        "pooling_kernel_sizes": None,
        "n_freq_downsample": None,
    }

    return constants


def _params_NLINEAR(trial):
    in_len = trial.suggest_int("in_len", IN_MIN * PERIOD_UNIT, IN_MAX * PERIOD_UNIT)

    out_len = trial.suggest_int("out_len", 1, (in_len-1) * PERIOD_UNIT)

    shared_weights = trial.suggest_categorical("shared_weights", [False, True])
    const_init = trial.suggest_categorical("const_init", [False, True])
    normalize = trial.suggest_categorical("normalize", [False, True])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])

    return None


def _params_DLINEAR(trial):

    in_len = trial.suggest_int("in_len", IN_MIN * PERIOD_UNIT, IN_MAX * PERIOD_UNIT)

    out_len = trial.suggest_int("out_len", 1, (in_len-1) * PERIOD_UNIT)

    kernel_size = trial.suggest_int("kernel_size", 5, 25)
    shared_weights = trial.suggest_categorical("shared_weights", [False, True])
    const_init = trial.suggest_categorical("const_init", [False, True])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])

    return None


def _params_TCNMODEL(trial):

    in_len = trial.suggest_int("in_len", IN_MIN * PERIOD_UNIT, IN_MAX * PERIOD_UNIT)

    out_len = trial.suggest_int("out_len", 1, (in_len-1) * PERIOD_UNIT)

    kernel_size = trial.suggest_int("kernel_size", 5, 25)
    num_filters = trial.suggest_int("num_filters", 5, 25)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])

    return None


def _params_LGBMModel(trial):

    lags = trial.suggest_int("lags", IN_MIN * PERIOD_UNIT, IN_MAX * PERIOD_UNIT)
    out_len = trial.suggest_int("out_len", 1, (lags-1) * PERIOD_UNIT)

    boosting = trial.suggest_categorical("boosting", ["gbdt", "dart"])
    num_leaves = trial.suggest_int("num_leaves", 2, 50)
    max_bin = trial.suggest_int("max_bin", 100, 500)
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-1, log=True)
    num_iterations = trial.suggest_int("num_iterations", 50, 500)
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])

    return None


def _params_XGBModel(trial):
    pass


def _params_LinearRegression(trial):

    lags = trial.suggest_int("lags", IN_MIN * PERIOD_UNIT, IN_MAX * PERIOD_UNIT)
    out_len = trial.suggest_int("out_len", 1, lags - PERIOD_UNIT)

    return None


PARAMS_GENERATORS = {
    TCNModel.__name__: _params_TCNMODEL,
    DLinearModel.__name__: _params_DLINEAR,
    NLinearModel.__name__: _params_NLINEAR,
    NHiTSModel.__name__: _params_NHITS,
    LightGBMModel.__name__: _params_LGBMModel,
    XGBModel.__name__: _params_XGBModel,
    LinearRegressionModel.__name__: _params_LinearRegression,
}


if __name__ == "__main__":

    # Fix random states
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)

    exp_start_time = datetime.now()
    exp_name = f"{model_cl.__name__}_{exp_start_time.strftime('%Y-%m-%d')}_pid{os.getpid()}_seed{random_seed}"

    # create directories
    experiment_root = os.path.join(
        os.getcwd(), f"benchmark_experiments/{dataset.__name__}"
    )
    experiment_dir = os.path.join(os.getcwd(), f"{experiment_root}/{exp_name}")
    print(f"experiment directory: {experiment_dir}")
    os.makedirs(experiment_dir, exist_ok=True)

    # setup logging file
    fields_names = [
        "experiment name",
        "model",
        "metric",
        "metric on test-mean",
        "metric on test-std",
        "model training time-mean",
        "model training time-std",
        "model inference time-mean",
        "model inference time-std",
        "seed",
        "optimize with metric",
    ]

    logging_file = f"{experiment_root}/logs.csv"

    if not os.path.exists(logging_file):
        with open(logging_file, "a") as f:
            writer = DictWriter(f, fieldnames=fields_names)
            writer.writeheader()
            f.close()

    print(f"log file location : {logging_file}")

    # read data
    if "multivariate" in dataset.__init__.__code__.co_varnames:
        data = dataset(multivariate=load_as_multivariate).load()
    else:
        data = dataset().load()

    data = series2seq(data)

    data = [s[-subset_size:].astype(np.float32) for s in tqdm(data)]

    # split : train, validation , test (validation and test have same length)
    all_splits = [list(s.split_after(split)) for s in data]
    train_original = [split[0] for split in all_splits]
    vals = [split[1] for split in all_splits]
    vals = [list(s.split_after(0.5)) for s in vals]
    val_original = [s[0] for s in vals]
    test_original = [s[1] for s in vals]

    train_len = len(train_original[0])
    val_len = len(val_original[0])
    test_len = len(test_original[0])
    num_series = len(train_original)
    n_components = train_original[0].n_components

    print("number of series:", num_series)
    print("number of components:", n_components)
    print("training series length:", train_len)
    print("validation series length:", val_len)
    print("test series length:", test_len)

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

        missing_ratio = missing_values.missing_values_ratio(test_original[i])
        print(f"missing values ratio in test series {i} = {missing_ratio}")
        print("filling test missing values by interpolation")
        if missing_ratio > 0.0:
            missing_values.fill_missing_values(test_original[i])

    # scale data
    scaler = (
        Scaler(scaler=MaxAbsScaler())
        if issubclass(model_cl, TorchForecastingModel)
        else None
    )

    if scaler is not None:
        train = scaler.fit_transform(train_original)
        val = scaler.transform(val_original)
        test = scaler.transform(test_original)
    else:
        train = train_original
        val = val_original
        test = test_original

    # Tuner objective functions

    def objective_val_loss(
        config, model_cl, encoders, fixed_params, train=train, val=val
    ):

        metrics = {"metric": "val_loss"}

        callbacks = [TuneReportCallback(metrics, on="validation_end")]

        model = MODEL_BUILDERS[model_cl.__name__](
            **config, fixed_params=fixed_params, encoders=encoders, callbacks=callbacks
        )

        # train the model
        if "val_series" in model.fit.__code__.co_varnames:
            model.fit(
                series=train,
                val_series=val,
                max_samples_per_ts=fixed_params["MAX_SAMPLES_PER_TS"],
            )
        else:
            model.fit(
                series=train, max_samples_per_ts=fixed_params["MAX_SAMPLES_PER_TS"]
            )

    def objective_metric(
        config, model_cl, metric, encoders, fixed_params, train=train, val=val
    ):
        model = MODEL_BUILDERS[model_cl.__name__](
            **config, encoders=encoders, fixed_params=fixed_params
        )

        # train the model
        if "val_series" in model.fit.__code__.co_varnames:
            model.fit(
                series=train,
                val_series=val,
                max_samples_per_ts=fixed_params["MAX_SAMPLES_PER_TS"],
            )
        else:
            model.fit(
                series=train, max_samples_per_ts=fixed_params["MAX_SAMPLES_PER_TS"]
            )

        # DL Models : use best model for subsequent evaluation
        if isinstance(model, TorchForecastingModel):
            model = model_cl.load_from_checkpoint(
                model_cl.__name__, work_dir=os.getcwd(), best=True
            )

        preds = model.predict(series=train, n=val_len)

        if metric.__name__ == "mase":
            metric_evals = metric(val, preds, train, n_jobs=-1, verbose=True)
        else:
            metric_evals = metric(val, preds, n_jobs=-1, verbose=True)

        metric_evals_reduced = (
            np.mean(metric_evals) if metric_evals != np.nan else float("inf")
        )

        session.report({"metric": metric_evals_reduced})

    objective_metric_with_params = tune.with_parameters(
        objective_metric,
        model_cl=model_cl,
        metric=eval_metric,
        encoders=encoders,
        fixed_params=fixed_params,
        train=train,
        val=val,
    )

    objective_val_loss_with_params = tune.with_parameters(
        objective_val_loss,
        model_cl=model_cl,
        encoders=encoders,
        fixed_params=fixed_params,
        train=train,
        val=val,
    )

    # https://docs.ray.io/en/latest/tune/examples/optuna_example.html
    # the default optuna algorithm is TPEsampler
    search_alg = OptunaSearch(
        space=PARAMS_GENERATORS[model_cl.__name__],
        metric="metric",
        mode="min",
    )

    tuner = tune.Tuner(
        trainable=objective_metric_with_params
        if optimize_with_metric
        else objective_val_loss_with_params,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            num_samples=-1,
            time_budget_s=time_budget,
        ),
        run_config=air.RunConfig(
            local_dir=experiment_dir,
            name=f"{model_cl.__name__}_tuner_{eval_metric.__name__}",
        ),
    )

    # run tuner
    tuner_results = tuner.fit()

    # get best results
    best_params = tuner_results.get_best_result(metric="metric", mode="min").config
    print("best parameters:", best_params)

    runs_accuracy = []
    runs_training_time = []
    runs_inference_time = []

    for _ in range(RUNS):
        # train best model and get training time
        best_model = MODEL_BUILDERS[model_cl.__name__](
            **best_params,
            encoders=encoders,
            fixed_params=fixed_params,
            work_dir=experiment_dir,
        )

        if isinstance(best_model, TorchForecastingModel):
            best_model.n_epochs = fixed_params["MAX_N_EPOCHS"] + 50

        train_start_time = datetime.now()
        # train the model
        if "val_series" in best_model.fit.__code__.co_varnames:
            best_model.fit(
                series=train,
                val_series=val,
                max_samples_per_ts=fixed_params["MAX_SAMPLES_PER_TS"],
            )
        else:
            best_model.fit(
                series=train,
                max_samples_per_ts=fixed_params["MAX_SAMPLES_PER_TS"],
            )

        train_end_time = datetime.now()
        training_time = (train_end_time - train_start_time).total_seconds()
        runs_training_time.append(training_time)

        # inference with best model and inference time
        inference_start_time = datetime.now()
        test_predictions = best_model.predict(series=val, n=test_len)
        inference_end_time = datetime.now()
        inference_time = (inference_end_time - inference_start_time).total_seconds()
        runs_inference_time.append(inference_time)

        # best model accuracy
        if eval_metric.__name__ == "mase":
            metric_evals = eval_metric(
                test, test_predictions, train, n_jobs=-1, verbose=True
            )
        else:
            metric_evals = eval_metric(test, test_predictions, n_jobs=-1, verbose=True)

        metric_evals_mean = (
            np.mean(metric_evals) if metric_evals != np.nan else float("inf")
        )
        # if multiple series
        # metric_evals_std = np.std(metric_evals)
        runs_accuracy.append(metric_evals_mean)

    # backup resutls
    # dump best_prams
    with open(f"{experiment_dir}/best_params_bkp.pkl", "wb") as f:
        pickle.dump(best_params, f)

    # dump encoders configuration
    with open(f"{experiment_dir}/encoders.txt", "w") as f:
        f.write(json.dumps(encoders))
        f.close()

    with open(f"{experiment_dir}/all_runs_stats.txt", "w") as f:
        f.write(f"accuracy: {str(runs_accuracy)} \n")
        f.write(f"training time: {str(runs_training_time)} \n")
        f.write(f"inference time: {str(runs_inference_time)} \n")
        f.close()

    data_line = {
        "experiment name": exp_name,
        "model": model_cl.__name__,
        "metric": eval_metric.__name__,
        "metric on test-mean": mean(runs_accuracy),
        "metric on test-std": stdev(runs_accuracy),
        "model training time-mean": mean(runs_training_time),
        "model training time-std": stdev(runs_training_time),
        "model inference time-mean": mean(runs_inference_time),
        "model inference time-std": stdev(runs_inference_time),
        "seed": random_seed,
        "optimize with metric": optimize_with_metric,
    }

    with open(logging_file, "a") as f:
        writer = DictWriter(f, fieldnames=fields_names)
        writer.writerow(data_line)
        f.close()
