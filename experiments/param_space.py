# flake8: noqa
"""
This file defines, for each model, the hyperparameter space for optuna to explore
"""

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

LAG_RATIO_MIN = 1e-3
LAG_RATIO_MAX = 5e-2


def _params_NHITS(trial):
    in_len = trial.suggest_float("in_len", LAG_RATIO_MIN, LAG_RATIO_MAX)
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])

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

    constants = {
        "layer_widths": 512,
        "pooling_kernel_sizes": None,
        "n_freq_downsample": None,
    }


def _fixed_params_NHITS(dataset, suggested_lags=None, **kwargs):

    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(dataset) * 0.05))
    )
    output["output_chunk_length"] = 1
    output["n_epochs"] = 2
    return output


def _params_NLINEAR(trial):
    in_len = trial.suggest_float("in_len", LAG_RATIO_MIN, LAG_RATIO_MAX)
    shared_weights = trial.suggest_categorical("shared_weights", [False, True])
    const_init = trial.suggest_categorical("const_init", [False, True])
    normalize = trial.suggest_categorical("normalize", [False, True])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])


def _params_DLINEAR(trial):

    in_len = trial.suggest_float("in_len", LAG_RATIO_MIN, LAG_RATIO_MAX)
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])

    kernel_size = trial.suggest_int("kernel_size", 5, 25)
    shared_weights = trial.suggest_categorical("shared_weights", [False, True])
    const_init = trial.suggest_categorical("const_init", [False, True])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)


def _params_TCNMODEL(trial):

    in_len = trial.suggest_float("in_len", LAG_RATIO_MIN, LAG_RATIO_MAX)
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])

    kernel_size = trial.suggest_int("kernel_size", 5, 25)
    num_filters = trial.suggest_int("num_filters", 5, 25)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)


def _fixed_params_TCNMODEL(dataset, suggested_lags=None, **kwargs):
    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(dataset) * 0.05))
    )
    output["output_chunk_length"] = 1
    output["n_epochs"] = 2
    return output


def _params_LGBMModel(trial):

    in_len = trial.suggest_float("in_len", LAG_RATIO_MIN, LAG_RATIO_MAX)
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])

    boosting = trial.suggest_categorical("boosting", ["gbdt", "dart"])
    num_leaves = trial.suggest_int("num_leaves", 2, 50)
    max_bin = trial.suggest_int("max_bin", 100, 500)
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-1, log=True)
    num_iterations = trial.suggest_int("num_iterations", 50, 500)


def _params_LinearRegression(trial, dataset, **kwargs):
    # lag length as a ratio of the train data size
    lags_ratio = trial.suggest_float(
        "lags", max(5, int(len(dataset) * 0.001)), max(6, int(len(dataset) * 0.2))
    )
    add_encoders = trial.suggest_categorical("add_encoders", [False, True])


def _fixed_params_LinearRegression(
    dataset,
    suggested_lags=None,
    has_past_cov=False,
    lags_past_covariates=[-1],
    has_future_cov=False,
    lags_future_covariates=[0],
    **kwargs
):
    output = dict()

    if has_future_cov:
        output["lags_future_covariates"] = lags_future_covariates
    if has_past_cov:
        output["lags_past_covariates"] = lags_past_covariates
    output["lags"] = (
        suggested_lags if suggested_lags else max(5, int(len(dataset) * 0.05))
    )
    return output


def _fixed_params_Catboost(
    dataset,
    suggested_lags=None,
    has_past_cov=False,
    lags_past_covariates=[-1],
    has_future_cov=False,
    lags_future_covariates=[0],
    **kwargs
):
    output = dict()
    if has_future_cov:
        output["lags_future_covariates"] = lags_future_covariates
    if has_past_cov:
        output["lags_past_covariates"] = lags_past_covariates
    output["lags"] = (
        suggested_lags if suggested_lags else max(5, int(len(dataset) * 0.05))
    )
    return output


def _fixed_params_Nbeats(dataset, suggested_lags=None, **kwargs):
    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(dataset) * 0.05))
    )
    output["output_chunk_length"] = 1
    output["generic_architecture"] = False
    output["n_epochs"] = 2
    return output


def _empty_params(**kwargs):
    return dict()


OPTUNA_SEARCH_SPACE = {
    TCNModel.__name__: _params_TCNMODEL,
    DLinearModel.__name__: _params_DLINEAR,
    NLinearModel.__name__: _params_NLINEAR,
    NHiTSModel.__name__: _params_NHITS,
    LightGBMModel.__name__: _params_LGBMModel,
    LinearRegressionModel.__name__: _params_LinearRegression,
}

FIXED_PARAMS = {
    LinearRegressionModel.__name__: _fixed_params_LinearRegression,
    CatBoostModel.__name__: _fixed_params_Catboost,
    NBEATSModel.__name__: _fixed_params_Nbeats,
    NHiTSModel.__name__: _fixed_params_NHITS,
    ARIMA.__name__: _empty_params,
    FFT.__name__: _empty_params,
    Prophet.__name__: _empty_params,
    TCNModel.__name__: _fixed_params_TCNMODEL,
    NaiveSeasonal.__name__: _empty_params,
}
