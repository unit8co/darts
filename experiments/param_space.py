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
)


def _params_NHITS(trial, target_dataset, **kwargs):
    trial.suggest_int(
        "input_chunk_length",
        max(5, int(len(target_dataset) * 0.001)),
        max(6, int(len(target_dataset) * 0.2)),
        log=True,
    )
    trial.suggest_categorical("add_encoders", [False, True])

    trial.suggest_int("num_stacks", 2, 5)
    trial.suggest_int("num_blocks", 1, 3)
    trial.suggest_int("num_layers", 2, 5)
    trial.suggest_categorical(
        "activation",
        ["ReLU", "RReLU", "PReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid"],
    )

    trial.suggest_categorical("MaxPool1d", [False, True])
    trial.suggest_float("dropout", 0.0, 0.4)


def _fixed_params_NHITS(target_dataset, suggested_lags=None, **kwargs):

    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(target_dataset) * 0.05))
    )
    output["output_chunk_length"] = 1
    output["n_epochs"] = 2
    return output


def _params_NLINEAR(trial, target_dataset, **kwargs):
    trial.suggest_int(
        "input_chunk_length",
        max(5, int(len(target_dataset) * 0.001)),
        max(6, int(len(target_dataset) * 0.2)),
        log=True,
    )
    trial.suggest_categorical("shared_weights", [False, True])
    trial.suggest_categorical("const_init", [False, True])
    trial.suggest_categorical("normalize", [False, True])
    trial.suggest_categorical("add_encoders", [False, True])


def _params_DLINEAR(trial, target_dataset, **kwargs):

    input_size = trial.suggest_int(
        "input_chunk_length",
        max(5, int(len(target_dataset) * 0.001)),
        max(6, int(len(target_dataset) * 0.2)),
        log=True,
    )
    trial.suggest_categorical("add_encoders", [False, True])

    trial.suggest_int("kernel_size", 2, input_size)
    trial.suggest_categorical("shared_weights", [False, True])
    trial.suggest_categorical("const_init", [False, True])


def _params_TCNMODEL(trial, target_dataset, **kwargs):

    input_size = trial.suggest_int(
        "input_chunk_length",
        max(5, int(len(target_dataset) * 0.001)),
        max(6, int(len(target_dataset) * 0.2)),
        log=True,
    )
    trial.suggest_categorical("add_encoders", [False, True])

    trial.suggest_int("kernel_size", 2, input_size)
    trial.suggest_int("num_filters", 2, 10)
    trial.suggest_categorical("weight_norm", [False, True])
    trial.suggest_int("dilation_base", 2, 4)
    trial.suggest_float("dropout", 0.0, 0.4)


def _fixed_params_TCNMODEL(target_dataset, suggested_lags=None, **kwargs):
    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(target_dataset) * 0.05))
    )
    output["output_chunk_length"] = 1
    output["n_epochs"] = 2
    return output


def _params_LGBMModel(trial, target_dataset, **kwargs):

    trial.suggest_int(
        "lags",
        max(5, int(len(target_dataset) * 0.001)),
        max(6, int(len(target_dataset) * 0.2)),
        log=True,
    )
    trial.suggest_categorical("add_encoders", [False, True])

    trial.suggest_categorical("boosting", ["gbdt", "dart"])
    trial.suggest_int("num_leaves", 2, 50)
    trial.suggest_int("max_bin", 100, 500)
    trial.suggest_float("learning_rate", 1e-8, 1e-1, log=True)
    trial.suggest_int("num_iterations", 50, 500)


def _fixed_params_LGBMModel(
    target_dataset,
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
        suggested_lags if suggested_lags else max(5, int(len(target_dataset) * 0.05))
    )
    return output


def _params_LinearRegression(trial, target_dataset, **kwargs):
    # lag length as a ratio of the train data size
    trial.suggest_int(
        "lags",
        max(5, int(len(target_dataset) * 0.001)),
        max(6, int(len(target_dataset) * 0.2)),
    )
    trial.suggest_categorical("add_encoders", [False, True])


def _fixed_params_LinearRegression(
    target_dataset,
    suggested_lags: int = None,
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
        suggested_lags if suggested_lags else max(5, int(len(target_dataset) * 0.05))
    )
    return output


def _fixed_params_Catboost(
    target_dataset,
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
        suggested_lags if suggested_lags else max(5, int(len(target_dataset) * 0.05))
    )
    return output


def _params_Nbeats(trial, target_dataset, **kwargs):
    trial.suggest_int("input_chunk_length", 2, max(5, int(len(target_dataset) * 0.15)))


def _fixed_params_Nbeats(target_dataset, suggested_lags=None, **kwargs):
    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(target_dataset) * 0.05))
    )
    output["output_chunk_length"] = 1
    output["generic_architecture"] = False
    output["n_epochs"] = 5
    return output


def _empty_params(**kwargs):
    return dict()


OPTUNA_SEARCH_SPACE = {
    TCNModel.__name__: _params_TCNMODEL,
    DLinearModel.__name__: _params_DLINEAR,
    NLinearModel.__name__: _params_NLINEAR,
    NHiTSModel.__name__: _params_NHITS,
    NBEATSModel.__name__: _params_Nbeats,
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
    LightGBMModel.__name__: _fixed_params_LGBMModel,
}
