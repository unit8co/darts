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

LAG_RATIO_MIN = 1e-3
LAG_RATIO_MAX = 5e-2


def _params_NHITS(trial):
    trial.suggest_float("in_len", LAG_RATIO_MIN, LAG_RATIO_MAX)
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

    trial.suggest_float("lr", 5e-5, 1e-3, log=True)


def _fixed_params_NHITS(dataset, suggested_lags=None, **kwargs):

    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(dataset) * 0.05))
    )
    output["output_chunk_length"] = 1
    output["n_epochs"] = 2
    return output


def _params_NLINEAR(trial):
    trial.suggest_float("in_len", LAG_RATIO_MIN, LAG_RATIO_MAX)
    trial.suggest_categorical("shared_weights", [False, True])
    trial.suggest_categorical("const_init", [False, True])
    trial.suggest_categorical("normalize", [False, True])
    trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    trial.suggest_categorical("add_encoders", [False, True])


def _params_DLINEAR(trial):

    trial.suggest_float("in_len", LAG_RATIO_MIN, LAG_RATIO_MAX)
    trial.suggest_categorical("add_encoders", [False, True])

    trial.suggest_int("kernel_size", 5, 25)
    trial.suggest_categorical("shared_weights", [False, True])
    trial.suggest_categorical("const_init", [False, True])
    trial.suggest_float("lr", 5e-5, 1e-3, log=True)


def _params_TCNMODEL(trial):

    trial.suggest_float("in_len", LAG_RATIO_MIN, LAG_RATIO_MAX)
    trial.suggest_categorical("add_encoders", [False, True])

    trial.suggest_int("kernel_size", 5, 25)
    trial.suggest_int("num_filters", 5, 25)
    trial.suggest_categorical("weight_norm", [False, True])
    trial.suggest_int("dilation_base", 2, 4)
    trial.suggest_float("dropout", 0.0, 0.4)
    trial.suggest_float("lr", 5e-5, 1e-3, log=True)


def _fixed_params_TCNMODEL(dataset, suggested_lags=None, **kwargs):
    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(dataset) * 0.05))
    )
    output["output_chunk_length"] = 1
    output["n_epochs"] = 2
    return output


def _params_LGBMModel(trial):

    trial.suggest_float("in_len", LAG_RATIO_MIN, LAG_RATIO_MAX)
    trial.suggest_categorical("add_encoders", [False, True])

    trial.suggest_categorical("boosting", ["gbdt", "dart"])
    trial.suggest_int("num_leaves", 2, 50)
    trial.suggest_int("max_bin", 100, 500)
    trial.suggest_float("learning_rate", 1e-8, 1e-1, log=True)
    trial.suggest_int("num_iterations", 50, 500)


def _params_LinearRegression(trial, dataset, **kwargs):
    # lag length as a ratio of the train data size
    trial.suggest_float(
        "lags", max(5, int(len(dataset) * 0.001)), max(6, int(len(dataset) * 0.2))
    )
    trial.suggest_categorical("add_encoders", [False, True])


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
