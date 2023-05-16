"""
This file defines, for each model, a generic hyperparameter space.
Every model have
- fixed params: default generic params for fast results
- optuna params: a range of params to explore with optuna search

Parameters explored by optuna will override the fix param if there is a conflict.
"""
from typing import Any, Dict

from sklearn.preprocessing import StandardScaler

from darts.dataprocessing.transformers import Scaler
from darts.models import (
    ARIMA,
    FFT,
    AutoARIMA,
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

# --------------------------------------- UTILS


N_EPOCHS = 15
encoders_dict_past = {
    "cyclic": {"past": ["month", "week", "hour", "dayofweek"]},
    "datetime_attribute": {"past": ["year"]},
    "transformer": Scaler(StandardScaler()),
}

encoders_dict_future = {
    "cyclic": {"future": ["month", "week", "hour", "dayofweek"]},
    "datetime_attribute": {"future": ["year"]},
    "transformer": Scaler(StandardScaler()),
}


def optuna2params(optuna_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optuna only takes ints/float/bool/categorical parameters.
    If we want to pass more complex parameters (dict/list, ...)
    we have to find a workaround. This function converts optuna parameters to
    enable models to take more complex parameters.
    e.g: we pass 2 ints to optuna and in here, we put those 2 ints in a list
    """

    # Encoders want dicts params, so we converts boolean to dicts of encoders
    output_params = optuna_params.copy()
    if "add_past_encoders" in output_params:
        output_params["add_encoders"] = (
            encoders_dict_past if output_params["add_past_encoders"] else None
        )
        del output_params["add_past_encoders"]
    if "add_future_encoders" in output_params:
        output_params["add_encoders"] = (
            encoders_dict_future if output_params["add_future_encoders"] else None
        )
        del output_params["add_future_encoders"]

    # Lags require a tuple with 2 ints, so we convert 2 ints to a tuple
    if (
        "lags_future_covariates_past" in output_params
        or "lags_future_covariates_future" in output_params
    ):
        # converts int to list for lags_future_covariates
        output_params["lags_future_covariates"] = (
            output_params["lags_future_covariates_past"],
            output_params["lags_future_covariates_future"],
        )
        del output_params["lags_future_covariates_past"]
        del output_params["lags_future_covariates_future"]

    if optuna_params.get("prophet_seasonality_period"):
        optuna_params["add_seasonalities"] = {
            "name": "optuna_seasonality",
            "seasonal_periods": optuna_params["prophet_seasonality_period"],
            "fourier_order": 10,
            "mode": "additive",  # assuming additive to limit search space
        }

        del output_params["prophet_seasonality_period"]

    return output_params


def _empty_params(**kwargs) -> Dict[str, Any]:
    """Place holder for models which already have good default params"""
    return dict()


def suggest_lags(trial, series, var_name: str):
    """A generic function to suggest a lag value depending on the TS length"""
    lags = trial.suggest_int(
        var_name,
        max(5, int(len(series) * 0.001)),
        max(6, int(len(series) * 0.2)),
        log=True,
    )
    return lags


def fixed_lags(series, suggested_lags=None):
    return suggested_lags or max(3, int(len(series) * 0.02))


# --------------------------------------- NHITS
def _params_NHITS(trial, series, forecast_horizon, **kwargs):
    suggest_lags(trial, series, "input_chunk_length")
    trial.suggest_categorical("output_chunk_length", [1, forecast_horizon])
    trial.suggest_categorical("add_past_encoders", [False, True])
    trial.suggest_int("num_stacks", 2, 3)
    trial.suggest_int("layer_widths", 64, 1024)
    trial.suggest_categorical("MaxPool1d", [False, True])
    trial.suggest_float("dropout", 0.0, 0.4)


def _fixed_params_NHITS(series, forecast_horizon, suggested_lags=None, **kwargs):

    output = dict()
    output["input_chunk_length"] = fixed_lags(series, suggested_lags)
    output["output_chunk_length"] = forecast_horizon
    output["n_epochs"] = N_EPOCHS
    output["pl_trainer_kwargs"] = {"enable_progress_bar": False}

    return output


# --------------------------------------- NLINEAR
def _params_NLINEAR(trial, series, forecast_horizon, **kwargs):

    suggest_lags(trial, series, "input_chunk_length")
    trial.suggest_categorical("output_chunk_length", [1, forecast_horizon])

    trial.suggest_categorical("const_init", [False, True])
    normalize = trial.suggest_categorical("normalize", [False, True])
    shared_weights = trial.suggest_categorical("shared_weights", [False, True])

    if not shared_weights and not normalize:
        # in current version, darts does not support covariates with normalize.
        # Should be fixed with https://github.com/unit8co/darts/pull/1583
        trial.suggest_categorical("add_past_encoders", [False, True])


def _fixed_params_NLINEAR(series, forecast_horizon, suggested_lags=None, **kwargs):

    output = dict()
    output["input_chunk_length"] = fixed_lags(series, suggested_lags)
    output["output_chunk_length"] = forecast_horizon
    output["n_epochs"] = N_EPOCHS
    return output


# --------------------------------------- DLINEAR
def _params_DLINEAR(trial, series, forecast_horizon, **kwargs):

    input_size = suggest_lags(trial, series, "input_chunk_length")
    trial.suggest_categorical("output_chunk_length", [1, forecast_horizon])

    trial.suggest_int("kernel_size", 2, input_size)
    shared_weights = trial.suggest_categorical("shared_weights", [False, True])
    if not shared_weights:
        trial.suggest_categorical("add_past_encoders", [False, True])

    trial.suggest_categorical("const_init", [False, True])


def _fixed_params_DLINEAR(series, forecast_horizon, suggested_lags=None, **kwargs):
    output = dict()
    output["input_chunk_length"] = fixed_lags(series, suggested_lags)
    output["output_chunk_length"] = forecast_horizon
    output["n_epochs"] = N_EPOCHS
    return output


# --------------------------------------- TCNMODEL
def _params_TCNMODEL(trial, series, forecast_horizon, **kwargs):

    input_size = suggest_lags(trial, series, "input_chunk_length")
    trial.suggest_int("output_chunk_length", 1, min(forecast_horizon, input_size - 1))

    trial.suggest_int("kernel_size", 2, input_size - 1)
    trial.suggest_int("num_layers", 1, 4)
    trial.suggest_float("dropout", 0.0, 0.4)
    trial.suggest_categorical("add_past_encoders", [False, True])


def _fixed_params_TCNMODEL(series, forecast_horizon, suggested_lags=None, **kwargs):
    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(series) * 0.05))
    )
    output["output_chunk_length"] = forecast_horizon
    output["n_epochs"] = N_EPOCHS

    return output


# --------------------------------------- LGMBMODEL
def _params_LGBMModel(trial, series, future_covariates=None, **kwargs):

    suggest_lags(trial, series, "lags")

    trial.suggest_int("num_leaves", 2, 50, log=True)
    trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
    trial.suggest_float("reg_lambda", 1e-2, 1e1, log=True)

    encoders_future = trial.suggest_categorical("add_future_encoders", [False, True])
    if encoders_future or future_covariates:
        trial.suggest_int("lags_future_covariates_past", 1, 5)
        trial.suggest_int("lags_future_covariates_future", 1, 5)


def _fixed_params_LGBMModel(
    series,
    suggested_lags=None,
    past_covariates=None,
    lags_past_covariates=[-1],
    future_covariates=None,
    lags_future_covariates=[0],
    **kwargs
):
    output = dict()

    output["lags"] = fixed_lags(series, suggested_lags)
    output["reg_lambda"] = 0.1
    if future_covariates:
        output["lags_future_covariates"] = lags_future_covariates
    if past_covariates:
        output["lags_past_covariates"] = lags_past_covariates

    return output


# --------------------------------------- LINEARREGRESSION
def _params_LinearRegression(trial, series, future_covariates=None, **kwargs):
    # lag length as a ratio of the train data size
    suggest_lags(trial, series, "lags")

    encoders_future = trial.suggest_categorical("add_future_encoders", [False, True])
    if encoders_future or future_covariates:
        trial.suggest_int("lags_future_covariates_past", 1, 5)
        trial.suggest_int("lags_future_covariates_future", 1, 5)


def _fixed_params_LinearRegression(
    series,
    suggested_lags: int = None,
    past_covariates=None,
    lags_past_covariates=[-1],
    future_covariates=None,
    lags_future_covariates=[0],
    **kwargs
):
    output = dict()
    if future_covariates:
        output["lags_future_covariates"] = lags_future_covariates
    if past_covariates:
        output["lags_past_covariates"] = lags_past_covariates
    output["lags"] = (
        suggested_lags if suggested_lags else max(5, int(len(series) * 0.05))
    )
    return output


# --------------------------------------- CATBOOST
def _fixed_params_Catboost(
    series,
    suggested_lags=None,
    past_covariates=None,
    lags_past_covariates=[-1],
    future_covariates=None,
    lags_future_covariates=[0],
    **kwargs
):
    output = dict()
    if future_covariates:
        output["lags_future_covariates"] = lags_future_covariates
    if past_covariates:
        output["lags_past_covariates"] = lags_past_covariates
    output["lags"] = (
        suggested_lags if suggested_lags else max(5, int(len(series) * 0.05))
    )
    return output


# --------------------------------------- NBEATS
def _params_Nbeats(trial, series, forecast_horizon, **kwargs):
    suggest_lags(trial, series, "input_chunk_length")
    trial.suggest_categorical("output_chunk_length", [1, forecast_horizon])

    trial.suggest_categorical("add_past_encoders", [False, True])
    trial.suggest_categorical("generic_architecture", [False, True])
    trial.suggest_float("dropout", 0.0, 0.4)


def _fixed_params_Nbeats(series, forecast_horizon, suggested_lags=None, **kwargs):
    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(series) * 0.05))
    )
    output["output_chunk_length"] = forecast_horizon
    output["generic_architecture"] = True
    output["n_epochs"] = N_EPOCHS

    return output


# --------------------------------------- ARIMA
def _params_arima(trial, series, **kwargs):
    """Raytune gets stuck on this model. Since no fix could be found, we deactivate optuna search for ARIMA"""
    suggest_lags(trial, series, "p")
    trial.suggest_int("q", 0, 10)
    trend = trial.suggest_categorical("trend", ["n", "c", "t"])
    if trend == "n":
        trial.suggest_int("d", 0, 2)
    elif trend == "t":
        trial.suggest_int("d", 0, 1)
    elif trend == "c":
        trial.suggest_int("d", 0, 0)


def _fixed_params_arima(**kwargs):
    output = dict()
    output["p"] = 10
    output["d"] = 1
    output["q"] = 5
    output["trend"] = "t"

    return output


# --------------------------------------- prophet
def _params_prophet(trial, series, **kwargs):
    growth = trial.suggest_categorical("growth", ["linear", "logistic"])
    if growth == "logistic":
        min_ = series.values().min()
        max_ = series.values().max()
        diff = max_ - min_
        trial.suggest_float("floor", min_ - diff, min_)
        trial.suggest_float("cap", max_, max_ + diff)

    trial.suggest_int(
        "prophet_seasonality_period", 4, max(4, int(len(series * 0.25))), log=True
    )
    trial.suggest_categorical("add_future_encoders", [False, True])


# --------------------------------------- FFT
def _params_fft(trial, **kwargs):
    trend = trial.suggest_categorical("trend", ["poly", "exp", None])
    if trend == "poly":
        trial.suggest_int("trend_poly_degree", 1, 3)


def _fixed_params_fft(**kwargs):
    return {"trend": "poly", "trend_poly_degree": 0}


OPTUNA_SEARCH_SPACE = {
    TCNModel.__name__: _params_TCNMODEL,
    DLinearModel.__name__: _params_DLINEAR,
    NLinearModel.__name__: _params_NLINEAR,
    NHiTSModel.__name__: _params_NHITS,
    NBEATSModel.__name__: _params_Nbeats,
    FFT.__name__: _params_fft,
    LightGBMModel.__name__: _params_LGBMModel,
    LinearRegressionModel.__name__: _params_LinearRegression,
    Prophet.__name__: _params_prophet,
    ARIMA.__name__: _params_arima,
}

FIXED_PARAMS = {
    LinearRegressionModel.__name__: _fixed_params_LinearRegression,
    CatBoostModel.__name__: _fixed_params_Catboost,
    NBEATSModel.__name__: _fixed_params_Nbeats,
    NHiTSModel.__name__: _fixed_params_NHITS,
    ARIMA.__name__: _fixed_params_arima,
    FFT.__name__: _fixed_params_fft,
    Prophet.__name__: _empty_params,
    TCNModel.__name__: _fixed_params_TCNMODEL,
    NaiveSeasonal.__name__: _empty_params,
    LightGBMModel.__name__: _fixed_params_LGBMModel,
    NLinearModel.__name__: _fixed_params_NLINEAR,
    DLinearModel.__name__: _fixed_params_DLINEAR,
    AutoARIMA.__name__: _empty_params,
}
