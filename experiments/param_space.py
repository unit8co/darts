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

# --------------------------------------- UTILS
N_EPOCHS = 5
encoders_dict = {
    # maybe future wouéld be better but torch models only take past covariates
    "datetime_attribute": {"past": ["month", "week", "hour", "dayofweek"]},
    "cyclic": {"past": ["month", "week", "hour", "dayofweek"]},
}

encoders_dict_future = {
    "datetime_attribute": {"future": ["month", "week", "hour", "dayofweek"]},
    "cyclic": {"future": ["month", "week", "hour", "dayofweek"]},
}


def optuna2params(optuna_params):
    """
    Optuna only takes ints/float/bool/categorical parameters. This function converts the optuna parameters to
    enable models to take more complex parameters.
    """
    output_params = optuna_params.copy()
    if "add_encoders" in output_params:
        # converts boolean True/False to dict for encoders (adapts the future_lags if necessary)
        output_params["add_encoders"] = (
            encoders_dict if output_params["add_encoders"] else None
        )

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

    return output_params


def _empty_params(**kwargs):
    return dict()


def suggest_lags(trial, target_dataset, var_name: str):
    lags = trial.suggest_int(
        var_name,
        max(5, int(len(target_dataset) * 0.001)),
        max(6, int(len(target_dataset) * 0.2)),
        log=True,
    )
    return lags


def fixed_lags(target_dataset, suggested_lags=None):
    return suggested_lags or max(5, int(len(target_dataset) * 0.05))


# --------------------------------------- NHITS
def _params_NHITS(trial, target_dataset, **kwargs):
    suggest_lags(trial, target_dataset, "input_chunk_length")

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
    output["input_chunk_length"] = fixed_lags(target_dataset, suggested_lags)
    output["output_chunk_length"] = 1
    output["n_epochs"] = N_EPOCHS
    return output


# --------------------------------------- NLINEAR
def _params_NLINEAR(trial, target_dataset, **kwargs):

    suggest_lags(trial, target_dataset, "input_chunk_length")

    trial.suggest_categorical("const_init", [False, True])
    normalize = trial.suggest_categorical("normalize", [False, True])
    shared_weights = trial.suggest_categorical("shared_weights", [False, True])

    if not shared_weights and not normalize:
        # in current version, darts does not support covariates with normalize.
        # Should be fixed with https://github.com/unit8co/darts/pull/1583
        trial.suggest_categorical("add_encoders", [False, True])


def _fixed_params_NLINEAR(target_dataset, suggested_lags=None, **kwargs):

    output = dict()
    output["input_chunk_length"] = fixed_lags(target_dataset, suggested_lags)
    output["output_chunk_length"] = 1
    output["n_epochs"] = N_EPOCHS
    return output


# --------------------------------------- DLINEAR
def _params_DLINEAR(trial, target_dataset, **kwargs):

    input_size = suggest_lags(trial, target_dataset, "input_chunk_length")
    trial.suggest_int("kernel_size", 2, input_size)
    shared_weights = trial.suggest_categorical("shared_weights", [False, True])
    if not shared_weights:
        trial.suggest_categorical("add_encoders", [False, True])

    trial.suggest_categorical("const_init", [False, True])


def _fixed_params_DLINEAR(target_dataset, suggested_lags=None, **kwargs):
    output = dict()
    output["input_chunk_length"] = fixed_lags(target_dataset, suggested_lags)
    output["output_chunk_length"] = 1
    output["n_epochs"] = N_EPOCHS
    return output


# --------------------------------------- TCNMODEL
def _params_TCNMODEL(trial, target_dataset, **kwargs):

    input_size = suggest_lags(trial, target_dataset, "input_chunk_length")

    trial.suggest_int("kernel_size", 2, input_size - 1)
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
    output["n_epochs"] = N_EPOCHS
    return output


# --------------------------------------- LGMBMODEL
def _params_LGBMModel(trial, target_dataset, **kwargs):

    suggest_lags(trial, target_dataset, "lags")

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


# --------------------------------------- LINEARREGRESSION
def _params_LinearRegression(trial, target_dataset, **kwargs):
    # lag length as a ratio of the train data size
    suggest_lags(trial, target_dataset, "lags")

    encoders_past = trial.suggest_categorical("add_encoders", [False, True])
    if encoders_past:
        trial.suggest_int("lags_past_covariates", 1, 5)


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


# --------------------------------------- CATBOOST
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


# --------------------------------------- NBEATS
def _params_Nbeats(trial, target_dataset, **kwargs):
    suggest_lags(trial, target_dataset, "input_chunk_length")


def _fixed_params_Nbeats(target_dataset, suggested_lags=None, **kwargs):
    output = dict()
    output["input_chunk_length"] = (
        suggested_lags if suggested_lags else max(5, int(len(target_dataset) * 0.05))
    )
    output["output_chunk_length"] = 1
    output["generic_architecture"] = False
    output["n_epochs"] = N_EPOCHS
    return output


# --------------------------------------- ARIMA
def _params_arima(trial, target_dataset, **kwargs):
    suggest_lags(trial, target_dataset, "p")
    trial.suggest_int("d", 0, 2)
    trial.suggest_int("q", 0, 10)
    trial.suggest_categorical("q", ["n", "c", "t", "ct"])


OPTUNA_SEARCH_SPACE = {
    TCNModel.__name__: _params_TCNMODEL,
    DLinearModel.__name__: _params_DLINEAR,
    NLinearModel.__name__: _params_NLINEAR,
    NHiTSModel.__name__: _params_NHITS,
    NBEATSModel.__name__: _params_Nbeats,
    LightGBMModel.__name__: _params_LGBMModel,
    LinearRegressionModel.__name__: _params_LinearRegression,
    ARIMA.__name__: _params_arima,
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
    NLinearModel.__name__: _fixed_params_NLINEAR,
    DLinearModel.__name__: _fixed_params_DLINEAR,
}
