"""Safe-by-default save/load roundtrips for forecasting models.

Exercises the default restricted loading paths (``trusted=False`` for pickle-based
models, ``weights_only=True`` for torch models) across available model flavors.
Predictions before save must match predictions after load.
"""

from __future__ import annotations

import contextlib
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor

from darts import TimeSeries
from darts.models import LinearRegressionModel, NaiveMean, NaiveSeasonal, SKLearnModel
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.tests.conftest import (
    CB_AVAILABLE,
    LGBM_AVAILABLE,
    NF_AVAILABLE,
    PROPHET_AVAILABLE,
    SF_AVAILABLE,
    T0_AVAILABLE,
    TIREX_AVAILABLE,
    TORCH_AVAILABLE,
    XGB_AVAILABLE,
    tfm_kwargs,
)
from darts.utils.serialization.base import (
    is_allowed_global,
    resolve_reference,
    safe_base_classes,
)
from darts.utils.utils import NotImportedModule

if TORCH_AVAILABLE:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import Callback

    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
    from darts.utils.likelihood_models.torch import GaussianLikelihood

SERIES_LENGTH = 30
PREDICT_H = 3
ICL = 6
OCL = 2

PL_TRAINER_KWARGS = {
    **tfm_kwargs["pl_trainer_kwargs"],
    "fast_dev_run": True,
}

TORCH_BASE = {
    "input_chunk_length": ICL,
    "output_chunk_length": OCL,
    "n_epochs": 1,
    "random_state": 0,
    "pl_trainer_kwargs": PL_TRAINER_KWARGS,
}

LGBM_KW = {
    "lags": 4,
    "n_estimators": 8,
    "max_depth": 2,
    "num_leaves": 4,
    "verbosity": -1,
    "random_state": 0,
}
XGB_KW = {
    "lags": 4,
    "n_estimators": 2,
    "max_depth": 2,
    "max_leaves": 4,
    "tree_method": "exact",
    "random_state": 0,
}
CB_KW = {"lags": 4, "iterations": 2, "depth": 2, "verbose": -1, "random_state": 0}

SeriesKey = Literal["uni", "pos", "mv", "foundation"]


def _model_class(name: str):
    import darts.models as models_pkg

    cls = getattr(models_pkg, name, None)
    if cls is None or isinstance(cls, NotImportedModule):
        return None
    return cls


class QuantileLinearRegressionModel(LinearRegressionModel):
    def __init__(self):
        super().__init__(
            lags=4,
            likelihood="quantile",
            quantiles=[0.1, 0.5, 0.9],
            random_state=0,
        )


class KNeighborsSKLearnModel(SKLearnModel):
    def __init__(self):
        super().__init__(
            lags=4,
            model=KNeighborsRegressor(n_neighbors=1),
        )


if TORCH_AVAILABLE:

    class _RecordEpochCallback(Callback):
        """Module-level callback; must be importable by qualname when unpickling."""

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            return None


@dataclass
class _SeriesBundle:
    uni: TimeSeries
    pos: TimeSeries
    mv: TimeSeries
    foundation: TimeSeries


def _times(index_len: int = SERIES_LENGTH) -> pd.DatetimeIndex:
    return pd.date_range("20180101", periods=index_len, freq="D")


def _univariate_series(index_len: int = SERIES_LENGTH) -> TimeSeries:
    idx = _times(index_len)
    return TimeSeries.from_series(
        pd.Series(np.arange(index_len, dtype=np.float64), index=idx)
    )


def _positive_series(index_len: int = SERIES_LENGTH) -> TimeSeries:
    idx = _times(index_len)
    return TimeSeries.from_series(
        pd.Series(np.arange(1, index_len + 1, dtype=np.float64), index=idx)
    )


def _multivariate_series(index_len: int = SERIES_LENGTH) -> TimeSeries:
    idx = _times(index_len)
    df = pd.DataFrame(
        {
            "a": np.arange(index_len, dtype=np.float64),
            "b": np.arange(index_len - 1, -1, -1, dtype=np.float64),
        },
        index=idx,
    )
    return TimeSeries.from_dataframe(df)


def _series_bundle() -> _SeriesBundle:
    from darts.utils.timeseries_generation import linear_timeseries

    return _SeriesBundle(
        uni=_univariate_series(),
        pos=_positive_series(),
        mv=_multivariate_series(),
        foundation=linear_timeseries(length=18, dtype=np.float32, column_name="y"),
    )


@dataclass
class SaveLoadCase:
    id: str
    model_cls: type
    build: Callable[[], Any]
    series: TimeSeries
    fit_kwargs: dict = field(default_factory=dict)
    predict_n: int = PREDICT_H
    patch_factory: Callable[[], contextlib.AbstractContextManager] = (
        contextlib.nullcontext
    )
    skip_fit: bool = False


def _predict(model, series: TimeSeries, n: int) -> TimeSeries:
    """Predict with fixed ``random_state``; use direct quantiles when supported."""
    kwargs: dict[str, Any] = {
        "n": n,
        "random_state": 0,
        "num_samples": 1,
    }
    ocl = model.output_chunk_length
    ocl = ocl if ocl is not None else n
    if model.supports_likelihood_parameter_prediction and n <= ocl:
        kwargs["predict_likelihood_parameters"] = True
    if isinstance(model, GlobalForecastingModel):
        kwargs["series"] = series

    try:
        return model.predict(**kwargs)
    except Exception:
        kwargs.pop("predict_likelihood_parameters", None)
        return model.predict(**kwargs)


def _resolve_model_cls(cls: type | str) -> type | None:
    if isinstance(cls, type):
        return cls
    return _model_class(cls)


def _model_config_specs(series: _SeriesBundle) -> list[dict[str, Any]]:
    """Flat list of save/load test configurations (optional ``when`` skips entry)."""
    uni, _, _ = series.uni, series.pos, series.mv
    specs: list[dict[str, Any]] = [
        {"id": "naive_seasonal", "cls": "NaiveSeasonal", "series_key": "mv"},
        {"id": "naive_mean", "cls": "NaiveMean"},
        {"id": "naive_drift", "cls": "NaiveDrift"},
        {
            "id": "naive_moving_average",
            "cls": "NaiveMovingAverage",
            "kwargs": {"input_chunk_length": 5},
        },
        {"id": "arima", "cls": "ARIMA", "init_args": (1, 0, 0)},
        {"id": "exp_smoothing", "cls": "ExponentialSmoothing", "kwargs": {}},
        {"id": "fft", "cls": "FFT", "kwargs": {"nr_freqs_to_keep": 6}},
        {"id": "theta", "cls": "Theta", "init_args": (1,)},
        {"id": "four_theta", "cls": "FourTheta", "init_args": (1,)},
        {"id": "kalman", "cls": "KalmanForecaster", "kwargs": {"dim_x": 3}},
        {
            "id": "linear_regression_quantile",
            "cls": QuantileLinearRegressionModel,
        },
        {
            "id": "random_forest",
            "cls": "RandomForestModel",
            "kwargs": {"lags": 4, "n_estimators": 2, "max_depth": 2, "random_state": 0},
        },
        {
            "id": "sklearn_knn",
            "cls": KNeighborsSKLearnModel,
        },
        {
            "id": "multivariate_naive_seasonal",
            "cls": "MultivariateModel",
            "kwargs": {"model": "NaiveSeasonal"},
            "series_key": "mv",
        },
        {"id": "varima", "cls": "VARIMA", "init_args": (1, 0, 0), "series_key": "mv"},
        {
            "id": "naive_ensemble",
            "cls": "NaiveEnsembleModel",
            "kwargs": {"forecasting_models": [NaiveSeasonal(), NaiveMean()]},
        },
        {
            "id": "regression_ensemble",
            "cls": "RegressionEnsembleModel",
            "kwargs": {
                "forecasting_models": [NaiveSeasonal(), NaiveMean()],
                "regression_train_n_points": 10,
            },
        },
        {
            "id": "conformal_naive",
            "cls": "ConformalNaiveModel",
            "kwargs": {
                "model": LinearRegressionModel(lags=4).fit(uni),
                "quantiles": [0.1, 0.5, 0.9],
                "cal_num_samples": 1,
                "random_state": 0,
            },
            "skip_fit": True,
        },
        {
            "id": "conformal_qr",
            "cls": "ConformalQRModel",
            "kwargs": {
                "model": LinearRegressionModel(
                    lags=4,
                    likelihood="quantile",
                    quantiles=[0.1, 0.5, 0.9],
                    random_state=0,
                ).fit(uni),
                "quantiles": [0.1, 0.5, 0.9],
                "cal_num_samples": 10,
                "random_state": 0,
            },
            "skip_fit": True,
        },
    ]

    if PROPHET_AVAILABLE:
        specs.append({"id": "prophet", "cls": "Prophet"})

    if SF_AVAILABLE:
        from statsforecast.models import AutoARIMA as SFAutoARIMA

        specs.extend([
            {"id": "auto_arima", "cls": "AutoARIMA", "kwargs": {"season_length": 7}},
            {
                "id": "statsforecast_auto_arima",
                "cls": "StatsForecastModel",
                "kwargs": {"model": SFAutoARIMA(season_length=7)},
            },
            {
                "id": "statsforecast_auto_arima",
                "cls": "StatsForecastModel",
                "kwargs": {"model": "Naive"},
            },
            {"id": "auto_theta", "cls": "AutoTheta", "kwargs": {"season_length": 7}},
            {
                "id": "auto_ces",
                "cls": "AutoCES",
                "kwargs": {"season_length": 7, "model": "Z"},
            },
            {
                "id": "auto_ets",
                "cls": "AutoETS",
                "kwargs": {"season_length": 7, "model": "AAZ"},
            },
            {
                "id": "auto_mfles",
                "cls": "AutoMFLES",
                "kwargs": {"season_length": 7, "test_size": 7},
            },
            {"id": "auto_tbats", "cls": "AutoTBATS", "kwargs": {"season_length": 7}},
            {"id": "croston", "cls": "Croston", "kwargs": {"version": "classic"}},
            {
                "id": "tbats",
                "cls": "TBATS",
                "kwargs": {"season_length": 7, "use_trend": False},
            },
        ])

    if LGBM_AVAILABLE:
        import lightgbm as lgb

        specs.extend([
            {
                "id": "lightgbm_early_stopping",
                "cls": "LightGBMModel",
                "kwargs": {
                    **LGBM_KW,
                    "callbacks": [lgb.early_stopping(stopping_rounds=2, verbose=False)],
                },
                "fit_kwargs": {"val_series": uni},
            },
            {
                "id": "lightgbm_poisson",
                "cls": "LightGBMModel",
                "kwargs": {"likelihood": "poisson", **LGBM_KW},
                "series_key": "pos",
            },
        ])

    if XGB_AVAILABLE:
        specs.append({
            "id": "xgboost_quantile",
            "cls": "XGBModel",
            "kwargs": {"likelihood": "quantile", **XGB_KW},
        })

    if CB_AVAILABLE:
        specs.append({
            "id": "catboost",
            "cls": "CatBoostModel",
            "kwargs": CB_KW,
        })

    if not TORCH_AVAILABLE:
        return specs

    dlinear_light = {"kernel_size": 2}
    nbeats_light = {
        "num_stacks": 1,
        "num_blocks": 1,
        "num_layers": 1,
        "layer_widths": 2,
    }
    tcn_light = {"kernel_size": 2, "num_filters": 1, "dilation_base": 1}
    trafo_light = {
        "d_model": 2,
        "nhead": 1,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 2,
    }
    tft_light = {
        "hidden_size": 2,
        "lstm_layers": 1,
        "num_attention_heads": 1,
        "hidden_continuous_size": 2,
    }

    specs.extend([
        {
            "id": "block_rnn",
            "cls": "BlockRNNModel",
            "torch_base": True,
            "kwargs": {"model": "RNN", "hidden_dim": 4, "n_rnn_layers": 1},
        },
        {
            "id": "dlinear_encoders_callback",
            "cls": "DLinearModel",
            "torch_base": True,
            "kwargs": {
                **dlinear_light,
                "add_encoders": {"cyclic": {"past": ["month"]}},
                "pl_trainer_kwargs": {
                    **PL_TRAINER_KWARGS,
                    "callbacks": [_RecordEpochCallback()],
                },
            },
        },
        {
            "id": "nbeats",
            "cls": "NBEATSModel",
            "torch_base": True,
            "kwargs": nbeats_light,
        },
        {
            "id": "nhits",
            "cls": "NHiTSModel",
            "torch_base": True,
            "kwargs": nbeats_light,
        },
        {"id": "nlinear", "cls": "NLinearModel", "torch_base": True},
        {
            "id": "dlinear_gaussian_likelihood",
            "cls": "DLinearModel",
            "torch_base": True,
            "kwargs": {**dlinear_light, "likelihood": GaussianLikelihood()},
        },
        {"id": "tcn", "cls": "TCNModel", "torch_base": True, "kwargs": tcn_light},
        {
            "id": "tft",
            "cls": "TFTModel",
            "torch_base": True,
            "kwargs": {"add_relative_index": True, **tft_light},
            "series_key": "uni",
            "predict_n": OCL,
        },
        {"id": "tide", "cls": "TiDEModel", "torch_base": True},
        {
            "id": "transformer",
            "cls": "TransformerModel",
            "torch_base": True,
            "kwargs": trafo_light,
        },
        {"id": "tsmixer", "cls": "TSMixerModel", "torch_base": True},
        {
            "id": "global_naive_seasonal",
            "cls": "GlobalNaiveSeasonal",
            "torch_base": True,
        },
        {
            "id": "global_naive_aggregate",
            "cls": "GlobalNaiveAggregate",
            "torch_base": True,
        },
        {"id": "global_naive_drift", "cls": "GlobalNaiveDrift", "torch_base": True},
        {
            "id": "dlinear_custom_loss",
            "cls": "DLinearModel",
            "torch_base": True,
            "kwargs": {**dlinear_light, "loss_fn": torch.nn.L1Loss()},
        },
        {
            "id": "dlinear_early_stopping",
            "cls": "DLinearModel",
            "torch_base": True,
            "kwargs": {
                **dlinear_light,
                "pl_trainer_kwargs": {
                    **PL_TRAINER_KWARGS,
                    "callbacks": [
                        pl.callbacks.EarlyStopping(
                            monitor="val_loss", patience=3, min_delta=0.01
                        )
                    ],
                },
            },
            "fit_kwargs": {"val_series": uni},
        },
    ])

    if NF_AVAILABLE:
        specs.append({
            "id": "neuralforecast_nlinear_loss",
            "cls": "NeuralForecastModel",
            "torch_base": True,
            "kwargs": {
                "model": "NLinear",
                "loss_fn": torch.nn.SmoothL1Loss(),
            },
        })

    from darts.tests.models.forecasting.foundation_test_utils import (
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        PATCHTST_FM_TINY_DIR,
        TIMESFM3_TINY_DIR,
        TIREX_LOAD_MODEL_PATCH_TARGET,
        TiRexStub,
        mock_hf_hub_download,
        timesfm2p5_tiny_context,
        tiny_t0_dir,
    )

    foundation_entries = [
        (
            "chronos2",
            "Chronos2Model",
            {},
            lambda: patch(
                HF_HUB_DOWNLOAD_PATCH_TARGET, side_effect=mock_hf_hub_download
            ),
        ),
        (
            "patchtst_fm",
            "PatchTSTFMModel",
            {"local_dir": PATCHTST_FM_TINY_DIR},
            contextlib.nullcontext,
        ),
        ("timesfm2p5", "TimesFM2p5Model", {}, timesfm2p5_tiny_context),
        (
            "timesfm3",
            "TimesFM3Model",
            {"accept_license": True, "local_dir": TIMESFM3_TINY_DIR},
            contextlib.nullcontext,
        ),
    ]
    if TIREX_AVAILABLE:
        foundation_entries.append((
            "tirex",
            "TiRexModel",
            {"accept_license": True},
            lambda: patch(TIREX_LOAD_MODEL_PATCH_TARGET, return_value=TiRexStub()),
        ))
    if T0_AVAILABLE:
        foundation_entries.append((
            "t0",
            "T0Model",
            {"local_dir": tiny_t0_dir()},
            contextlib.nullcontext,
        ))

    for fid, cls_name, extra, patch_factory in foundation_entries:
        specs.append({
            "id": fid,
            "cls": cls_name,
            "torch_base": True,
            "kwargs": extra,
            "series_key": "foundation",
            "predict_n": OCL,
            "patch_factory": patch_factory,
        })

    return specs


def _series_for_spec(spec: dict[str, Any], bundle: _SeriesBundle) -> TimeSeries:
    key: SeriesKey = spec.get("series_key", "uni")
    return getattr(bundle, key)


def _case_from_spec(spec: dict[str, Any], bundle: _SeriesBundle) -> SaveLoadCase | None:
    model_cls = _resolve_model_cls(spec["cls"])
    if model_cls is None:
        return None

    if "factory" in spec:
        factory = spec["factory"]
        build = lambda b=bundle, fn=factory: fn(b)
    else:
        init_args = spec.get("init_args", ())
        kwargs = dict(spec.get("kwargs", {}))
        if spec.get("torch_base"):
            kwargs = {**TORCH_BASE, **kwargs}

        def build(c=model_cls, a=init_args, k=kwargs):
            if a:
                return c(*a, **k)
            return c(**k)

    patch_factory = spec.get("patch_factory", contextlib.nullcontext)

    return SaveLoadCase(
        id=spec["id"],
        model_cls=model_cls,
        build=build,
        series=_series_for_spec(spec, bundle),
        fit_kwargs=dict(spec.get("fit_kwargs", {})),
        predict_n=spec.get("predict_n", PREDICT_H),
        patch_factory=patch_factory,
        skip_fit=spec.get("skip_fit", False),
    )


def _build_cases() -> list[SaveLoadCase]:
    bundle = _series_bundle()
    cases: list[SaveLoadCase] = []
    for spec in _model_config_specs(bundle):
        case = _case_from_spec(spec, bundle)
        if case is not None:
            cases.append(case)
    return cases


SAVE_LOAD_CASES = _build_cases()


def _qualnames_from_checkpoint(items) -> list[str]:
    names = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], str):
            names.append(f"{item[0]}.{item[1]}")
    return names


def _globals_referenced_by(path: str, *, torch_wrapper: bool) -> set[str]:
    """Record every global a saved model file asks ``find_class`` to import."""
    found: set[str] = set()

    class _Recording:
        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "__builtin__":
                    module = "builtins"
                found.add(f"{module}.{name}")
                return super().find_class(module, name)

    if torch_wrapper:
        import torch

        torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_Recording,
        )
    else:
        with open(path, "rb") as handle:
            _Recording.Unpickler(handle).load()
    return found


def _denied_globals(names) -> list[str]:
    bases = safe_base_classes()
    denied = []
    for name in sorted(set(names)):
        if name.startswith("__builtin__."):
            name = "builtins." + name.removeprefix("__builtin__.")
        if not is_allowed_global(name, resolve_reference(name), safe_bases=bases):
            denied.append(name)
    return denied


def _assert_saved_file_is_allowlisted(path: str, *, torch_model: bool) -> None:
    """Fail when a saved model references a global outside the safe-load registries.

    New dependency types must be classified (hierarchy, exact reconstructor, or
    exact state class) instead of being absorbed by a package prefix.
    """
    denied = _denied_globals(_globals_referenced_by(path, torch_wrapper=torch_model))
    if torch_model:
        import os

        import torch

        ckpt = path + ".ckpt"
        if os.path.exists(ckpt):
            unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(ckpt)
            denied.extend(_denied_globals(_qualnames_from_checkpoint(unsafe)))
    assert not denied, (
        "Saved model references globals that are not allow-listed for safe "
        f"loading: {sorted(set(denied))}"
    )


@pytest.mark.parametrize("case", SAVE_LOAD_CASES, ids=lambda c: c.id)
def test_safe_save_load_prediction_parity(case: SaveLoadCase, tmp_path):
    """Default safe loading must reproduce pre-save forecasts."""
    with case.patch_factory():
        model = case.build()
        if not case.skip_fit:
            model.fit(series=case.series, **case.fit_kwargs)
        pred_before = _predict(model, case.series, case.predict_n)

        ext = (
            ".pt"
            if TORCH_AVAILABLE and isinstance(model, TorchForecastingModel)
            else ".pkl"
        )
        save_path = tmp_path / f"{case.id}{ext}"
        model.save(str(save_path))
        _assert_saved_file_is_allowlisted(
            str(save_path),
            torch_model=ext == ".pt",
        )

        loaded = case.model_cls.load(str(save_path))
        pred_after = _predict(loaded, case.series, case.predict_n)

    assert pred_after == pred_before
