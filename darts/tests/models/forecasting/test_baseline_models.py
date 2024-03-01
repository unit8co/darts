import numpy as np
import pytest

from darts.logging import get_logger
from darts.models import NaiveDrift, NaiveMean, NaiveMovingAverage, NaiveSeasonal
from darts.models.forecasting.forecasting_model import (
    GlobalForecastingModel,
    LocalForecastingModel,
)
from darts.tests.conftest import tfm_kwargs
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)


icl = 5
local_models = [
    (NaiveDrift, {}),
    (NaiveMean, {}),
    (NaiveMovingAverage, {}),
    (NaiveSeasonal, {}),
]
global_models = []
try:
    import torch

    from darts.models import GlobalNaiveAggregate, GlobalNaiveDrift, GlobalNaiveSeasonal

    TORCH_AVAILABLE = True

    global_models += [
        (
            GlobalNaiveAggregate,
            {"input_chunk_length": icl, "output_chunk_length": 3, **tfm_kwargs},
        ),
        (
            GlobalNaiveAggregate,
            {"input_chunk_length": icl, "output_chunk_length": 1, **tfm_kwargs},
        ),
        (
            GlobalNaiveDrift,
            {"input_chunk_length": icl, "output_chunk_length": 3, **tfm_kwargs},
        ),
        (
            GlobalNaiveDrift,
            {"input_chunk_length": icl, "output_chunk_length": 1, **tfm_kwargs},
        ),
        (
            GlobalNaiveSeasonal,
            {"input_chunk_length": icl, "output_chunk_length": 3, **tfm_kwargs},
        ),
        (
            GlobalNaiveSeasonal,
            {"input_chunk_length": icl, "output_chunk_length": 1, **tfm_kwargs},
        ),
    ]
except ImportError:
    logger.warning("Torch not installed - will be skipping Torch models tests")
    TORCH_AVAILABLE = False


class TestBaselineModels:
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)

    @pytest.mark.parametrize("model_config", local_models + global_models)
    def test_invalid_model_calls(self, model_config):
        # min train series length for global naive models
        series = tg.linear_timeseries(length=icl)

        model_cls, model_kwargs = model_config
        model = model_cls(**model_kwargs)

        # calling fit before predict
        with pytest.raises(ValueError):
            model.predict(n=10)

        # calling fit with covariates
        if isinstance(model, GlobalForecastingModel):
            err_type = ValueError
        else:  # for local models, covariates are not part of signature
            err_type = TypeError
        with pytest.raises(err_type):
            model.fit(series=series, past_covariates=series)
        with pytest.raises(err_type):
            model.fit(series=series, future_covariates=series)

        model.fit(series=series)
        # calling predict with covariates
        with pytest.raises(err_type):
            model.predict(n=10, past_covariates=series)
        with pytest.raises(err_type):
            model.predict(n=10, future_covariates=series)

        # single series predict works with all models
        model.predict(n=10)

        if isinstance(model, LocalForecastingModel):
            # no series at prediction time
            with pytest.raises(err_type):
                _ = model.predict(n=10, series=series)
            # no multiple series prediction
            with pytest.raises(err_type):
                _ = model.predict(n=10, series=[series, series])
        else:
            _ = model.predict(n=10, series=series)
            _ = model.predict(n=10, series=[series, series])

        # multiple series training only with global baselines
        if isinstance(model, LocalForecastingModel):
            with pytest.raises(ValueError):
                model.fit(series=[series, series])
        else:
            model.fit(series=[series, series])

    @pytest.mark.parametrize("model_config", local_models + global_models)
    def test_single_model_univariate_fit_predict(self, model_config):
        series = tg.linear_timeseries(length=icl)

        model_cls, model_kwargs = model_config
        model = model_cls(**model_kwargs)

        model.fit(series=series)
        # call fit before predict
        with pytest.raises(ValueError):
            model.predict(n=10)
