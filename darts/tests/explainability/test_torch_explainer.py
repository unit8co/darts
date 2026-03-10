import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.tests.conftest import NF_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts import TimeSeries
from darts.explainability import TorchExplainer
from darts.models import (
    BlockRNNModel,
    Chronos2Model,
    DLinearModel,
    NBEATSModel,
    NeuralForecastModel,
    NHiTSModel,
    RNNModel,
    TiDEModel,
    TSMixerModel,
)

N_PAST_COVARIATES = 3
N_FUTURE_COVARIATES = 2
N_TARGETS = 3

chronos2_local_dir = (
    Path(__file__).parent.parent
    / "models"
    / "forecasting"
    / "artefacts"
    / "chronos2"
    / "tiny_chronos2"
).absolute()

ALL_MODELS = [
    (RNNModel, {"model": "LSTM"}),
    (BlockRNNModel, None),
    (NBEATSModel, {"num_stacks": 2, "num_layers": 2, "layer_widths": 16}),
    (NHiTSModel, {"layer_widths": 16}),
    (TiDEModel, {"hidden_size": 32}),
    (DLinearModel, None),
    (TSMixerModel, {"hidden_size": 16, "ff_size": 16}),
    (Chronos2Model, {"local_dir": chronos2_local_dir}),
]

if NF_AVAILABLE:
    ALL_MODELS += [
        (
            NeuralForecastModel,
            {"model": "MLPMultivariate", "model_kwargs": {"hidden_size": 16}},
        ),
        (
            NeuralForecastModel,
            {
                "model": "PatchTST",
                "model_kwargs": {
                    "patch_len": 3,
                    "n_heads": 4,
                    "hidden_size": 16,
                    "linear_hidden_size": 32,
                },
            },
        ),
    ]


kwargs = {
    "n_epochs": 1,
    **tfm_kwargs,
}


class TestSKLearnExplainer:
    # set random seed
    np.random.seed(42)

    X, Y = make_regression(
        n_samples=300,
        n_features=N_PAST_COVARIATES + N_FUTURE_COVARIATES,
        n_informative=3,
        n_targets=N_TARGETS,
        noise=1,
        random_state=42,
    )
    X, Y = X.astype(np.float32), Y.astype(np.float32)
    multivariate_series = TimeSeries.from_times_and_values(
        times=pd.date_range("20200101", periods=250, freq="D"),
        values=Y[:250],
        columns=[f"T_{i}" for i in range(N_TARGETS)],
    ).with_static_covariates(pd.DataFrame({"static_cov": [1] * N_TARGETS}))
    univariate_series = multivariate_series.univariate_component(0)
    past_covariates = TimeSeries.from_times_and_values(
        times=pd.date_range("20200101", periods=250, freq="D"),
        values=X[:250, :N_PAST_COVARIATES],
        columns=[f"P_{i}" for i in range(N_PAST_COVARIATES)],
    )
    future_covariates = TimeSeries.from_times_and_values(
        times=pd.date_range("20200101", periods=297, freq="D"),
        # shift forward by 3 so that future covariate may influence target at time t
        values=X[3:, N_PAST_COVARIATES:],
        columns=[f"F_{i}" for i in range(N_FUTURE_COVARIATES)],
    )
    multiple_multivariate_series = [
        multivariate_series,
        multivariate_series + 1,
    ]

    @pytest.mark.parametrize("model_cls, model_kwargs", ALL_MODELS)
    def test_creation(
        self,
        model_cls: type[TorchForecastingModel],
        model_kwargs: dict | None,
        tmpdir_fn,
    ):
        model = model_cls(
            input_chunk_length=10,
            output_chunk_length=5,
            **(model_kwargs or {}),
            **kwargs,
        )

        # cannot create explainer with unfitted model
        with pytest.raises(ValueError, match="must be fitted before instantiating."):
            explainer = TorchExplainer(model)

        # fit the model
        model.fit(series=self.multivariate_series)

        # create explainer with fitted model
        explainer = TorchExplainer(model)

        # check explainer attributes
        assert explainer.model == model
        assert explainer.n == model.output_chunk_length

        # save and load the model to check explainer works with loaded models
        save_path = os.path.join(tmpdir_fn, "model.pt")
        model.save(save_path)
        loaded_model = model_cls.load(save_path)
        loaded_explainer = TorchExplainer(loaded_model)

        assert loaded_explainer.model == loaded_model
        assert loaded_explainer.n == loaded_model.output_chunk_length

    @pytest.mark.parametrize("model_cls, model_kwargs", ALL_MODELS)
    def test_creation_multiple_series(
        self, model_cls: type[TorchForecastingModel], model_kwargs: dict | None
    ):
        model = model_cls(
            input_chunk_length=10,
            output_chunk_length=5,
            **(model_kwargs or {}),
            **kwargs,
        )

        # cannot create explainer with unfitted model
        with pytest.raises(ValueError, match="must be fitted before instantiating."):
            explainer = TorchExplainer(model)

        # fit the model
        model.fit(series=self.multiple_multivariate_series)

        # create explainer with multiple series but no background raises error
        with pytest.raises(ValueError, match="`background_series` must be provided"):
            explainer = TorchExplainer(model)

        # create explainer with multiple series and background
        explainer = TorchExplainer(
            model, background_series=self.multiple_multivariate_series
        )

        # check explainer attributes
        assert explainer.model == model
        assert explainer.n == model.output_chunk_length
