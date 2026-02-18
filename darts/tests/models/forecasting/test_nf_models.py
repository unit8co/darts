import logging
import os

import numpy as np
import pandas as pd
import pytest

from darts.tests.conftest import NF_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

if not NF_AVAILABLE:
    pytest.skip(
        f"NeuralForecast not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import torch
from neuralforecast.models import Autoformer, MLPMultivariate, NLinear

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.models import NeuralForecastModel
from darts.utils.likelihood_models import (
    GaussianLikelihood,
    LaplaceLikelihood,
    QuantileRegression,
)
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

kwargs = {
    "n_epochs": 1,
    **tfm_kwargs,
}

UNIVARIATE_MODELS = [
    ("PatchTST", {"patch_len": 3}),
    (NLinear, None),
]
MULTIVARIATE_MODELS = [
    ("SOFTS", {"hidden_size": 16, "d_core": 16, "d_ff": 64}),
    ("TimeMixer", None),
]
UNIVARIATE_MODELS_WITH_FUTURE_COVS = [
    (Autoformer, None),
    ("FEDformer", None),
]
UNIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS = [
    ("DeepNPTS", None),
    ("KAN", None),
]
MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS = [
    (MLPMultivariate, None),
    ("TSMixerx", None),
]
ALL_MODELS = (
    UNIVARIATE_MODELS
    + MULTIVARIATE_MODELS
    + UNIVARIATE_MODELS_WITH_FUTURE_COVS
    + UNIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS
    + MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS
)
LOSS_FUNCTIONS = [
    torch.nn.L1Loss(),
    torch.nn.HuberLoss(),
    torch.nn.SmoothL1Loss(),
]
LIKELIHOOD_MODELS = [
    GaussianLikelihood(),
    LaplaceLikelihood(),
    QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
]


class TestNeuralForecastModel:
    univariate_series = linear_timeseries(
        length=200, dtype=np.float32, column_name="U1"
    )
    past_cov = concatenate(
        [
            linear_timeseries(length=200, dtype=np.float32, column_name="P1"),
            sine_timeseries(length=200, dtype=np.float32, column_name="P2"),
            gaussian_timeseries(length=200, dtype=np.float32, column_name="P3"),
        ],
        axis=1,
    )
    future_cov = concatenate(
        [
            linear_timeseries(length=300, dtype=np.float32, column_name="F1"),
            sine_timeseries(length=300, dtype=np.float32, column_name="F2"),
            gaussian_timeseries(length=300, dtype=np.float32, column_name="F3"),
        ],
        axis=1,
    )
    multivariate_series = concatenate(
        [
            linear_timeseries(length=200, dtype=np.float32, column_name="M1"),
            sine_timeseries(length=200, dtype=np.float32, column_name="M2"),
            gaussian_timeseries(length=200, dtype=np.float32, column_name="M3"),
        ],
        axis=1,
    )
    multiple_series = [
        linear_timeseries(
            length=100, dtype=np.float32, column_name="S1"
        ).with_static_covariates(pd.DataFrame({"type": ["linear"]})),
        sine_timeseries(
            length=200, dtype=np.float32, column_name="S1"
        ).with_static_covariates(pd.DataFrame({"type": ["sine"]})),
        gaussian_timeseries(
            length=300, dtype=np.float32, column_name="S1"
        ).with_static_covariates(pd.DataFrame({"type": ["gaussian"]})),
    ]
    multiple_multivariate_series = [
        concatenate(
            [
                linear_timeseries(length=100, dtype=np.float32, column_name="S1_M1"),
                linear_timeseries(length=100, dtype=np.float32, column_name="S1_M2"),
                linear_timeseries(length=100, dtype=np.float32, column_name="S1_M3"),
            ],
            axis=1,
        ).with_static_covariates(pd.DataFrame({"type": ["linear"]})),
        concatenate(
            [
                sine_timeseries(length=200, dtype=np.float32, column_name="S1_M1"),
                sine_timeseries(length=200, dtype=np.float32, column_name="S1_M2"),
                sine_timeseries(length=200, dtype=np.float32, column_name="S1_M3"),
            ],
            axis=1,
        ).with_static_covariates(pd.DataFrame({"type": ["sine"]})),
        concatenate(
            [
                gaussian_timeseries(length=300, dtype=np.float32, column_name="S1_M1"),
                gaussian_timeseries(length=300, dtype=np.float32, column_name="S1_M2"),
                gaussian_timeseries(length=300, dtype=np.float32, column_name="S1_M3"),
            ],
            axis=1,
        ).with_static_covariates(pd.DataFrame({"type": ["gaussian"]})),
    ]

    @pytest.mark.parametrize("model_name, model_kwargs", ALL_MODELS)
    def test_univariate(self, model_name: str, model_kwargs: dict | None, tmpdir_fn):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=10,
            output_chunk_length=7,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        model.fit(series=self.univariate_series)
        pred = model.predict(6)
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 6
        assert pred.n_components == 1

        # save model
        save_path = os.path.join(tmpdir_fn, f"{model_name}.pt")
        ckpt_path = os.path.join(tmpdir_fn, f"{model_name}.pt.ckpt")
        model.save(save_path)
        assert os.path.exists(save_path)
        assert os.path.exists(ckpt_path)

        # load model and compare predictions
        loaded_model = NeuralForecastModel.load(save_path)
        pred_loaded = loaded_model.predict(6)
        assert isinstance(pred_loaded, TimeSeries)
        assert pred_loaded.n_timesteps == 6
        assert pred_loaded.n_components == 1
        assert pred_loaded.values().shape == pred.values().shape
        np.testing.assert_almost_equal(pred_loaded.values(), pred.values(), decimal=5)

    @pytest.mark.parametrize(
        "model_name, model_kwargs",
        UNIVARIATE_MODELS_WITH_FUTURE_COVS
        + UNIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS
        + MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS,
    )
    def test_univariate_with_future_covs(
        self, model_name: str, model_kwargs: dict | None
    ):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=8,
            output_chunk_length=5,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        model.fit(series=self.univariate_series, future_covariates=self.future_cov)
        pred = model.predict(n=4)
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 4
        assert pred.n_components == 1

    @pytest.mark.parametrize(
        "model_name, model_kwargs",
        UNIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS
        + MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS,
    )
    def test_univariate_with_past_covs(
        self, model_name: str, model_kwargs: dict | None
    ):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=9,
            output_chunk_length=11,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        model.fit(series=self.univariate_series, past_covariates=self.past_cov)
        pred = model.predict(n=10)
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 10
        assert pred.n_components == 1

    @pytest.mark.parametrize(
        "model_name, model_kwargs",
        UNIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS
        + MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS,
    )
    def test_univariate_with_past_and_future_covs(
        self, model_name: str, model_kwargs: dict | None
    ):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=7,
            output_chunk_length=13,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        model.fit(
            series=self.univariate_series,
            past_covariates=self.past_cov,
            future_covariates=self.future_cov,
        )
        pred = model.predict(n=12)
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 12
        assert pred.n_components == 1

    @pytest.mark.parametrize(
        "model_name, model_kwargs",
        MULTIVARIATE_MODELS + MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS,
    )
    def test_multivariate(self, model_name: str, model_kwargs: dict | None):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=12,
            output_chunk_length=13,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        model.fit(series=self.multivariate_series)
        pred = model.predict(n=11)
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 11
        assert pred.n_components == self.multivariate_series.n_components

    @pytest.mark.parametrize(
        "model_name, model_kwargs", MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS
    )
    def test_multivariate_with_past_covs(
        self, model_name: str, model_kwargs: dict | None
    ):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=11,
            output_chunk_length=9,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        model.fit(series=self.multivariate_series, past_covariates=self.past_cov)
        pred = model.predict(n=8)
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 8
        assert pred.n_components == self.multivariate_series.n_components

    @pytest.mark.parametrize(
        "model_name, model_kwargs", MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS
    )
    def test_multivariate_with_future_covs(
        self, model_name: str, model_kwargs: dict | None
    ):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=13,
            output_chunk_length=12,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        model.fit(series=self.multivariate_series, future_covariates=self.future_cov)
        pred = model.predict(n=10)
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 10
        assert pred.n_components == self.multivariate_series.n_components

    @pytest.mark.parametrize(
        "model_name, model_kwargs", MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS
    )
    def test_multivariate_with_past_and_future_covs(
        self, model_name: str, model_kwargs: dict | None
    ):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=10,
            output_chunk_length=14,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        model.fit(
            series=self.multivariate_series,
            past_covariates=self.past_cov,
            future_covariates=self.future_cov,
        )
        pred = model.predict(n=13)
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 13
        assert pred.n_components == self.multivariate_series.n_components

    # TODO: add tests for static covariates

    @pytest.mark.parametrize("model_name, model_kwargs", ALL_MODELS)
    def test_multiple_series(self, model_name: str, model_kwargs: dict | None):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=10,
            output_chunk_length=8,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        model.fit(series=self.multiple_series)
        pred = model.predict(n=4, series=self.multiple_series[0])
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 4
        assert pred.n_components == 1

    @pytest.mark.parametrize(
        "model_name, model_kwargs",
        UNIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS
        + MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS,
    )
    def test_multiple_series_with_static_covs(
        self, model_name: str, model_kwargs: dict | None
    ):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=11,
            output_chunk_length=13,
            model_kwargs=model_kwargs,
            use_static_covariates=True,
            **kwargs,
        )
        scaler = StaticCovariatesTransformer()
        train_series = scaler.fit_transform(self.multiple_series)
        model.fit(series=train_series)
        assert model.uses_static_covariates
        pred = model.predict(n=11, series=train_series[0])
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 11
        assert pred.n_components == 1

    @pytest.mark.parametrize(
        "model_name, model_kwargs",
        MULTIVARIATE_MODELS_WITH_PAST_AND_FUTURE_COVS,
    )
    def test_multiple_multivariate_series_with_static_covs(
        self, model_name: str, model_kwargs: dict | None
    ):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=9,
            output_chunk_length=12,
            model_kwargs=model_kwargs,
            use_static_covariates=True,
            **kwargs,
        )
        scaler = StaticCovariatesTransformer()
        train_series = scaler.fit_transform(self.multiple_multivariate_series)
        model.fit(series=train_series)
        assert model.uses_static_covariates
        pred = model.predict(n=10, series=train_series[0])
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 10
        assert pred.n_components == train_series[0].n_components

    @pytest.mark.parametrize("model_name, model_kwargs", ALL_MODELS)
    def test_output_chunk_shift(self, model_name: str, model_kwargs: dict | None):
        model = NeuralForecastModel(
            model=model_name,
            input_chunk_length=10,
            output_chunk_length=9,
            output_chunk_shift=2,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        model.fit(series=self.univariate_series)
        pred = model.predict(n=4)
        assert isinstance(pred, TimeSeries)
        assert pred.n_timesteps == 4
        assert pred.n_components == 1
        assert pred.start_time() - self.univariate_series.end_time() == pd.Timedelta(
            "3D"
        )

    @pytest.mark.parametrize("model_name, model_kwargs", ALL_MODELS)
    def test_mae_loss(self, model_name: str, model_kwargs: dict | None):
        for loss_fn in LOSS_FUNCTIONS:
            model = NeuralForecastModel(
                model=model_name,
                input_chunk_length=10,
                output_chunk_length=8,
                model_kwargs=model_kwargs,
                loss_fn=loss_fn,
                **kwargs,
            )
            model.fit(series=self.univariate_series)
            pred = model.predict(n=4)
            assert isinstance(pred, TimeSeries)
            assert pred.n_timesteps == 4
            assert pred.n_components == 1

    @pytest.mark.parametrize("model_name, model_kwargs", ALL_MODELS)
    def test_probabilistic_forecasting(
        self, model_name: str, model_kwargs: dict | None
    ):
        if model_name in ["DeepNPTS"]:
            pytest.skip(
                f"{model_name} does not support likelihood models. Skipping test.",
                allow_module_level=True,
            )

        for likelihood in LIKELIHOOD_MODELS:
            model = NeuralForecastModel(
                model=model_name,
                input_chunk_length=10,
                output_chunk_length=5,
                model_kwargs=model_kwargs,
                likelihood=likelihood,
                **kwargs,
            )
            model.fit(series=self.univariate_series)

            # predict with likelihood parameters
            pred = model.predict(n=4, predict_likelihood_parameters=True)
            assert isinstance(pred, TimeSeries)
            assert pred.n_timesteps == 4
            assert pred.n_components == likelihood.num_parameters

            # predict with auto-regressive sampling
            pred = model.predict(n=10, num_samples=50)
            assert isinstance(pred, TimeSeries)
            assert pred.n_timesteps == 10
            assert pred.n_components == 1
            assert pred.n_samples == 50

    def test_invalid_model_name(self):
        with pytest.raises(
            ValueError,
            match="Could not find a NeuralForecast model class named `InvalidModel`",
        ):
            NeuralForecastModel(
                model="InvalidModel",
                input_chunk_length=10,
                output_chunk_length=5,
                **kwargs,
            )

    def test_invalid_model_class(self):
        # not a base class
        from neuralforecast import NeuralForecast
        from neuralforecast.auto import AutoAutoformer, AutoKAN

        for model_class in ["HINT", NeuralForecast, AutoAutoformer, AutoKAN]:
            with pytest.raises(
                ValueError, match="must be a NeuralForecast base model class"
            ):
                NeuralForecastModel(
                    model=model_class,
                    input_chunk_length=10,
                    output_chunk_length=5,
                    **kwargs,
                )

    def test_invalid_model_kwargs(self):
        with pytest.raises(
            ValueError,
            match="The following parameters are not valid",
        ):
            NeuralForecastModel(
                model="PatchTST",
                input_chunk_length=10,
                output_chunk_length=5,
                model_kwargs={"invalid_arg_zzz": 123},
                **kwargs,
            )

    @pytest.mark.parametrize(
        "model_name, rinorm_name",
        [
            ("iTransformer", "use_norm"),
            ("PatchTST", "revin"),
        ],
    )
    def test_use_reversible_instance_norm(
        self, model_name: str, rinorm_name: str, caplog
    ):
        # Both iTransformer and PatchTST enable RINorm by default,
        # but can be disabled by setting `use_norm=False` or `revin=False`.
        with caplog.at_level(logging.WARNING):
            NeuralForecastModel(
                model=model_name,
                input_chunk_length=10,
                output_chunk_length=5,
                use_reversible_instance_norm=True,
                **kwargs,
            )
        assert (
            f"NeuralForecast model's `{rinorm_name}=True` may be incompatible"
            in caplog.text
        )

        NeuralForecastModel(
            model=model_name,
            input_chunk_length=10,
            output_chunk_length=5,
            use_reversible_instance_norm=True,
            model_kwargs={rinorm_name: False},
            **kwargs,
        )

    def test_use_reversible_instance_norm_rmok(self, caplog):
        # RMoK always enables RINorm and cannot be disabled
        with caplog.at_level(logging.WARNING):
            NeuralForecastModel(
                model="RMoK",
                input_chunk_length=10,
                output_chunk_length=5,
                use_reversible_instance_norm=True,
                **kwargs,
            )
        assert (
            "NeuralForecast model has reversible instance normalization enabled"
            in caplog.text
        )

        NeuralForecastModel(
            model="RMoK",
            input_chunk_length=10,
            output_chunk_length=5,
            use_reversible_instance_norm=False,
            **kwargs,
        )

    @pytest.mark.parametrize("model_name", ["DeepAR", "LSTM"])
    def test_unsupported_recurrent_models(self, model_name: str):
        # Recurrent models are not currently supported in Darts due to the way they handle covariates.
        # If we try to use one, we should get a clear error message.
        with pytest.raises(
            NotImplementedError,
            match="Recurrent NeuralForecast models are currently not supported",
        ):
            NeuralForecastModel(
                model=model_name,
                input_chunk_length=10,
                output_chunk_length=5,
                **kwargs,
            )
