import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import numpy as np
import pandas as pd
import torch
from torch import nn

from darts import concatenate
from darts.models.forecasting.tsmixer_model import TimeBatchNorm2d, TSMixerModel
from darts.tests.conftest import tfm_kwargs
from darts.utils import timeseries_generation as tg
from darts.utils.likelihood_models import GaussianLikelihood


class TestTSMixerModel:
    np.random.seed(42)
    torch.manual_seed(42)

    def test_creation(self):
        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            likelihood=GaussianLikelihood(),
        )

        assert model.input_chunk_length == 1

    def test_fit(self):
        large_ts = tg.constant_timeseries(length=10, value=1.0)
        small_ts = tg.constant_timeseries(length=10, value=0.1)

        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            n_epochs=10,
            random_state=42,
            **tfm_kwargs,
        )

        model.fit(large_ts)
        pred = model.predict(n=2).values()[0]

        # Test whether model trained on one series is better
        # than one trained on another
        model2 = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            n_epochs=10,
            random_state=42,
            **tfm_kwargs,
        )

        model2.fit(small_ts)
        pred2 = model2.predict(n=2).values()[0]
        assert abs(pred2 - 0.1) < abs(pred - 0.1)

        # test short predict
        pred3 = model2.predict(n=1)
        assert len(pred3) == 1

    def test_likelihood_fit(self):
        ts = tg.constant_timeseries(length=3)

        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            n_epochs=1,
            random_state=42,
            likelihood=GaussianLikelihood(),
            **tfm_kwargs,
        )
        model.fit(ts)
        # sampled from distribution
        pred = model.predict(n=1, num_samples=20)
        assert pred.n_samples == 20

        # direct distribution parameter prediction
        pred = model.predict(n=1, num_samples=1, predict_likelihood_parameters=True)
        assert pred.n_components == 2
        assert pred.n_samples == 1

        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit(ts)
        # mc dropout
        pred = model.predict(n=1, mc_dropout=True, num_samples=10)
        assert pred.n_samples == 10

    def test_logtensorboard(self, tmpdir_module):
        ts = tg.constant_timeseries(length=4)

        # Test basic fit and predict
        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            n_epochs=1,
            log_tensorboard=True,
            batch_size=2,
            work_dir=tmpdir_module,
            pl_trainer_kwargs={
                "log_every_n_steps": 1,
                **tfm_kwargs["pl_trainer_kwargs"],
            },
        )
        model.fit(ts)
        _ = model.predict(n=2)

    def test_static_covariates_support(self):
        target_multi = concatenate(
            [tg.sine_timeseries(length=10, freq="h")] * 2, axis=1
        )

        target_multi = target_multi.with_static_covariates(
            pd.DataFrame(
                [[0.0, 1.0, 0, 2], [2.0, 3.0, 1, 3]],
                columns=["st1", "st2", "cat1", "cat2"],
            )
        )

        # should work with cyclic encoding for time index
        model = TSMixerModel(
            input_chunk_length=3,
            output_chunk_length=4,
            add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
            pl_trainer_kwargs={
                "fast_dev_run": True,
                **tfm_kwargs["pl_trainer_kwargs"],
            },
        )
        model.fit(target_multi, verbose=False)

        assert model.model.static_cov_dim == np.prod(
            target_multi.static_covariates.values.shape
        )

        # raise an error when trained with static covariates of wrong dimensionality
        target_multi = target_multi.with_static_covariates(
            pd.concat([target_multi.static_covariates] * 2, axis=1)
        )
        with pytest.raises(ValueError):
            model.predict(n=1, series=target_multi, verbose=False)

        # raise an error when trained with static covariates and trying to predict without
        with pytest.raises(ValueError):
            model.predict(
                n=1, series=target_multi.with_static_covariates(None), verbose=False
            )

        # with `use_static_covariates=False`, we can predict without static covs
        model = TSMixerModel(
            input_chunk_length=3,
            output_chunk_length=4,
            use_static_covariates=False,
            n_epochs=1,
            **tfm_kwargs,
        )
        model.fit(target_multi)
        preds = model.predict(n=2, series=target_multi.with_static_covariates(None))
        assert preds.static_covariates is None

        model = TSMixerModel(
            input_chunk_length=3,
            output_chunk_length=4,
            use_static_covariates=False,
            n_epochs=1,
            **tfm_kwargs,
        )
        model.fit(target_multi.with_static_covariates(None))
        preds = model.predict(n=2, series=target_multi)
        assert preds.static_covariates.equals(target_multi.static_covariates)

    @pytest.mark.parametrize("enable_rin", [True, False])
    def test_future_covariate_handling(self, enable_rin):
        ts_time_index = tg.sine_timeseries(length=2, freq="h")

        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            add_encoders={"cyclic": {"future": "hour"}},
            use_reversible_instance_norm=enable_rin,
            **tfm_kwargs,
        )
        model.fit(ts_time_index, verbose=False, epochs=1)

    def test_past_covariate_handling(self):
        ts_time_index = tg.sine_timeseries(length=2, freq="h")

        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            add_encoders={"cyclic": {"past": "hour"}},
            **tfm_kwargs,
        )
        model.fit(ts_time_index, verbose=False, epochs=1)

    def test_future_and_past_covariate_handling(self):
        ts_time_index = tg.sine_timeseries(length=2, freq="h")

        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
            **tfm_kwargs,
        )
        model.fit(ts_time_index, verbose=False, epochs=1)

    def test_future_past_and_static_covariate_as_timeseries_handling(self):
        ts_time_index = tg.sine_timeseries(length=2, freq="h")
        ts_time_index = ts_time_index.with_static_covariates(
            pd.DataFrame(
                [
                    [
                        0.0,
                    ]
                ],
                columns=["st1"],
            )
        )
        for enable_rin in [True, False]:
            # test with past_covariates timeseries
            model = TSMixerModel(
                input_chunk_length=1,
                output_chunk_length=1,
                add_encoders={"cyclic": {"future": "hour"}},
                use_reversible_instance_norm=enable_rin,
                **tfm_kwargs,
            )
            model.fit(
                ts_time_index,
                past_covariates=ts_time_index,
                verbose=False,
                epochs=1,
            )

            # test with past_covariates and future_covariates timeseries
            model = TSMixerModel(
                input_chunk_length=1,
                output_chunk_length=1,
                add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
                use_reversible_instance_norm=enable_rin,
                **tfm_kwargs,
            )
            model.fit(
                ts_time_index,
                past_covariates=ts_time_index,
                future_covariates=ts_time_index,
                verbose=False,
                epochs=1,
            )

    @pytest.mark.parametrize(
        "norm_type, expect_exception",
        [
            ("LayerNorm", False),
            ("LayerNormNoBias", False),
            (nn.LayerNorm, False),
            ("TimeBatchNorm2d", False),
            ("invalid", True),
        ],
    )
    def test_layer_norms_with_parametrization(self, norm_type, expect_exception):
        series = tg.sine_timeseries(length=3)
        base_model = TSMixerModel

        if expect_exception:
            with pytest.raises(ValueError):
                model = base_model(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    norm_type=norm_type,
                    **tfm_kwargs,
                )
                model.fit(series, epochs=1)
        else:
            model = base_model(
                input_chunk_length=1,
                output_chunk_length=1,
                norm_type=norm_type,
                **tfm_kwargs,
            )
            model.fit(series, epochs=1)

    @pytest.mark.parametrize(
        "activation, expect_error",
        [
            ("ReLU", False),
            ("RReLU", False),
            ("PReLU", False),
            ("ELU", False),
            ("Softplus", False),
            ("Tanh", False),
            ("SELU", False),
            ("LeakyReLU", False),
            ("Sigmoid", False),
            ("invalid", True),
        ],
    )
    def test_activation_functions(self, activation, expect_error):
        series = tg.sine_timeseries(length=3)
        base_model = TSMixerModel

        if expect_error:
            with pytest.raises(ValueError):
                model = base_model(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    activation=activation,
                    **tfm_kwargs,
                )
                model.fit(series, epochs=1)
        else:
            model = base_model(
                input_chunk_length=1,
                output_chunk_length=1,
                activation=activation,
                **tfm_kwargs,
            )
            model.fit(series, epochs=1)

    def test_time_batch_norm_3d(self):
        torch.manual_seed(0)

        layer = TimeBatchNorm2d()
        # 4D does not work
        with pytest.raises(ValueError):
            layer.forward(torch.randn(3, 3, 3, 3))

        # 2D does not work
        with pytest.raises(ValueError):
            layer.forward(torch.randn(3, 3))

        # 3D works
        norm = layer.forward(torch.randn(3, 3, 3)).detach()
        assert norm.mean().numpy() == pytest.approx(0.0, abs=0.1)
        assert norm.std().numpy() == pytest.approx(1.0, abs=0.1)

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
    def test_time_batch_norm_2d_different_batch_sizes(self, batch_size):
        layer = TimeBatchNorm2d()
        input_tensor = torch.randn(batch_size, 3, 3)
        output = layer.forward(input_tensor)
        assert output.shape == input_tensor.shape

    def test_time_batch_norm_2d_gradients(self):
        normalized_shape = (10, 32)
        layer = TimeBatchNorm2d(normalized_shape)
        input_tensor = torch.randn(5, 10, 32, requires_grad=True)

        output = layer.forward(input_tensor)
        output.mean().backward()

        assert input_tensor.grad is not None
