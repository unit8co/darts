from darts.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np
    import pandas as pd
    import pytest
    import torch
    from torch import nn

    from darts import TimeSeries, concatenate
    from darts.dataprocessing.transformers import Scaler
    from darts.models.forecasting.tsmixer_model import TimeBatchNorm2d, TSMixerModel
    from darts.tests.conftest import tfm_kwargs
    from darts.utils import timeseries_generation as tg
    from darts.utils.likelihood_models import GaussianLikelihood

    TORCH_AVAILABLE = True

except ImportError:
    logger.warning("Torch not available. TSMixerModel tests will be skipped.")
    TORCH_AVAILABLE = False


@pytest.mark.skipif(
    TORCH_AVAILABLE is False,
    reason="Torch not available. TSMixerModel tests will be skipped.",
)
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
        large_ts = tg.constant_timeseries(length=100, value=1000)
        small_ts = tg.constant_timeseries(length=100, value=10)

        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            n_epochs=10,
            random_state=42,
            **tfm_kwargs,
        )

        model.fit(large_ts[:98])
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

        model2.fit(small_ts[:98])
        pred2 = model2.predict(n=2).values()[0]
        assert abs(pred2 - 10) < abs(pred - 10)

        # test short predict
        pred3 = model2.predict(n=1)
        assert len(pred3) == 1

    def test_likelihood_fit(self):
        large_ts = tg.constant_timeseries(length=100, value=1000)

        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            n_epochs=10,
            random_state=42,
            likelihood=GaussianLikelihood(),
            **tfm_kwargs,
        )

        model.fit(large_ts[:98])
        pred = model.predict(n=2, num_samples=20)

        assert pred.n_samples == 20

        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            n_epochs=10,
            random_state=42,
            **tfm_kwargs,
        )

        model.fit(large_ts[:98])
        pred = model.predict(n=2, mc_dropout=True, num_samples=10)

        assert pred.n_samples == 10

    def test_logtensorboard(self, tmpdir_module):
        ts = tg.constant_timeseries(length=50, value=10)

        # Test basic fit and predict
        model = TSMixerModel(
            input_chunk_length=1,
            output_chunk_length=1,
            n_epochs=1,
            log_tensorboard=True,
            work_dir=tmpdir_module,
            pl_trainer_kwargs={
                "log_every_n_steps": 1,
                **tfm_kwargs["pl_trainer_kwargs"],
            },
        )
        model.fit(ts)
        model.predict(n=2)

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
            add_encoders={"cyclic": {"future": "hour"}},
            pl_trainer_kwargs={
                "fast_dev_run": True,
                **tfm_kwargs["pl_trainer_kwargs"],
            },
        )
        model.fit(target_multi, verbose=False)

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

    def test_future_covariate_handling(self):
        ts_time_index = tg.sine_timeseries(length=2, freq="h")

        for enable_rin in [True, False]:
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

    def test_multivariate_static_covariates_support(self):
        target_multi = concatenate(
            [tg.sine_timeseries(length=10, freq="h")] * 2, axis=1
        )

        target_multi = target_multi.with_static_covariates(
            pd.DataFrame(
                [[0.0, 1.0, 0, 2], [2.0, 3.0, 1, 3]],
                columns=["st1", "st2", "cat1", "cat2"],
            )
        )

        # test with static covariates in the timeseries
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

        # raise an error when trained with static covariates of wrong dimensionality
        target_multi = target_multi.with_static_covariates(
            pd.concat([target_multi.static_covariates] * 2, axis=1)
        )

        with pytest.raises(ValueError):
            model.predict(n=1, series=target_multi, verbose=False)

        # raise an error when trained with static covariates and trying to
        # predict without
        with pytest.raises(ValueError):
            model.predict(
                n=1, series=target_multi.with_static_covariates(None), verbose=False
            )

    @pytest.mark.parametrize(
        "norm_type, expect_exception",
        [
            ("RINorm", False),
            (nn.LayerNorm, False),
            ("TimeBatchNorm2d", False),
            ("invalid", True),
        ],
    )
    def test_layer_norms_with_parametrization(self, norm_type, expect_exception):
        times = pd.date_range("20130101", "20130410")
        pd_series = pd.Series(range(100), index=times)
        series = TimeSeries.from_series(pd_series)
        base_model = TSMixerModel

        if expect_exception:
            with pytest.raises(AttributeError):
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
        times = pd.date_range("20130101", "20130410")
        pd_series = pd.Series(range(100), index=times)
        series = TimeSeries.from_series(pd_series)
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

    def test_time_batch_norm_2d(self):
        # test init
        normalized_shape = (10, 32)
        layer = TimeBatchNorm2d(normalized_shape)

        expected_num_features = normalized_shape[0] * normalized_shape[1]
        assert layer.num_features == expected_num_features

        # test 3d tensor
        normalized_shape = (10, 32)
        layer = TimeBatchNorm2d(normalized_shape)

        incorrect_input = torch.randn(5, 10, 32, 32)

        with pytest.raises(ValueError):
            layer.forward(incorrect_input)

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
    def test_time_batch_norm_2d_different_batch_sizes(self, batch_size):
        normalized_shape = (10, 32)
        layer = TimeBatchNorm2d(normalized_shape)
        input_tensor = torch.randn(batch_size, *normalized_shape)

        if batch_size < 2:
            with pytest.raises(ValueError):
                layer.forward(input_tensor)
        else:
            output = layer.forward(input_tensor)
            assert output.shape == input_tensor.shape

    def test_time_batch_norm_2d_gradients(self):
        normalized_shape = (10, 32)
        layer = TimeBatchNorm2d(normalized_shape)
        input_tensor = torch.randn(5, 10, 32, requires_grad=True)

        output = layer.forward(input_tensor)
        output.mean().backward()

        assert input_tensor.grad is not None

    @staticmethod
    def helper_fit_predict(
        predict_n,
        ts_train,
        ts_val,
        past_covariates,
        future_covariates,
        kwargs_tsmixer,
    ):
        """simple helper that returns prediction for the individual test cases"""
        model = TSMixerModel(**kwargs_tsmixer)

        model.fit(
            ts_train,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            val_series=ts_val,
            val_past_covariates=past_covariates,
            val_future_covariates=future_covariates,
            verbose=False,
        )

        series = None if isinstance(ts_train, TimeSeries) else ts_train
        y_hat = model.predict(
            n=predict_n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=(100 if model.supports_probabilistic_prediction else 1),
        )

        if isinstance(y_hat, TimeSeries):
            y_hat = y_hat.quantile_timeseries(0.5) if y_hat.n_samples > 1 else y_hat
        else:
            y_hat = [
                ts.quantile_timeseries(0.5) if ts.n_samples > 1 else ts for ts in y_hat
            ]
        return y_hat

    def helper_test_prediction_accuracy(
        self,
        predict_n,
        ts,
        ts_train,
        ts_val,
        past_covariates,
        future_covariates,
        kwargs_tsmixer,
    ):
        """prediction should be almost equal to y_true. Absolute tolerance is set
        to 0.2 to give some flexibility"""
        absolute_tolerance = 0.2
        y_hat = self.helper_fit_predict(
            predict_n,
            ts_train,
            ts_val,
            past_covariates,
            future_covariates,
            kwargs_tsmixer,
        )

        y_true = ts[y_hat.start_time() : y_hat.end_time()]

        assert np.allclose(
            y_true[1:-1].all_values(),
            y_hat[1:-1].all_values(),
            atol=absolute_tolerance,
        )

    def helper_test_prediction_shape(
        self, predict_n, ts, ts_train, ts_val, future_covariates, kwargs_tsmixer
    ):
        """checks whether prediction has same number of variable as input series and
        whether prediction has correct length"""
        y_hat = self.helper_fit_predict(
            predict_n, ts_train, ts_val, None, future_covariates, kwargs_tsmixer
        )

        y_hat_list = [y_hat] if isinstance(y_hat, TimeSeries) else y_hat
        ts_list = [ts] if isinstance(ts, TimeSeries) else ts

        for y_hat_i, ts_i in zip(y_hat_list, ts_list):
            assert len(y_hat_i) == predict_n
            assert y_hat_i.n_components == ts_i.n_components

    @pytest.mark.parametrize(
        "season_length,selection,future_covariates_adjustment,multi_ts",
        [
            (1, "single", True, False),  # univariate
            (2, "single", True, False),  # univariate and short prediction length
            (1, "all", True, False),  # multivariate
            (1, "multi", False, True),  # multi-TS
        ],
    )
    def test_prediction_shape_parametrized(
        self, season_length, selection, future_covariates_adjustment, multi_ts
    ):
        n_repeat = 20
        # Generate data
        (
            ts,
            ts_train,
            ts_val,
            covariates,
        ) = self.helper_generate_multivariate_case_data(season_length, n_repeat)

        # Default kwargs
        kwargs = {
            "input_chunk_length": 1,
            "output_chunk_length": 1,
            "n_epochs": 1,
            "random_state": 42,
            "add_encoders": {"cyclic": {"future": "hour"}},
        }

        future_covariates = covariates if future_covariates_adjustment else None

        if selection == "single":
            first_var = ts.columns[0]
            ts_selected = ts[first_var]
            ts_train_selected = ts_train[first_var]
            ts_val_selected = ts_val[first_var]
        elif selection == "all":
            ts_selected = ts
            ts_train_selected = ts_train
            ts_val_selected = ts_val
        elif multi_ts:
            first_var, second_var = ts.columns[0], ts.columns[-1]
            ts_selected = [ts[first_var], ts[second_var]]
            ts_train_selected = [ts_train[first_var], ts_train[second_var]]
            ts_val_selected = [ts_val[first_var], ts_val[second_var]]

        # Call your test helper function
        self.helper_test_prediction_shape(
            season_length,
            ts_selected,
            ts_train_selected,
            ts_val_selected,
            future_covariates=future_covariates,
            kwargs_tsmixer=kwargs,
        )

    def helper_generate_multivariate_case_data(self, season_length, n_repeat):
        """generates multivariate test case data. Target series is a sine wave
        stacked with a repeating linear curve of equal seasonal length.
        Covariates are datetime attributes for 'hours'.
        """

        # generate sine wave
        ts_sine = tg.sine_timeseries(
            value_frequency=1 / season_length,
            length=n_repeat * season_length,
            freq="h",
        )

        # generate repeating linear curve
        ts_linear = tg.linear_timeseries(
            0, 1, length=season_length, start=ts_sine.end_time() + ts_sine.freq
        )

        for i in range(n_repeat - 1):
            start = ts_linear.end_time() + ts_linear.freq
            new_ts = tg.linear_timeseries(0, 1, length=season_length, start=start)
            ts_linear = ts_linear.append(new_ts)

        ts_linear = TimeSeries.from_times_and_values(
            times=ts_sine.time_index, values=ts_linear.values()
        )

        # create multivariate TimeSeries by stacking sine and linear curves
        ts = ts_sine.stack(ts_linear)

        # create train/test sets
        val_length = 10 * season_length

        ts_train, ts_val = ts[:-val_length], ts[-val_length:]

        # scale data
        scaler_ts = Scaler()
        ts_train_scaled = scaler_ts.fit_transform(ts_train)
        ts_val_scaled = scaler_ts.transform(ts_val)
        ts_scaled = scaler_ts.transform(ts)

        # generate long enough covariates (past and future covariates will be
        # the same for simplicity)
        long_enough_ts = tg.sine_timeseries(
            value_frequency=1 / season_length, length=1000, freq=ts.freq
        )
        covariates = tg.datetime_attribute_timeseries(long_enough_ts, attribute="hour")
        scaler_covs = Scaler()
        covariates_scaled = scaler_covs.fit_transform(covariates)
        return ts_scaled, ts_train_scaled, ts_val_scaled, covariates_scaled

    def test_mixed_covariates_and_accuracy(self):
        """Performs tests using past and future covariates for a multivariate
        prediction of a sine wave together with a repeating linear curve.
        Both curves have the seasonal length.
        """
        season_length = 24
        n_repeat = 30
        (
            ts,
            ts_train,
            ts_val,
            covariates,
        ) = self.helper_generate_multivariate_case_data(season_length, n_repeat)

        kwargs_full_coverage = {
            "input_chunk_length": 12,
            "output_chunk_length": 12,
            "n_epochs": 100,
            "random_state": 42,
            "blocks": 1,
            "hidden_size": 32,
            "dropout": 0.2,
            "ff_size": 32,
            "batch_size": 8,
        }
        kwargs_full_coverage = dict(kwargs_full_coverage, **tfm_kwargs)

        self.helper_test_prediction_accuracy(
            season_length,
            ts,
            ts_train,
            ts_val,
            past_covariates=covariates,
            future_covariates=covariates,
            kwargs_tsmixer=kwargs_full_coverage,
        )
