import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    import torch.nn as nn
    from torch.nn import MSELoss

    from darts.models.forecasting.tft_model import TFTModel
    from darts.models.forecasting.tft_submodels import get_embedding_size
    from darts.utils.likelihood_models import QuantileRegression

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. TFT tests will be skipped.")
    TORCH_AVAILABLE = False
    TFTModel, QuantileRegression, MSELoss = None, None, None


if TORCH_AVAILABLE:

    class TFTModelTestCase(DartsBaseTestClass):
        def test_quantile_regression(self):
            q_no_50 = [0.1, 0.4, 0.9]
            q_non_symmetric = [0.2, 0.5, 0.9]

            # if a QuantileLoss is used, it must have to q=0.5 quantile
            with self.assertRaises(ValueError):
                QuantileRegression(q_no_50)

            # if a QuantileLoss is used, it must be symmetric around q=0.5 quantile (i.e. [0.1, 0.5, 0.9])
            with self.assertRaises(ValueError):
                QuantileRegression(q_non_symmetric)

        def test_future_covariate_handling(self):
            ts_time_index = tg.sine_timeseries(length=2, freq="h")
            ts_integer_index = TimeSeries.from_values(values=ts_time_index.values())

            # model requires future covariates without cyclic encoding
            model = TFTModel(input_chunk_length=1, output_chunk_length=1)
            with self.assertRaises(ValueError):
                model.fit(ts_time_index, verbose=False)

            # should work with cyclic encoding for time index
            model = TFTModel(
                input_chunk_length=1,
                output_chunk_length=1,
                add_encoders={"cyclic": {"future": "hour"}},
            )
            model.fit(ts_time_index, verbose=False)

            # should work with relative index both with time index and integer index
            model = TFTModel(
                input_chunk_length=1, output_chunk_length=1, add_relative_index=True
            )
            model.fit(ts_time_index, verbose=False)
            model.fit(ts_integer_index, verbose=False)

        def test_prediction_shape(self):
            """checks whether prediction has same number of variable as input series and
            whether prediction has correct length.
            Test cases:
                -   univariate
                -   multivariate
                -   multi-TS
            """
            season_length = 1
            n_repeat = 20

            # data comes as multivariate
            (
                ts,
                ts_train,
                ts_val,
                covariates,
            ) = self.helper_generate_multivariate_case_data(season_length, n_repeat)

            kwargs_TFT_quick_test = {
                "input_chunk_length": 1,
                "output_chunk_length": 1,
                "n_epochs": 1,
                "lstm_layers": 1,
                "hidden_size": 8,
                "loss_fn": MSELoss(),
                "random_state": 42,
            }

            # univariate
            first_var = ts.columns[0]
            self.helper_test_prediction_shape(
                season_length,
                ts[first_var],
                ts_train[first_var],
                ts_val[first_var],
                future_covariates=covariates,
                kwargs_tft=kwargs_TFT_quick_test,
            )
            # univariate and short prediction length
            self.helper_test_prediction_shape(
                2,
                ts[first_var],
                ts_train[first_var],
                ts_val[first_var],
                future_covariates=covariates,
                kwargs_tft=kwargs_TFT_quick_test,
            )
            # multivariate
            self.helper_test_prediction_shape(
                season_length,
                ts,
                ts_train,
                ts_val,
                future_covariates=covariates,
                kwargs_tft=kwargs_TFT_quick_test,
            )
            # multi-TS
            kwargs_TFT_quick_test["add_encoders"] = {"cyclic": {"future": "hour"}}
            second_var = ts.columns[-1]
            self.helper_test_prediction_shape(
                season_length,
                [ts[first_var], ts[second_var]],
                [ts_train[first_var], ts_train[second_var]],
                [ts_val[first_var], ts_val[second_var]],
                future_covariates=None,
                kwargs_tft=kwargs_TFT_quick_test,
            )

        def test_mixed_covariates_and_accuracy(self):
            """Performs tests usingpast and future covariates for a multivariate prediction of a
            sine wave together with a repeating linear curve. Both curves have the seasonal length.
            """
            season_length = 24
            n_repeat = 30
            (
                ts,
                ts_train,
                ts_val,
                covariates,
            ) = self.helper_generate_multivariate_case_data(season_length, n_repeat)

            kwargs_TFT_full_coverage = {
                "input_chunk_length": 12,
                "output_chunk_length": 12,
                "n_epochs": 10,
                "lstm_layers": 2,
                "hidden_size": 32,
                "likelihood": QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
                "random_state": 42,
                "add_encoders": {"cyclic": {"future": "hour"}},
            }

            self.helper_test_prediction_accuracy(
                season_length,
                ts,
                ts_train,
                ts_val,
                past_covariates=covariates,
                future_covariates=covariates,
                kwargs_tft=kwargs_TFT_full_coverage,
            )

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
            # set categorical embedding sizes once with automatic embedding size with an `int` and once by
            # manually setting it with `tuple(int, int)`
            model = TFTModel(
                input_chunk_length=3,
                output_chunk_length=4,
                add_encoders={"cyclic": {"future": "hour"}},
                categorical_embedding_sizes={"cat1": 2, "cat2": (2, 2)},
                pl_trainer_kwargs={"fast_dev_run": True},
            )
            model.fit(target_multi, verbose=False)

            assert len(model.model.static_variables) == len(
                target_multi.static_covariates.columns
            )

            # check model embeddings
            target_embedding = {
                "static_covariate_2": (
                    2,
                    get_embedding_size(2),
                ),  # automatic embedding size
                "static_covariate_3": (2, 2),  # manual embedding size
            }
            assert model.categorical_embedding_sizes == target_embedding
            for cat_var, embedding_dims in target_embedding.items():
                assert (
                    model.model.input_embeddings.embeddings[cat_var].num_embeddings
                    == embedding_dims[0]
                )
                assert (
                    model.model.input_embeddings.embeddings[cat_var].embedding_dim
                    == embedding_dims[1]
                )

            preds = model.predict(n=1, series=target_multi, verbose=False)
            assert preds.static_covariates.equals(target_multi.static_covariates)

            # raise an error when trained with static covariates of wrong dimensionality
            target_multi = target_multi.with_static_covariates(
                pd.concat([target_multi.static_covariates] * 2, axis=1)
            )
            with pytest.raises(ValueError):
                model.predict(n=1, series=target_multi, verbose=False)

            # raise an error when trained with static covariates and trying to predict without
            target_multi = target_multi.with_static_covariates(None)
            with pytest.raises(ValueError):
                model.predict(n=1, series=target_multi, verbose=False)

        def helper_generate_multivariate_case_data(self, season_length, n_repeat):
            """generates multivariate test case data. Target series is a sine wave stacked with a repeating
            linear curve of equal seasonal length. Covariates are datetime attributes for 'hours'.
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

            # generate long enough covariates (past and future covariates will be the same for simplicity)
            long_enough_ts = tg.sine_timeseries(
                value_frequency=1 / season_length, length=1000, freq=ts.freq
            )
            covariates = tg.datetime_attribute_timeseries(
                long_enough_ts, attribute="hour"
            )
            scaler_covs = Scaler()
            covariates_scaled = scaler_covs.fit_transform(covariates)
            return ts_scaled, ts_train_scaled, ts_val_scaled, covariates_scaled

        def helper_test_prediction_shape(
            self, predict_n, ts, ts_train, ts_val, future_covariates, kwargs_tft
        ):
            """checks whether prediction has same number of variable as input series and
            whether prediction has correct length"""
            y_hat = self.helper_fit_predict(
                predict_n, ts_train, ts_val, None, future_covariates, kwargs_tft
            )

            y_hat_list = [y_hat] if isinstance(y_hat, TimeSeries) else y_hat
            ts_list = [ts] if isinstance(ts, TimeSeries) else ts

            for y_hat_i, ts_i in zip(y_hat_list, ts_list):
                self.assertEqual(len(y_hat_i), predict_n)
                self.assertEqual(y_hat_i.n_components, ts_i.n_components)

        def helper_test_prediction_accuracy(
            self,
            predict_n,
            ts,
            ts_train,
            ts_val,
            past_covariates,
            future_covariates,
            kwargs_tft,
        ):
            """prediction should be almost equal to y_true. Absolute tolarance is set
            to 0.2 to give some flexibility"""

            absolute_tolarance = 0.2
            y_hat = self.helper_fit_predict(
                predict_n,
                ts_train,
                ts_val,
                past_covariates,
                future_covariates,
                kwargs_tft,
            )

            y_true = ts[y_hat.start_time() : y_hat.end_time()]
            self.assertTrue(
                np.allclose(
                    y_true[1:-1].all_values(),
                    y_hat[1:-1].all_values(),
                    atol=absolute_tolarance,
                )
            )

        @staticmethod
        def helper_fit_predict(
            predict_n, ts_train, ts_val, past_covariates, future_covariates, kwargs_tft
        ):
            """simple helper that returns prediction for the individual test cases"""
            model = TFTModel(**kwargs_tft)

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
                num_samples=(100 if model._is_probabilistic() else 1),
            )

            if isinstance(y_hat, TimeSeries):
                y_hat = y_hat.quantile_timeseries(0.5) if y_hat.n_samples > 1 else y_hat
            else:
                y_hat = [
                    ts.quantile_timeseries(0.5) if ts.n_samples > 1 else ts
                    for ts in y_hat
                ]
            return y_hat

        def test_layer_norm(self):
            times = pd.date_range("20130101", "20130410")
            pd_series = pd.Series(range(100), index=times)
            series: TimeSeries = TimeSeries.from_series(pd_series)
            base_model = TFTModel

            model1 = base_model(
                input_chunk_length=1,
                output_chunk_length=1,
                add_relative_index=True,
                norm_type="RMSNorm",
            )
            model1.fit(series, epochs=1)

            model2 = base_model(
                input_chunk_length=1,
                output_chunk_length=1,
                add_relative_index=True,
                norm_type=nn.LayerNorm,
            )
            model2.fit(series, epochs=1)

            with self.assertRaises(AttributeError):
                model4 = base_model(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    add_relative_index=True,
                    norm_type="invalid",
                )
                model4.fit(series, epochs=1)
