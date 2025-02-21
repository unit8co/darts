import copy
import itertools
import os

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.logging import get_logger
from darts.models import (
    ExponentialSmoothing,
    LinearRegressionModel,
    NaiveDrift,
    NaiveEnsembleModel,
    NaiveSeasonal,
    RegressionEnsembleModel,
    StatsForecastAutoARIMA,
    Theta,
)
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg

if TORCH_AVAILABLE:
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
else:
    TorchForecastingModel = None

logger = get_logger(__name__)

if TORCH_AVAILABLE:
    from darts.models import DLinearModel, NBEATSModel, RNNModel, TCNModel
    from darts.utils.likelihood_models import QuantileRegression


def _make_ts(start_value=0, n=100):
    times = pd.date_range(start="1/1/2013", periods=n, freq="D")
    pd_series = pd.Series(range(start_value, start_value + n), index=times)
    return TimeSeries.from_series(pd_series)


class TestEnsembleModels:
    series1 = tg.sine_timeseries(value_frequency=(1 / 5), value_y_offset=10, length=50)
    series2 = tg.linear_timeseries(length=50)

    seq1 = [_make_ts(0), _make_ts(10), _make_ts(20)]
    cov1 = [_make_ts(5), _make_ts(15), _make_ts(25)]

    def test_untrained_models(self):
        model = NaiveDrift()
        _ = NaiveEnsembleModel([model])

        # trained local models should raise error
        model.fit(self.series1)
        with pytest.raises(ValueError):
            NaiveEnsembleModel([model])

        # an untrained ensemble model should also give untrained underlying models
        model_ens = NaiveEnsembleModel([NaiveDrift()])
        model_ens.fit(self.series1)
        assert model_ens.forecasting_models[0]._fit_called
        new_model = model_ens.untrained_model()
        assert not new_model.forecasting_models[0]._fit_called

    def test_trained_models(self):
        """EnsembleModels can be instantiated with pre-trained GlobalForecastingModels"""
        local_model = NaiveDrift()
        global_model = LinearRegressionModel(lags=2)
        local_model.fit(self.series1)
        global_model.fit(self.series1)

        # local and global trained
        with pytest.raises(ValueError):
            NaiveEnsembleModel([local_model, global_model])

        # local untrained, global trained
        with pytest.raises(ValueError):
            NaiveEnsembleModel([local_model.untrained_model(), global_model])

        # local trained, global untrained
        with pytest.raises(ValueError):
            NaiveEnsembleModel([local_model, global_model.untrained_model()])

        # global trained, global untrained
        with pytest.raises(ValueError):
            NaiveEnsembleModel([global_model, global_model.untrained_model()])

        # both global trained, retrain = True
        with pytest.raises(ValueError):
            # models need to be explicitly reset before retraining them
            NaiveEnsembleModel(
                [global_model, global_model], train_forecasting_models=True
            )
        model_ens_retrain = NaiveEnsembleModel(
            [global_model.untrained_model(), global_model.untrained_model()],
            train_forecasting_models=True,
        )
        with pytest.raises(ValueError):
            model_ens_retrain.predict(1, series=self.series1)
        model_ens_retrain.fit(self.series1)
        model_ens_retrain.predict(1, series=self.series1)

        # both global trained, retrain = False
        model_ens_no_retrain = NaiveEnsembleModel(
            [global_model, global_model], train_forecasting_models=False
        )
        model_ens_no_retrain.predict(1, series=self.series1)

    def test_extreme_lag_inference(self):
        ensemble = NaiveEnsembleModel([NaiveDrift()])
        assert ensemble.extreme_lags == (
            -3,
            -1,
            None,
            None,
            None,
            None,
            0,
            None,
        )  # test if default is okay

        model1 = LinearRegressionModel(
            lags=3, lags_past_covariates=[-3, -5], lags_future_covariates=[7, 8]
        )
        model2 = LinearRegressionModel(
            lags=5, lags_past_covariates=6, lags_future_covariates=[6, 9]
        )

        ensemble = NaiveEnsembleModel([
            model1,
            model2,
        ])  # test if infers extreme lags is okay
        expected = (-5, 0, -6, -1, 6, 9, 0, None)
        assert expected == ensemble.extreme_lags

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_extreme_lags_rnn(self):
        # RNNModel has the 8th element in `extreme_lags` for the `max_target_lag_train`.
        # it is given by `training_length - input_chunk_length`.
        # for the ensemble model we want the max lag of all forecasting models.
        model1 = RNNModel(input_chunk_length=14, training_length=24)
        model2 = RNNModel(input_chunk_length=12, training_length=37)

        ensemble = NaiveEnsembleModel([model1, model2])
        expected = (-14, 0, None, None, -14, 0, 0, 37 - 12)
        assert expected == ensemble.extreme_lags

    def test_input_models_local_models(self):
        with pytest.raises(ValueError):
            NaiveEnsembleModel([])
        # models are not instantiated
        with pytest.raises(ValueError):
            NaiveEnsembleModel([NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing])
        # one model is not instantiated
        with pytest.raises(ValueError):
            NaiveEnsembleModel([
                NaiveDrift(),
                NaiveSeasonal,
                Theta(),
                ExponentialSmoothing(),
            ])
        NaiveEnsembleModel([
            NaiveDrift(),
            NaiveSeasonal(),
            Theta(),
            ExponentialSmoothing(),
        ])

    def test_call_predict_local_models(self):
        naive_ensemble = NaiveEnsembleModel([NaiveSeasonal(), Theta()])
        with pytest.raises(Exception):
            naive_ensemble.predict(5)
        naive_ensemble.fit(self.series1)
        pred1 = naive_ensemble.predict(5)
        assert self.series1.components == pred1.components

    def test_call_backtest_naive_ensemble_local_models(self):
        ensemble = NaiveEnsembleModel([NaiveSeasonal(5), Theta(2, 5)])
        ensemble.fit(self.series1)
        assert ensemble.extreme_lags == (-10, -1, None, None, None, None, 0, None)
        ensemble.backtest(self.series1)

    def test_predict_univariate_ensemble_local_models(self):
        naive = NaiveSeasonal(K=5)
        theta = Theta()
        naive_ensemble: NaiveEnsembleModel = NaiveEnsembleModel([naive, theta])
        naive_ensemble.fit(self.series1 + self.series2)
        forecast_naive_ensemble = naive_ensemble.predict(5)
        naive.fit(self.series1 + self.series2)
        theta.fit(self.series1 + self.series2)
        forecast_mean = 0.5 * naive.predict(5) + 0.5 * theta.predict(5)

        np.testing.assert_array_equal(
            forecast_naive_ensemble.values(), forecast_mean.values()
        )

    def test_predict_multivariate_ensemble_local_models(self):
        multivariate_series = self.series1.stack(self.series2)

        seasonal1 = NaiveSeasonal(K=5)
        seasonal2 = NaiveSeasonal(K=8)
        naive_ensemble: NaiveEnsembleModel = NaiveEnsembleModel([seasonal1, seasonal2])
        naive_ensemble.fit(multivariate_series)
        forecast_naive_ensemble = naive_ensemble.predict(5)
        seasonal1.fit(multivariate_series)
        seasonal2.fit(multivariate_series)
        forecast_mean = 0.5 * seasonal1.predict(5) + 0.5 * seasonal2.predict(5)

        np.testing.assert_array_equal(
            forecast_naive_ensemble.values(), forecast_mean.values()
        )
        assert all(forecast_naive_ensemble.components == multivariate_series.components)

    def test_stochastic_naive_ensemble(self):
        num_samples = 100

        # probabilistic forecasting models
        model_proba_1 = LinearRegressionModel(
            lags=1, likelihood="quantile", random_state=42
        )
        model_proba_2 = LinearRegressionModel(
            lags=2, likelihood="quantile", random_state=42
        )

        # only probabilistic forecasting models
        naive_ensemble_proba = NaiveEnsembleModel([model_proba_1, model_proba_2])
        assert naive_ensemble_proba.supports_probabilistic_prediction

        naive_ensemble_proba.fit(self.series1 + self.series2)
        # by default, only 1 sample
        pred_proba_1_sample = naive_ensemble_proba.predict(n=5)
        assert pred_proba_1_sample.n_samples == 1

        # possible to obtain probabilistic forecast by averaging samples across the models
        pred_proba_many_sample = naive_ensemble_proba.predict(
            n=5, num_samples=num_samples
        )
        assert pred_proba_many_sample.n_samples == num_samples

        # need to redefine the models to reset the random state
        model_alone_1 = LinearRegressionModel(
            lags=1, likelihood="quantile", random_state=42
        )
        model_alone_2 = LinearRegressionModel(
            lags=2, likelihood="quantile", random_state=42
        )
        model_alone_1.fit(self.series1 + self.series2)
        model_alone_2.fit(self.series1 + self.series2)
        forecast_mean = 0.5 * model_alone_1.predict(
            5, num_samples=num_samples
        ) + 0.5 * model_alone_2.predict(5, num_samples=num_samples)

        assert forecast_mean.values().shape == pred_proba_many_sample.values().shape
        assert forecast_mean.n_samples == pred_proba_many_sample.n_samples
        np.testing.assert_array_equal(
            pred_proba_many_sample.values(), forecast_mean.values()
        )

    def test_predict_likelihood_parameters_wrong_args(self):
        m_deterministic = LinearRegressionModel(lags=2, output_chunk_length=2)
        m_proba_quantile1 = LinearRegressionModel(
            lags=2,
            output_chunk_length=2,
            likelihood="quantile",
            quantiles=[0.05, 0.5, 0.95],
        )
        m_proba_quantile2 = LinearRegressionModel(
            lags=3,
            output_chunk_length=3,
            likelihood="quantile",
            quantiles=[0.05, 0.5, 0.95],
        )
        m_proba_poisson = LinearRegressionModel(
            lags=2, output_chunk_length=2, likelihood="poisson"
        )
        # one model is not probabilistic
        naive_ensemble = NaiveEnsembleModel([m_deterministic, m_proba_quantile1])
        naive_ensemble.fit(self.series1 + self.series2)
        with pytest.raises(ValueError):
            naive_ensemble.predict(n=1, predict_likelihood_parameters=True)

        # one model has a different likelihood
        naive_ensemble = NaiveEnsembleModel([
            m_proba_quantile1.untrained_model(),
            m_proba_poisson,
        ])
        naive_ensemble.fit(self.series1 + self.series2)
        with pytest.raises(ValueError):
            naive_ensemble.predict(n=1, predict_likelihood_parameters=True)

        # n > shortest output_chunk_length
        naive_ensemble = NaiveEnsembleModel([
            m_proba_quantile1.untrained_model(),
            m_proba_quantile2,
        ])
        naive_ensemble.fit(self.series1 + self.series2)
        with pytest.raises(ValueError):
            naive_ensemble.predict(n=4, predict_likelihood_parameters=True)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_predict_likelihood_parameters_univariate_naive_ensemble(self):
        m_proba_quantile1 = LinearRegressionModel(
            lags=2,
            output_chunk_length=2,
            likelihood="quantile",
            quantiles=[0.05, 0.5, 0.95],
        )
        m_proba_quantile2 = LinearRegressionModel(
            lags=3,
            output_chunk_length=2,
            likelihood="quantile",
            quantiles=[0.05, 0.5, 0.95],
        )
        m_proba_quantile3 = DLinearModel(
            input_chunk_length=4,
            output_chunk_length=2,
            likelihood=QuantileRegression([0.05, 0.5, 0.95]),
            **tfm_kwargs,
        )

        naive_ensemble = NaiveEnsembleModel([m_proba_quantile1, m_proba_quantile2])
        naive_ensemble.fit(self.series1)
        pred_ens = naive_ensemble.predict(n=1, predict_likelihood_parameters=True)
        naive_ensemble = NaiveEnsembleModel([
            m_proba_quantile2.untrained_model(),
            m_proba_quantile3.untrained_model(),
        ])
        naive_ensemble.fit(self.series1)
        pred_mix_ens = naive_ensemble.predict(n=1, predict_likelihood_parameters=True)
        assert pred_ens.time_index == pred_mix_ens.time_index
        assert all(pred_ens.components == pred_mix_ens.components)
        assert (
            pred_ens["sine_q0.05"].values()
            < pred_ens["sine_q0.50"].values()
            < pred_ens["sine_q0.95"].values()
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_predict_likelihood_parameters_multivariate_naive_ensemble(self):
        m_proba_quantile1 = LinearRegressionModel(
            lags=2,
            output_chunk_length=2,
            likelihood="quantile",
            quantiles=[0.05, 0.5, 0.95],
        )
        m_proba_quantile2 = LinearRegressionModel(
            lags=3,
            output_chunk_length=2,
            likelihood="quantile",
            quantiles=[0.05, 0.5, 0.95],
        )
        m_proba_quantile3 = DLinearModel(
            input_chunk_length=4,
            output_chunk_length=2,
            likelihood=QuantileRegression([0.05, 0.5, 0.95]),
            **tfm_kwargs,
        )

        multivariate_series = self.series1.stack(self.series2)

        naive_ensemble = NaiveEnsembleModel([m_proba_quantile1, m_proba_quantile2])
        naive_ensemble.fit(multivariate_series)
        pred_ens = naive_ensemble.predict(n=1, predict_likelihood_parameters=True)
        naive_ensemble = NaiveEnsembleModel([
            m_proba_quantile2.untrained_model(),
            m_proba_quantile3.untrained_model(),
        ])
        naive_ensemble.fit(multivariate_series)
        pred_mix_ens = naive_ensemble.predict(n=1, predict_likelihood_parameters=True)
        assert pred_ens.time_index == pred_mix_ens.time_index
        assert all(
            pred_ens.components
            == [
                "sine_q0.05",
                "sine_q0.50",
                "sine_q0.95",
                "linear_q0.05",
                "linear_q0.50",
                "linear_q0.95",
            ]
        )
        assert all(pred_ens.components == pred_mix_ens.components)
        assert (
            pred_ens["sine_q0.05"].values()
            < pred_ens["sine_q0.50"].values()
            < pred_ens["sine_q0.95"].values()
        )
        assert (
            pred_ens["linear_q0.05"].values()
            < pred_ens["linear_q0.50"].values()
            < pred_ens["linear_q0.95"].values()
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_input_models_global_models(self):
        # one model is not instantiated
        with pytest.raises(ValueError):
            NaiveEnsembleModel([RNNModel(12), TCNModel(10, 2), NBEATSModel])
        NaiveEnsembleModel([RNNModel(12), TCNModel(10, 2), NBEATSModel(10, 2)])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_call_predict_global_models_univariate_input_no_covariates(self):
        naive_ensemble = NaiveEnsembleModel([
            RNNModel(12, n_epochs=1, **tfm_kwargs),
            TCNModel(10, 2, n_epochs=1, **tfm_kwargs),
            NBEATSModel(10, 2, n_epochs=1, **tfm_kwargs),
        ])
        with pytest.raises(Exception):
            naive_ensemble.predict(5)

        naive_ensemble.fit(self.series1)
        naive_ensemble.predict(5)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_call_predict_global_models_multivariate_input_no_covariates(self):
        naive_ensemble = NaiveEnsembleModel([
            RNNModel(12, n_epochs=1, **tfm_kwargs),
            TCNModel(10, 2, n_epochs=1, **tfm_kwargs),
            NBEATSModel(10, 2, n_epochs=1, **tfm_kwargs),
        ])
        naive_ensemble.fit(self.seq1)
        naive_ensemble.predict(n=5, series=self.seq1)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_call_predict_global_models_multivariate_input_with_covariates(self):
        naive_ensemble = NaiveEnsembleModel([
            RNNModel(12, n_epochs=1, **tfm_kwargs),
            TCNModel(10, 2, n_epochs=1, **tfm_kwargs),
            NBEATSModel(10, 2, n_epochs=1, **tfm_kwargs),
        ])
        naive_ensemble.fit(self.seq1, self.cov1)
        predict_series = [s[:12] for s in self.seq1]
        predict_covariates = [c[:14] for c in self.cov1]
        naive_ensemble.predict(
            n=2, series=predict_series, past_covariates=predict_covariates
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_input_models_mixed(self):
        # NaiveDrift is local, RNNModel is global
        naive_ensemble = NaiveEnsembleModel([
            NaiveDrift(),
            RNNModel(12, n_epochs=1, **tfm_kwargs),
        ])
        # ensemble is neither local, nor global
        assert not naive_ensemble.is_local_ensemble
        assert not naive_ensemble.is_global_ensemble

        # ensemble contains one local model, no support for multiple ts fit
        with pytest.raises(ValueError):
            naive_ensemble.fit([self.series1, self.series2])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_call_predict_different_covariates_support(self):
        # AutoARIMA support future covariates only
        local_ensemble_one_covs = NaiveEnsembleModel([
            NaiveDrift(),
            StatsForecastAutoARIMA(),
        ])
        with pytest.raises(ValueError):
            local_ensemble_one_covs.fit(self.series1, past_covariates=self.series2)
        local_ensemble_one_covs.fit(self.series1, future_covariates=self.series2)

        # RNN support future covariates only
        mixed_ensemble_one_covs = NaiveEnsembleModel([
            NaiveDrift(),
            RNNModel(12, n_epochs=1, **tfm_kwargs),
        ])
        with pytest.raises(ValueError):
            mixed_ensemble_one_covs.fit(self.series1, past_covariates=self.series2)
        mixed_ensemble_one_covs.fit(self.series1, future_covariates=self.series2)

        # both models support future covariates only
        mixed_ensemble_future_covs = NaiveEnsembleModel([
            StatsForecastAutoARIMA(),
            RNNModel(12, n_epochs=1, **tfm_kwargs),
        ])
        mixed_ensemble_future_covs.fit(self.series1, future_covariates=self.series2)
        with pytest.raises(ValueError):
            mixed_ensemble_future_covs.fit(self.series1, past_covariates=self.series2)

        # RegressionModels with different covariates
        global_ensemble_both_covs = NaiveEnsembleModel([
            LinearRegressionModel(lags=1, lags_past_covariates=[-1]),
            LinearRegressionModel(lags=1, lags_future_covariates=[1]),
        ])
        # missing future covariates
        with pytest.raises(ValueError):
            global_ensemble_both_covs.fit(self.series1, past_covariates=self.series2)
        # missing past covariates
        with pytest.raises(ValueError):
            global_ensemble_both_covs.fit(self.series1, future_covariates=self.series2)
        global_ensemble_both_covs.fit(
            self.series1, past_covariates=self.series2, future_covariates=self.series2
        )

    def test_fit_multivar_ts_with_local_models(self):
        naive = NaiveEnsembleModel([
            NaiveDrift(),
            NaiveSeasonal(),
            Theta(),
            ExponentialSmoothing(),
        ])
        with pytest.raises(ValueError):
            naive.fit(self.seq1)

    def test_fit_univar_ts_with_covariates_for_local_models(self):
        naive = NaiveEnsembleModel([
            NaiveDrift(),
            NaiveSeasonal(),
            Theta(),
            ExponentialSmoothing(),
        ])
        with pytest.raises(ValueError):
            naive.fit(self.series1, self.series2)

    @pytest.mark.parametrize("model_cls", [NaiveEnsembleModel, RegressionEnsembleModel])
    def test_sample_weight_mixed_models(self, model_cls):
        """Check sample weights for ensemble models with mixed forecasting models.

        NaiveEnsembleModel
            Sample weights will only be passed to global models.
            A weighted linear model that ignores `1000` should learn that y_t = y_(t-1) + 1. When calling predict():
            - linear model should predict y_(t,lin) = 1000 + 1 = 1001
            - naive seasonal should predict y_(t,ns) = y_(t-1) = 1000

            The ensemble takes the average:
            - y_t = 0.5 * y_(t,lin) + 0.5 * y_(t,ns) = 1001 + 1000 = 1000.5

        RegressionEnsembleModel
            Sample weights will be passed to global forecasting models and regression ensemble model.
            A weighted linear model that ignores `1000` should learn that y_t = y_(t-1) + 1. When calling predict():
            - linear model should predict y_(t,lin) = y_(t-1) + 1
            - naive seasonal should predict y_(t,ns) = y_(t-1)

            The training set for regression ensemble covers the forecasts for and labels of last 5 points, where labels
            10000 and 1002 are ignored (0 weight):
            - the linear forecasting model generates forecasts: [1001, 1002, 1003, 1004, 1005]
            - the naive seasonal model generates forecasts: [1000, 1000, 1000, 1000, 1000]

            The ensemble model should then learn a perfect fit based only on the output of the linear model:
            - y_t = 1.0 * y_(t,lin) + 0.0 * y_(t,ns) = 1.0 * (y_(t-1) + 1)
            - for y_(t-1) = 1005 -> y_t = 1006
        """
        if issubclass(model_cls, NaiveEnsembleModel):
            series = TimeSeries.from_values(np.array([0.0, 1.0, 2.0, 3.0, 1000]))
            weights = TimeSeries.from_values(np.array([1.0, 1.0, 1.0, 1.0, 0.0]))
            pred_expected = np.array([[1000.5]])
            kwargs = {}
        else:
            series = TimeSeries.from_values(
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 1000, 10000, 1002, 1003, 1004, 1005])
            )
            weights = TimeSeries.from_values(
                np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            )
            pred_expected = np.array([[1006.0]])
            kwargs = {"regression_train_n_points": 5}

        model = model_cls(
            [LinearRegressionModel(lags=[-1]), NaiveSeasonal(K=1)], **kwargs
        )
        model.fit(series, sample_weight=weights)
        preds_weighted = model.predict(n=1)
        np.testing.assert_array_almost_equal(preds_weighted.values(), pred_expected)

        # make sure that without weights we get different results
        model = model_cls(
            [LinearRegressionModel(lags=[-1]), NaiveSeasonal(K=1)], **kwargs
        )
        model.fit(series)
        preds = model.predict(n=1)
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(
                preds_weighted.values(), preds.values()
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product([NaiveEnsembleModel, RegressionEnsembleModel], [True, False]),
    )
    def test_sample_weight_global(self, config):
        """Check sample weights for ensemble models with global forecasting models.

        NaiveEnsembleModel
            Sample weights will only be passed to global forecasting models.
            A weighted linear model that ignores `1000` should learn that y_t = y_(t-1) + 1. When calling predict():
            - linear model should predict y_(t,lin) = 1000 + 1 = 1001

            The ensemble takes the average:
            - y_t = 0.5 * y_(t,lin) + 0.5 * y_(t,lin) = 1001 + 1001 = 1001

        RegressionEnsembleModel
            Sample weights will be passed to global forecasting models and regression ensemble model.
            A weighted linear model that ignores `1000` should learn that y_t = y_(t-1) + 1. When calling predict():
            - linear model should predict y_(t,lin) = y_(t-1) + 1

            The training set for regression ensemble covers the forecasts for and labels of last 5 points, where labels
            10000 and 1002 are ignored (0 weight):
            - the linear forecasting model generates forecasts: [1001, 1002, 1003, 1004, 1005]

            The ensemble model should then learn a perfect fit based on the output of the linear model:
            - y_t = 1.0 * y_(t,lin) + 0.0 * y_(t,ns) = 1.0 * (y_(t-1) + 1)
        """
        model_cls, single_series = config
        if issubclass(model_cls, NaiveEnsembleModel):
            series = TimeSeries.from_values(np.array([0.0, 1.0, 2.0, 3.0, 1000]))
            weights = TimeSeries.from_values(np.array([1.0, 1.0, 1.0, 1.0, 0.0]))
            pred_expected = np.array([[1001.0]])
            kwargs = {}
        else:
            series = TimeSeries.from_values(
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 1000, 10000, 1002, 1003, 1004, 1005])
            )
            weights = TimeSeries.from_values(
                np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            )
            pred_expected = np.array([[1006.0]])
            kwargs = {"regression_train_n_points": 5}

        if not single_series:
            series = [series] * 2
            weights = [weights] * 2

        model = model_cls(
            [LinearRegressionModel(lags=[-1]), LinearRegressionModel(lags=[-1])],
            **kwargs,
        )
        model.fit(series, sample_weight=weights)
        preds_weighted = model.predict(n=1, series=series)
        if single_series:
            preds_weighted = [preds_weighted]

        for preds in preds_weighted:
            np.testing.assert_array_almost_equal(preds.values(), pred_expected)

        # make sure that without weights we get different results
        model = model_cls(
            [LinearRegressionModel(lags=[-1]), LinearRegressionModel(lags=[-1])],
            **kwargs,
        )
        model.fit(series)
        preds = model.predict(n=1, series=series)
        if single_series:
            preds = [preds]

        for pred_w, pred_nw in zip(preds_weighted, preds):
            with pytest.raises(AssertionError):
                np.testing.assert_array_almost_equal(pred_w.values(), pred_nw.values())

    @pytest.mark.parametrize("model_cls", [NaiveEnsembleModel, RegressionEnsembleModel])
    def test_invalid_sample_weight(self, model_cls):
        kwargs = {
            "forecasting_models": [
                LinearRegressionModel(lags=[-1]),
                NaiveSeasonal(K=1),
            ],
        }
        if issubclass(model_cls, RegressionEnsembleModel):
            kwargs["regression_train_n_points"] = 3

        ts = TimeSeries.from_values(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
        # weights too short
        model = model_cls(**copy.deepcopy(kwargs))
        with pytest.raises(ValueError) as err:
            model.fit(ts, sample_weight=ts[:-1])
        assert (
            str(err.value)
            == "The `sample_weight` series must have at least the same times as the target `series`."
        )

        # same number of series
        model = model_cls(**copy.deepcopy(kwargs))
        with pytest.raises(ValueError) as err:
            model.fit(ts, sample_weight=[ts, ts])
        assert (
            str(err.value)
            == "The provided sequence of target `series` must have the same length as the "
            "provided sequence of `sample_weight`."
        )

        # same number of components
        model = model_cls(**copy.deepcopy(kwargs))
        with pytest.raises(ValueError) as err:
            model.fit(ts, sample_weight=ts.stack(ts))
        assert (
            str(err.value)
            == "The number of components in `sample_weight` must either be `1` or match the "
            "number of target series components `1`."
        )
        # with correct number it works
        model = model_cls(**copy.deepcopy(kwargs))
        model.fit(ts.stack(ts), sample_weight=ts.stack(ts))
        # or with multivar ts and single component weights (globally applied)
        model = model_cls(**copy.deepcopy(kwargs))
        model.fit(ts.stack(ts), sample_weight=ts)

        # invalid string
        model = model_cls(**copy.deepcopy(kwargs))
        with pytest.raises(ValueError) as err:
            model.fit(ts, sample_weight="invalid")
        assert str(err.value).startswith("Invalid `sample_weight` value: `'invalid'`. ")

        # but with valid string it works
        model.fit(ts, sample_weight="linear")

    def test_predict_with_target(self):
        series_long = self.series1
        series_short = series_long[:25]

        # train with a single series
        ensemble_model = self.get_global_ensemble_model()
        ensemble_model.fit(series_short, past_covariates=series_long)
        # predict after end of train series
        preds = ensemble_model.predict(n=5, past_covariates=series_long)
        assert isinstance(preds, TimeSeries)
        # predict a new target series
        preds = ensemble_model.predict(
            n=5, series=series_long, past_covariates=series_long
        )
        assert isinstance(preds, TimeSeries)
        # predict multiple target series
        preds = ensemble_model.predict(
            n=5, series=[series_long] * 2, past_covariates=[series_long] * 2
        )
        assert isinstance(preds, list) and len(preds) == 2
        # predict single target series in list
        preds = ensemble_model.predict(
            n=5, series=[series_long], past_covariates=[series_long]
        )
        assert isinstance(preds, list) and len(preds) == 1

        # train with multiple series
        ensemble_model = self.get_global_ensemble_model()
        ensemble_model.fit([series_short] * 2, past_covariates=[series_long] * 2)
        with pytest.raises(ValueError):
            # predict without passing series should raise an error
            ensemble_model.predict(n=5, past_covariates=series_long)
        # predict a new target series
        preds = ensemble_model.predict(
            n=5, series=series_long, past_covariates=series_long
        )
        assert isinstance(preds, TimeSeries)
        # predict multiple target series
        preds = ensemble_model.predict(
            n=5, series=[series_long] * 2, past_covariates=[series_long] * 2
        )
        assert isinstance(preds, list) and len(preds) == 2
        # predict single target series in list
        preds = ensemble_model.predict(
            n=5, series=[series_long], past_covariates=[series_long]
        )
        assert isinstance(preds, list) and len(preds) == 1

    @staticmethod
    def get_global_ensemble_model(output_chunk_length=5):
        lags = [-1, -2, -5]
        return NaiveEnsembleModel(
            forecasting_models=[
                LinearRegressionModel(
                    lags=lags,
                    lags_past_covariates=lags,
                    output_chunk_length=output_chunk_length,
                ),
                LinearRegressionModel(
                    lags=lags,
                    lags_past_covariates=lags,
                    output_chunk_length=output_chunk_length,
                ),
            ],
        )

    @pytest.mark.parametrize("model_cls", [NaiveEnsembleModel, RegressionEnsembleModel])
    def test_save_load_ensemble_models(self, tmpdir_fn, model_cls):
        # check if save and load methods work and
        # if loaded ensemble model creates same forecasts as original ensemble models
        full_model_path_str = os.getcwd()
        kwargs = {}
        expected_suffixes = [".pkl", ".pkl.RNNModel_2.pt", ".pkl.RNNModel_2.pt.ckpt"]

        if issubclass(model_cls, RegressionEnsembleModel):
            kwargs["regression_train_n_points"] = 5

        if TORCH_AVAILABLE:
            model = model_cls(
                [
                    LinearRegressionModel(lags=[-1]),
                    NaiveSeasonal(K=1),
                    RNNModel(10, n_epochs=1, **tfm_kwargs),
                ],
                **kwargs,
            )
        else:
            model = model_cls(
                [LinearRegressionModel(lags=[-1]), NaiveSeasonal(K=1)], **kwargs
            )

        model.fit(self.series1 + self.series2)
        model_prediction = model.predict(5)

        # test save
        model.save()
        model.save(os.path.join(full_model_path_str, f"{model_cls.__name__}.pkl"))

        assert os.path.exists(full_model_path_str)
        files = os.listdir(full_model_path_str)
        if TORCH_AVAILABLE:
            assert len(files) == 6
            for f in files:
                assert f.startswith(model_cls.__name__)
            suffix_counts = {
                suffix: sum(
                    1 for p in os.listdir(full_model_path_str) if p.endswith(suffix)
                )
                for suffix in expected_suffixes
            }
            assert all(count == 2 for count in suffix_counts.values())
        else:
            assert len(files) == 2
            for f in files:
                assert f.startswith(model_cls.__name__) and f.endswith(".pkl")

        # test load
        pkl_files = []
        for filename in os.listdir(full_model_path_str):
            if filename.endswith(".pkl"):
                pkl_files.append(os.path.join(full_model_path_str, filename))
        for p in pkl_files:
            loaded_model = model_cls.load(p)
            assert model_prediction == loaded_model.predict(5)

            # test pl_trainer_kwargs (only for torch models)
            loaded_model = model_cls.load(p, pl_trainer_kwargs={"accelerator": "cuda"})
            for i, m in enumerate(loaded_model.forecasting_models):
                if TORCH_AVAILABLE and issubclass(type(m), TorchForecastingModel):
                    assert m.trainer_params["accelerator"] == "cuda"

        # test clean save
        path = os.path.join(full_model_path_str, f"clean_{model_cls.__name__}.pkl")
        model.save(path, clean=True)
        clean_model = model_cls.load(path, pl_trainer_kwargs={"accelerator": "cpu"})
        for i, m in enumerate(clean_model.forecasting_models):
            if not issubclass(type(m), LocalForecastingModel):
                assert m.training_series is None
                assert m.past_covariate_series is None
                assert m.future_covariate_series is None
        assert model.predict(5) == clean_model.predict(5, self.series1 + self.series2)
