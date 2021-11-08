import numpy as np
import pandas as pd
import unittest

from .base_test_class import DartsBaseTestClass
from .. import TimeSeries
from ..utils import timeseries_generation as tg
from ..models import NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing
from ..models import NaiveEnsembleModel
from ..logging import get_logger
logger = get_logger(__name__)

try:
    from ..models import RNNModel, TCNModel, NBEATSModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not installed - Some ensemble models tests will be skipped.")
    TORCH_AVAILABLE = False


def _make_ts(start_value=0, n=100):
    times = pd.date_range(start='1/1/2013', periods=n, freq='D')
    pd_series = pd.Series(range(start_value, start_value + n), index=times)
    return TimeSeries.from_series(pd_series)


class EnsembleModelsTestCase(DartsBaseTestClass):
    series1 = tg.sine_timeseries(value_frequency=(1 / 5), value_y_offset=10, length=50)
    series2 = tg.linear_timeseries(length=50)

    seq1 = [_make_ts(0), _make_ts(10), _make_ts(20)]
    cov1 = [_make_ts(5), _make_ts(15), _make_ts(25)]

    def test_input_models_local_models(self):
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([])
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing])
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([NaiveDrift(), NaiveSeasonal, Theta(), ExponentialSmoothing()])
        NaiveEnsembleModel([NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()])

    def test_call_predict_local_models(self):
        naive_ensemble = NaiveEnsembleModel([NaiveSeasonal(), Theta()])
        with self.assertRaises(Exception):
            naive_ensemble.predict(5)
        naive_ensemble.fit(self.series1)
        naive_ensemble.predict(5)

    def test_predict_ensemble_local_models(self):
        naive = NaiveSeasonal(K=5)
        theta = Theta()
        naive_ensemble = NaiveEnsembleModel([naive, theta])
        naive_ensemble.fit(self.series1 + self.series2)
        forecast_naive_ensemble = naive_ensemble.predict(5)
        naive.fit(self.series1 + self.series2)
        theta.fit(self.series1 + self.series2)
        forecast_mean = 0.5 * naive.predict(5) + 0.5 * theta.predict(5)

        self.assertTrue(np.array_equal(forecast_naive_ensemble.values(), forecast_mean.values()))

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_input_models_global_models(self):
        NaiveEnsembleModel([RNNModel(), TCNModel(10, 2), NBEATSModel(10, 2)])

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_call_predict_global_models_univariate_input_no_covariates(self):
        naive_ensemble = NaiveEnsembleModel([RNNModel(n_epochs=1),
                                             TCNModel(10, 2, n_epochs=1),
                                             NBEATSModel(10, 2, n_epochs=1)])
        with self.assertRaises(Exception):
            naive_ensemble.predict(5)

        naive_ensemble.fit(self.series1)
        naive_ensemble.predict(5)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_call_predict_global_models_multivariate_input_no_covariates(self):
        naive_ensemble = NaiveEnsembleModel([RNNModel(n_epochs=1),
                                             TCNModel(10, 2, n_epochs=1),
                                             NBEATSModel(10, 2, n_epochs=1)])
        naive_ensemble.fit(self.seq1)
        naive_ensemble.predict(n=5, series=self.seq1)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_call_predict_global_models_multivariate_input_with_covariates(self):
        naive_ensemble = NaiveEnsembleModel([RNNModel(n_epochs=1),
                                             TCNModel(10, 2, n_epochs=1),
                                             NBEATSModel(10, 2, n_epochs=1)])
        naive_ensemble.fit(self.seq1, self.cov1)
        predict_series = [s[:12] for s in self.seq1]
        predict_covariates = [c[:14] for c in self.cov1]
        naive_ensemble.predict(n=2, series=predict_series, past_covariates=predict_covariates)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_input_models_mixed(self):
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([NaiveDrift(), Theta(), RNNModel()])

    def test_fit_multivar_ts_with_local_models(self):
        naive = NaiveEnsembleModel([NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()])
        with self.assertRaises(ValueError):
            naive.fit(self.seq1)

    def test_fit_univar_ts_with_covariates_for_local_models(self):
        naive = NaiveEnsembleModel([NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()])
        with self.assertRaises(ValueError):
            naive.fit(self.series1, self.series2)


if __name__ == '__main__':
    import unittest

    unittest.main()
