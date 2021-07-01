from .base_test_class import DartsBaseTestClass
from ..logging import get_logger
from unittest.mock import patch
from unittest import skip
import tempfile
import shutil
import pandas as pd
from ..timeseries import TimeSeries

logger = get_logger(__name__)

try:
    from ..models.rnn_model import RNNModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not available. RNN tests will be skipped.')
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TestTorchForecastingModel(DartsBaseTestClass):

        def setUp(self):
            self.temp_work_dir = tempfile.mkdtemp(prefix='darts')

        def tearDown(self):
            shutil.rmtree(self.temp_work_dir)

        def test_create_instance_new_model_no_name_set(self):
            model = RNNModel('RNN', 10, 10, work_dir=self.temp_work_dir)
            # no exception is raised
            model2 = RNNModel('RNN', 10, 10, work_dir=self.temp_work_dir)
            # no exception is raised

        def test_create_instance_existing_model_with_name_no_fit(self):
            model_name = 'test_model'
            model1 = RNNModel('RNN', 10, 10, work_dir=self.temp_work_dir, model_name=model_name)
            # no exception is raised

            model2 = RNNModel('RNN', 10, 10, work_dir=self.temp_work_dir, model_name=model_name)
            # no exception is raised

        @patch('darts.models.torch_forecasting_model.TorchForecastingModel.reset_model')
        def test_create_instance_existing_model_with_name_force(self, patch_reset_model):
            model_name = 'test_model'
            model1 = RNNModel('RNN', 10, 10, work_dir=self.temp_work_dir, model_name=model_name)
            # no exception is raised
            # since no fit, there is no data stored for the model, hence `force_reset` does noting

            model2 = RNNModel('RNN', 10, 10, work_dir=self.temp_work_dir, model_name=model_name, force_reset=True)
            patch_reset_model.assert_not_called()

        @patch('darts.models.torch_forecasting_model.TorchForecastingModel.reset_model')
        def test_create_instance_existing_model_with_name_force_fit_with_reset(self, patch_reset_model):
            model_name = 'test_model'
            model1 = RNNModel('RNN', 10, 10, work_dir=self.temp_work_dir, model_name=model_name)
            # no exception is raised

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=1)

            model2 = RNNModel('RNN', 10, 10, work_dir=self.temp_work_dir, model_name=model_name, force_reset=True)
            patch_reset_model.assert_called_once()

        # n_epochs=20, fit|epochs=None, total_epochs=0 - train for 20 epochs
        def test_train_from_0_n_epochs_20_no_fit_epochs(self):
            model1 = RNNModel('RNN', 10, 10, n_epochs=20, work_dir=self.temp_work_dir)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series)

            self.assertEqual(model1.total_epochs, 20)

        # n_epochs = 20, fit|epochs=None, total_epochs=20 - train for another 20 epochs
        def test_train_from_20_n_epochs_40_no_fit_epochs(self):
            model1 = RNNModel('RNN', 10, 10, n_epochs=20, work_dir=self.temp_work_dir)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series)
            self.assertEqual(model1.total_epochs, 20)

            model1.fit(series)
            self.assertEqual(model1.total_epochs, 40)

        # n_epochs = 20, fit|epochs=None, total_epochs=10 - train for another 20 epochs
        def test_train_from_10_n_epochs_20_no_fit_epochs(self):
            model1 = RNNModel('RNN', 10, 10, n_epochs=20, work_dir=self.temp_work_dir)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            # simulate the case that user interrupted training with Ctrl-C after 10 epochs
            model1.fit(series, epochs=10)
            self.assertEqual(model1.total_epochs, 10)

            model1.fit(series)
            self.assertEqual(model1.total_epochs, 30)

        # n_epochs = 20, fit|epochs=15, total_epochs=0 - train for 15 epochs
        def test_train_from_0_n_epochs_20_fit_15_epochs(self):
            model1 = RNNModel('RNN', 10, 10, n_epochs=20, work_dir=self.temp_work_dir)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=15)
            self.assertEqual(model1.total_epochs, 15)

        # n_epochs = 20, fit|epochs=15, total_epochs=10 - train for 15 epochs
        def test_train_from_10_n_epochs_20_fit_15_epochs(self):
            model1 = RNNModel('RNN', 10, 10, n_epochs=20, work_dir=self.temp_work_dir)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            # simulate the case that user interrupted training with Ctrl-C after 10 epochs
            model1.fit(series, epochs=10)
            self.assertEqual(model1.total_epochs, 10)

            model1.fit(series, epochs=15)
            self.assertEqual(model1.total_epochs, 25)
