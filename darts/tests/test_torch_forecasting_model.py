import os
import tempfile
import shutil
import pandas as pd

from ..timeseries import TimeSeries
from .base_test_class import DartsBaseTestClass
from ..logging import get_logger
from unittest.mock import patch

logger = get_logger(__name__)

try:
    from darts.models.forecasting.rnn_model import RNNModel
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

        @patch('darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.save_model')
        def test_suppress_automatic_save(self, patch_save_model):
            model_name = 'test_model'
            model1 = RNNModel('RNN', 10, 10, model_name=model_name, work_dir=self.temp_work_dir, save_checkpoints=False)
            model2 = RNNModel('RNN', 10, 10, model_name=model_name, work_dir=self.temp_work_dir, force_reset=True,
                             save_checkpoints=False)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=1)
            model2.fit(series, epochs=1)

            model1.predict(n=1)
            model2.predict(n=2)

            patch_save_model.assert_not_called()

            model1.save_model(path=os.path.join(self.temp_work_dir, model_name))
            patch_save_model.assert_called()

        def test_aa_manual_save(self):
            model_name = 'test_model'
            model_manual_save = RNNModel('RNN', 10, 10, model_name=model_name + '_manual', work_dir=self.temp_work_dir,
                                         save_checkpoints=False)
            model_auto_save = RNNModel('RNN', 10, 10, model_name=model_name + '_automatic', work_dir=self.temp_work_dir,
                                       save_checkpoints=True)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)

            model_manual_save.fit(series, epochs=1)
            model_auto_save.fit(series, epochs=1)

            checkpoints_dir = os.path.join(self.temp_work_dir, 'checkpoints')

            self.assertFalse(os.path.exists(os.path.join(checkpoints_dir, 'test_model_manual')))
            self.assertTrue(os.path.exists(os.path.join(checkpoints_dir, 'test_model_automatic')))

            out_path_manual = os.path.join(checkpoints_dir, 'test_model_manual')

            os.mkdir(out_path_manual)
            model_manual_save.save_model(os.path.join(out_path_manual, 'checkpoint_0.pth.tar'))

            self.assertTrue(os.path.exists())




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

        @patch('darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.reset_model')
        def test_create_instance_existing_model_with_name_force(self, patch_reset_model):
            model_name = 'test_model'
            model1 = RNNModel('RNN', 10, 10, work_dir=self.temp_work_dir, model_name=model_name)
            # no exception is raised
            # since no fit, there is no data stored for the model, hence `force_reset` does noting

            model2 = RNNModel('RNN', 10, 10, work_dir=self.temp_work_dir, model_name=model_name, force_reset=True)
            patch_reset_model.assert_not_called()

        @patch('darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.reset_model')
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
