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
            model = RNNModel(10, 10, work_dir=self.temp_work_dir)
            # no exception is raised
            model2 = RNNModel(10, 10, work_dir=self.temp_work_dir)
            # no exception is raised

        def test_create_instance_existing_model_with_name(self):
            model_name = 'test_model'
            model1 = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name=model_name)
            #no exception is raised

            model2 = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name=model_name)
            self.assertRaisesRegex(AttributeError,
                                   "You already have model data for the \'{}\' name.*".format(model_name))

        def test_create_instance_existing_model_with_name_force(self):
            model_name = 'test_model'
            model1 = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name=model_name)
            # no exception is raised
            # since no fit, there is no data stored for the model

            model2 = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name=model_name, force=True)
            # no exception and no warning is raised

        def test_create_instance_existing_model_with_name_force_fit_no_reset(self):
            model_name = 'test_model'
            model1 = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name=model_name)
            #no exception is raised

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=1)

            model2 = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name=model_name, force=True)
            self.assertWarnsRegex(UserWarning, "You already have model data for the '{}' name and you "
                                  "initialized it with".format(model_name))

            model2.fit(series, epochs=1)
            self.assertRaisesRegex(ValueError, "You forced initialization of the model but the data already exist.*")

        def test_create_instance_existing_model_with_name_force_fit_with_reset(self):
            model_name = 'test_model'
            model1 = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name=model_name)
            #no exception is raised

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=1)

            model2 = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name=model_name, force=True)
            model2.reset_model()
            model2.fit(series, epochs=1)

        # n_epochs=20, fit|epochs=None, total_epochs=0 - train for 20 epochs
        def test_train_from_0_to_20_no_fit_epochs(self):
            model1 = RNNModel(10, 10, n_epochs=20, work_dir=self.temp_work_dir)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series)

            self.assertEqual(model1.total_epochs, 20)

        # n_epochs = 20, fit|epochs=None, total_epochs=20 - train for another 20 epochs
        def test_train_from_20_to_40_no_fit_epochs(self):
            model1 = RNNModel(10, 10, n_epochs=20, work_dir=self.temp_work_dir)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series)
            self.assertEqual(model1.total_epochs, 20)

            model1.fit(series)
            self.assertEqual(model1.total_epochs, 40)

        # n_epochs = 20, fit|epochs=None, total_epochs=10 - train for another 20 epochs
        def test_train_from_10_to_20_no_fit_epochs(self):
            model1 = RNNModel(10, 10, n_epochs=20, work_dir=self.temp_work_dir)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=10)  # cheating here a little bit
            self.assertEqual(model1.total_epochs, 10)

            model1.fit(series)
            self.assertEqual(model1.total_epochs, 30)

        # n_epochs = 20, fit|epochs=15, total_epochs=0 - train for 15 epochs
        def test_train_from_0_to_20_fit_15_epochs(self):
            model1 = RNNModel(10, 10, n_epochs=20, work_dir=self.temp_work_dir)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=15)
            self.assertEqual(model1.total_epochs, 15)

        # n_epochs = 20, fit|epochs=15, total_epochs=10 - train for 15 epochs
        def test_train_from_0_to_20_fit_15_epochs(self):
            model1 = RNNModel(10, 10, n_epochs=20, work_dir=self.temp_work_dir)

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=10)  # cheating here a little bit
            self.assertEqual(model1.total_epochs, 10)

            model1.fit(series, epochs=15)
            self.assertEqual(model1.total_epochs, 25)
