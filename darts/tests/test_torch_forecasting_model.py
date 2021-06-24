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

        @skip
        @patch('darts.models.torch_forecasting_model.TorchForecastingModel.load')
        def test_create_instance_new_model_no_name_set(self, load_patch):
            model = RNNModel(10, 10, work_dir=self.temp_work_dir)
            load_patch.assert_not_called()

        @skip
        @patch('darts.models.torch_forecasting_model.TorchForecastingModel.load')
        def test_create_instance_new_model_with_name(self, load_patch):
            model = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name='test_model')
            load_patch.assert_not_called()

        @skip
        @patch('darts.models.torch_forecasting_model.TorchForecastingModel.load')
        def test_create_instance_existing_model_with_name(self, load_patch):
            model1 = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name='test_model')
            load_patch.assert_not_called()

            times = pd.date_range('20130101', '20130410')
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=1)

            model2 = RNNModel(10, 10, work_dir=self.temp_work_dir, model_name='test_model')
            load_patch.assert_called()


        def test_reset_model(self):
            return False

        def test_load(self):
            # immutable params
            # mutable params
            # new model
            return False

        def test_load_checkpoint_folder_does_not_exist(self):
            return False
