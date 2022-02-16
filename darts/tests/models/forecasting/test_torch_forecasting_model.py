import os
import shutil
import tempfile
from unittest.mock import patch

import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass

logger = get_logger(__name__)

try:
    from darts.models.forecasting.rnn_model import RNNModel

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. RNN tests will be skipped.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TestTorchForecastingModel(DartsBaseTestClass):
        def setUp(self):
            self.temp_work_dir = tempfile.mkdtemp(prefix="darts")

        def tearDown(self):
            shutil.rmtree(self.temp_work_dir)

        def test_save_model_parameters(self):
            # check if re-created model has same params as original
            model = RNNModel(12, "RNN", 10, 10)
            self.assertTrue(model._model_params, model.untrained_model()._model_params)

        @patch(
            "darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.save_model"
        )
        def test_suppress_automatic_save(self, patch_save_model):
            model_name = "test_model"
            model1 = RNNModel(
                12,
                "RNN",
                10,
                10,
                model_name=model_name,
                work_dir=self.temp_work_dir,
                save_checkpoints=False,
            )
            model2 = RNNModel(
                12,
                "RNN",
                10,
                10,
                model_name=model_name,
                work_dir=self.temp_work_dir,
                force_reset=True,
                save_checkpoints=False,
            )

            times = pd.date_range("20130101", "20130410")
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=1)
            model2.fit(series, epochs=1)

            model1.predict(n=1)
            model2.predict(n=2)

            patch_save_model.assert_not_called()

            model1.save_model(path=os.path.join(self.temp_work_dir, model_name))
            patch_save_model.assert_called()

        def test_manual_save_and_load(self):
            """validate manual save with automatic save files by comparing output between the two"""

            manual_name = "test_save_manual"
            auto_name = "test_save_automatic"
            model_manual_save = RNNModel(
                12,
                "RNN",
                10,
                10,
                model_name=manual_name,
                work_dir=self.temp_work_dir,
                save_checkpoints=False,
                random_state=42,
            )
            model_auto_save = RNNModel(
                12,
                "RNN",
                10,
                10,
                model_name=auto_name,
                work_dir=self.temp_work_dir,
                save_checkpoints=True,
                random_state=42,
            )

            times = pd.date_range("20130101", "20130410")
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)

            model_manual_save.fit(series, epochs=1)
            model_auto_save.fit(series, epochs=1)

            model_dir = os.path.join(self.temp_work_dir)

            # check that file was not created with manual save
            self.assertFalse(
                os.path.exists(os.path.join(model_dir, manual_name, "checkpoints"))
            )
            # check that file was created with automatic save
            self.assertTrue(
                os.path.exists(os.path.join(model_dir, auto_name, "checkpoints"))
            )

            # create manually saved model checkpoints folder
            checkpoint_path_manual = os.path.join(model_dir, manual_name)
            os.mkdir(checkpoint_path_manual)

            # save manually saved model
            checkpoint_file_name = "checkpoint_0.pth.tar"
            model_path_manual = os.path.join(
                checkpoint_path_manual, checkpoint_file_name
            )
            model_manual_save.save_model(model_path_manual)
            self.assertTrue(os.path.exists(model_path_manual))

            # load manual save model and compare with automatic model results
            model_manual_save = RNNModel.load_model(model_path_manual)
            self.assertEqual(
                model_manual_save.predict(n=4), model_auto_save.predict(n=4)
            )

            # load automatically saved model with manual load_model() and load_from_checkpoint()
            model_auto_save1 = RNNModel.load_from_checkpoint(
                model_name=auto_name, work_dir=self.temp_work_dir, best=False
            )

            # compare loaded checkpoint with manual save
            self.assertEqual(
                model_manual_save.predict(n=4), model_auto_save1.predict(n=4)
            )

        def test_create_instance_new_model_no_name_set(self):
            RNNModel(12, "RNN", 10, 10, work_dir=self.temp_work_dir)
            # no exception is raised
            RNNModel(12, "RNN", 10, 10, work_dir=self.temp_work_dir)
            # no exception is raised

        def test_create_instance_existing_model_with_name_no_fit(self):
            model_name = "test_model"
            RNNModel(
                12, "RNN", 10, 10, work_dir=self.temp_work_dir, model_name=model_name
            )
            # no exception is raised

            RNNModel(
                12, "RNN", 10, 10, work_dir=self.temp_work_dir, model_name=model_name
            )
            # no exception is raised

        @patch(
            "darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.reset_model"
        )
        def test_create_instance_existing_model_with_name_force(
            self, patch_reset_model
        ):
            model_name = "test_model"
            RNNModel(
                12, "RNN", 10, 10, work_dir=self.temp_work_dir, model_name=model_name
            )
            # no exception is raised
            # since no fit, there is no data stored for the model, hence `force_reset` does noting

            RNNModel(
                12,
                "RNN",
                10,
                10,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                force_reset=True,
            )
            patch_reset_model.assert_not_called()

        @patch(
            "darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.reset_model"
        )
        def test_create_instance_existing_model_with_name_force_fit_with_reset(
            self, patch_reset_model
        ):
            model_name = "test_model"
            model1 = RNNModel(
                12,
                "RNN",
                10,
                10,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                save_checkpoints=True,
            )
            # no exception is raised

            times = pd.date_range("20130101", "20130410")
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series, epochs=1)

            RNNModel(
                12,
                "RNN",
                10,
                10,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                save_checkpoints=True,
                force_reset=True,
            )
            patch_reset_model.assert_called_once()

        # TODO for PTL: currently we (have to (?)) create a mew PTL trainer object every time fit() is called which
        #  resets some of the model's attributes such as epoch and step counts. We have check whether there is another
        #  way of doing this.

        # n_epochs=20, fit|epochs=None, epochs_trained=0 - train for 20 epochs
        def test_train_from_0_n_epochs_20_no_fit_epochs(self):
            model1 = RNNModel(
                12, "RNN", 10, 10, n_epochs=20, work_dir=self.temp_work_dir
            )

            times = pd.date_range("20130101", "20130410")
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series)

            self.assertEqual(20, model1.epochs_trained)

        # n_epochs = 20, fit|epochs=None, epochs_trained=20 - train for another 20 epochs
        def test_train_from_20_n_epochs_40_no_fit_epochs(self):
            model1 = RNNModel(
                12, "RNN", 10, 10, n_epochs=20, work_dir=self.temp_work_dir
            )

            times = pd.date_range("20130101", "20130410")
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            model1.fit(series)
            self.assertEqual(20, model1.epochs_trained)

            model1.fit(series)
            self.assertEqual(20, model1.epochs_trained)

        # n_epochs = 20, fit|epochs=None, epochs_trained=10 - train for another 20 epochs
        def test_train_from_10_n_epochs_20_no_fit_epochs(self):
            model1 = RNNModel(
                12, "RNN", 10, 10, n_epochs=20, work_dir=self.temp_work_dir
            )

            times = pd.date_range("20130101", "20130410")
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            # simulate the case that user interrupted training with Ctrl-C after 10 epochs
            model1.fit(series, epochs=10)
            self.assertEqual(10, model1.epochs_trained)

            model1.fit(series)
            self.assertEqual(20, model1.epochs_trained)

        # n_epochs = 20, fit|epochs=15, epochs_trained=10 - train for 15 epochs
        def test_train_from_10_n_epochs_20_fit_15_epochs(self):
            model1 = RNNModel(
                12, "RNN", 10, 10, n_epochs=20, work_dir=self.temp_work_dir
            )

            times = pd.date_range("20130101", "20130410")
            pd_series = pd.Series(range(100), index=times)
            series = TimeSeries.from_series(pd_series)
            # simulate the case that user interrupted training with Ctrl-C after 10 epochs
            model1.fit(series, epochs=10)
            self.assertEqual(10, model1.epochs_trained)

            model1.fit(series, epochs=15)
            self.assertEqual(15, model1.epochs_trained)
