import shutil
import tempfile

import numpy as np

from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.timeseries_generation import linear_timeseries

logger = get_logger(__name__)

try:
    import pytorch_lightning as pl

    from darts.models.forecasting.rnn_model import RNNModel

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. RNN tests will be skipped.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TestTorchForecastingModel(DartsBaseTestClass):
        trainer_params = {
            "max_epochs": 1,
            "logger": False,
            "enable_checkpointing": False,
        }

        series = linear_timeseries(length=100).astype(np.float32)

        def setUp(self):
            self.temp_work_dir = tempfile.mkdtemp(prefix="darts")

        def tearDown(self):
            shutil.rmtree(self.temp_work_dir)

        def test_prediction_loaded_custom_trainer(self):
            """validate manual save with automatic save files by comparing output between the two"""
            auto_name = "test_save_automatic"
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                model_name=auto_name,
                work_dir=self.temp_work_dir,
                save_checkpoints=True,
                random_state=42,
            )

            # fit model with custom trainer
            trainer = pl.Trainer(
                max_epochs=1,
                enable_checkpointing=True,
                logger=False,
                callbacks=model.trainer_params["callbacks"],
                precision=32,
            )
            model.fit(self.series, trainer=trainer)

            # load automatically saved model with manual load_model() and load_from_checkpoint()
            model_loaded = RNNModel.load_from_checkpoint(
                model_name=auto_name, work_dir=self.temp_work_dir, best=False
            )

            # compare prediction of loaded model with original model
            self.assertEqual(model.predict(n=4), model_loaded.predict(n=4))

        def test_prediction_custom_trainer(self):
            model = RNNModel(12, "RNN", 10, 10, random_state=42)
            model2 = RNNModel(12, "RNN", 10, 10, random_state=42)

            # fit model with custom trainer
            trainer = pl.Trainer(**self.trainer_params, precision=32)
            model.fit(self.series, trainer=trainer)

            # fit model with built-in trainer
            model2.fit(self.series, epochs=1)

            # both should produce identical prediction
            self.assertEqual(model.predict(n=4), model2.predict(n=4))

        def test_custom_trainer_setup(self):
            model = RNNModel(12, "RNN", 10, 10, random_state=42)

            # trainer with wrong precision should raise ValueError
            trainer = pl.Trainer(**self.trainer_params, precision=64)
            with self.assertRaises(ValueError):
                model.fit(self.series, trainer=trainer)

            # no error with correct precision
            trainer = pl.Trainer(**self.trainer_params, precision=32)
            model.fit(self.series, trainer=trainer)

            # check if number of epochs trained is same as trainer.max_epochs
            self.assertEqual(trainer.max_epochs, model.epochs_trained)

        def test_builtin_extended_trainer(self):
            invalid_trainer_kwarg = {"precisionn": 32}

            # error will be raised at training time
            with self.assertRaises(TypeError):
                model = RNNModel(
                    12,
                    "RNN",
                    10,
                    10,
                    random_state=42,
                    pl_trainer_kwargs=invalid_trainer_kwarg,
                )
                model.fit(self.series, epochs=1)

            valid_trainer_kwargs = {
                "precision": 32,
            }

            # valid parameters shouldn't raise error
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                random_state=42,
                pl_trainer_kwargs=valid_trainer_kwargs,
            )
            model.fit(self.series, epochs=1)

        def test_custom_callback(self):
            class CounterCallback(pl.callbacks.Callback):
                # counts the number of trained epochs starting from count_default
                def __init__(self, count_default):
                    self.counter = count_default

                def on_train_epoch_end(self, *args, **kwargs):
                    self.counter += 1

            my_counter_0 = CounterCallback(count_default=0)
            my_counter_2 = CounterCallback(count_default=2)

            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                random_state=42,
                pl_trainer_kwargs={"callbacks": [my_counter_0, my_counter_2]},
            )

            # check if callbacks were added
            self.assertEqual(len(model.trainer_params["callbacks"]), 2)
            model.fit(self.series, epochs=2)

            self.assertEqual(my_counter_0.counter, model.epochs_trained)
            self.assertEqual(my_counter_2.counter, model.epochs_trained + 2)

            # check that callbacks don't overwrite Darts' built-in checkpointer
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                random_state=42,
                work_dir=self.temp_work_dir,
                save_checkpoints=True,
                pl_trainer_kwargs={
                    "callbacks": [CounterCallback(0), CounterCallback(2)]
                },
            )
            # we expect 3 callbacks
            self.assertEqual(len(model.trainer_params["callbacks"]), 3)

            # first one is our Checkpointer
            self.assertTrue(
                isinstance(
                    model.trainer_params["callbacks"][0], pl.callbacks.ModelCheckpoint
                )
            )

            # second and third are CounterCallbacks
            for i in range(1, 3):
                self.assertTrue(
                    isinstance(model.trainer_params["callbacks"][i], CounterCallback)
                )

        def test_early_stopping(self):
            my_stopper = pl.callbacks.early_stopping.EarlyStopping(
                monitor="val_loss",
                stopping_threshold=1e9,
            )
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                nr_epochs_val_period=1,
                random_state=42,
                pl_trainer_kwargs={"callbacks": [my_stopper]},
            )

            # training should stop immediately with high stopping_threshold
            model.fit(self.series, val_series=self.series, epochs=100, verbose=True)
            self.assertEqual(model.epochs_trained, 1)

            # check that early stopping only takes valid monitor variables
            my_stopper = pl.callbacks.early_stopping.EarlyStopping(
                monitor="invalid_variable",
                stopping_threshold=1e9,
            )
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                nr_epochs_val_period=1,
                random_state=42,
                pl_trainer_kwargs={"callbacks": [my_stopper]},
            )

            with self.assertRaises(RuntimeError):
                model.fit(self.series, val_series=self.series, epochs=100, verbose=True)
