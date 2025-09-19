import numpy as np
import pytest

from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils.timeseries_generation import linear_timeseries

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import pytorch_lightning as pl

from darts.models.forecasting.rnn_model import RNNModel


class TestPTLTrainer:
    trainer_params = {
        "max_epochs": 1,
        "logger": False,
        "enable_checkpointing": False,
    }
    series = linear_timeseries(length=100).astype(np.float32)
    precisions = {
        16: "bf16-true",
        32: "32-true",
        64: "64-true",
        "mixed": "bf16-mixed",
    }

    def test_prediction_loaded_custom_trainer(self, tmpdir_module):
        """validate manual save with automatic save files by comparing output between the two"""
        auto_name = "test_save_automatic"
        model = RNNModel(
            12,
            "RNN",
            10,
            10,
            model_name=auto_name,
            work_dir=tmpdir_module,
            save_checkpoints=True,
            random_state=42,
            **tfm_kwargs,
        )

        # fit model with custom trainer
        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=True,
            logger=False,
            callbacks=model.trainer_params["callbacks"],
            **tfm_kwargs["pl_trainer_kwargs"],
        )
        model.fit(self.series, trainer=trainer)

        # load automatically saved model with manual load_model() and load_from_checkpoint()
        model_loaded = RNNModel.load_from_checkpoint(
            model_name=auto_name,
            work_dir=tmpdir_module,
            best=False,
            map_location="cpu",
        )

        # compare prediction of loaded model with original model
        assert model.predict(n=4) == model_loaded.predict(n=4)

    def test_prediction_custom_trainer(self):
        model = RNNModel(12, "RNN", 10, 10, random_state=42, **tfm_kwargs)
        model2 = RNNModel(12, "RNN", 10, 10, random_state=42, **tfm_kwargs)

        # fit model with custom trainer
        trainer = pl.Trainer(
            **self.trainer_params,
            precision=self.precisions[32],
            **tfm_kwargs["pl_trainer_kwargs"],
        )
        model.fit(self.series, trainer=trainer)

        # fit model with built-in trainer
        model2.fit(self.series, epochs=1)

        # both should produce identical prediction
        assert model.predict(n=4) == model2.predict(n=4)

    @pytest.mark.parametrize(
        "precision, dtype",
        [
            (16, np.float16),
            (32, np.float32),
            (64, np.float64),
            ("mixed", np.float32),
        ],
    )
    def test_custom_trainer_setup(self, precision, dtype):
        model = RNNModel(12, "RNN", 10, 10, random_state=42, **tfm_kwargs)

        # no error with correct precision
        trainer = pl.Trainer(
            **self.trainer_params,
            precision=self.precisions[precision],
            **tfm_kwargs["pl_trainer_kwargs"],
        )
        model.fit(self.series.astype(dtype), trainer=trainer)

        # check if number of epochs trained is same as trainer.max_epochs
        assert trainer.max_epochs == model.epochs_trained

        preds = model.predict(n=3)
        assert model.trainer.precision == self.precisions[precision]
        if dtype != np.float16:
            # predictions should have same dtype as input except for float16
            assert preds.dtype == dtype
        else:
            # predictions are float32 when input is float16
            assert preds.dtype == np.float32
        # predictions should not contain NaNs or infs
        assert np.all(np.isfinite(preds.values()))

    def test_higher_precision_custom_trainer(self):
        model = RNNModel(12, "RNN", 10, 10, random_state=42, **tfm_kwargs)

        # trainer with higher precision than data should not raise ValueError
        # since Trainer would automatically cast data to correct precision
        trainer = pl.Trainer(
            **self.trainer_params,
            precision=self.precisions[64],
            **tfm_kwargs["pl_trainer_kwargs"],
        )
        model.fit(self.series, trainer=trainer)

        # check if number of epochs trained is same as trainer.max_epochs
        assert trainer.max_epochs == model.epochs_trained

    def test_lower_precision_custom_trainer(self):
        model = RNNModel(12, "RNN", 10, 10, random_state=42, **tfm_kwargs)

        # trainer with lower precision than data should raise ValueError
        # because precision here is defined by the user and handled by Lightning
        # the error should come from trainer instead of darts
        trainer = pl.Trainer(
            **self.trainer_params,
            precision=self.precisions[32],
            **tfm_kwargs["pl_trainer_kwargs"],
        )
        with pytest.raises(ValueError):
            model.fit(self.series.astype(np.float64), trainer=trainer)

    def test_mixed_precision_custom_trainer(self):
        model = RNNModel(12, "RNN", 10, 10, random_state=42, **tfm_kwargs)

        # trainer with mixed precision should not raise ValueError
        trainer = pl.Trainer(
            **self.trainer_params,
            precision=self.precisions["mixed"],
            **tfm_kwargs["pl_trainer_kwargs"],
        )
        model.fit(self.series, trainer=trainer)

        # check if number of epochs trained is same as trainer.max_epochs
        assert trainer.max_epochs == model.epochs_trained

    def test_builtin_extended_trainer(self):
        # wrong precision parameter name
        with pytest.raises(TypeError):
            invalid_trainer_kwarg = {
                "precisionn": self.precisions[32],
                **tfm_kwargs["pl_trainer_kwargs"],
            }
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                random_state=42,
                pl_trainer_kwargs=invalid_trainer_kwarg,
            )
            model.fit(self.series, epochs=1)

        # precision value is lower than series precision
        with pytest.raises(ValueError):
            invalid_trainer_kwarg = {
                "precision": self.precisions[32],
                **tfm_kwargs["pl_trainer_kwargs"],
            }
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                random_state=42,
                pl_trainer_kwargs=invalid_trainer_kwarg,
            )
            model.fit(self.series.astype(np.float64), epochs=1)

    @pytest.mark.parametrize(
        "precision, dtype",
        [
            (16, np.float16),
            (32, np.float32),
            (64, np.float64),
            ("mixed", np.float32),
        ],
    )
    def test_precision_builtin_extended_trainer(self, precision, dtype):
        valid_trainer_kwargs = {
            "precision": self.precisions[precision],
            **tfm_kwargs["pl_trainer_kwargs"],
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
        model.fit(self.series.astype(dtype), epochs=1)
        preds = model.predict(n=3)
        assert model.trainer.precision == self.precisions[precision]
        if dtype != np.float16:
            # predictions should have same dtype as input except for float16
            assert preds.dtype == dtype
        else:
            # predictions are float32 when input is float16
            assert preds.dtype == np.float32
        # predictions should not contain NaNs or infs
        assert np.all(np.isfinite(preds.values()))

    def test_custom_callback(self, tmpdir_module):
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
            pl_trainer_kwargs={
                "callbacks": [my_counter_0, my_counter_2],
                **tfm_kwargs["pl_trainer_kwargs"],
            },
        )

        # check if callbacks were added
        assert len(model.trainer_params["callbacks"]) == 2
        model.fit(self.series, epochs=2, verbose=True)
        # check that lightning did not mutate callbacks (verbosity adds a progress bar callback)
        assert len(model.trainer_params["callbacks"]) == 2

        assert my_counter_0.counter == model.epochs_trained
        assert my_counter_2.counter == model.epochs_trained + 2

        # check that callbacks don't overwrite Darts' built-in checkpointer
        model = RNNModel(
            12,
            "RNN",
            10,
            10,
            random_state=42,
            work_dir=tmpdir_module,
            save_checkpoints=True,
            pl_trainer_kwargs={
                "callbacks": [CounterCallback(0), CounterCallback(2)],
                **tfm_kwargs["pl_trainer_kwargs"],
            },
        )
        # we expect 3 callbacks
        assert len(model.trainer_params["callbacks"]) == 3

        # first one is our Checkpointer
        assert isinstance(
            model.trainer_params["callbacks"][0], pl.callbacks.ModelCheckpoint
        )

        # second and third are CounterCallbacks
        for i in range(1, 3):
            assert isinstance(model.trainer_params["callbacks"][i], CounterCallback)

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
            pl_trainer_kwargs={
                "callbacks": [my_stopper],
                **tfm_kwargs["pl_trainer_kwargs"],
            },
        )

        # training should stop immediately with high stopping_threshold
        model.fit(self.series, val_series=self.series, epochs=100, verbose=True)
        assert model.epochs_trained == 1

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
            pl_trainer_kwargs={
                "callbacks": [my_stopper],
                **tfm_kwargs["pl_trainer_kwargs"],
            },
        )

        with pytest.raises(RuntimeError):
            model.fit(self.series, val_series=self.series, epochs=100, verbose=True)
