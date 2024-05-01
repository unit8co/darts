import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import torch.nn as nn

from darts.models.forecasting.rnn_model import CustomRNNModule, RNNModel, _RNNModule


class ModuleValid1(_RNNModule):
    """Wrapper around the _RNNModule"""

    def __init__(self, **kwargs):
        super().__init__(name="RNN", **kwargs)


class ModuleValid2(CustomRNNModule):
    """Just a linear layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(self.input_size, self.target_size)

    def forward(self, x_in, h=None):
        x = self.linear(x_in[0])
        return x.view(len(x), -1, self.target_size, self.nr_params)


class TestRNNModel:
    times = pd.date_range("20130101", "20130410")
    pd_series = pd.Series(range(100), index=times)
    series: TimeSeries = TimeSeries.from_series(pd_series)
    module_invalid = _RNNModule(
        name="RNN",
        input_chunk_length=1,
        output_chunk_length=1,
        output_chunk_shift=0,
        input_size=1,
        hidden_dim=25,
        num_layers=1,
        target_size=1,
        nr_params=1,
        dropout=0,
    )

    def test_training_length_input(self):
        # too small training length
        with pytest.raises(ValueError) as msg:
            RNNModel(input_chunk_length=2, training_length=1)
        assert (
            str(msg.value)
            == "`training_length` (1) must be `>=input_chunk_length` (2)."
        )

        # training_length >= input_chunk_length works
        model = RNNModel(
            input_chunk_length=2,
            training_length=2,
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit(self.series[:3])

    def test_creation(self):
        # cannot choose any string
        with pytest.raises(ValueError) as msg:
            RNNModel(input_chunk_length=1, model="UnknownRNN?")
        assert str(msg.value).startswith("`model` is not a valid RNN model.")

        # cannot create from a class instance
        with pytest.raises(ValueError) as msg:
            _ = RNNModel(
                input_chunk_length=1,
                model=self.module_invalid,
            )
        assert str(msg.value).startswith("`model` is not a valid RNN model.")

        # can create from valid module name
        model1 = RNNModel(
            input_chunk_length=1, model="RNN", n_epochs=1, random_state=42, **tfm_kwargs
        )
        model1.fit(self.series)
        preds1 = model1.predict(n=3)

        # can create from a custom class itself
        model2 = RNNModel(
            input_chunk_length=1,
            model=ModuleValid1,
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
        )
        model2.fit(self.series)
        preds2 = model2.predict(n=3)
        np.testing.assert_array_equal(preds1.all_values(), preds2.all_values())

        model3 = RNNModel(
            input_chunk_length=1,
            model=ModuleValid2,
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
        )
        model3.fit(self.series)
        preds3 = model2.predict(n=3)
        assert preds3.all_values().shape == preds2.all_values().shape
        assert preds3.time_index.equals(preds2.time_index)

    def test_fit(self, tmpdir_module):
        # Test basic fit()
        model = RNNModel(input_chunk_length=1, n_epochs=2, **tfm_kwargs)
        model.fit(self.series)

        # Test fit-save-load cycle
        model2 = RNNModel(
            input_chunk_length=1,
            model="LSTM",
            n_epochs=1,
            model_name="unittest-model-lstm",
            work_dir=tmpdir_module,
            save_checkpoints=True,
            force_reset=True,
            **tfm_kwargs,
        )
        model2.fit(self.series)
        model_loaded = model2.load_from_checkpoint(
            model_name="unittest-model-lstm",
            work_dir=tmpdir_module,
            best=False,
            map_location="cpu",
        )
        pred1 = model2.predict(n=6)
        pred2 = model_loaded.predict(n=6)

        # Two models with the same parameters should deterministically yield the same output
        np.testing.assert_array_equal(pred1.values(), pred2.values())

        # Another random model should not
        model3 = RNNModel(input_chunk_length=1, model="RNN", n_epochs=2, **tfm_kwargs)
        model3.fit(self.series)
        pred3 = model3.predict(n=6)
        assert not np.array_equal(pred1.values(), pred3.values())

        # test short predict
        pred4 = model3.predict(n=1)
        assert len(pred4) == 1

        # test validation series input
        model3.fit(self.series[:60], val_series=self.series[60:])
        pred4 = model3.predict(n=6)
        assert len(pred4) == 6

    def helper_test_pred_length(self, pytorch_model, series):
        model = pytorch_model(input_chunk_length=1, n_epochs=1, **tfm_kwargs)
        model.fit(series)
        pred = model.predict(7)
        assert len(pred) == 7
        pred = model.predict(2)
        assert len(pred) == 2
        assert pred.width == 1
        pred = model.predict(4)
        assert len(pred) == 4
        assert pred.width == 1

    def test_pred_length(self):
        self.helper_test_pred_length(RNNModel, self.series)
