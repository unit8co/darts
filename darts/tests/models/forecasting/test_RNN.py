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
import torch
import torch.nn as nn

from darts.models.forecasting.rnn_model import CustomRNNModule, RNNModel, _RNNModule
from darts.utils.data.torch_datasets.utils import (
    ModuleStage,
    PLModuleInput,
    PLModuleOutput,
)


class ModuleValid1(_RNNModule):
    """Wrapper around the _RNNModule"""

    def __init__(self, **kwargs):
        super().__init__(name="RNN", **kwargs)


class ModuleValid2(CustomRNNModule):
    """Just a linear layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(self.input_size, self.target_size)

    def forward(self, x_in):
        x = self.linear(x_in.past_target)
        return PLModuleOutput(
            prediction=x.view(len(x), -1, self.target_size, self.nr_params),
            state=x_in.state,
        )


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
        preds3 = model3.predict(n=3)
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

    def test_forward_uses_module_stage_not_trainer(self):
        """`forward()` branches on `x_in.stage` and must not require a Lightning trainer."""
        module = _RNNModule(
            name="RNN",
            input_chunk_length=4,
            output_chunk_length=1,
            output_chunk_shift=0,
            input_size=2,
            hidden_dim=8,
            num_layers=1,
            target_size=1,
            nr_params=1,
            dropout=0,
        )
        module.eval()
        torch.manual_seed(0)

        past_train = torch.randn(2, 6, 1)
        hfc_train = torch.randn(2, 6, 1)
        fc_train = torch.randn(2, 6, 1)
        x_train = PLModuleInput(
            past_target=past_train,
            historic_future_covariates=hfc_train,
            future_covariates=fc_train,
            stage=ModuleStage.TRAIN,
        )
        out_train = module(x_train)
        # in non-predict mode: all inputs get predictions (6)
        assert out_train.prediction.shape == (2, 6, 1, 1)

        out_val = module(x_train.replace(stage=ModuleStage.VALIDATE))
        assert out_val.prediction.shape == (2, 6, 1, 1)
        torch.testing.assert_close(out_train.prediction, out_val.prediction)

        past_pred = torch.randn(2, 4, 1)
        hfc_pred = torch.randn(2, 4, 1)
        fc_pred = torch.randn(2, 1, 1)
        x_pred = PLModuleInput(
            past_target=past_pred,
            historic_future_covariates=hfc_pred,
            future_covariates=fc_pred,
        )

        # only one predicted step in predict mode
        assert x_pred.stage is ModuleStage.PREDICT
        out_pred = module(x_pred)
        assert out_pred.prediction.shape == (2, 1, 1, 1)

        out_step = module(x_pred.replace(state=out_pred.state))
        assert out_step.prediction.shape == (2, 1, 1, 1)
