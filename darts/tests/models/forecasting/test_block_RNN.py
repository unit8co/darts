import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.logging import get_logger
from darts.tests.conftest import tfm_kwargs
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    import torch.nn as nn

    from darts.models.forecasting.block_rnn_model import (
        BlockRNNModel,
        CustomBlockRNNModule,
        _BlockRNNModule,
    )

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. RNN tests will be skipped.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ModuleValid1(_BlockRNNModule):
        """Wrapper around the _BlockRNNModule"""

        def __init__(self, **kwargs):
            super().__init__(name="RNN", **kwargs)

    class ModuleValid2(CustomBlockRNNModule):
        """Just a linear layer."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.linear = nn.Linear(self.input_size, self.target_size)

        def forward(self, x_in):
            x = self.linear(x_in[0])
            return x.view(len(x), -1, self.target_size, self.nr_params)

    class TestBlockRNNModel:
        times = pd.date_range("20130101", "20130410")
        pd_series = pd.Series(range(100), index=times)
        series: TimeSeries = TimeSeries.from_series(pd_series)
        module_invalid = _BlockRNNModule(
            "RNN",
            input_size=1,
            input_chunk_length=1,
            output_chunk_length=1,
            output_chunk_shift=0,
            hidden_dim=25,
            target_size=1,
            nr_params=1,
            num_layers=1,
            num_layers_out_fc=[],
            dropout=0,
        )

        def test_creation(self):
            # cannot choose any string
            with pytest.raises(ValueError) as msg:
                BlockRNNModel(
                    input_chunk_length=1, output_chunk_length=1, model="UnknownRNN?"
                )
            assert str(msg.value).startswith("`model` is not a valid RNN model.")

            # cannot create from a class instance
            with pytest.raises(ValueError) as msg:
                _ = BlockRNNModel(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    model=self.module_invalid,
                )
            assert str(msg.value).startswith("`model` is not a valid RNN model.")

            # can create from valid module name
            model1 = BlockRNNModel(
                input_chunk_length=1,
                output_chunk_length=1,
                model="RNN",
                n_epochs=1,
                random_state=42,
                **tfm_kwargs,
            )
            model1.fit(self.series)
            preds1 = model1.predict(n=3)

            # can create from a custom class itself
            model2 = BlockRNNModel(
                input_chunk_length=1,
                output_chunk_length=1,
                model=ModuleValid1,
                n_epochs=1,
                random_state=42,
                **tfm_kwargs,
            )
            model2.fit(self.series)
            preds2 = model2.predict(n=3)
            np.testing.assert_array_equal(preds1.all_values(), preds2.all_values())

            model3 = BlockRNNModel(
                input_chunk_length=1,
                output_chunk_length=1,
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
            model = BlockRNNModel(
                input_chunk_length=1, output_chunk_length=1, n_epochs=2, **tfm_kwargs
            )
            model.fit(self.series)

            # Test fit-save-load cycle
            model2 = BlockRNNModel(
                input_chunk_length=1,
                output_chunk_length=1,
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
            model3 = BlockRNNModel(
                input_chunk_length=1,
                output_chunk_length=1,
                model="RNN",
                n_epochs=2,
                **tfm_kwargs,
            )
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
            model = pytorch_model(
                input_chunk_length=1, output_chunk_length=3, n_epochs=1, **tfm_kwargs
            )
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
            self.helper_test_pred_length(BlockRNNModel, self.series)

        @pytest.mark.parametrize("shift", [3, 7, 10])
        def test_output_shift(self, shift):
            """Tests shifted output for shift smaller than, equal to, and larger than output_chunk_length."""
            icl = 7
            ocl = 7
            series = tg.linear_timeseries(
                length=28, start=pd.Timestamp("2000-01-01"), freq="d"
            )

            model = self.helper_create_model(icl, ocl, shift)
            model.fit(series)

            # no auto-regression with shifted output
            with pytest.raises(ValueError) as err:
                _ = model.predict(n=ocl + 1)
            assert str(err.value).startswith("Cannot perform auto-regression")

            # pred starts with a shift
            for ocl_test in [ocl - 1, ocl]:
                pred = model.predict(n=ocl_test)
                assert (
                    pred.start_time() == series.end_time() + (shift + 1) * series.freq
                )
                assert len(pred) == ocl_test
                assert pred.freq == series.freq

            # check that shifted output chunk results with encoders are the
            # same as using identical covariates

            # model trained on encoders
            model_enc_shift = self.helper_create_model(
                icl,
                ocl,
                shift,
                add_encoders={"datetime_attribute": {"past": ["dayofweek"]}},
            )
            model_enc_shift.fit(series)

            # model trained with identical covariates
            model_fc_shift = self.helper_create_model(icl, ocl, shift)

            covs = tg.datetime_attribute_timeseries(
                series,
                attribute="dayofweek",
                add_length=ocl + shift,
            )
            model_fc_shift.fit(
                series,
                past_covariates=covs,
            )

            pred_enc = model_enc_shift.predict(n=ocl)
            pred_fc = model_fc_shift.predict(n=ocl)
            assert pred_enc == pred_fc

            # past covs too short
            with pytest.raises(ValueError) as err:
                _ = model_fc_shift.predict(
                    n=ocl, past_covariates=covs[: -(ocl + shift + 1)]
                )
            assert "provided past covariates at dataset index" in str(err.value)

        def helper_create_model(self, icl, ocl, shift, **kwargs):
            return BlockRNNModel(
                input_chunk_length=icl,
                output_chunk_length=ocl,
                output_chunk_shift=shift,
                n_epochs=1,
                random_state=42,
                **tfm_kwargs,
                **kwargs,
            )
