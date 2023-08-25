import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.logging import get_logger
from darts.tests.conftest import tfm_kwargs

logger = get_logger(__name__)

try:
    from darts.models.forecasting.block_rnn_model import BlockRNNModel, _BlockRNNModule

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. RNN tests will be skipped.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TestBlockRNNModel:
        times = pd.date_range("20130101", "20130410")
        pd_series = pd.Series(range(100), index=times)
        series: TimeSeries = TimeSeries.from_series(pd_series)
        module = _BlockRNNModule(
            "RNN",
            input_size=1,
            input_chunk_length=1,
            output_chunk_length=1,
            hidden_dim=25,
            target_size=1,
            nr_params=1,
            num_layers=1,
            num_layers_out_fc=[],
            dropout=0,
        )

        def test_creation(self):
            with pytest.raises(ValueError):
                # cannot choose any string
                BlockRNNModel(
                    input_chunk_length=1, output_chunk_length=1, model="UnknownRNN?"
                )
            # can give a custom module
            model1 = BlockRNNModel(
                input_chunk_length=1, output_chunk_length=1, model=self.module
            )
            model2 = BlockRNNModel(
                input_chunk_length=1, output_chunk_length=1, model="RNN"
            )
            assert model1.model.__repr__() == model2.model.__repr__()

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
                **tfm_kwargs
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
                **tfm_kwargs
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
