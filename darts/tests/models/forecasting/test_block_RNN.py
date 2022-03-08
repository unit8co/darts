import shutil
import tempfile

import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass

logger = get_logger(__name__)

try:
    from darts.models.forecasting.block_rnn_model import BlockRNNModel, _BlockRNNModule

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. RNN tests will be skipped.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class BlockRNNModelTestCase(DartsBaseTestClass):
        __test__ = True
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

        def setUp(self):
            self.temp_work_dir = tempfile.mkdtemp(prefix="darts")

        def tearDown(self):
            shutil.rmtree(self.temp_work_dir)

        def test_creation(self):
            with self.assertRaises(ValueError):
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
            self.assertEqual(model1.model.__repr__(), model2.model.__repr__())

        def test_fit(self):
            # Test basic fit()
            model = BlockRNNModel(
                input_chunk_length=1, output_chunk_length=1, n_epochs=2
            )
            model.fit(self.series)

            # Test fit-save-load cycle
            model2 = BlockRNNModel(
                input_chunk_length=1,
                output_chunk_length=1,
                model="LSTM",
                n_epochs=1,
                model_name="unittest-model-lstm",
                work_dir=self.temp_work_dir,
                save_checkpoints=True,
                force_reset=True,
            )
            model2.fit(self.series)
            model_loaded = model2.load_from_checkpoint(
                model_name="unittest-model-lstm",
                work_dir=self.temp_work_dir,
                best=False,
            )
            pred1 = model2.predict(n=6)
            pred2 = model_loaded.predict(n=6)

            # Two models with the same parameters should deterministically yield the same output
            self.assertEqual(sum(pred1.values() - pred2.values()), 0.0)

            # Another random model should not
            model3 = BlockRNNModel(
                input_chunk_length=1, output_chunk_length=1, model="RNN", n_epochs=2
            )
            model3.fit(self.series)
            pred3 = model3.predict(n=6)
            self.assertNotEqual(sum(pred1.values() - pred3.values()), 0.0)

            # test short predict
            pred4 = model3.predict(n=1)
            self.assertEqual(len(pred4), 1)

            # test validation series input
            model3.fit(self.series[:60], val_series=self.series[60:])
            pred4 = model3.predict(n=6)
            self.assertEqual(len(pred4), 6)

        def helper_test_pred_length(self, pytorch_model, series):
            model = pytorch_model(
                input_chunk_length=1, output_chunk_length=3, n_epochs=1
            )
            model.fit(series)
            pred = model.predict(7)
            self.assertEqual(len(pred), 7)
            pred = model.predict(2)
            self.assertEqual(len(pred), 2)
            self.assertEqual(pred.width, 1)
            pred = model.predict(4)
            self.assertEqual(len(pred), 4)
            self.assertEqual(pred.width, 1)

        def test_pred_length(self):
            self.helper_test_pred_length(BlockRNNModel, self.series)
