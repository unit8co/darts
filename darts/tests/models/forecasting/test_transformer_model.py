import shutil
import tempfile

import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    import torch.nn as nn

    from darts.models.components.transformer import (
        CustomFeedForwardDecoderLayer,
        CustomFeedForwardEncoderLayer,
    )
    from darts.models.forecasting.transformer_model import (
        TransformerModel,
        _TransformerModule,
    )

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. Transformer tests will be skipped.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TransformerModelTestCase(DartsBaseTestClass):
        __test__ = True
        times = pd.date_range("20130101", "20130410")
        pd_series = pd.Series(range(100), index=times)
        series: TimeSeries = TimeSeries.from_series(pd_series)
        series_multivariate = series.stack(series * 2)
        module = _TransformerModule(
            input_size=1,
            input_chunk_length=1,
            output_chunk_length=1,
            output_size=1,
            nr_params=1,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            norm_type=None,
            custom_encoder=None,
            custom_decoder=None,
        )

        def setUp(self):
            self.temp_work_dir = tempfile.mkdtemp(prefix="darts")

        def tearDown(self):
            shutil.rmtree(self.temp_work_dir)

        def test_fit(self):
            # Test fit-save-load cycle
            model2 = TransformerModel(
                input_chunk_length=1,
                output_chunk_length=1,
                n_epochs=2,
                model_name="unittest-model-transformer",
                work_dir=self.temp_work_dir,
                save_checkpoints=True,
                force_reset=True,
            )
            model2.fit(self.series)
            model_loaded = model2.load_from_checkpoint(
                model_name="unittest-model-transformer",
                work_dir=self.temp_work_dir,
                best=False,
            )
            pred1 = model2.predict(n=6)
            pred2 = model_loaded.predict(n=6)

            # Two models with the same parameters should deterministically yield the same output
            self.assertEqual(sum(pred1.values() - pred2.values()), 0.0)

            # Another random model should not
            model3 = TransformerModel(
                input_chunk_length=1, output_chunk_length=1, n_epochs=1
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
            series = tg.linear_timeseries(length=100)
            self.helper_test_pred_length(TransformerModel, series)

        def test_activations(self):
            with self.assertRaises(ValueError):
                model1 = TransformerModel(
                    input_chunk_length=1, output_chunk_length=1, activation="invalid"
                )
                model1.fit(self.series, epochs=1)

            # internal activation function uses PyTorch TransformerEncoderLayer
            model2 = TransformerModel(
                input_chunk_length=1, output_chunk_length=1, activation="gelu"
            )
            model2.fit(self.series, epochs=1)
            assert isinstance(
                model2.model.transformer.encoder.layers[0], nn.TransformerEncoderLayer
            )
            assert isinstance(
                model2.model.transformer.decoder.layers[0], nn.TransformerDecoderLayer
            )

            # glue variant FFN uses our custom _FeedForwardEncoderLayer
            model3 = TransformerModel(
                input_chunk_length=1, output_chunk_length=1, activation="SwiGLU"
            )
            model3.fit(self.series, epochs=1)
            assert isinstance(
                model3.model.transformer.encoder.layers[0],
                CustomFeedForwardEncoderLayer,
            )
            assert isinstance(
                model3.model.transformer.decoder.layers[0],
                CustomFeedForwardDecoderLayer,
            )

        def test_layer_norm(self):
            base_model = TransformerModel

            # default norm_type is None
            model0 = base_model(input_chunk_length=1, output_chunk_length=1)
            y0 = model0.fit(self.series, epochs=1)

            model1 = base_model(
                input_chunk_length=1, output_chunk_length=1, norm_type="RMSNorm"
            )
            y1 = model1.fit(self.series, epochs=1)

            model2 = base_model(
                input_chunk_length=1, output_chunk_length=1, norm_type=nn.LayerNorm
            )
            y2 = model2.fit(self.series, epochs=1)

            model3 = base_model(
                input_chunk_length=1,
                output_chunk_length=1,
                activation="gelu",
                norm_type="RMSNorm",
            )
            y3 = model3.fit(self.series, epochs=1)

            assert y0 != y1
            assert y0 != y2
            assert y0 != y3
            assert y1 != y3

            with self.assertRaises(AttributeError):
                model4 = base_model(
                    input_chunk_length=1, output_chunk_length=1, norm_type="invalid"
                )
                model4.fit(self.series, epochs=1)
