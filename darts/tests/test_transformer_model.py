import shutil
import unittest

import logging
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger
from ..utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    from ..models.transformer_model import _TransformerModule, TransformerModel
    from .test_RNN import RNNModelTestCase
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not available. Transformer tests will be skipped.')
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TransformerModelTestCase(unittest.TestCase):
        __test__ = True
        times = pd.date_range('20130101', '20130410')
        pd_series = pd.Series(range(100), index=times)
        series: TimeSeries = TimeSeries.from_series(pd_series)
        series_multivariate = series.stack(series * 2)
        module = _TransformerModule(input_size=1,
                                    input_length=1,
                                    output_length=1,
                                    output_size=1,
                                    d_model=512,
                                    nhead=8,
                                    num_encoder_layers=6,
                                    num_decoder_layers=6,
                                    dim_feedforward=2048,
                                    dropout=0.1,
                                    activation="relu",
                                    custom_encoder=None,
                                    custom_decoder=None,
                                    )

        @classmethod
        def setUpClass(cls):
            logging.disable(logging.CRITICAL)

        @classmethod
        def tearDownClass(cls):
            shutil.rmtree('.darts')

        def test_fit(self):
            # Test fit-save-load cycle
            model2 = TransformerModel(n_epochs=2, model_name='unittest-model-transformer')
            model2.fit(self.series)
            model_loaded = model2.load_from_checkpoint(model_name='unittest-model-transformer', best=False)
            pred1 = model2.predict(n=6)
            pred2 = model_loaded.predict(n=6)

            # Two models with the same parameters should deterministically yield the same output
            self.assertEqual(sum(pred1.values() - pred2.values()), 0.)

            # Another random model should not
            model3 = TransformerModel(n_epochs=1)
            model3.fit(self.series)
            pred3 = model3.predict(n=6)
            self.assertNotEqual(sum(pred1.values() - pred3.values()), 0.)

            # test short predict
            pred4 = model3.predict(n=1)
            self.assertEqual(len(pred4), 1)

            # test validation series input
            model3.fit(self.series[:60], val_series=self.series[60:])
            pred4 = model3.predict(n=6)
            self.assertEqual(len(pred4), 6)

            shutil.rmtree('.darts')

        def test_pred_length(self):
            series = tg.linear_timeseries(length=100)
            RNNModelTestCase.helper_test_pred_length(self, TransformerModel, series)
