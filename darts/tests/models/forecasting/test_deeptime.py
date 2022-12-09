import shutil
import tempfile

import numpy as np

from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    from darts.models.forecasting.deeptime import DeepTimeModel

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. DeepTime tests will be skipped.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class DeepTimeModelTestCase(DartsBaseTestClass):
        def setUp(self):
            self.temp_work_dir = tempfile.mkdtemp(prefix="darts")

        def tearDown(self):
            shutil.rmtree(self.temp_work_dir)

        def test_creation(self):
            with self.assertRaises(ValueError):
                # if the `inr_layer_widths` argument is a list, its length must be equal to `inr_num_layers`
                DeepTimeModel(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    inr_num_layers=3,
                    inr_layers_width=[1],
                )
            with self.assertRaises(ValueError):
                # n_epochs should be greater than 0 for instantiation of the lr schedulers
                DeepTimeModel(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    inr_num_layers=3,
                    inr_layers_width=20,
                    n_epochs=0,
                )
            with self.assertRaises(ValueError):
                # n_epochs should be greater than warmup_epochs of lr schedulers
                DeepTimeModel(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    inr_num_layers=3,
                    inr_layers_width=20,
                    n_epochs=1,
                )

        def test_fit(self):
            large_ts = tg.constant_timeseries(length=100, value=1000)
            small_ts = tg.constant_timeseries(length=100, value=10)

            # Test basic fit and predict
            model = DeepTimeModel(
                input_chunk_length=1,
                output_chunk_length=1,
                n_epochs=10,
                inr_num_layers=2,
                inr_layers_width=20,
                n_fourier_feats=64,
                scales=[0.01, 0.1, 1, 5, 10, 20, 50, 100],
                random_state=42,
            )
            model.fit(large_ts[:98])
            pred = model.predict(n=2).values()[0]

            # Test whether model trained on one series is better than one trained on another
            model2 = DeepTimeModel(
                input_chunk_length=1,
                output_chunk_length=1,
                n_epochs=10,
                inr_num_layers=2,
                inr_layers_width=20,
                n_fourier_feats=64,
                scales=[0.01, 0.1, 1, 5, 10, 20, 50, 100],
                random_state=42,
            )
            model2.fit(small_ts[:98])
            pred2 = model2.predict(n=2).values()[0]
            self.assertTrue(abs(pred2 - 10) < abs(pred - 10))

            # test short predict
            pred3 = model2.predict(n=1)
            self.assertEqual(len(pred3), 1)

        def test_multivariate(self):
            # testing a 2-variate linear ts, first one from 0 to 1, second one from 0 to 0.5, length 100
            series_multivariate = tg.linear_timeseries(length=100).stack(
                tg.linear_timeseries(length=100, start_value=0, end_value=0.5)
            )

            model = DeepTimeModel(
                input_chunk_length=3,
                output_chunk_length=1,
                n_epochs=10,
                inr_num_layers=5,
                inr_layers_width=64,
                n_fourier_feats=256,  # must have enough fourier components to capture low freq
                scales=[0.01, 0.1, 1, 5, 10, 20, 50, 100],
                random_state=42,
            )
            model.fit(series_multivariate)
            res = model.predict(n=3).values()

            # the theoretical result should be [[1.01, 1.02, 1.03], [0.505, 0.51, 0.515]].
            # We just test if the given result is not too far on average.
            self.assertTrue(
                abs(
                    np.average(
                        res - np.array([[1.01, 1.02, 1.03], [0.505, 0.51, 0.515]]).T
                    )
                    < 0.03
                )
            )

            # Test Covariates
            series_covariates = tg.linear_timeseries(length=100).stack(
                tg.linear_timeseries(length=100, start_value=0, end_value=0.1)
            )
            model = DeepTimeModel(
                input_chunk_length=3,
                output_chunk_length=4,
                n_epochs=10,
                inr_num_layers=2,
                inr_layers_width=20,
                n_fourier_feats=64,
                scales=[0.01, 0.1, 1, 5, 10, 20, 50, 100],
                random_state=42,
            )
            model.fit(series_multivariate, past_covariates=series_covariates)

            res = model.predict(
                n=3, series=series_multivariate, past_covariates=series_covariates
            ).values()

            self.assertEqual(len(res), 3)
            self.assertTrue(abs(np.average(res)) < 5)

        def test_deeptime_n_fourier_feats(self):
            with self.assertRaises(ValueError):
                # wrong number of scales and n_fourier feats
                # n_fourier_feats must be divisiable by 2*len(scales)
                DeepTimeModel(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    n_epochs=6,
                    inr_num_layers=2,
                    inr_layers_width=20,
                    n_fourier_feats=17,
                    scales=[0.01, 0.1, 1, 5, 10, 20, 50, 100],
                    random_state=42,
                )

        def test_logtensorboard(self):
            ts = tg.constant_timeseries(length=50, value=10)

            # Test basic fit and predict
            model = DeepTimeModel(
                input_chunk_length=1,
                output_chunk_length=1,
                n_epochs=6,
                inr_num_layers=2,
                inr_layers_width=20,
                n_fourier_feats=64,
                scales=[0.01, 0.1, 1, 5, 10, 20, 50, 100],
                random_state=42,
                log_tensorboard=True,
                work_dir=self.temp_work_dir,
            )
            model.fit(ts)
            model.predict(n=2)

        def test_activation_fns(self):
            ts = tg.constant_timeseries(length=50, value=10)

            model = DeepTimeModel(
                input_chunk_length=1,
                output_chunk_length=1,
                n_epochs=6,
                inr_num_layers=2,
                inr_layers_width=8,
                n_fourier_feats=8,
                scales=[0.01, 0.1],
                activation="LeakyReLU",
                random_state=42,
            )
            model.fit(ts)

            with self.assertRaises(ValueError):
                model = DeepTimeModel(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    n_epochs=6,
                    inr_num_layers=2,
                    inr_layers_width=8,
                    n_fourier_feats=8,
                    scales=[0.01, 0.1],
                    activation="invalid",
                    random_state=42,
                )
                model.fit(ts)
