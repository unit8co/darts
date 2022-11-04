import shutil
import tempfile

import numpy as np

from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    from darts.models.forecasting.dlinear import DLinearModel
    from darts.models.forecasting.nlinear import NLinearModel

    TORCH_AVAILABLE = True
    from darts.utils.likelihood_models import GaussianLikelihood

except ImportError:
    logger.warning("Torch not available. Dlinear and NLinear tests will be skipped.")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class DlinearNlinearModelsTestCase(DartsBaseTestClass):
        def setUp(self):
            self.temp_work_dir = tempfile.mkdtemp(prefix="darts")

        def tearDown(self):
            shutil.rmtree(self.temp_work_dir)

        def test_creation(self):
            with self.assertRaises(ValueError):
                DLinearModel(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    normalize=True,
                    likelihood=GaussianLikelihood(),
                )

            with self.assertRaises(ValueError):
                NLinearModel(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    normalize=True,
                    likelihood=GaussianLikelihood(),
                )

        def test_fit(self):
            large_ts = tg.constant_timeseries(length=100, value=1000)
            small_ts = tg.constant_timeseries(length=100, value=10)

            for model_cls in [DLinearModel, NLinearModel]:
                # Test basic fit and predict
                model = model_cls(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    n_epochs=10,
                )
                model.fit(large_ts[:98])
                pred = model.predict(n=2).values()[0]

                # Test whether model trained on one series is better than one trained on another
                model2 = model_cls(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    n_epochs=10,
                )
                model2.fit(small_ts[:98])
                pred2 = model2.predict(n=2).values()[0]
                self.assertTrue(abs(pred2 - 10) < abs(pred - 10))

                # test short predict
                pred3 = model2.predict(n=1)
                self.assertEqual(len(pred3), 1)

        def test_logtensorboard(self):
            ts = tg.constant_timeseries(length=50, value=10)

            for model_cls in [DLinearModel, NLinearModel]:
                # Test basic fit and predict
                model = model_cls(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    n_epochs=1,
                    log_tensorboard=True,
                    work_dir=self.temp_work_dir,
                )
                model.fit(ts)
                model.predict(n=2)

        def test_multivariate(self):
            # TODO: this test will not pass, the accuracy is too low
            # testing a 2-variate linear ts, first one from 0 to 1, second one from 0 to 0.5, length 100
            series_multivariate = tg.linear_timeseries(length=100).stack(
                tg.linear_timeseries(length=100, start_value=0, end_value=0.5)
            )

            for model_cls in [DLinearModel, NLinearModel]:
                model = model_cls(
                    input_chunk_length=3,
                    output_chunk_length=1,
                    n_epochs=20,
                    random_state=42,
                )

                model.fit(series_multivariate)
                res = model.predict(n=2).values()

                # the theoretical result should be [[1.01, 1.02], [0.505, 0.51]].
                # We just test if the given result is not too far on average.
                print(model_cls)
                print(res)
                self.assertTrue(
                    abs(
                        np.average(res - np.array([[1.01, 1.02], [0.505, 0.51]])) < 0.03
                    )
                )

                # Test Covariates
                series_covariates = tg.linear_timeseries(length=100).stack(
                    tg.linear_timeseries(length=100, start_value=0, end_value=0.1)
                )
                model = model_cls(
                    input_chunk_length=3,
                    output_chunk_length=4,
                    n_epochs=5,
                    random_state=42,
                )
                model.fit(series_multivariate, past_covariates=series_covariates)

                res = model.predict(
                    n=3, series=series_multivariate, past_covariates=series_covariates
                ).values()

                self.assertEqual(len(res), 3)
                self.assertTrue(abs(np.average(res)) < 5)
