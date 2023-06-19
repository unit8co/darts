import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from darts import concatenate
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    import torch

    from darts.models.forecasting.tide_model import TiDEModel
    from darts.utils.likelihood_models import GaussianLikelihood

    TORCH_AVAILABLE = True

except ImportError:
    logger.warning("Torch not available. TiDEModel tests will be skipped.")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class TiDEModelModelTestCase(DartsBaseTestClass):
        np.random.seed(42)
        torch.manual_seed(42)

        def setUp(self):
            self.temp_work_dir = tempfile.mkdtemp(prefix="darts")

        def tearDown(self):
            shutil.rmtree(self.temp_work_dir)

        def test_creation(self):
            model = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                likelihood=GaussianLikelihood(),
            )

            self.assertEqual(model.input_chunk_length, 1)

        def test_fit(self):
            large_ts = tg.constant_timeseries(length=100, value=1000)
            small_ts = tg.constant_timeseries(length=100, value=10)

            model = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                n_epochs=10,
                random_state=42,
            )

            model.fit(large_ts[:98])
            pred = model.predict(n=2).values()[0]

            # Test whether model trained on one series is better than one trained on another
            model2 = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                n_epochs=10,
                random_state=42,
            )

            model2.fit(small_ts[:98])
            pred2 = model2.predict(n=2).values()[0]
            self.assertTrue(abs(pred2 - 10) < abs(pred - 10))

            # test short predict
            pred3 = model2.predict(n=1)
            self.assertEqual(len(pred3), 1)

        def test_logtensorboard(self):
            ts = tg.constant_timeseries(length=50, value=10)

            # Test basic fit and predict
            model = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                n_epochs=1,
                log_tensorboard=True,
                work_dir=self.temp_work_dir,
                pl_trainer_kwargs={"log_every_n_steps": 1},
            )
            model.fit(ts)
            model.predict(n=2)

        def test_future_covariate_handling(self):
            ts_time_index = tg.sine_timeseries(length=2, freq="h")

            model = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                add_encoders={"cyclic": {"future": "hour"}},
            )
            model.fit(ts_time_index, verbose=False, epochs=1)

        def test_future_and_past_covariate_handling(self):
            ts_time_index = tg.sine_timeseries(length=2, freq="h")

            model = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
            )
            model.fit(ts_time_index, verbose=False, epochs=1)

        def test_past_covariate_handling(self):
            ts_time_index = tg.sine_timeseries(length=2, freq="h")

            model = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                add_encoders={"cyclic": {"past": "hour"}},
            )
            model.fit(ts_time_index, verbose=False, epochs=1)

        def test_future_and_past_covariate_as_timeseries_handling(self):
            ts_time_index = tg.sine_timeseries(length=2, freq="h")

            # test with past_covariates timeseries
            model = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
            )
            model.fit(ts_time_index, ts_time_index, verbose=False, epochs=1)

            # test with past_covariates and future_covariates timeseries
            model = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
            )
            model.fit(
                ts_time_index, ts_time_index, ts_time_index, verbose=False, epochs=1
            )

        def test_static_covariates_support(self):
            target_multi = concatenate(
                [tg.sine_timeseries(length=10, freq="h")] * 2, axis=1
            )

            target_multi = target_multi.with_static_covariates(
                pd.DataFrame(
                    [[0.0, 1.0, 0, 2], [2.0, 3.0, 1, 3]],
                    columns=["st1", "st2", "cat1", "cat2"],
                )
            )

            # should work with cyclic encoding for time index
            # set categorical embedding sizes once with automatic embedding size with an `int` and once by
            # manually setting it with `tuple(int, int)`
            model = TiDEModel(
                input_chunk_length=3,
                output_chunk_length=4,
                add_encoders={"cyclic": {"future": "hour"}},
                # categorical_embedding_sizes={"cat1": 2, "cat2": (2, 2)},
                pl_trainer_kwargs={"fast_dev_run": True},
            )
            model.fit(target_multi, verbose=False)

            assert model.model.static_cov_dim == np.prod(
                target_multi.static_covariates.values.shape
            )

            # raise an error when trained with static covariates of wrong dimensionality
            target_multi = target_multi.with_static_covariates(
                pd.concat([target_multi.static_covariates] * 2, axis=1)
            )
            with pytest.raises(ValueError):
                model.predict(n=1, series=target_multi, verbose=False)

            # raise an error when trained with static covariates and trying to predict without
            with pytest.raises(ValueError):
                model.predict(
                    n=1, series=target_multi.with_static_covariates(None), verbose=False
                )

            # with `use_static_covariates=False`, we can predict without static covs
            model = TiDEModel(
                input_chunk_length=3,
                output_chunk_length=4,
                use_static_covariates=False,
                n_epochs=1,
            )
            model.fit(target_multi)
            preds = model.predict(n=2, series=target_multi.with_static_covariates(None))
            assert preds.static_covariates is None

            model = TiDEModel(
                input_chunk_length=3,
                output_chunk_length=4,
                use_static_covariates=False,
                n_epochs=1,
            )
            model.fit(target_multi.with_static_covariates(None))
            preds = model.predict(n=2, series=target_multi)
            assert preds.static_covariates.equals(target_multi.static_covariates)
