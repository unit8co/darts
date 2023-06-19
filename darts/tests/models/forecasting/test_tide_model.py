import shutil
import tempfile
from itertools import product

import numpy as np
import pandas as pd
import pytest

from darts import concatenate
from darts.logging import get_logger
from darts.metrics import rmse
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

        def test_multivariate_and_covariates(self):
            np.random.seed(42)
            torch.manual_seed(42)
            # test on multiple multivariate series with future and static covariates

            def _create_multiv_series(f1, f2, n1, n2, nf1, nf2):
                bases = [
                    tg.sine_timeseries(
                        length=400, value_frequency=f, value_amplitude=1.0
                    )
                    for f in (f1, f2)
                ]
                noises = [tg.gaussian_timeseries(length=400, std=n) for n in (n1, n2)]
                noise_modulators = [
                    tg.sine_timeseries(length=400, value_frequency=nf)
                    + tg.constant_timeseries(length=400, value=1) / 2
                    for nf in (nf1, nf2)
                ]
                noises = [noises[i] * noise_modulators[i] for i in range(len(noises))]

                target = concatenate(
                    [bases[i] + noises[i] for i in range(len(bases))], axis="component"
                )

                target = target.with_static_covariates(
                    pd.DataFrame([[f1, n1, nf1], [f2, n2, nf2]])
                )

                return target, concatenate(noise_modulators, axis="component")

            def _eval_model(
                train1,
                train2,
                val1,
                val2,
                fut_cov1,
                fut_cov2,
                cls=TiDEModel,
                lkl=None,
            ):
                model = cls(
                    input_chunk_length=50,
                    output_chunk_length=10,
                    likelihood=lkl,
                    random_state=42,
                    optimizer_kwargs={"lr": 1e-3},
                    lr_scheduler_cls=torch.optim.lr_scheduler.ExponentialLR,
                    lr_scheduler_kwargs={"gamma": 0.99},
                )

                model.fit(
                    [train1, train2],
                    future_covariates=[fut_cov1, fut_cov2]
                    if fut_cov1 is not None
                    else None,
                    epochs=10,
                )

                pred1, pred2 = model.predict(
                    series=[train1, train2],
                    future_covariates=[fut_cov1, fut_cov2]
                    if fut_cov1 is not None
                    else None,
                    n=len(val1),
                    num_samples=500 if lkl is not None else 1,
                )

                return rmse(val1, pred1), rmse(val2, pred2)

            series1, fut_cov1 = _create_multiv_series(0.05, 0.07, 0.2, 0.4, 0.02, 0.03)
            series2, fut_cov2 = _create_multiv_series(0.04, 0.03, 0.4, 0.1, 0.02, 0.04)

            train1, val1 = series1.split_after(0.7)
            train2, val2 = series2.split_after(0.7)

            regular_results = [
                (1.18, 1.29),
                (0.77, 0.93),
                (1.02, 1.07),
                (1.12, 1.10),
            ]
            prob_results = [
                (0.55, 0.68),
                (0.91, 0.92),
                (0.69, 0.63),
                (0.81, 0.67),
            ]
            assert len(regular_results) == len(prob_results)

            # for model, lkl in product([TiDEModel], [GaussianLikelihood()]):
            for model, lkl in product([TiDEModel], [None, GaussianLikelihood()]):

                if lkl is None:
                    results = regular_results
                else:
                    results = prob_results

                e1, e2 = _eval_model(
                    train1, train2, val1, val2, fut_cov1, fut_cov2, cls=model, lkl=lkl
                )
                self.assertLessEqual(e1, results[0][0])
                self.assertLessEqual(e2, results[0][1])

                e1, e2 = _eval_model(
                    train1.with_static_covariates(None),
                    train2.with_static_covariates(None),
                    val1,
                    val2,
                    fut_cov1,
                    fut_cov2,
                    cls=model,
                    lkl=lkl,
                )
                self.assertLessEqual(e1, results[1][0])
                self.assertLessEqual(e2, results[1][1])

                e1, e2 = _eval_model(
                    train1, train2, val1, val2, None, None, cls=model, lkl=lkl
                )
                self.assertLessEqual(e1, results[2][0])
                self.assertLessEqual(e2, results[2][1])

                e1, e2 = _eval_model(
                    train1.with_static_covariates(None),
                    train2.with_static_covariates(None),
                    val1,
                    val2,
                    None,
                    None,
                    cls=model,
                    lkl=lkl,
                )
                self.assertLessEqual(e1, results[3][0])
                self.assertLessEqual(e2, results[3][1])

        def test_optional_static_covariates(self):
            series = tg.sine_timeseries(length=20).with_static_covariates(
                pd.DataFrame({"a": [1]})
            )

            # training model with static covs and predicting without will raise an error
            model = TiDEModel(
                input_chunk_length=12,
                output_chunk_length=6,
                use_static_covariates=True,
                n_epochs=1,
            )
            model.fit(series)
            with pytest.raises(ValueError):
                model.predict(n=2, series=series.with_static_covariates(None))

            # with `use_static_covariates=False`, static covariates are ignored and prediction works
            model = TiDEModel(
                input_chunk_length=12,
                output_chunk_length=6,
                use_static_covariates=False,
                n_epochs=1,
            )
            model.fit(series)
            preds = model.predict(n=2, series=series.with_static_covariates(None))
            assert preds.static_covariates is None

            # with `use_static_covariates=False`, static covariates are ignored and prediction works
            model = TiDEModel(
                input_chunk_length=12,
                output_chunk_length=6,
                use_static_covariates=False,
                n_epochs=1,
            )
            model.fit(series.with_static_covariates(None))
            preds = model.predict(n=2, series=series)
            assert preds.static_covariates.equals(series.static_covariates)
