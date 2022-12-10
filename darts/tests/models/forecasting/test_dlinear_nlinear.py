import shutil
import tempfile
from itertools import product

import numpy as np
import pandas as pd

from darts import concatenate
from darts.logging import get_logger
from darts.metrics import rmse
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    import torch

    from darts.models.forecasting.dlinear import DLinearModel
    from darts.models.forecasting.nlinear import NLinearModel
    from darts.utils.likelihood_models import GaussianLikelihood

    TORCH_AVAILABLE = True

except ImportError:
    logger.warning("Torch not available. Dlinear and NLinear tests will be skipped.")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class DlinearNlinearModelsTestCase(DartsBaseTestClass):
        np.random.seed(42)
        torch.manual_seed(42)

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
                    random_state=42,
                )
                model.fit(large_ts[:98])
                pred = model.predict(n=2).values()[0]

                # Test whether model trained on one series is better than one trained on another
                model2 = model_cls(
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

        def test_shared_weights(self):
            ts = tg.constant_timeseries(length=50, value=10).stack(
                tg.gaussian_timeseries(length=50)
            )

            for model_cls in [DLinearModel, NLinearModel]:
                # Test basic fit and predict
                model_shared = model_cls(
                    input_chunk_length=5,
                    output_chunk_length=1,
                    n_epochs=2,
                    const_init=False,
                    shared_weights=True,
                    random_state=42,
                )
                model_not_shared = model_cls(
                    input_chunk_length=5,
                    output_chunk_length=1,
                    n_epochs=2,
                    const_init=False,
                    shared_weights=False,
                    random_state=42,
                )
                model_shared.fit(ts)
                model_not_shared.fit(ts)
                pred_shared = model_shared.predict(n=2)
                pred_not_shared = model_not_shared.predict(n=2)
                self.assertTrue(
                    np.any(np.not_equal(pred_shared.values(), pred_not_shared.values()))
                )

        def test_multivariate_and_covariates(self):
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
                cls=DLinearModel,
                lkl=None,
            ):
                model = cls(
                    input_chunk_length=50,
                    output_chunk_length=10,
                    shared_weights=False,
                    const_init=True,
                    likelihood=lkl,
                    random_state=42,
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

            for model, lkl in product(
                [DLinearModel, NLinearModel], [None, GaussianLikelihood()]
            ):

                e1, e2 = _eval_model(
                    train1, train2, val1, val2, fut_cov1, fut_cov2, cls=model, lkl=lkl
                )
                self.assertLessEqual(e1, 0.31)
                self.assertLessEqual(e2, 0.28)

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
                self.assertLessEqual(e1, 0.32)
                self.assertLessEqual(e2, 0.28)

                e1, e2 = _eval_model(
                    train1, train2, val1, val2, None, None, cls=model, lkl=lkl
                )
                self.assertLessEqual(e1, 0.40)
                self.assertLessEqual(e2, 0.34)

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
                self.assertLessEqual(e1, 0.40)
                self.assertLessEqual(e2, 0.34)
