import itertools
from itertools import product

import numpy as np
import pandas as pd
import pytest

from darts import concatenate
from darts.logging import get_logger
from darts.metrics import rmse
from darts.tests.conftest import tfm_kwargs
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

    class TestDlinearNlinearModels:
        np.random.seed(42)
        torch.manual_seed(42)

        def test_creation(self):
            with pytest.raises(ValueError):
                DLinearModel(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    normalize=True,
                    likelihood=GaussianLikelihood(),
                )

            with pytest.raises(ValueError):
                NLinearModel(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    normalize=True,
                    likelihood=GaussianLikelihood(),
                )

        def test_fit(self):
            large_ts = tg.constant_timeseries(length=100, value=1000)
            small_ts = tg.constant_timeseries(length=100, value=10)

            for (model_cls, kwargs) in [
                (DLinearModel, {"kernel_size": 5}),
                (DLinearModel, {"kernel_size": 6}),
                (NLinearModel, {}),
            ]:
                # Test basic fit and predict
                model = model_cls(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    n_epochs=10,
                    random_state=42,
                    **kwargs,
                    **tfm_kwargs,
                )
                model.fit(large_ts[:98])
                pred = model.predict(n=2).values()[0]

                # Test whether model trained on one series is better than one trained on another
                model2 = model_cls(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    n_epochs=10,
                    random_state=42,
                    **tfm_kwargs,
                )
                model2.fit(small_ts[:98])
                pred2 = model2.predict(n=2).values()[0]
                assert abs(pred2 - 10) < abs(pred - 10)

                # test short predict
                pred3 = model2.predict(n=1)
                assert len(pred3) == 1

        def test_logtensorboard(self, tmpdir_module):
            ts = tg.constant_timeseries(length=50, value=10)

            for model_cls in [DLinearModel, NLinearModel]:
                # Test basic fit and predict
                model = model_cls(
                    input_chunk_length=1,
                    output_chunk_length=1,
                    n_epochs=1,
                    log_tensorboard=True,
                    work_dir=tmpdir_module,
                    pl_trainer_kwargs={
                        "log_every_n_steps": 1,
                        **tfm_kwargs["pl_trainer_kwargs"],
                    },
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
                    **tfm_kwargs,
                )
                model_not_shared = model_cls(
                    input_chunk_length=5,
                    output_chunk_length=1,
                    n_epochs=2,
                    const_init=False,
                    shared_weights=False,
                    random_state=42,
                    **tfm_kwargs,
                )
                model_shared.fit(ts)
                model_not_shared.fit(ts)
                pred_shared = model_shared.predict(n=2)
                pred_not_shared = model_not_shared.predict(n=2)
                assert np.any(
                    np.not_equal(pred_shared.values(), pred_not_shared.values())
                )

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
                past_cov1=None,
                past_cov2=None,
                val_past_cov1=None,
                val_past_cov2=None,
                cls=DLinearModel,
                lkl=None,
                **kwargs
            ):
                model = cls(
                    input_chunk_length=50,
                    output_chunk_length=10,
                    shared_weights=False,
                    const_init=True,
                    likelihood=lkl,
                    random_state=42,
                    **tfm_kwargs,
                )

                model.fit(
                    [train1, train2],
                    past_covariates=[past_cov1, past_cov2]
                    if past_cov1 is not None
                    else None,
                    val_past_covariates=[val_past_cov1, val_past_cov2]
                    if val_past_cov1 is not None
                    else None,
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
                    past_covariates=[fut_cov1, fut_cov2]
                    if past_cov1 is not None
                    else None,
                    n=len(val1),
                    num_samples=500 if lkl is not None else 1,
                )

                return rmse(val1, pred1), rmse(val2, pred2)

            series1, fut_cov1 = _create_multiv_series(0.05, 0.07, 0.2, 0.4, 0.02, 0.03)
            series2, fut_cov2 = _create_multiv_series(0.04, 0.03, 0.4, 0.1, 0.02, 0.04)

            train1, val1 = series1.split_after(0.7)
            train2, val2 = series2.split_after(0.7)
            past_cov1 = train1.copy()
            past_cov2 = train2.copy()
            val_past_cov1 = val1.copy()
            val_past_cov2 = val2.copy()

            for model, lkl in product(
                [DLinearModel, NLinearModel], [None, GaussianLikelihood()]
            ):

                e1, e2 = _eval_model(
                    train1, train2, val1, val2, fut_cov1, fut_cov2, cls=model, lkl=lkl
                )
                assert e1 <= 0.34
                assert e2 <= 0.28

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
                assert e1 <= 0.32
                assert e2 <= 0.28

                e1, e2 = _eval_model(
                    train1, train2, val1, val2, None, None, cls=model, lkl=lkl
                )
                assert e1 <= 0.40
                assert e2 <= 0.34

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
                assert e1 <= 0.40
                assert e2 <= 0.34

            e1, e2 = _eval_model(
                train1,
                train2,
                val1,
                val2,
                fut_cov1,
                fut_cov2,
                past_cov1=past_cov1,
                past_cov2=past_cov2,
                val_past_cov1=val_past_cov1,
                val_past_cov2=val_past_cov2,
                cls=NLinearModel,
                lkl=None,
                normalize=True,
            )
            # can only fit models with past/future covariates when shared_weights=False
            for model in [DLinearModel, NLinearModel]:
                for shared_weights in [True, False]:
                    model_instance = model(
                        5, 5, shared_weights=shared_weights, **tfm_kwargs
                    )
                    assert model_instance.supports_past_covariates == (
                        not shared_weights
                    )
                    assert model_instance.supports_future_covariates == (
                        not shared_weights
                    )
                    if shared_weights:
                        with pytest.raises(ValueError):
                            model_instance.fit(series1, future_covariates=fut_cov1)

        def test_optional_static_covariates(self):
            series = tg.sine_timeseries(length=20).with_static_covariates(
                pd.DataFrame({"a": [1]})
            )
            for model_cls in [NLinearModel, DLinearModel]:
                # training model with static covs and predicting without will raise an error
                model = model_cls(
                    input_chunk_length=12,
                    output_chunk_length=6,
                    use_static_covariates=True,
                    n_epochs=1,
                    **tfm_kwargs,
                )
                model.fit(series)
                with pytest.raises(ValueError):
                    model.predict(n=2, series=series.with_static_covariates(None))

                # with `use_static_covariates=False`, static covariates are ignored and prediction works
                model = model_cls(
                    input_chunk_length=12,
                    output_chunk_length=6,
                    use_static_covariates=False,
                    n_epochs=1,
                    **tfm_kwargs,
                )
                model.fit(series)
                preds = model.predict(n=2, series=series.with_static_covariates(None))
                assert preds.static_covariates is None

                # with `use_static_covariates=False`, static covariates are ignored and prediction works
                model = model_cls(
                    input_chunk_length=12,
                    output_chunk_length=6,
                    use_static_covariates=False,
                    n_epochs=1,
                    **tfm_kwargs,
                )
                model.fit(series.with_static_covariates(None))
                preds = model.predict(n=2, series=series)
                assert preds.static_covariates.equals(series.static_covariates)

        @pytest.mark.parametrize(
            "config", itertools.product([3, 7, 10], [NLinearModel, DLinearModel])
        )
        def test_output_shift(self, config):
            """Tests shifted output for shift smaller than, equal to, and larger than output_chunk_length."""
            shift, model_cls = config
            icl = 7
            ocl = 7
            series = tg.linear_timeseries(
                length=28, start=pd.Timestamp("2000-01-01"), freq="d"
            )

            model = self.helper_create_model(model_cls, icl, ocl, shift)
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
                model_cls,
                icl,
                ocl,
                shift,
                add_encoders={
                    "datetime_attribute": {
                        "future": ["dayofweek"],
                        "past": ["dayofweek"],
                    }
                },
            )
            model_enc_shift.fit(series)

            # model trained with identical covariates
            model_fc_shift = self.helper_create_model(model_cls, icl, ocl, shift)

            covs = tg.datetime_attribute_timeseries(
                series,
                attribute="dayofweek",
                add_length=ocl + shift,
            )
            model_fc_shift.fit(series, past_covariates=covs, future_covariates=covs)

            pred_enc = model_enc_shift.predict(n=ocl)
            pred_fc = model_fc_shift.predict(n=ocl)
            assert pred_enc == pred_fc

            # future covs too short
            with pytest.raises(ValueError) as err:
                _ = model_fc_shift.predict(n=ocl, future_covariates=covs[:-1])
            assert "provided future covariates at dataset index" in str(err.value)

            # past covs too short
            with pytest.raises(ValueError) as err:
                _ = model_fc_shift.predict(
                    n=ocl, past_covariates=covs[: -(ocl + shift + 1)]
                )
            assert "provided past covariates at dataset index" in str(err.value)

        def helper_create_model(self, model_cls, icl, ocl, shift, **kwargs):
            return model_cls(
                input_chunk_length=icl,
                output_chunk_length=ocl,
                output_chunk_shift=shift,
                n_epochs=1,
                random_state=42,
                **tfm_kwargs,
                **kwargs,
            )
