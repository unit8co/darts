import numpy as np
import pandas as pd
import pytest

from darts import concatenate
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import torch

from darts.models.forecasting.tide_model import TiDEModel
from darts.utils.likelihood_models import GaussianLikelihood


class TestTiDEModel:
    np.random.seed(42)
    torch.manual_seed(42)

    def test_creation(self):
        model = TiDEModel(
            input_chunk_length=1,
            output_chunk_length=1,
            likelihood=GaussianLikelihood(),
        )

        assert model.input_chunk_length == 1

    def test_fit(self):
        large_ts = tg.constant_timeseries(length=100, value=1000)
        small_ts = tg.constant_timeseries(length=100, value=10)

        model = TiDEModel(
            input_chunk_length=1,
            output_chunk_length=1,
            n_epochs=10,
            random_state=42,
            **tfm_kwargs,
        )

        model.fit(large_ts[:98])
        pred = model.predict(n=2).values()[0]

        # Test whether model trained on one series is better than one trained on another
        model2 = TiDEModel(
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

        # Test basic fit and predict
        model = TiDEModel(
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

    def test_future_covariate_handling(self):
        ts_time_index = tg.sine_timeseries(length=2, freq="h")

        model = TiDEModel(
            input_chunk_length=1,
            output_chunk_length=1,
            add_encoders={"cyclic": {"future": "hour"}},
            use_reversible_instance_norm=False,
            **tfm_kwargs,
        )
        model.fit(ts_time_index, verbose=False, epochs=1)

        model = TiDEModel(
            input_chunk_length=1,
            output_chunk_length=1,
            add_encoders={"cyclic": {"future": "hour"}},
            use_reversible_instance_norm=True,
            **tfm_kwargs,
        )
        model.fit(ts_time_index, verbose=False, epochs=1)

    def test_future_and_past_covariate_handling(self):
        ts_time_index = tg.sine_timeseries(length=2, freq="h")

        model = TiDEModel(
            input_chunk_length=1,
            output_chunk_length=1,
            add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
            **tfm_kwargs,
        )
        model.fit(ts_time_index, verbose=False, epochs=1)

        model = TiDEModel(
            input_chunk_length=1,
            output_chunk_length=1,
            add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
            **tfm_kwargs,
        )
        model.fit(ts_time_index, verbose=False, epochs=1)

    @pytest.mark.parametrize("temporal_widths", [(-1, 1), (1, -1)])
    def test_failing_future_and_past_temporal_widths(self, temporal_widths):
        # invalid temporal widths
        with pytest.raises(ValueError):
            TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                temporal_width_past=temporal_widths[0],
                temporal_width_future=temporal_widths[1],
                **tfm_kwargs,
            )

    @pytest.mark.parametrize(
        "temporal_widths",
        [
            (2, 2),  # feature projection to same amount of features
            (1, 2),  # past: feature reduction, future: same amount of features
            (2, 1),  # past: same amount of features, future: feature reduction
            (3, 3),  # feature expansion
            (0, 2),  # bypass past feature projection
            (2, 0),  # bypass future feature projection
            (0, 0),  # bypass all feature projection
        ],
    )
    def test_future_and_past_temporal_widths(self, temporal_widths):
        ts_time_index = tg.sine_timeseries(length=2, freq="h")

        # feature projection to 2 features (same amount as input features)
        model = TiDEModel(
            input_chunk_length=1,
            output_chunk_length=1,
            temporal_width_past=temporal_widths[0],
            temporal_width_future=temporal_widths[1],
            add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
            **tfm_kwargs,
        )
        model.fit(ts_time_index, verbose=False, epochs=1)
        assert model.model.temporal_width_past == temporal_widths[0]
        assert model.model.temporal_width_future == temporal_widths[1]

    def test_past_covariate_handling(self):
        ts_time_index = tg.sine_timeseries(length=2, freq="h")

        model = TiDEModel(
            input_chunk_length=1,
            output_chunk_length=1,
            add_encoders={"cyclic": {"past": "hour"}},
            **tfm_kwargs,
        )
        model.fit(ts_time_index, verbose=False, epochs=1)

    def test_future_and_past_covariate_as_timeseries_handling(self):
        ts_time_index = tg.sine_timeseries(length=2, freq="h")

        for enable_rin in [True, False]:
            # test with past_covariates timeseries
            model = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
                use_reversible_instance_norm=enable_rin,
                **tfm_kwargs,
            )
            model.fit(
                ts_time_index,
                past_covariates=ts_time_index,
                verbose=False,
                epochs=1,
            )

            # test with past_covariates and future_covariates timeseries
            model = TiDEModel(
                input_chunk_length=1,
                output_chunk_length=1,
                add_encoders={"cyclic": {"future": "hour", "past": "hour"}},
                use_reversible_instance_norm=enable_rin,
                **tfm_kwargs,
            )
            model.fit(
                ts_time_index,
                past_covariates=ts_time_index,
                future_covariates=ts_time_index,
                verbose=False,
                epochs=1,
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

        # test with static covariates in the timeseries
        model = TiDEModel(
            input_chunk_length=3,
            output_chunk_length=4,
            add_encoders={"cyclic": {"future": "hour"}},
            pl_trainer_kwargs={
                "fast_dev_run": True,
                **tfm_kwargs["pl_trainer_kwargs"],
            },
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
            **tfm_kwargs,
        )
        model.fit(target_multi)
        preds = model.predict(n=2, series=target_multi.with_static_covariates(None))
        assert preds.static_covariates is None

        model = TiDEModel(
            input_chunk_length=3,
            output_chunk_length=4,
            use_static_covariates=False,
            n_epochs=1,
            **tfm_kwargs,
        )
        model.fit(target_multi.with_static_covariates(None))
        preds = model.predict(n=2, series=target_multi)
        assert preds.static_covariates.equals(target_multi.static_covariates)
