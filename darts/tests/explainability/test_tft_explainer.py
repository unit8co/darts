import itertools
from unittest.mock import patch

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import pytorch_lightning as pl
import torch

from darts.explainability import TFTExplainabilityResult, TFTExplainer
from darts.explainability.tft_explainer import _TFTPredictionOutputCollector
from darts.models import TFTModel


class _PredictionBatchCallback(pl.Callback):
    def __init__(self):
        self.batch_sizes = []
        self.collector_batch_counts = []
        self.collector = None
        self.raise_on_batch_idx = None

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self.batch_sizes.append(batch[0].shape[0])
        self.collector = next(
            callback
            for callback in trainer.callbacks
            if isinstance(callback, _TFTPredictionOutputCollector)
        )
        self.collector_batch_counts.append(
            len(self.collector._batch_outputs["_attn_out_weights"])
        )
        if batch_idx == self.raise_on_batch_idx:
            raise RuntimeError("user callback failure")


def helper_create_test_cases(series_options: list):
    covariates_options = [
        {},
        {"past_covariates"},
        {"future_covariates"},
        {"past_covariates", "future_covariates"},
    ]
    relative_index_options = [False, True]
    use_encoders_options = [False, True]
    return itertools.product(*[
        series_options,
        covariates_options,
        relative_index_options,
        use_encoders_options,
    ])


class TestTFTExplainer:
    freq = "MS"
    series_lin_pos = tg.linear_timeseries(length=10, freq=freq).with_static_covariates(
        pd.Series([0.0, 0.5], index=["cat", "num"])
    )
    series_sine = tg.sine_timeseries(length=10, freq=freq)
    series_mv1 = series_lin_pos.stack(series_sine)

    series_lin_neg = tg.linear_timeseries(
        start_value=1, end_value=0, length=10, freq=freq
    ).with_static_covariates(pd.Series([1.0, 0.5], index=["cat", "num"]))
    series_cos = tg.sine_timeseries(length=10, value_phase=90, freq=freq)
    series_mv2 = series_lin_neg.stack(series_cos)

    series_multi = [series_mv1, series_mv2]
    pc = tg.constant_timeseries(length=10, freq=freq)
    pc_multi = [pc] * 2
    fc = tg.constant_timeseries(length=13, freq=freq)
    fc_multi = [fc] * 2

    def helper_get_input(self, series_option: str):
        if series_option == "univariate":
            return self.series_lin_pos, self.pc, self.fc
        elif series_option == "multivariate":
            return self.series_mv1, self.pc, self.fc
        else:  # multiple
            return self.series_multi, self.pc_multi, self.fc_multi

    @pytest.mark.parametrize(
        "test_case", helper_create_test_cases(["univariate", "multivariate"])
    )
    def test_explainer_single_univariate_multivariate_series(self, test_case):
        """Test TFTExplainer with single univariate and multivariate series and a combination of
        encoders, covariates, and addition of relative index."""
        series_option, cov_option, add_relative_idx, use_encoders = test_case
        series, pc, fc = self.helper_get_input(series_option)
        cov_test_case = dict()
        use_pc, use_fc = False, False
        if "past_covariates" in cov_option:
            cov_test_case["past_covariates"] = pc
            use_pc = True
        if "future_covariates" in cov_option:
            cov_test_case["future_covariates"] = fc
            use_fc = True

        # expected number of features for past covs, future covs, and static covs, and encoder/decoder
        n_target_expected = series.n_components
        n_pc_expected = 1 if "past_covariates" in cov_test_case else 0
        n_fc_expected = 1 if "future_covariates" in cov_test_case else 0
        n_sc_expected = 2
        # encoder is number of past and future covs plus 4 optional encodings (future and past)
        # plus 1 univariate target plus 1 optional relative index
        n_enc_expected = (
            n_pc_expected
            + n_fc_expected
            + n_target_expected
            + (4 if use_encoders else 0)
            + (1 if add_relative_idx else 0)
        )
        # encoder is number of future covs plus 2 optional encodings (future)
        # plus 1 optional relative index
        n_dec_expected = (
            n_fc_expected + (2 if use_encoders else 0) + (1 if add_relative_idx else 0)
        )
        model = self.helper_create_model(
            use_encoders=use_encoders, add_relative_idx=add_relative_idx
        )
        # TFTModel requires future covariates
        if (
            not add_relative_idx
            and "future_covariates" not in cov_test_case
            and not use_encoders
        ):
            with pytest.raises(ValueError):
                model.fit(series=series, **cov_test_case)
            return

        model.fit(series=series, **cov_test_case)
        explainer = TFTExplainer(model)
        explainer2 = TFTExplainer(
            model,
            background_series=series,
            background_past_covariates=pc if use_pc else None,
            background_future_covariates=fc if use_fc else None,
        )
        assert explainer.background_series == explainer2.background_series
        assert (
            explainer.background_past_covariates
            == explainer2.background_past_covariates
        )
        assert (
            explainer.background_future_covariates
            == explainer2.background_future_covariates
        )

        assert hasattr(explainer, "model")
        assert explainer.background_series[0] == series
        if use_pc:
            assert explainer.background_past_covariates[0] == pc
            assert explainer.background_past_covariates[0].n_components == n_pc_expected
        else:
            assert explainer.background_past_covariates is None
        if use_fc:
            assert explainer.background_future_covariates[0] == fc
            assert (
                explainer.background_future_covariates[0].n_components == n_fc_expected
            )
        else:
            assert explainer.background_future_covariates is None
        result = explainer.explain()
        assert isinstance(result, TFTExplainabilityResult)

        enc_imp = result.get_encoder_importance()
        dec_imp = result.get_decoder_importance()
        stc_imp = result.get_static_covariates_importance()
        imps = [enc_imp, dec_imp, stc_imp]
        assert all([isinstance(imp, pd.DataFrame) for imp in imps])
        # importances must sum up to 100 percent
        assert all([
            imp.squeeze().sum() == pytest.approx(100.0, rel=0.2) for imp in imps
        ])
        # importances must have the expected number of columns
        assert all([
            len(imp.columns) == n
            for imp, n in zip(imps, [n_enc_expected, n_dec_expected, n_sc_expected])
        ])

        attention = result.get_attention()
        assert isinstance(attention, TimeSeries)
        # input chunk length + output chunk length = 5 + 2 = 7
        icl, ocl = 5, 2
        freq = series.freq
        assert len(attention) == icl + ocl
        assert attention.start_time() == series.end_time() - (icl - 1) * freq
        assert attention.end_time() == series.end_time() + ocl * freq
        assert attention.n_components == ocl

        enc_imp_ot = result.get_encoder_importance_over_time()
        dec_imp_ot = result.get_decoder_importance_over_time()
        assert isinstance(enc_imp_ot, TimeSeries)
        assert isinstance(dec_imp_ot, TimeSeries)
        assert len(enc_imp_ot) == icl
        assert enc_imp_ot.start_time() == series.end_time() - (icl - 1) * freq
        assert enc_imp_ot.end_time() == series.end_time()
        assert enc_imp_ot.n_components == n_enc_expected
        assert len(dec_imp_ot) == ocl
        assert dec_imp_ot.start_time() == series.end_time() + freq
        assert dec_imp_ot.end_time() == series.end_time() + ocl * freq
        assert dec_imp_ot.n_components == n_dec_expected
        # importances must sum up to 100 percent at every single timestep (not just on average)
        np.testing.assert_allclose(enc_imp_ot.values().sum(axis=1), 100.0, atol=0.1)
        np.testing.assert_allclose(dec_imp_ot.values().sum(axis=1), 100.0, atol=0.1)

    @pytest.mark.parametrize("test_case", helper_create_test_cases(["multiple"]))
    def test_explainer_multiple_multivariate_series(self, test_case):
        """Test TFTExplainer with multiple multivaraites series and a combination of encoders, covariates,
        and addition of relative index."""
        series_option, cov_option, add_relative_idx, use_encoders = test_case
        series, pc, fc = self.helper_get_input(series_option)
        cov_test_case = dict()
        use_pc, use_fc = False, False
        if "past_covariates" in cov_option:
            cov_test_case["past_covariates"] = pc
            use_pc = True
        if "future_covariates" in cov_option:
            cov_test_case["future_covariates"] = fc
            use_fc = True

        # expected number of features for past covs, future covs, and static covs, and encoder/decoder
        n_target_expected = series[0].n_components
        n_pc_expected = 1 if "past_covariates" in cov_test_case else 0
        n_fc_expected = 1 if "future_covariates" in cov_test_case else 0
        n_sc_expected = 2
        # encoder is number of past and future covs plus 4 optional encodings (future and past)
        # plus 1 univariate target plus 1 optional relative index
        n_enc_expected = (
            n_pc_expected
            + n_fc_expected
            + n_target_expected
            + (4 if use_encoders else 0)
            + (1 if add_relative_idx else 0)
        )
        # encoder is number of future covs plus 2 optional encodings (future)
        # plus 1 optional relative index
        n_dec_expected = (
            n_fc_expected + (2 if use_encoders else 0) + (1 if add_relative_idx else 0)
        )
        model = self.helper_create_model(
            use_encoders=use_encoders, add_relative_idx=add_relative_idx
        )
        # TFTModel requires future covariates
        if (
            not add_relative_idx
            and "future_covariates" not in cov_test_case
            and not use_encoders
        ):
            with pytest.raises(ValueError):
                model.fit(series=series, **cov_test_case)
            return

        model.fit(series=series, **cov_test_case)
        # explainer requires background if model trained on multiple time series
        with pytest.raises(ValueError):
            explainer = TFTExplainer(model)
        explainer = TFTExplainer(
            model,
            background_series=series,
            background_past_covariates=pc if use_pc else None,
            background_future_covariates=fc if use_fc else None,
        )
        assert hasattr(explainer, "model")
        assert explainer.background_series, series
        if use_pc:
            assert explainer.background_past_covariates == pc
            assert explainer.background_past_covariates[0].n_components == n_pc_expected
        else:
            assert explainer.background_past_covariates is None
        if use_fc:
            assert explainer.background_future_covariates == fc
            assert (
                explainer.background_future_covariates[0].n_components == n_fc_expected
            )
        else:
            assert explainer.background_future_covariates is None
        result = explainer.explain()
        assert isinstance(result, TFTExplainabilityResult)

        enc_imp = result.get_encoder_importance()
        dec_imp = result.get_decoder_importance()
        stc_imp = result.get_static_covariates_importance()
        imps = [enc_imp, dec_imp, stc_imp]
        assert all([isinstance(imp, list) for imp in imps])
        assert all([len(imp) == len(series) for imp in imps])
        assert all([isinstance(imp_, pd.DataFrame) for imp in imps for imp_ in imp])
        # importances must sum up to 100 percent
        assert all([
            imp_.squeeze().sum() == pytest.approx(100.0, abs=0.21)
            for imp in imps
            for imp_ in imp
        ])
        # importances must have the expected number of columns
        assert all([
            len(imp_.columns) == n
            for imp, n in zip(imps, [n_enc_expected, n_dec_expected, n_sc_expected])
            for imp_ in imp
        ])

        attention = result.get_attention()
        assert isinstance(attention, list)
        assert len(attention) == len(series)
        assert all([isinstance(att, TimeSeries) for att in attention])
        # input chunk length + output chunk length = 5 + 2 = 7
        icl, ocl = 5, 2
        freq = series[0].freq
        assert all([len(att) == icl + ocl for att in attention])
        assert all([
            att.start_time() == series_.end_time() - (icl - 1) * freq
            for att, series_ in zip(attention, series)
        ])
        assert all([
            att.end_time() == series_.end_time() + ocl * freq
            for att, series_ in zip(attention, series)
        ])
        assert all([att.n_components == ocl for att in attention])

        enc_imp_ot = result.get_encoder_importance_over_time()
        dec_imp_ot = result.get_decoder_importance_over_time()
        assert isinstance(enc_imp_ot, list) and len(enc_imp_ot) == len(series)
        assert isinstance(dec_imp_ot, list) and len(dec_imp_ot) == len(series)
        assert all([isinstance(ts, TimeSeries) for ts in enc_imp_ot])
        assert all([isinstance(ts, TimeSeries) for ts in dec_imp_ot])
        assert all([len(ts) == icl for ts in enc_imp_ot])
        assert all([len(ts) == ocl for ts in dec_imp_ot])
        assert all([ts.n_components == n_enc_expected for ts in enc_imp_ot])
        assert all([ts.n_components == n_dec_expected for ts in dec_imp_ot])
        assert all([
            ts.start_time() == series_.end_time() - (icl - 1) * freq
            for ts, series_ in zip(enc_imp_ot, series)
        ])
        assert all([
            ts.end_time() == series_.end_time() + ocl * freq
            for ts, series_ in zip(dec_imp_ot, series)
        ])
        # importances must sum up to 100 percent at every single timestep (not just on average)
        for ts in enc_imp_ot + dec_imp_ot:
            np.testing.assert_allclose(ts.values().sum(axis=1), 100.0, atol=0.1)

    def test_explain_multiple_prediction_batches(self):
        series, past_covariates, future_covariates = (
            self.helper_get_distinct_batched_input()
        )
        user_callback = _PredictionBatchCallback()
        model = self.helper_create_model(
            batch_size=2,
            callbacks=[user_callback],
        )
        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        explainer = TFTExplainer(
            model,
            background_series=series,
            background_past_covariates=past_covariates,
            background_future_covariates=future_covariates,
        )
        configured_callbacks = tuple(model.trainer_params["callbacks"])
        expected_five_series = None

        for n_series, expected_batch_sizes in [
            (1, [1]),
            (2, [2]),
            (4, [2, 2]),
            (5, [2, 2, 1]),
        ]:
            inputs = {
                "foreground_series": series[:n_series],
                "foreground_past_covariates": past_covariates[:n_series],
                "foreground_future_covariates": future_covariates[:n_series],
            }
            with patch.object(model, "batch_size", n_series):
                expected = explainer.explain(**inputs)

            user_callback.batch_sizes.clear()
            user_callback.collector_batch_counts.clear()
            with patch.object(model, "predict", wraps=model.predict) as predict_mock:
                actual = explainer.explain(**inputs)

            assert predict_mock.call_count == 1
            assert user_callback.batch_sizes == expected_batch_sizes
            assert user_callback.collector_batch_counts == list(
                range(1, len(expected_batch_sizes) + 1)
            )
            assert all(
                not batch_outputs
                for batch_outputs in user_callback.collector._batch_outputs.values()
            )
            assert tuple(model.trainer_params["callbacks"]) == configured_callbacks
            assert user_callback in model.trainer.callbacks
            self.helper_assert_no_tft_collector(model)
            self.helper_assert_tft_results_equal(actual, expected, n_series)

            if n_series == 1:
                assert isinstance(actual.get_attention(), TimeSeries)
            else:
                assert isinstance(actual.get_attention(), list)

            if n_series == 5:
                expected_five_series = expected
                encoder_importance = actual.get_encoder_importance()
                assert [
                    importance.index.tolist() for importance in encoder_importance
                ] == [[idx] for idx in range(n_series)]
                self.helper_assert_distinct_tft_results(actual)

        user_callback.batch_sizes.clear()
        user_callback.collector_batch_counts.clear()
        user_callback.raise_on_batch_idx = 1
        with pytest.raises(RuntimeError, match="user callback failure"):
            explainer.explain(
                foreground_series=series,
                foreground_past_covariates=past_covariates,
                foreground_future_covariates=future_covariates,
            )
        assert user_callback.batch_sizes == [2, 2]
        assert user_callback.collector_batch_counts == [1, 2]
        assert all(
            not batch_outputs
            for batch_outputs in user_callback.collector._batch_outputs.values()
        )
        assert tuple(model.trainer_params["callbacks"]) == configured_callbacks
        assert user_callback in model.trainer.callbacks
        self.helper_assert_no_tft_collector(model)

        user_callback.batch_sizes.clear()
        user_callback.collector_batch_counts.clear()
        user_callback.raise_on_batch_idx = None
        recovered = explainer.explain(
            foreground_series=series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
        )
        assert user_callback.batch_sizes == [2, 2, 1]
        assert user_callback.collector_batch_counts == [1, 2, 3]
        self.helper_assert_no_tft_collector(model)
        self.helper_assert_tft_results_equal(
            recovered, expected_five_series, len(series)
        )

    def test_explain_multiple_prediction_batches_without_static_covariates(self):
        series, past_covariates, future_covariates = (
            self.helper_get_distinct_batched_input(with_static_covariates=False)
        )
        model = self.helper_create_model(batch_size=2)
        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        explainer = TFTExplainer(
            model,
            background_series=series,
            background_past_covariates=past_covariates,
            background_future_covariates=future_covariates,
        )
        inputs = {
            "foreground_series": series,
            "foreground_past_covariates": past_covariates,
            "foreground_future_covariates": future_covariates,
        }

        with patch.object(model, "batch_size", len(series)):
            expected = explainer.explain(**inputs)
        with patch.object(model, "predict", wraps=model.predict) as predict_mock:
            actual = explainer.explain(**inputs)

        assert predict_mock.call_count == 1
        self.helper_assert_no_tft_collector(model)
        self.helper_assert_tft_results_equal(actual, expected, len(series))
        assert all(
            importance.shape == (0, 0)
            for importance in actual.get_static_covariates_importance()
        )

    def test_explain_rejects_incomplete_prediction_output(self):
        series, past_covariates, future_covariates = (
            self.helper_get_distinct_batched_input()
        )
        model = self.helper_create_model(batch_size=2)
        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        explainer = TFTExplainer(
            model,
            background_series=series,
            background_past_covariates=past_covariates,
            background_future_covariates=future_covariates,
        )
        model.trainer_params["limit_predict_batches"] = 1

        with pytest.raises(
            RuntimeError,
            match="TFT prediction returned 2 series for 5 inputs",
        ):
            explainer.explain()
        self.helper_assert_no_tft_collector(model)

    @pytest.mark.parametrize(
        "trainer_overrides",
        [
            {"devices": 2, "strategy": "ddp_spawn"},
            {"devices": 1, "strategy": "ddp_spawn"},
        ],
    )
    def test_explain_rejects_distributed_prediction(self, trainer_overrides):
        model = self.helper_create_model(batch_size=2)
        model.fit(self.series_mv1, past_covariates=self.pc, future_covariates=self.fc)
        explainer = TFTExplainer(model)
        model.trainer_params.update(trainer_overrides)
        model.trainer_params["accelerator"] = "cpu"
        previous_trainer = model.trainer
        created_trainer = None
        init_trainer = model._init_trainer

        def capture_trainer(*args, **kwargs):
            nonlocal created_trainer
            created_trainer = init_trainer(*args, **kwargs)
            return created_trainer

        with (
            patch.object(model, "_init_trainer", side_effect=capture_trainer),
            patch.object(model, "predict", wraps=model.predict) as predict_mock,
            pytest.raises(ValueError, match="only supports single-process prediction"),
        ):
            explainer.explain()

        predict_mock.assert_not_called()
        assert model.trainer is previous_trainer
        self.helper_assert_no_tft_collector(created_trainer)

    def test_prediction_output_collector_validates_cache_shapes_and_presence(self):
        required_attributes = _TFTPredictionOutputCollector._OUTPUT_ATTRIBUTES[:-1]
        pl_module = type("TFTModule", (), {})()
        for name in required_attributes:
            setattr(pl_module, name, torch.ones((2, 1)))
        pl_module._static_covariate_var = None
        outputs = (torch.ones((1, 2, 1, 1)), [], [])
        batch = (torch.ones((2, 1)),)

        with pytest.raises(RuntimeError, match="No TFT explanation outputs"):
            _TFTPredictionOutputCollector().collect(expected_num_series=2)

        collector = _TFTPredictionOutputCollector()
        pl_module._attn_out_weights = "invalid"
        with pytest.raises(TypeError, match="torch.Tensor or None"):
            collector.on_predict_batch_end(None, pl_module, outputs, batch, 0)

        pl_module._attn_out_weights = torch.ones((2, 1))
        collector = _TFTPredictionOutputCollector()
        collector.on_predict_batch_end(None, pl_module, outputs, batch, 0)
        with pytest.raises(RuntimeError, match="collected 2"):
            collector.collect(expected_num_series=3)

        pl_module._static_covariate_var = torch.ones((2, 1))
        collector.on_predict_batch_end(None, pl_module, outputs, batch, 1)
        with pytest.raises(RuntimeError, match="missing from some prediction batches"):
            collector.collect(expected_num_series=4)

        collector = _TFTPredictionOutputCollector()
        pl_module._static_covariate_var = None
        pl_module._attn_out_weights = torch.ones((1, 1))
        with pytest.raises(RuntimeError, match="contain 2 explanation outputs"):
            collector.on_predict_batch_end(None, pl_module, outputs, batch, 0)

        collector = _TFTPredictionOutputCollector()
        pl_module._attn_out_weights = None
        collector.on_predict_batch_end(None, pl_module, outputs, batch, 0)
        with pytest.raises(RuntimeError, match="No `_attn_out_weights` outputs"):
            collector.collect(expected_num_series=2)

    @pytest.mark.parametrize("n_series", [1, 2])
    def test_variable_selection_explanation(self, n_series, mpl_safe_plotting):
        """Test variable selection (feature importance) explanation results and plotting."""
        model = self.helper_create_model(use_encoders=True, add_relative_idx=True)
        series, pc, fc = self.helper_get_input(series_option="multivariate")
        model.fit(series, past_covariates=pc, future_covariates=fc)
        explainer = TFTExplainer(model)
        results = explainer.explain(
            foreground_series=series if n_series == 1 else [series] * 2,
            foreground_past_covariates=pc if n_series == 1 else [pc] * 2,
            foreground_future_covariates=fc if n_series == 1 else [fc] * 2,
        )

        imps = results.get_feature_importances()
        enc_imp = results.get_encoder_importance()
        dec_imp = results.get_decoder_importance()
        stc_imp = results.get_static_covariates_importance()
        imps_direct = [enc_imp, dec_imp, stc_imp]

        # check that all importances are the same across series (since the series have identical values)
        if n_series > 1:
            for imp in imps.values():
                np.testing.assert_array_almost_equal(imp[0].values, imp[1].values)
            for imp in imps_direct:
                np.testing.assert_array_almost_equal(imp[0].values, imp[1].values)

            imps = {k: v[0] for k, v in imps.items()}
            imps_direct = [imp[0] for imp in imps_direct]

        imp_names = [
            "encoder_importance",
            "decoder_importance",
            "static_covariates_importance",
        ]
        assert list(imps.keys()) == imp_names
        for imp, imp_name in zip(imps_direct, imp_names):
            assert imps[imp_name].equals(imp)

        enc_expected = pd.DataFrame(
            {
                "linear_target": 1.7,
                "sine_target": 3.1,
                "add_relative_index_futcov": 3.6,
                "constant_pastcov": 3.9,
                "darts_enc_fc_cyc_month_sin_futcov": 5.0,
                "darts_enc_pc_cyc_month_sin_pastcov": 10.1,
                "darts_enc_pc_cyc_month_cos_pastcov": 19.9,
                "constant_futcov": 21.8,
                "darts_enc_fc_cyc_month_cos_futcov": 31.0,
            },
            index=[0],
        )
        # relaxed comparison because M1 chip gives slightly different results than intel chip
        enc_imp = imps_direct[0]
        assert ((enc_imp.round(decimals=1) - enc_expected).abs() <= 3).all().all()

        dec_expected = pd.DataFrame(
            {
                "darts_enc_fc_cyc_month_sin_futcov": 5.3,
                "darts_enc_fc_cyc_month_cos_futcov": 7.4,
                "constant_futcov": 24.5,
                "add_relative_index_futcov": 62.9,
            },
            index=[0],
        )
        dec_imp = imps_direct[1]
        # relaxed comparison because M1 chip gives slightly different results than intel chip
        assert ((dec_imp.round(decimals=1) - dec_expected).abs() <= 0.6).all().all()

        stc_expected = pd.DataFrame(
            {"num_statcov": 11.9, "cat_statcov": 88.1}, index=[0]
        )
        stc_imp = imps_direct[2]
        # relaxed comparison because M1 chip gives slightly different results than intel chip
        assert ((stc_imp.round(decimals=1) - stc_expected).abs() <= 0.1).all().all()

        enc_imp_ot = results.get_encoder_importance_over_time()
        dec_imp_ot = results.get_decoder_importance_over_time()
        if n_series > 1:
            # check that all importances are the same across series (since the series have identical values)
            np.testing.assert_array_almost_equal(
                enc_imp_ot[0].values(), enc_imp_ot[1].values()
            )
            np.testing.assert_array_almost_equal(
                dec_imp_ot[0].values(), dec_imp_ot[1].values()
            )
            enc_imp_ot = enc_imp_ot[0]
            dec_imp_ot = dec_imp_ot[0]
        assert isinstance(enc_imp_ot, TimeSeries)
        assert isinstance(dec_imp_ot, TimeSeries)
        assert set(enc_imp_ot.components) == set(enc_imp.columns)
        assert set(dec_imp_ot.components) == set(dec_imp.columns)
        # averaging the per-timestep importance over time must recover the (already validated)
        # time-aggregated importance, since both derive from the same underlying softmax weights
        mean_enc_ot = pd.Series(
            enc_imp_ot.values().mean(axis=0), index=enc_imp_ot.components
        )
        mean_dec_ot = pd.Series(
            dec_imp_ot.values().mean(axis=0), index=dec_imp_ot.components
        )
        np.testing.assert_allclose(
            mean_enc_ot[enc_imp.columns].to_numpy(),
            enc_imp.iloc[0].to_numpy(),
            atol=0.1,
        )
        np.testing.assert_allclose(
            mean_dec_ot[dec_imp.columns].to_numpy(),
            dec_imp.iloc[0].to_numpy(),
            atol=0.1,
        )

        figs = explainer.plot_variable_selection(results)
        if n_series == 1:
            figs = [figs]
        for fig in figs:
            assert isinstance(fig, matplotlib.figure.Figure)
            assert len(fig.get_axes()) == 3

    @pytest.mark.parametrize("n_series", [1, 2])
    def test_attention_explanation(self, n_series, mpl_safe_plotting):
        """Test attention (feature importance) explanation results and plotting."""
        # past attention (full_attention=False) on attends to values in the past relative to each horizon
        # (look at the last 0 values in the array)
        att_exp_past_att = np.array([
            [1.0, 0.8],
            [0.8, 0.7],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.9, 0.4],
            [0.0, 1.3],
            [0.0, 0.0],
        ])
        # full attention (full_attention=True) attends to all values in past, present, and future
        # see the that all values are non-0
        att_exp_full_att = np.array([
            [0.8, 0.8],
            [0.7, 0.6],
            [0.4, 0.4],
            [0.3, 0.3],
            [0.3, 0.3],
            [0.7, 0.8],
            [0.8, 0.8],
        ])
        for full_attention, att_exp in zip(
            [False, True], [att_exp_past_att, att_exp_full_att]
        ):
            model = self.helper_create_model(
                use_encoders=True,
                add_relative_idx=True,
                full_attention=full_attention,
            )
            series, pc, fc = self.helper_get_input(series_option="multivariate")
            model.fit(series, past_covariates=pc, future_covariates=fc)
            explainer = TFTExplainer(model)
            results = explainer.explain(
                foreground_series=series if n_series == 1 else [series] * 2,
                foreground_past_covariates=pc if n_series == 1 else [pc] * 2,
                foreground_future_covariates=fc if n_series == 1 else [fc] * 2,
            )

            attns = results.get_attention()
            # relaxed comparison because M1 chip gives slightly different results than intel chip
            if n_series == 1:
                attns = [attns]

            for att in attns:
                assert np.all(
                    np.abs(np.round(att.values(), decimals=1) - att_exp) <= 0.2
                )
                assert att.columns.tolist() == ["horizon 1", "horizon 2"]

            def _check_plot(n_figs_expected, n_axes_expected, **kwargs):
                figs = explainer.plot_attention(results, **kwargs)
                if n_figs_expected == 1:
                    figs = [figs]
                for fig in figs:
                    assert isinstance(fig, matplotlib.figure.Figure)
                    assert isinstance(fig, matplotlib.figure.Figure)
                    assert len(fig.get_axes()) == n_axes_expected

            # only a single axis should be plotted
            _check_plot(n_series, 1, plot_type="all", show_index_as="relative")
            _check_plot(n_series, 1, plot_type="all", show_index_as="time")
            _check_plot(n_series, 1, plot_type="time", show_index_as="relative")
            _check_plot(n_series, 1, plot_type="time", show_index_as="time")
            # heatmap also plot colorbar axis
            _check_plot(n_series, 2, plot_type="heatmap", show_index_as="relative")
            _check_plot(n_series, 2, plot_type="heatmap", show_index_as="time")

            with pytest.raises(ValueError, match="`plot_type` must be either"):
                _check_plot(n_series, 2, plot_type="invalid", show_index_as="time")

    def helper_create_model(
        self,
        use_encoders=True,
        add_relative_idx=True,
        full_attention=False,
        batch_size=32,
        callbacks=None,
    ):
        add_encoders = (
            {"cyclic": {"past": ["month"], "future": ["month"]}}
            if use_encoders
            else None
        )
        model_kwargs = dict(tfm_kwargs)
        trainer_kwargs = dict(model_kwargs["pl_trainer_kwargs"])
        trainer_kwargs["callbacks"] = list(callbacks or [])
        model_kwargs["pl_trainer_kwargs"] = trainer_kwargs
        return TFTModel(
            input_chunk_length=5,
            output_chunk_length=2,
            n_epochs=1,
            batch_size=batch_size,
            add_encoders=add_encoders,
            add_relative_index=add_relative_idx,
            full_attention=full_attention,
            random_state=42,
            **model_kwargs,
        )

    def helper_get_distinct_batched_input(self, with_static_covariates=True):
        inputs = []
        for idx in range(5):
            target = (self.series_mv1 * (idx + 1) + idx).shift(idx * 12)
            if with_static_covariates:
                target = target.with_static_covariates(
                    pd.Series(
                        [idx % 2, idx + 0.25],
                        index=["cat", "num"],
                    )
                )
            else:
                target = target.with_static_covariates(None)
            past_covariate = (self.pc * (idx + 1)).shift(idx * 12)
            future_covariate = (self.fc * (idx + 2) + idx).shift(idx * 12)
            inputs.append((target, past_covariate, future_covariate))

        ordered = [inputs[idx] for idx in [3, 0, 4, 1, 2]]
        return tuple([values[idx] for values in ordered] for idx in range(3))

    @staticmethod
    def helper_as_list(value):
        return value if isinstance(value, list) else [value]

    @classmethod
    def helper_assert_tft_results_equal(cls, actual, expected, n_series):
        time_series_getters = [
            "get_attention",
            "get_encoder_importance_over_time",
            "get_decoder_importance_over_time",
        ]
        dataframe_getters = [
            "get_encoder_importance",
            "get_decoder_importance",
            "get_static_covariates_importance",
        ]

        for getter_name in time_series_getters:
            actual_values = cls.helper_as_list(getattr(actual, getter_name)())
            expected_values = cls.helper_as_list(getattr(expected, getter_name)())
            assert len(actual_values) == len(expected_values) == n_series
            for actual_ts, expected_ts in zip(actual_values, expected_values):
                assert actual_ts.time_index.equals(expected_ts.time_index)
                assert actual_ts.components.equals(expected_ts.components)
                np.testing.assert_allclose(
                    actual_ts.all_values(copy=False),
                    expected_ts.all_values(copy=False),
                    rtol=1e-6,
                    atol=1e-6,
                )

        for getter_name in dataframe_getters:
            actual_values = cls.helper_as_list(getattr(actual, getter_name)())
            expected_values = cls.helper_as_list(getattr(expected, getter_name)())
            assert len(actual_values) == len(expected_values) == n_series
            for actual_df, expected_df in zip(actual_values, expected_values):
                pd.testing.assert_frame_equal(
                    actual_df,
                    expected_df,
                    check_exact=False,
                    rtol=1e-6,
                    atol=1e-6,
                )

    @classmethod
    def helper_assert_distinct_tft_results(cls, result):
        fingerprints = [
            (
                tuple(attention.all_values(copy=False).ravel()),
                tuple(encoder.all_values(copy=False).ravel()),
                tuple(decoder.all_values(copy=False).ravel()),
            )
            for attention, encoder, decoder in zip(
                cls.helper_as_list(result.get_attention()),
                cls.helper_as_list(result.get_encoder_importance_over_time()),
                cls.helper_as_list(result.get_decoder_importance_over_time()),
            )
        ]
        assert len(set(fingerprints)) == len(fingerprints)

    @staticmethod
    def helper_assert_no_tft_collector(model_or_trainer):
        trainer = getattr(model_or_trainer, "trainer", model_or_trainer)
        assert not any(
            isinstance(callback, _TFTPredictionOutputCollector)
            for callback in trainer.callbacks
        )
