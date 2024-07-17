import itertools
from unittest.mock import patch

import matplotlib.pyplot as plt
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
from darts.explainability import TFTExplainabilityResult, TFTExplainer
from darts.models import TFTModel


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

    def test_variable_selection_explanation(self):
        """Test variable selection (feature importance) explanation results and plotting."""
        model = self.helper_create_model(use_encoders=True, add_relative_idx=True)
        series, pc, fc = self.helper_get_input(series_option="multivariate")
        model.fit(series, past_covariates=pc, future_covariates=fc)
        explainer = TFTExplainer(model)
        results = explainer.explain()

        imps = results.get_feature_importances()
        enc_imp = results.get_encoder_importance()
        dec_imp = results.get_decoder_importance()
        stc_imp = results.get_static_covariates_importance()
        imps_direct = [enc_imp, dec_imp, stc_imp]

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
        # relaxed comparison because M1 chip gives slightly different results than intel chip
        assert ((dec_imp.round(decimals=1) - dec_expected).abs() <= 0.6).all().all()

        stc_expected = pd.DataFrame(
            {"num_statcov": 11.9, "cat_statcov": 88.1}, index=[0]
        )
        # relaxed comparison because M1 chip gives slightly different results than intel chip
        assert ((stc_imp.round(decimals=1) - stc_expected).abs() <= 0.1).all().all()

        with patch("matplotlib.pyplot.show") as _:
            _ = explainer.plot_variable_selection(results)

    def test_attention_explanation(self):
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
            results = explainer.explain()

            att = results.get_attention()
            # relaxed comparison because M1 chip gives slightly different results than intel chip
            assert np.all(np.abs(np.round(att.values(), decimals=1) - att_exp) <= 0.2)
            assert att.columns.tolist() == ["horizon 1", "horizon 2"]
            with patch("matplotlib.pyplot.show") as _:
                _ = explainer.plot_attention(
                    results, plot_type="all", show_index_as="relative"
                )
                plt.close()
            with patch("matplotlib.pyplot.show") as _:
                _ = explainer.plot_attention(
                    results, plot_type="all", show_index_as="time"
                )
                plt.close()
            with patch("matplotlib.pyplot.show") as _:
                _ = explainer.plot_attention(
                    results, plot_type="time", show_index_as="relative"
                )
                plt.close()
            with patch("matplotlib.pyplot.show") as _:
                _ = explainer.plot_attention(
                    results, plot_type="time", show_index_as="time"
                )
                plt.close()
            with patch("matplotlib.pyplot.show") as _:
                _ = explainer.plot_attention(
                    results, plot_type="heatmap", show_index_as="relative"
                )
                plt.close()
            with patch("matplotlib.pyplot.show") as _:
                _ = explainer.plot_attention(
                    results, plot_type="heatmap", show_index_as="time"
                )
                plt.close()

    def helper_create_model(
        self, use_encoders=True, add_relative_idx=True, full_attention=False
    ):
        add_encoders = (
            {"cyclic": {"past": ["month"], "future": ["month"]}}
            if use_encoders
            else None
        )
        return TFTModel(
            input_chunk_length=5,
            output_chunk_length=2,
            n_epochs=1,
            add_encoders=add_encoders,
            add_relative_index=add_relative_idx,
            full_attention=full_attention,
            random_state=42,
            **tfm_kwargs,
        )
