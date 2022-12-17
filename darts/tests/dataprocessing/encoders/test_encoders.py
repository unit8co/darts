import copy
import unittest
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.dataprocessing.encoders import (
    FutureCallableIndexEncoder,
    FutureCyclicEncoder,
    FutureDatetimeAttributeEncoder,
    FutureIntegerIndexEncoder,
    PastCallableIndexEncoder,
    PastCyclicEncoder,
    PastDatetimeAttributeEncoder,
    PastIntegerIndexEncoder,
    SequentialEncoder,
)
from darts.dataprocessing.encoders.encoder_base import (
    PastCovariatesIndexGenerator,
    SingleEncoder,
)
from darts.dataprocessing.encoders.encoders import (
    CallableIndexEncoder,
    CyclicTemporalEncoder,
    DatetimeAttributeEncoder,
    IntegerIndexEncoder,
)
from darts.dataprocessing.transformers import Scaler
from darts.logging import get_logger, raise_log
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    from darts.models import TFTModel

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not installed - will be skipping Torch models tests")
    TORCH_AVAILABLE = False


class EncoderTestCase(DartsBaseTestClass):
    encoders_cls = [
        FutureCallableIndexEncoder,
        FutureCyclicEncoder,
        FutureDatetimeAttributeEncoder,
        FutureIntegerIndexEncoder,
        PastCallableIndexEncoder,
        PastCyclicEncoder,
        PastDatetimeAttributeEncoder,
        PastIntegerIndexEncoder,
    ]
    n_target_1 = 12
    n_target_2 = 24
    shift = 50
    target_1 = tg.linear_timeseries(length=n_target_1, freq="MS")
    target_2 = tg.linear_timeseries(
        start=target_1.end_time() + shift * target_1.freq, length=n_target_2, freq="MS"
    )
    covariates_1 = tg.linear_timeseries(length=2 * n_target_1, freq="MS")
    covariates_2 = tg.linear_timeseries(
        start=target_1.end_time() + shift * target_1.freq,
        length=2 * n_target_2,
        freq="MS",
    )

    target_multi = [target_1, target_2]
    covariates_multi = [covariates_1, covariates_2]

    input_chunk_length = 12
    output_chunk_length = 6
    n_short = 6
    n_long = 8

    # for the given input_chunk_length, ..., n_long from above, the time_index of the expected encoded covariates
    # multi-TS at prediction should be as follows
    inf_ts_short_future = [
        TimeSeries.from_times_and_values(
            tg.generate_index(
                start=ts.end_time() + (1 - 12) * ts.freq, length=12 + 6, freq=ts.freq
            ),
            np.arange(12 + 6),
        )
        for ts in target_multi
    ]

    inf_ts_long_future = [
        TimeSeries.from_times_and_values(
            tg.generate_index(
                start=ts.end_time() + (1 - 12) * ts.freq, length=12 + 8, freq=ts.freq
            ),
            np.arange(12 + 8),
        )
        for ts in target_multi
    ]

    inf_ts_short_past = [
        TimeSeries.from_times_and_values(
            tg.generate_index(
                start=ts.end_time() + (1 - 12) * ts.freq, length=12, freq=ts.freq
            ),
            np.arange(12),
        )
        for ts in target_multi
    ]

    inf_ts_long_past = [
        TimeSeries.from_times_and_values(
            tg.generate_index(
                start=ts.end_time() + (1 - 12) * ts.freq,
                length=12 + (8 - 6),
                freq=ts.freq,
            ),
            np.arange(12 + (8 - 6)),
        )
        for ts in target_multi
    ]

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "Torch not available. SequentialEncoder tests with models will be skipped.",
    )
    def test_sequence_encoder_from_model_params(self):
        """test if sequence encoder is initialized properly from model params"""
        # valid encoder model parameters are ('past', 'future') for the main key and datetime attribute for sub keys
        valid_encoder_args = {
            "cyclic": {"past": ["month"], "future": ["dayofyear", "dayofweek"]}
        }
        encoders = self.helper_encoder_from_model(add_encoder_dict=valid_encoder_args)

        self.assertTrue(len(encoders.past_encoders) == 1)
        self.assertTrue(len(encoders.future_encoders) == 2)

        # test if encoders have the correct attributes
        self.assertTrue(encoders.past_encoders[0].attribute == "month")
        self.assertTrue(
            [enc.attribute for enc in encoders.future_encoders]
            == ["dayofyear", "dayofweek"]
        )

        valid_encoder_args = {"cyclic": {"past": ["month"]}}
        encoders = self.helper_encoder_from_model(
            add_encoder_dict=valid_encoder_args, takes_future_covariates=False
        )
        self.assertTrue(len(encoders.past_encoders) == 1)
        self.assertTrue(len(encoders.future_encoders) == 0)

        # test invalid encoder kwarg at model creation
        bad_encoder = {"no_encoder": {"past": ["month"]}}
        with self.assertRaises(ValueError):
            _ = self.helper_encoder_from_model(add_encoder_dict=bad_encoder)

        # test invalid kwargs at model creation
        bad_time = {"cyclic": {"ppast": ["month"]}}
        with self.assertRaises(ValueError):
            _ = self.helper_encoder_from_model(add_encoder_dict=bad_time)

        bad_attribute = {"cyclic": {"past": ["year"]}}
        with self.assertRaises(ValueError):
            _ = self.helper_encoder_from_model(add_encoder_dict=bad_attribute)

        bad_type = {"cyclic": {"past": 1}}
        with self.assertRaises(ValueError):
            _ = self.helper_encoder_from_model(add_encoder_dict=bad_type)

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "Torch not available. SequentialEncoder tests with models will be skipped.",
    )
    def test_encoder_sequence_train(self):
        """Test `SequentialEncoder.encode_train()` output"""
        # ====> Sequential Cyclic Encoder Tests <====
        encoder_args = {"cyclic": {"past": ["month"], "future": ["month", "month"]}}
        encoders = self.helper_encoder_from_model(add_encoder_dict=encoder_args)

        # ==> test training <==
        past_covs_train, future_covs_train = encoders.encode_train(
            target=self.target_multi,
            past_covariates=self.covariates_multi,
            future_covariates=self.covariates_multi,
        )

        # encoded multi TS covariates should have same number as input covariates
        self.assertEqual(len(past_covs_train), 2)
        self.assertEqual(len(future_covs_train), 2)

        # each attribute (i.e., 'month', ...) generates 2 output variables (+ 1 covariates from input covariates)
        self.assertEqual(past_covs_train[0].n_components, 3)
        self.assertEqual(future_covs_train[0].n_components, 5)

        # check with different inputs
        encoder_args = {"cyclic": {"past": ["month"], "future": ["month"]}}
        encoders = self.helper_encoder_from_model(add_encoder_dict=encoder_args)

        # ==> test training <==
        past_covs_train, future_covs_train = encoders.encode_train(
            target=self.target_multi,
            past_covariates=self.covariates_multi,
            future_covariates=self.covariates_multi,
        )

        # encoded multi TS covariates should have same number as input covariates
        self.assertEqual(len(past_covs_train), 2)
        self.assertEqual(len(future_covs_train), 2)

        # each attribute (i.e., 'month', ...) generates 2 output variables (+ 1 covariates from input covariates)
        self.assertEqual(past_covs_train[0].n_components, 3)
        self.assertEqual(future_covs_train[0].n_components, 3)

        # encoded past covariates must have equal index as input past covariates
        for pc, pc_in in zip(past_covs_train, self.covariates_multi):
            self.assertTrue(pc.time_index.equals(pc_in.time_index))

        # encoded future covariates must have equal index as input future covariates
        for fc, fc_in in zip(future_covs_train, self.covariates_multi):
            self.assertTrue(fc.time_index.equals(fc_in.time_index))

        # for training dataset: both encoded past and future covariates with cyclic encoder 'month' should be equal
        # (apart from component names)
        for pc, fc in zip(past_covs_train, future_covs_train):
            self.assertEqual(
                pc.with_columns_renamed(
                    list(pc.components), [f"comp{i}" for i in range(len(pc.components))]
                ),
                fc.with_columns_renamed(
                    list(fc.components), [f"comp{i}" for i in range(len(fc.components))]
                ),
            )

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "Torch not available. SequentialEncoder tests with models will be skipped.",
    )
    def test_encoder_sequence_inference(self):
        """Test `SequentialEncoder.encode_inference()` output"""
        # ==> test prediction <==
        encoder_args = {"cyclic": {"past": ["month"], "future": ["month"]}}
        encoders = self.helper_encoder_from_model(add_encoder_dict=encoder_args)

        # tests with n <= output_chunk_length
        # with supplying past and future covariates as input
        self.helper_sequence_encode_inference(
            encoders=encoders,
            n=self.n_short,
            past_covariates=self.covariates_multi,
            future_covariates=self.covariates_multi,
            expected_past_idx_ts=self.covariates_multi,
            expected_future_idx_ts=self.covariates_multi,
        )
        # without supplying covariates as input
        self.helper_sequence_encode_inference(
            encoders=encoders,
            n=self.n_short,
            past_covariates=None,
            future_covariates=None,
            expected_past_idx_ts=self.inf_ts_short_past,
            expected_future_idx_ts=self.inf_ts_short_future,
        )
        # tests with n > output_chunk_length
        # with supplying past covariates as input
        self.helper_sequence_encode_inference(
            encoders=encoders,
            n=self.n_long,
            past_covariates=self.covariates_multi,
            future_covariates=None,
            expected_past_idx_ts=self.covariates_multi,
            expected_future_idx_ts=self.inf_ts_long_future,
        )
        # with supplying future covariates as input
        self.helper_sequence_encode_inference(
            encoders=encoders,
            n=self.n_long,
            past_covariates=None,
            future_covariates=self.covariates_multi,
            expected_past_idx_ts=self.inf_ts_long_past,
            expected_future_idx_ts=self.covariates_multi,
        )

    def helper_sequence_encode_inference(
        self,
        encoders,
        n,
        past_covariates,
        future_covariates,
        expected_past_idx_ts,
        expected_future_idx_ts,
    ):
        """test comparisons for `SequentialEncoder.encode_inference()"""

        # generate encodings
        past_covs_pred, future_covs_pred = encoders.encode_inference(
            n=n,
            target=self.target_multi,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        # encoded past and future covariates must have equal index as expected past and future
        for pc, pc_in in zip(past_covs_pred, expected_past_idx_ts):
            self.assertTrue(pc.time_index.equals(pc_in.time_index))
        for fc, fc_in in zip(future_covs_pred, expected_future_idx_ts):
            self.assertTrue(fc.time_index.equals(fc_in.time_index))

    def helper_encoder_from_model(
        self, add_encoder_dict, takes_past_covariates=True, takes_future_covariates=True
    ):
        """extracts encoders from parameters at model creation"""
        model = TFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            add_encoders=add_encoder_dict,
        )

        encoders = model.initialize_encoders()
        # see if encoding works
        _ = encoders.encode_train(
            self.target_multi, self.covariates_multi, self.covariates_multi
        )
        _ = encoders.encode_inference(
            3, self.target_multi, self.covariates_multi, self.covariates_multi
        )
        return encoders

    def test_single_encoders_general(self):
        ts = tg.linear_timeseries(length=24, freq="MS")
        covs = tg.linear_timeseries(length=24, freq="MS")

        input_chunk_length = 12
        output_chunk_length = 6

        for enc_cls in self.encoders_cls:
            if issubclass(enc_cls, CyclicTemporalEncoder):
                attr = "month"
                comps_expected = ["cyc_month_sin", "cyc_month_cos"]
                requires_fit = False
            elif issubclass(enc_cls, DatetimeAttributeEncoder):
                attr = "month"
                comps_expected = ["dta_month"]
                requires_fit = False
            elif issubclass(enc_cls, IntegerIndexEncoder):
                attr = "relative"
                comps_expected = ["pos_relative"]
                requires_fit = True
            elif issubclass(enc_cls, CallableIndexEncoder):

                def some_f(idx):
                    return idx.month

                attr = some_f
                comps_expected = ["cus_custom"]
                requires_fit = False
            else:
                attr, comps_expected, requires_fit = None, None, False
                raise_log(ValueError("unknown encoder class"), logger=logger)

            enc = enc_cls(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                attribute=attr,
            )
            if isinstance(enc.index_generator, PastCovariatesIndexGenerator):
                base_comp_name = "darts_enc_pc_"
            else:
                base_comp_name = "darts_enc_fc_"
            comps_expected = pd.Index(
                [base_comp_name + comp_name for comp_name in comps_expected]
            )

            self.assertTrue(not enc.fit_called)
            # initially, no components
            self.assertTrue(enc.components.empty)

            # some encoders must be fit before encoding inference part
            self.assertTrue(requires_fit == enc.requires_fit)
            if requires_fit:
                with pytest.raises(ValueError):
                    enc.encode_inference(n=1, target=ts, covariates=covs)

            def test_routine(encoder, merge_covs: bool, comps_expected: pd.Index):
                """checks general behavior for `encode_train` and `encode_inference` with and without merging the
                output with original covariates"""
                n = 1
                covs_train = encoder.encode_train(
                    target=ts, covariates=covs, merge_covariates=merge_covs
                )
                assert covs_train.end_time() == ts.end_time()

                # check the encoded component names
                self.assertTrue(encoder.components.equals(comps_expected))
                if not merge_covs:
                    self.assertTrue(covs_train.components.equals(comps_expected))
                else:
                    self.assertTrue(comps_expected.isin(covs_train.components).all())
                    # check that original components are in output when merging
                    self.assertTrue(covs_train[list(covs.components)] == covs)

                # check the same for inference
                covs_inf = encoder.encode_inference(
                    n=n, target=ts, covariates=covs, merge_covariates=merge_covs
                )
                # if we give input `covs` the encoder will use the same index as `covs`
                assert covs_inf.end_time() == covs.end_time()
                self.assertTrue(encoder.components.equals(comps_expected))
                if not merge_covs:
                    self.assertTrue(covs_inf.components.equals(comps_expected))
                else:
                    self.assertTrue(comps_expected.isin(covs_inf.components).all())
                    self.assertTrue(covs_inf[list(covs.components)] == covs)

                # we can use the output of `encode_train()` as input for `encode_train()` and get the same
                # results (encoded components get overwritten)
                covs_train2 = encoder.encode_train(
                    target=ts, covariates=covs_train, merge_covariates=merge_covs
                )
                if merge_covs:
                    assert covs_train2 == covs_train
                else:
                    overlap = covs_train.slice_intersect(covs_train2)
                    assert covs_train2 == overlap

                # we can use the output of `encode_train()` as input for `encode_inference()`. The encoded components
                # are dropped internally and appended again at the end
                covs_inf2 = encoder.encode_inference(
                    n=n, target=ts, covariates=covs_train, merge_covariates=merge_covs
                )
                # We get the same results with `merge_covariates=True` (as original covariates will still be in input)
                if merge_covs:
                    assert covs_inf2 == covs_inf
                # We get only the minimum required time spans with `merge_covariates=False` as input covariates will
                # are not in output of `encode_train()/inference()`
                else:
                    overlap = covs_inf.slice_intersect(covs_inf2)
                    if isinstance(
                        encoder.index_generator, PastCovariatesIndexGenerator
                    ):
                        assert len(covs_inf2) == input_chunk_length
                        assert covs_inf2.end_time() == ts.end_time()
                        assert covs_inf2 == overlap
                    else:
                        assert (
                            len(covs_inf2) == input_chunk_length + output_chunk_length
                        )
                        assert (
                            covs_inf2.end_time()
                            == ts.end_time() + ts.freq * output_chunk_length
                        )
                        overlap_inf = covs_inf2.slice_intersect(overlap)
                        assert overlap_inf == overlap

                # we can use the output of `encode_inference()` as input for `encode_inference()` and get the
                # same results (encoded components get overwritten)
                covs_inf3 = encoder.encode_inference(
                    n=1, target=ts, covariates=covs_inf2, merge_covariates=merge_covs
                )
                assert covs_inf3 == covs_inf2

            test_routine(
                copy.deepcopy(enc), merge_covs=False, comps_expected=comps_expected
            )
            test_routine(
                copy.deepcopy(enc), merge_covs=True, comps_expected=comps_expected
            )

    def test_sequential_encoder_general(self):
        ts = tg.linear_timeseries(length=24, freq="MS")
        covs = tg.linear_timeseries(length=24, freq="MS")

        input_chunk_length = 12
        output_chunk_length = 6
        add_encoders = {
            "cyclic": {"past": ["month", "day"], "future": ["day", "month"]},
            "datetime_attribute": {
                "past": ["month", "year"],
                "future": ["year", "month"],
            },
            "position": {
                "past": ["relative"],
                "future": ["relative"],
            },
            "custom": {
                "past": [lambda idx: idx.month, lambda idx: idx.year],
                "future": [lambda idx: idx.month, lambda idx: idx.year],
            },
            "transformer": Scaler(),
        }
        # given `add_encoders` dict, we expect encoders to generate the following components
        comps_expected_past = pd.Index(
            [
                "darts_enc_pc_cyc_month_sin",
                "darts_enc_pc_cyc_month_cos",
                "darts_enc_pc_cyc_day_sin",
                "darts_enc_pc_cyc_day_cos",
                "darts_enc_pc_dta_month",
                "darts_enc_pc_dta_year",
                "darts_enc_pc_pos_relative",
                "darts_enc_pc_cus_custom",
                "darts_enc_pc_cus_custom_1",
            ]
        )
        comps_expected_future = pd.Index(
            [
                "darts_enc_fc_cyc_day_sin",
                "darts_enc_fc_cyc_day_cos",
                "darts_enc_fc_cyc_month_sin",
                "darts_enc_fc_cyc_month_cos",
                "darts_enc_fc_dta_year",
                "darts_enc_fc_dta_month",
                "darts_enc_fc_pos_relative",
                "darts_enc_fc_cus_custom",
                "darts_enc_fc_cus_custom_1",
            ]
        )
        kwargs = {
            "add_encoders": add_encoders,
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "takes_future_covariates": True,
            "takes_past_covariates": True,
        }
        kwargs_copy = copy.deepcopy(kwargs)

        # with `position` encoder, we have to call `encode_train()` before inference set
        kwargs_copy["add_encoders"] = {
            k: val
            for k, val in kwargs["add_encoders"].items()
            if k not in ["transformer"]
        }
        enc = SequentialEncoder(**kwargs_copy)
        assert enc.requires_fit
        # inference directly does not work
        with pytest.raises(ValueError):
            _ = enc.encode_inference(
                n=1, target=ts, past_covariates=covs, future_covariates=covs
            )
        # train first then inference does work
        _ = enc.encode_train(target=ts, past_covariates=covs, future_covariates=covs)
        _ = enc.encode_inference(
            n=1, target=ts, past_covariates=covs, future_covariates=covs
        )

        # with `transformer`, we have to call `encode_train()` before inference set
        kwargs_copy["add_encoders"] = {
            k: val for k, val in kwargs["add_encoders"].items() if k not in ["position"]
        }
        enc = SequentialEncoder(**kwargs_copy)
        assert enc.requires_fit
        # inference directly does not work
        with pytest.raises(ValueError):
            _ = enc.encode_inference(
                n=1, target=ts, past_covariates=covs, future_covariates=covs
            )
        # train first then inference does work
        _ = enc.encode_train(target=ts, past_covariates=covs, future_covariates=covs)
        _ = enc.encode_inference(
            n=1, target=ts, past_covariates=covs, future_covariates=covs
        )

        # without position encoder and transformer, we can directly encode the inference set
        kwargs_copy["add_encoders"] = {
            k: val
            for k, val in kwargs["add_encoders"].items()
            if k not in ["position", "transformer"]
        }
        enc = SequentialEncoder(**kwargs_copy)
        assert not enc.requires_fit
        _ = enc.encode_inference(
            n=1, target=ts, past_covariates=covs, future_covariates=covs
        )

        # ==> test `encode_train()` with all encoders and transformer
        enc = SequentialEncoder(**kwargs)
        assert not enc.fit_called
        assert enc.requires_fit
        pc, fc = enc.encode_train(
            target=ts, past_covariates=covs, future_covariates=covs
        )

        assert enc.past_components.equals(comps_expected_past)
        assert comps_expected_past.isin(pc.components).all()
        assert covs.components.isin(pc.components).all()

        assert enc.future_components.equals(comps_expected_future)
        assert comps_expected_future.isin(fc.components).all()
        assert covs.components.isin(fc.components).all()

        # we can also take the output from a previous `encode_train` call as input covariates.
        # the encoded components get overwritten by the second call
        pc_train, fc_train = enc.encode_train(
            target=ts, past_covariates=pc, future_covariates=fc
        )
        assert pc_train == pc
        assert fc_train == fc

        # ==> test `encode_inference()` with all encoders and transformer
        assert enc.fit_called
        pc, fc = enc.encode_inference(
            n=1, target=ts, past_covariates=covs, future_covariates=covs
        )
        assert enc.past_components.equals(comps_expected_past)
        assert comps_expected_past.isin(pc.components).all()
        assert covs.components.isin(pc.components).all()

        assert enc.future_components.equals(comps_expected_future)
        assert comps_expected_future.isin(fc.components).all()
        assert covs.components.isin(fc.components).all()

        # we can also take the output from `encode_train()` as input covariates to `encode_inference()` to get the
        # same results
        pc_inf, fc_inf = enc.encode_inference(
            n=1, target=ts, past_covariates=pc_train, future_covariates=fc_train
        )
        assert pc_inf == pc
        assert fc_inf == fc

        # or take the output from `encode_inference()` and get the same results
        pc_inf2, fc_inf2 = enc.encode_inference(
            n=1, target=ts, past_covariates=pc, future_covariates=fc
        )
        assert pc_inf2 == pc
        assert fc_inf2 == fc

    def test_cyclic_encoder(self):
        """Test past and future `CyclicTemporalEncoder``"""

        attribute = "month"
        month_series = TimeSeries.from_times_and_values(
            times=tg.generate_index(
                start=pd.to_datetime("2000-01-01"), length=24, freq="MS"
            ),
            values=np.arange(24),
        )
        encoder = FutureCyclicEncoder(
            input_chunk_length=1, output_chunk_length=1, attribute="month"
        )
        first_halve = encoder.encode_train(
            target=month_series[:12],
            covariates=month_series[:12],
            merge_covariates=False,
        )
        second_halve = encoder.encode_train(
            target=month_series[12:],
            covariates=month_series[12:],
            merge_covariates=False,
        )

        # check if encoded values for first 12 months are equal to values of last 12 months
        self.assertTrue((first_halve.values() == second_halve.values()).all())

        # test past cyclic encoder
        # pc: past covariates
        expected_components = [
            "darts_enc_pc_cyc_month_sin",
            "darts_enc_pc_cyc_month_cos",
        ]
        self.helper_test_cyclic_encoder(
            PastCyclicEncoder,
            attribute=attribute,
            inf_ts_short=self.inf_ts_short_past,
            inf_ts_long=self.inf_ts_long_past,
            cyclic=True,
            expected_components=expected_components,
        )
        # test future cyclic encoder
        # fc: future covariates
        expected_components = [
            "darts_enc_fc_cyc_month_sin",
            "darts_enc_fc_cyc_month_cos",
        ]
        self.helper_test_cyclic_encoder(
            FutureCyclicEncoder,
            attribute=attribute,
            inf_ts_short=self.inf_ts_short_future,
            inf_ts_long=self.inf_ts_long_future,
            cyclic=True,
            expected_components=expected_components,
        )

    def test_datetime_attribute_encoder(self):
        """Test past and future `DatetimeAttributeEncoder`"""

        attribute = "month"

        month_series = TimeSeries.from_times_and_values(
            times=tg.generate_index(
                start=pd.to_datetime("2000-01-01"), length=24, freq="MS"
            ),
            values=np.arange(24),
        )

        encoder = FutureDatetimeAttributeEncoder(
            input_chunk_length=1, output_chunk_length=1, attribute="month"
        )
        first_halve = encoder.encode_train(
            target=month_series[:12],
            covariates=month_series[:12],
            merge_covariates=False,
        )
        second_halve = encoder.encode_train(
            target=month_series[12:],
            covariates=month_series[12:],
            merge_covariates=False,
        )

        # check if encoded values for first 12 months are equal to values of last 12 months
        self.assertTrue((first_halve.values() == second_halve.values()).all())

        # test past cyclic encoder
        expected_components = "darts_enc_pc_dta_month"
        self.helper_test_cyclic_encoder(
            PastDatetimeAttributeEncoder,
            attribute=attribute,
            inf_ts_short=self.inf_ts_short_past,
            inf_ts_long=self.inf_ts_long_past,
            cyclic=False,
            expected_components=expected_components,
        )

        # test future cyclic encoder
        expected_components = "darts_enc_fc_dta_month"
        self.helper_test_cyclic_encoder(
            FutureDatetimeAttributeEncoder,
            attribute=attribute,
            inf_ts_short=self.inf_ts_short_future,
            inf_ts_long=self.inf_ts_long_future,
            cyclic=False,
            expected_components=expected_components,
        )

    def test_integer_positional_encoder(self):
        """Test past `IntegerIndexEncoder`"""

        ts = tg.linear_timeseries(length=24, freq="MS")
        input_chunk_length = 12
        output_chunk_length = 6
        # ===> test relative position encoder <===
        encoder_params = {"position": {"past": ["relative"], "future": ["relative"]}}
        encs = SequentialEncoder(
            add_encoders=encoder_params,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            takes_past_covariates=True,
            takes_future_covariates=True,
        )
        # relative encoder takes the end of the training series as reference
        vals = np.arange(-len(ts) + 1, 1).reshape((len(ts), 1))

        pc1, fc1 = encs.encode_train(ts)
        self.assertTrue(
            pc1.time_index.equals(ts.time_index[:-output_chunk_length])
            and (pc1.values() == vals[:-output_chunk_length]).all()
        )
        self.assertTrue(
            fc1.time_index.equals(ts.time_index) and (fc1.values() == vals).all()
        )

        pc2, fc2 = encs.encode_train(
            TimeSeries.from_times_and_values(
                ts.time_index[:20] + ts.freq, ts[:20].values()
            )
        )
        self.assertTrue(
            (pc2.time_index.equals(ts.time_index[: 20 - output_chunk_length] + ts.freq))
            and (pc2.values() == vals[-20:-output_chunk_length]).all()
        )
        self.assertTrue(
            fc2.time_index.equals(ts.time_index[:20] + ts.freq)
            and (fc2.values() == vals[-20:]).all()
        )

        pc3, fc3 = encs.encode_train(
            TimeSeries.from_times_and_values(
                ts.time_index[:18] - ts.freq, ts[:18].values()
            )
        )
        self.assertTrue(
            pc3.time_index.equals(ts.time_index[: 18 - output_chunk_length] - ts.freq)
            and (pc3.values() == vals[-18:-output_chunk_length]).all()
        )
        self.assertTrue(
            fc3.time_index.equals(ts.time_index[:18] - ts.freq)
            and (fc3.values() == vals[-18:]).all()
        )

        # quickly test inference encoding
        # n > output_chunk_length
        n = output_chunk_length + 1
        pc4, fc4 = encs.encode_inference(n, ts)
        self.assertTrue(
            (pc4.univariate_values() == np.arange(-input_chunk_length + 1, 1 + 1)).all()
        )
        self.assertTrue(
            (fc4.univariate_values() == np.arange(-input_chunk_length + 1, 1 + n)).all()
        )
        # n <= output_chunk_length
        n = output_chunk_length - 1
        t5, fc5 = encs.encode_inference(
            n,
            TimeSeries.from_times_and_values(
                ts.time_index[:20] + ts.freq, ts[:20].values()
            ),
        )
        self.assertTrue(
            (t5.univariate_values() == np.arange(-input_chunk_length + 1, 0 + 1)).all()
        )
        self.assertTrue(
            (
                fc5.univariate_values()
                == np.arange(-input_chunk_length + 1, output_chunk_length + 1)
            ).all()
        )

        # quickly test with lags
        min_pc_lag = -input_chunk_length - 2  # = -14
        max_pc_lag = -6
        min_fc_lag = 2
        max_fc_lag = 8
        encs = SequentialEncoder(
            add_encoders=encoder_params,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            takes_past_covariates=True,
            takes_future_covariates=True,
            lags_past_covariates=[min_pc_lag, max_pc_lag],
            lags_future_covariates=[min_fc_lag, max_fc_lag],
        )
        pc1, fc1 = encs.encode_train(ts)
        self.assertTrue(
            pc1.start_time() == pd.Timestamp("1999-11-01", freq=ts.freq)
            and pc1.end_time() == pd.Timestamp("2001-01-01", freq=ts.freq)
            and (pc1.univariate_values() == np.arange(-25, -10)).all()
            and pc1[ts.start_time()].univariate_values()[0] == -23
        )
        self.assertTrue(
            fc1.start_time() == pd.Timestamp("2001-03-01", freq=ts.freq)
            and fc1.end_time() == pd.Timestamp("2002-03-01", freq=ts.freq)
            and (fc1.univariate_values() == np.arange(-9, 4)).all()
            and fc1[ts.end_time()].univariate_values()[0] == 0
        )

        n = 2
        pc2, fc2 = encs.encode_inference(n=n, target=ts)
        self.assertTrue(
            pc2.start_time() == pd.Timestamp("2000-11-01", freq=ts.freq)
            and pc2.end_time() == pd.Timestamp("2001-07-01", freq=ts.freq)
            and (pc2.univariate_values() == np.arange(-13, -4)).all()
        )
        self.assertTrue(
            fc2.start_time() == pd.Timestamp("2002-03-01", freq=ts.freq)
            and fc2.end_time() == pd.Timestamp("2002-09-01", freq=ts.freq)
            and (fc2.univariate_values() == np.arange(3, 10)).all()
        )

    def test_callable_encoder(self):
        """Test `CallableIndexEncoder`"""
        ts = tg.linear_timeseries(length=24, freq="A")
        input_chunk_length = 12
        output_chunk_length = 6

        # ===> test callable index encoder <===
        encoder_params = {
            "custom": {
                "past": [lambda index: index.year, lambda index: index.year - 1],
                "future": [lambda index: index.year],
            }
        }
        encs = SequentialEncoder(
            add_encoders=encoder_params,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            takes_past_covariates=True,
            takes_future_covariates=True,
        )

        # train set
        pc, fc = encs.encode_train(ts)
        # past covariates
        np.testing.assert_array_equal(
            ts[:-output_chunk_length].time_index.year.values, pc.values()[:, 0]
        )
        np.testing.assert_array_equal(
            ts[:-output_chunk_length].time_index.year.values - 1, pc.values()[:, 1]
        )
        # future covariates
        np.testing.assert_array_equal(ts.time_index.year.values, fc.values()[:, 0])

        # inference set
        pc, fc = encs.encode_inference(n=12, target=ts)
        year_index = tg.generate_index(
            start=ts.end_time() - ts.freq * (input_chunk_length - 1),
            length=24,
            freq=ts.freq,
        )
        # past covariates
        np.testing.assert_array_equal(
            year_index[:-output_chunk_length].year.values, pc.values()[:, 0]
        )
        np.testing.assert_array_equal(
            year_index[:-output_chunk_length].year.values - 1, pc.values()[:, 1]
        )
        # future covariates
        np.testing.assert_array_equal(year_index.year.values, fc.values()[:, 0])

    def test_transformer_single_series(self):
        def test_routine_cyclic(past_covs):
            for curve in ["sin", "cos"]:
                self.assertAlmostEqual(
                    past_covs[f"darts_enc_pc_cyc_minute_{curve}"]
                    .all_values(copy=False)
                    .min(),
                    -1.0,
                    delta=1e-9,
                )
                self.assertAlmostEqual(
                    past_covs[f"darts_enc_pc_cyc_minute_{curve}"]
                    .values(copy=False)
                    .max(),
                    1.0,
                    delta=0.1e-9,
                )

        ts1 = tg.linear_timeseries(
            start_value=1, end_value=2, length=60, freq="T", column_name="cov_in"
        )
        encoder_params = {
            "position": {"future": ["relative"]},
            "cyclic": {"past": ["minute"]},
            "transformer": Scaler(),
        }

        encs = SequentialEncoder(
            add_encoders=encoder_params,
            input_chunk_length=12,
            output_chunk_length=6,
            takes_past_covariates=True,
            takes_future_covariates=True,
        )

        pc1, fc1 = encs.encode_train(ts1, future_covariates=ts1)

        # ===> train set test <===
        # user supplied covariates should not be transformed
        self.assertTrue(fc1["cov_in"] == ts1)
        # cyclic encodings should not be transformed
        test_routine_cyclic(pc1)
        # all others should be transformed to values between 0 and 1
        self.assertAlmostEqual(
            fc1["darts_enc_fc_pos_relative"].values(copy=False).min(), 0.0, delta=10e-9
        )
        self.assertAlmostEqual(
            fc1["darts_enc_fc_pos_relative"].values(copy=False).max(), 1.0, delta=10e-9
        )

        # ===> validation set test <===
        ts2 = tg.linear_timeseries(
            start_value=1,
            end_value=2,
            start=ts1.end_time(),
            length=60,
            freq=ts1.freq,
            column_name="cov_in",
        )
        pc2, fc2 = encs.encode_train(ts2, future_covariates=ts2)
        # cyclic encodings should not be transformed
        test_routine_cyclic(pc2)
        # make sure that when calling encoders the second time, scalers are not fit again (for validation and inference)
        self.assertAlmostEqual(
            fc2["darts_enc_fc_pos_relative"].values(copy=False).min(), 0.0, delta=10e-9
        )
        self.assertAlmostEqual(
            fc2["darts_enc_fc_pos_relative"].values(copy=False).max(), 1.0, delta=10e-9
        )

        fc_inf = tg.linear_timeseries(
            start_value=1, end_value=3, length=80, freq="T", column_name="cov_in"
        )
        pc3, fc3 = encs.encode_inference(n=60, target=ts1, future_covariates=fc_inf)

        # cyclic encodings should not be transformed
        test_routine_cyclic(pc3)
        # index 0 is also start of train target series and value should be 0
        self.assertAlmostEqual(fc3["darts_enc_fc_pos_relative"][0].values()[0, 0], 0.0)
        # index len(ts1) - 1 is the prediction point and value should be 0
        self.assertAlmostEqual(
            fc3["darts_enc_fc_pos_relative"][len(ts1) - 1].values()[0, 0], 1.0
        )
        # the future should scale proportional to distance to prediction point
        self.assertAlmostEqual(
            fc3["darts_enc_fc_pos_relative"][80 - 1].values()[0, 0], 80 / 60, delta=0.01
        )

    def test_transformer_multi_series(self):
        ts1 = tg.linear_timeseries(
            start_value=1, end_value=2, length=21, freq="T", column_name="cov"
        )
        ts2 = tg.linear_timeseries(
            start=None,
            end=ts1.end_time(),
            start_value=1.5,
            end_value=2,
            length=11,
            freq="T",
            column_name="cov",
        )
        ts1_inf = ts1.drop_before(ts2.start_time() - ts1.freq)
        ts2_inf = ts2
        encoder_params = {
            "datetime_attribute": {"past": ["minute"], "future": ["minute"]},
            "position": {"future": ["relative"]},
            "transformer": Scaler(),
        }

        ocl = 6
        enc_base = SequentialEncoder(
            add_encoders=encoder_params,
            input_chunk_length=11,
            output_chunk_length=ocl,
            takes_past_covariates=True,
            takes_future_covariates=True,
        )

        # ====> TEST Transformation starting from multi-TimeSeries input: transformer is globally fit per encoded
        # component
        enc = copy.deepcopy(enc_base)
        pc, fc = enc.encode_train([ts1, ts2], future_covariates=[ts1, ts2])
        # user supplied covariates should not be transformed
        self.assertTrue(fc[0]["cov"] == ts1)
        self.assertTrue(fc[1]["cov"] == ts2)
        # check that first covariate series ranges from 0. to 1. and second from ~0.7 to 1.
        for covs, cov_name in zip(
            [pc, fc], ["darts_enc_pc_dta_minute", "darts_enc_fc_dta_minute"]
        ):
            self.assertAlmostEqual(
                covs[0][cov_name].values(copy=False).min(), 0.0, delta=10e-9
            )
            self.assertAlmostEqual(
                covs[0][cov_name].values(copy=False).max(), 1.0, delta=10e-9
            )
            self.assertEqual(
                covs[0][cov_name].univariate_values(copy=False)[-4],
                covs[1][cov_name].univariate_values(copy=False)[-4],
            )
            if "pc" in cov_name:
                self.assertAlmostEqual(
                    covs[1][cov_name].values(copy=False).min(), 0.714, delta=1e-2
                )
            else:
                self.assertAlmostEqual(
                    covs[1][cov_name].values(copy=False).min(), 0.5, delta=1e-2
                )
            self.assertAlmostEqual(
                covs[1][cov_name].values(copy=False).max(), 1.0, delta=10e-9
            )

        # check the same for inference
        pc, fc = enc.encode_inference(
            n=6, target=[ts1, ts2], future_covariates=[ts1_inf, ts2_inf]
        )
        for covs, cov_name in zip(
            [pc, fc], ["darts_enc_pc_dta_minute", "darts_enc_fc_dta_minute"]
        ):
            for cov in covs:
                if "pc" in cov_name:
                    self.assertEqual(
                        cov[cov_name][-(ocl + 1)].univariate_values()[0], 1.0
                    )
                else:
                    self.assertEqual(
                        cov[cov_name][ts1.end_time()].univariate_values()[0], 1.0
                    )

        # check the same for only supplying single series as input
        pc, fc = enc.encode_inference(n=6, target=ts2, future_covariates=ts2_inf)
        for cov, cov_name in zip(
            [pc, fc], ["darts_enc_pc_dta_minute", "darts_enc_fc_dta_minute"]
        ):
            if "pc" in cov_name:
                self.assertEqual(cov[cov_name][-(ocl + 1)].univariate_values()[0], 1.0)
            else:
                self.assertEqual(
                    cov[cov_name][ts1.end_time()].univariate_values()[0], 1.0
                )

        # ====> TEST Transformation starting from single-TimeSeries input: transformer is fit per component of a single
        # encoded series
        enc = copy.deepcopy(enc_base)
        pc, fc = enc.encode_train(ts2, future_covariates=ts2)
        # user supplied covariates should not be transformed
        self.assertTrue(fc["cov"] == ts2)
        for covs, cov_name in zip(
            [pc, fc], ["darts_enc_pc_dta_minute", "darts_enc_fc_dta_minute"]
        ):
            self.assertAlmostEqual(
                covs[cov_name].values(copy=False).min(), 0.0, delta=10e-9
            )
            self.assertAlmostEqual(
                covs[cov_name].values(copy=False).max(), 1.0, delta=10e-9
            )

        # second time fitting will not fit transformers again
        pc, fc = enc.encode_train([ts1, ts2], future_covariates=[ts1, ts2])
        for covs, cov_name in zip(
            [pc, fc], ["darts_enc_pc_dta_minute", "darts_enc_fc_dta_minute"]
        ):
            if "pc" in cov_name:
                self.assertAlmostEqual(
                    covs[0][cov_name].values(copy=False).min(), -2.5, delta=10e-9
                )
            else:
                self.assertAlmostEqual(
                    covs[0][cov_name].values(copy=False).min(), -1.0, delta=10e-9
                )
            self.assertAlmostEqual(
                covs[0][cov_name].values(copy=False).max(), 1.0, delta=10e-9
            )
            self.assertAlmostEqual(
                covs[1][cov_name].values(copy=False).min(), 0.0, delta=10e-9
            )
            self.assertAlmostEqual(
                covs[1][cov_name].values(copy=False).max(), 1.0, delta=10e-9
            )

        # check inference with single series
        pc, fc = enc.encode_inference(n=6, target=ts2, future_covariates=ts2_inf)
        for cov, cov_name in zip(
            [pc, fc], ["darts_enc_pc_dta_minute", "darts_enc_fc_dta_minute"]
        ):
            self.assertAlmostEqual(
                cov[cov_name].values(copy=False).min(), 0.0, delta=10e-9
            )
            if "pc" in cov_name:
                self.assertAlmostEqual(
                    cov[cov_name].values(copy=False).max(), 2.5, delta=10e-9
                )
            else:
                self.assertAlmostEqual(
                    cov[cov_name].values(copy=False).max(), 1.0, delta=10e-9
                )

        # check the same for supplying multiple series as input
        pc, fc = enc.encode_inference(
            n=6, target=[ts1, ts2], future_covariates=[ts1_inf, ts2_inf]
        )
        for covs, cov_name in zip(
            [pc, fc], ["darts_enc_pc_dta_minute", "darts_enc_fc_dta_minute"]
        ):
            for cov in covs:
                self.assertAlmostEqual(
                    cov[cov_name].values(copy=False).min(), 0.0, delta=10e-9
                )
                if "pc" in cov_name:
                    self.assertAlmostEqual(
                        cov[cov_name].values(copy=False).max(), 2.5, delta=10e-9
                    )
                else:
                    self.assertAlmostEqual(
                        cov[cov_name].values(copy=False).max(), 1.0, delta=10e-9
                    )

    def helper_test_cyclic_encoder(
        self,
        encoder_class,
        attribute,
        inf_ts_short,
        inf_ts_long,
        cyclic,
        expected_components,
    ):
        """Test cases for both `PastCyclicEncoder` and `FutureCyclicEncoder`"""
        encoder = encoder_class(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            attribute=attribute,
        )
        # covs: covariates; ds: dataset
        # expected generated covs when covs are supplied as input for train and inference ds
        result_with_cov = [
            tg.datetime_attribute_timeseries(
                ts,
                attribute=attribute,
                cyclic=cyclic,
                with_columns=expected_components,
            )
            for ts in self.covariates_multi
        ]
        # expected generated covs when covs are not supplied as input for train ds
        result_no_cov = [
            tg.datetime_attribute_timeseries(
                ts,
                attribute=attribute,
                cyclic=cyclic,
                with_columns=expected_components,
            )
            for ts in self.target_multi
        ]
        # expected generated covs when covs are not supplied as input for inference ds and n <= output_chunk_length
        result_no_cov_inf_short = [
            tg.datetime_attribute_timeseries(
                ts,
                attribute=attribute,
                cyclic=cyclic,
                with_columns=expected_components,
            )
            for ts in inf_ts_short
        ]
        # expected generated covs when covs are not supplied as input for inference ds and n > output_chunk_length
        result_no_cov_inf_long = [
            tg.datetime_attribute_timeseries(
                ts,
                attribute=attribute,
                cyclic=cyclic,
                with_columns=expected_components,
            )
            for ts in inf_ts_long
        ]

        # test train encoding with covariates
        self.helper_test_encoder_single_train(
            encoder=encoder,
            target=self.target_multi,
            covariates=self.covariates_multi,
            result=result_with_cov,
            merge_covariates=False,
        )

        # test train encoding without covariates
        self.helper_test_encoder_single_train(
            encoder=encoder,
            target=self.target_multi,
            covariates=[None] * len(self.target_multi),
            result=result_no_cov,
            merge_covariates=False,
        )
        # test inference encoding with covariates and n <= output_chunk_length
        self.helper_test_encoder_single_inference(
            encoder=encoder,
            n=self.n_short,
            target=self.target_multi,
            covariates=self.covariates_multi,
            result=result_with_cov,
            merge_covariates=False,
        )
        # test inference encoding with covariates and n > output_chunk_length
        self.helper_test_encoder_single_inference(
            encoder=encoder,
            n=self.n_long,
            target=self.target_multi,
            covariates=self.covariates_multi,
            result=result_with_cov,
            merge_covariates=False,
        )
        # test inference encoding without covariates and n <= output_chunk_length
        self.helper_test_encoder_single_inference(
            encoder=encoder,
            n=self.n_short,
            target=self.target_multi,
            covariates=[None] * len(self.target_multi),
            result=result_no_cov_inf_short,
            merge_covariates=False,
        )
        # test inference encoding without covariates and n > output_chunk_length
        self.helper_test_encoder_single_inference(
            encoder=encoder,
            n=self.n_long,
            target=self.target_multi,
            covariates=[None] * len(self.target_multi),
            result=result_no_cov_inf_long,
            merge_covariates=False,
        )

    def helper_test_encoder_single_train(
        self,
        encoder: SingleEncoder,
        target: Sequence[TimeSeries],
        covariates: Sequence[Optional[TimeSeries]],
        result: Sequence[TimeSeries],
        merge_covariates: bool = True,
    ):
        """Test `SingleEncoder.encode_train()`"""

        encoded = []
        for ts, cov in zip(target, covariates):
            encoded.append(
                encoder.encode_train(ts, cov, merge_covariates=merge_covariates)
            )

        expected_result = result
        # when user does not give covariates, and a past covariate encoder is used, the generate train covariates are
        # `output_chunk_length` steps shorter than the target series
        if covariates[0] is None and isinstance(
            encoder.index_generator, PastCovariatesIndexGenerator
        ):
            expected_result = [res[: -self.output_chunk_length] for res in result]

        self.assertTrue(encoded == expected_result)

    def helper_test_encoder_single_inference(
        self,
        encoder: SingleEncoder,
        n: int,
        target: Sequence[TimeSeries],
        covariates: Sequence[Optional[TimeSeries]],
        result: Sequence[TimeSeries],
        merge_covariates: bool = True,
    ):
        """Test `SingleEncoder.encode_inference()`"""

        encoded = []
        for ts, cov in zip(target, covariates):
            encoded.append(
                encoder.encode_inference(n, ts, cov, merge_covariates=merge_covariates)
            )
        self.assertTrue(encoded == result)
