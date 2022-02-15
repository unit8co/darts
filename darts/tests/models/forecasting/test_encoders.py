import unittest
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg
from darts.utils.data.encoder_base import SingleEncoder
from darts.utils.data.encoders import (
    FutureCyclicEncoder,
    FutureDatetimeAttributeEncoder,
    PastCyclicEncoder,
    PastDatetimeAttributeEncoder,
    SequentialEncoder,
)

logger = get_logger(__name__)

try:
    from darts.models import TFTModel

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not installed - will be skipping Torch models tests")
    TORCH_AVAILABLE = False


class EncoderTestCase(DartsBaseTestClass):
    n_target_1 = 12
    n_target_2 = 24
    shift = 50
    target_1 = tg.linear_timeseries(length=n_target_1, freq="MS")
    target_2 = tg.linear_timeseries(
        start=target_1.end_time() + shift * target_1.freq, length=n_target_2, freq="MS"
    )
    covariate_1 = tg.linear_timeseries(length=2 * n_target_1, freq="MS")
    covariate_2 = tg.linear_timeseries(
        start=target_1.end_time() + shift * target_1.freq,
        length=2 * n_target_2,
        freq="MS",
    )

    target_multi = [target_1, target_2]
    covariate_multi = [covariate_1, covariate_2]

    input_chunk_length = 12
    output_chunk_length = 6
    n_short = 6
    n_long = 8

    # for the given input_chunk_length, ..., n_long from above, the time_index of the expected encoded covariate
    # multi-TS at prediction should be as follows
    inf_ts_short_future = [
        TimeSeries.from_times_and_values(
            tg._generate_index(
                start=ts.end_time() + (1 - 12) * ts.freq, length=12 + 6, freq=ts.freq
            ),
            np.arange(12 + 6),
        )
        for ts in target_multi
    ]

    inf_ts_long_future = [
        TimeSeries.from_times_and_values(
            tg._generate_index(
                start=ts.end_time() + (1 - 12) * ts.freq, length=12 + 8, freq=ts.freq
            ),
            np.arange(12 + 8),
        )
        for ts in target_multi
    ]

    inf_ts_short_past = [
        TimeSeries.from_times_and_values(
            tg._generate_index(
                start=ts.end_time() + (1 - 12) * ts.freq, length=12, freq=ts.freq
            ),
            np.arange(12),
        )
        for ts in target_multi
    ]

    inf_ts_long_past = [
        TimeSeries.from_times_and_values(
            tg._generate_index(
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
            past_covariate=self.covariate_multi,
            future_covariate=self.covariate_multi,
        )

        # encoded multi TS covariates should have same number as input covariates
        self.assertEqual(len(past_covs_train), 2)
        self.assertEqual(len(future_covs_train), 2)

        # each attribute (i.e., 'month', ...) generates 2 output variables (+ 1 covariate from input covariates)
        self.assertEqual(past_covs_train[0].n_components, 3)
        self.assertEqual(future_covs_train[0].n_components, 5)

        # check with different inputs
        encoder_args = {"cyclic": {"past": ["month"], "future": ["month"]}}
        encoders = self.helper_encoder_from_model(add_encoder_dict=encoder_args)

        # ==> test training <==
        past_covs_train, future_covs_train = encoders.encode_train(
            target=self.target_multi,
            past_covariate=self.covariate_multi,
            future_covariate=self.covariate_multi,
        )

        # encoded multi TS covariates should have same number as input covariates
        self.assertEqual(len(past_covs_train), 2)
        self.assertEqual(len(future_covs_train), 2)

        # each attribute (i.e., 'month', ...) generates 2 output variables (+ 1 covariate from input covariates)
        self.assertEqual(past_covs_train[0].n_components, 3)
        self.assertEqual(future_covs_train[0].n_components, 3)

        # encoded past covariates must have equal index as input past covariates
        for pc, pc_in in zip(past_covs_train, self.covariate_multi):
            self.assertTrue(pc.time_index.equals(pc_in.time_index))

        # encoded future covariates must have equal index as input future covariates
        for fc, fc_in in zip(future_covs_train, self.covariate_multi):
            self.assertTrue(fc.time_index.equals(fc_in.time_index))

        # for training dataset: both encoded past and future covariates with cyclic encoder 'month' should be equal
        for pc, fc in zip(past_covs_train, future_covs_train):
            self.assertEqual(pc, fc)

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
            past_covariates=self.covariate_multi,
            future_covariates=self.covariate_multi,
            expected_past_idx_ts=self.covariate_multi,
            expected_future_idx_ts=self.covariate_multi,
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
            past_covariates=self.covariate_multi,
            future_covariates=None,
            expected_past_idx_ts=self.covariate_multi,
            expected_future_idx_ts=self.inf_ts_long_future,
        )
        # with supplying future covariates as input
        self.helper_sequence_encode_inference(
            encoders=encoders,
            n=self.n_long,
            past_covariates=None,
            future_covariates=self.covariate_multi,
            expected_past_idx_ts=self.inf_ts_long_past,
            expected_future_idx_ts=self.covariate_multi,
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
            past_covariate=past_covariates,
            future_covariate=future_covariates,
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
            self.target_multi, self.covariate_multi, self.covariate_multi
        )
        _ = encoders.encode_inference(
            3, self.target_multi, self.covariate_multi, self.covariate_multi
        )
        return encoders

    def test_cyclic_encoder(self):
        """Test past and future `CyclicTemporalEncoder``"""

        attribute = "month"

        month_series = TimeSeries.from_times_and_values(
            times=tg._generate_index(
                start=pd.to_datetime("2000-01-01"), length=24, freq="MS"
            ),
            values=np.arange(24),
        )
        encoder = FutureCyclicEncoder(
            input_chunk_length=1, output_chunk_length=1, attribute="month"
        )
        first_halve = encoder.encode_train(
            target=month_series[:12], covariate=month_series[:12], merge_covariate=False
        )
        second_halve = encoder.encode_train(
            target=month_series[12:], covariate=month_series[12:], merge_covariate=False
        )
        # check if encoded values for first 12 months are equal to values of last 12 months
        self.assertTrue((first_halve.values() == second_halve.values()).all())

        # test past cyclic encoder
        self.helper_test_cyclic_encoder(
            PastCyclicEncoder,
            attribute=attribute,
            inf_ts_short=self.inf_ts_short_past,
            inf_ts_long=self.inf_ts_long_past,
            cyclic=True,
        )
        # test future cyclic encoder
        self.helper_test_cyclic_encoder(
            FutureCyclicEncoder,
            attribute=attribute,
            inf_ts_short=self.inf_ts_short_future,
            inf_ts_long=self.inf_ts_long_future,
            cyclic=True,
        )

    def test_datetime_attribute_encoder(self):
        """Test past and future `DatetimeAttributeEncoder`"""

        attribute = "month"

        month_series = TimeSeries.from_times_and_values(
            times=tg._generate_index(
                start=pd.to_datetime("2000-01-01"), length=24, freq="MS"
            ),
            values=np.arange(24),
        )

        encoder = FutureDatetimeAttributeEncoder(
            input_chunk_length=1, output_chunk_length=1, attribute="month"
        )
        first_halve = encoder.encode_train(
            target=month_series[:12], covariate=month_series[:12], merge_covariate=False
        )
        second_halve = encoder.encode_train(
            target=month_series[12:], covariate=month_series[12:], merge_covariate=False
        )
        # check if encoded values for first 12 months are equal to values of last 12 months
        self.assertTrue((first_halve.values() == second_halve.values()).all())

        # test past cyclic encoder
        self.helper_test_cyclic_encoder(
            PastDatetimeAttributeEncoder,
            attribute=attribute,
            inf_ts_short=self.inf_ts_short_past,
            inf_ts_long=self.inf_ts_long_past,
            cyclic=False,
        )

        # test future cyclic encoder
        self.helper_test_cyclic_encoder(
            FutureDatetimeAttributeEncoder,
            attribute=attribute,
            inf_ts_short=self.inf_ts_short_future,
            inf_ts_long=self.inf_ts_long_future,
            cyclic=False,
        )

    def test_integer_positional_encoder(self):
        """Test past `IntegerIndexEncoder`"""

        ts = tg.linear_timeseries(length=24, freq="MS")
        input_chunk_length = 12
        output_chunk_length = 6

        # ===> test absolute position encoder <===
        encoder_params = {"position": {"past": ["absolute"]}}
        encs = SequentialEncoder(
            add_encoders=encoder_params,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            takes_past_covariates=True,
            takes_future_covariates=True,
        )

        t1, _ = encs.encode_train(ts)
        t2, _ = encs.encode_train(
            TimeSeries.from_times_and_values(ts.time_index + ts.freq, ts.values())
        )
        t3, _ = encs.encode_train(
            TimeSeries.from_times_and_values(ts.time_index - ts.freq, ts.values())
        )

        # absolute encoder takes the first observed index as a reference (from training)
        vals = np.arange(len(ts)).reshape((len(ts), 1))
        self.assertTrue(
            (t1[0].time_index == ts.time_index).all() and (t1[0].values() == vals).all()
        )
        # test that the position values are updated correctly
        self.assertTrue(
            (t2[0].time_index == ts.time_index + ts.freq).all()
            and (t2[0].values() == vals + 1).all()
        )
        self.assertTrue(
            (t3[0].time_index == ts.time_index - ts.freq).all()
            and (t3[0].values() == vals - 1).all()
        )
        # quickly test inference encoding
        # n > output_chunk_length
        t4, _ = encs.encode_inference(output_chunk_length + 1, ts)

        self.assertTrue(
            (
                t4[0].values()[:, 0]
                == np.arange(len(ts) - input_chunk_length, len(ts) + 1)
            ).all()
        )
        # n <= output_chunk_length
        t5, _ = encs.encode_inference(output_chunk_length - 1, ts)
        self.assertTrue(
            (
                t5[0].values()[:, 0] == np.arange(len(ts) - input_chunk_length, len(ts))
            ).all()
        )

        # ===> test relative position encoder <===
        encoder_params = {"position": {"past": ["relative"]}}
        encs = SequentialEncoder(
            add_encoders=encoder_params,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            takes_past_covariates=True,
            takes_future_covariates=True,
        )

        t1, _ = encs.encode_train(ts)
        t2, _ = encs.encode_train(
            TimeSeries.from_times_and_values(ts.time_index + ts.freq, ts.values())
        )
        t3, _ = encs.encode_train(
            TimeSeries.from_times_and_values(ts.time_index - ts.freq, ts.values())
        )
        # relative encoder takes the end of the training series as reference
        vals = np.arange(-len(ts) + 1, 1).reshape((len(ts), 1))
        self.assertTrue(
            (t1[0].time_index == ts.time_index).all() and (t1[0].values() == vals).all()
        )
        self.assertTrue(
            (t2[0].time_index == ts.time_index + ts.freq).all()
            and (t2[0].values() == vals + 1).all()
        )
        self.assertTrue(
            (t3[0].time_index == ts.time_index - ts.freq).all()
            and (t3[0].values() == vals - 1).all()
        )
        # quickly test inference encoding
        # n > output_chunk_length
        t4, _ = encs.encode_inference(output_chunk_length + 1, ts)
        self.assertTrue(
            (t4[0].values()[:, 0] == np.arange(-input_chunk_length + 1, 1 + 1)).all()
        )
        # n <= output_chunk_length
        t5, _ = encs.encode_inference(output_chunk_length - 1, ts)
        self.assertTrue(
            (t5[0].values()[:, 0] == np.arange(-input_chunk_length + 1, 0 + 1)).all()
        )

    def test_callable_encoder(self):
        """Test `CallableIndexEncoder`"""
        ts = tg.linear_timeseries(length=24, freq="A")
        input_chunk_length = 12
        output_chunk_length = 6

        # ===> test absolute position encoder <===
        encoder_params = {
            "custom": {"past": [lambda index: index.year, lambda index: index.year - 1]}
        }
        encs = SequentialEncoder(
            add_encoders=encoder_params,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            takes_past_covariates=True,
            takes_future_covariates=True,
        )

        t1, _ = encs.encode_train(ts)
        self.assertTrue((ts.time_index.year.values == t1[0].values()[:, 0]).all())
        self.assertTrue((ts.time_index.year.values - 1 == t1[0].values()[:, 1]).all())

    def test_transformer(self):
        ts1 = tg.linear_timeseries(
            start_value=1, end_value=2, length=60, freq="T", column_name="cov_in"
        )
        encoder_params = {
            "position": {"future": ["absolute"]},
            "cyclic": {"future": ["minute"]},
            "transformer": Scaler(),
        }

        encs = SequentialEncoder(
            add_encoders=encoder_params,
            input_chunk_length=12,
            output_chunk_length=6,
            takes_past_covariates=True,
            takes_future_covariates=True,
        )

        _, t1 = encs.encode_train(ts1, future_covariate=ts1)

        # ===> train set test <===
        # user supplied covariates should not be transformed
        self.assertTrue(t1[0]["cov_in"] == ts1)
        # cyclic encodings should not be transformed
        for curve in ["sin", "cos"]:
            self.assertAlmostEqual(
                t1[0][f"minute_{curve}"].all_values(copy=False).min(), -1.0, delta=10e-9
            )
            self.assertAlmostEqual(
                t1[0][f"minute_{curve}"].values(copy=False).max(), 1.0, delta=10e-9
            )
        # all others should be transformed to values between 0 and 1
        self.assertAlmostEqual(
            t1[0]["absolute_idx"].values(copy=False).min(), 0.0, delta=10e-9
        )
        self.assertAlmostEqual(
            t1[0]["absolute_idx"].values(copy=False).max(), 1.0, delta=10e-9
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
        _, t2 = encs.encode_train(ts2, future_covariate=ts2)
        # make sure that when calling encoders the second time, scalers are not fit again (for validation and inference)
        self.assertAlmostEqual(
            t2[0]["absolute_idx"].values(copy=False).min(), 1.0, delta=10e-9
        )
        self.assertAlmostEqual(
            t2[0]["absolute_idx"].values(copy=False).max(), 2.0, delta=10e-9
        )

        fc_inf = tg.linear_timeseries(
            start_value=1, end_value=3, length=80, freq="T", column_name="cov_in"
        )
        _, t3 = encs.encode_inference(n=12, target=ts1, future_covariate=fc_inf)

        # index 0 is also start of train target series and value should be 0
        self.assertAlmostEqual(t3[0]["absolute_idx"][0].values()[0, 0], 0.0)
        # index len(ts1) - 1 is the prediction point and value should be 0
        self.assertAlmostEqual(t3[0]["absolute_idx"][len(ts1) - 1].values()[0, 0], 1.0)
        # the future should scale proportional to distance to prediction point
        self.assertAlmostEqual(
            t3[0]["absolute_idx"][80 - 1].values()[0, 0], 80 / 60, delta=0.01
        )

    def helper_test_cyclic_encoder(
        self, encoder_class, attribute, inf_ts_short, inf_ts_long, cyclic
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
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=cyclic)
            for ts in self.covariate_multi
        ]
        # expected generated covs when covs are not supplied as input for train ds
        result_no_cov = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=cyclic)
            for ts in self.target_multi
        ]
        # expected generated covs when covs are not supplied as input for inference ds and n <= output_chunk_length
        result_no_cov_inf_short = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=cyclic)
            for ts in inf_ts_short
        ]
        # expected generated covs when covs are not supplied as input for inference ds and n > output_chunk_length
        result_no_cov_inf_long = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=cyclic)
            for ts in inf_ts_long
        ]

        # test train encoding with covariates
        self.helper_test_encoder_single_train(
            encoder=encoder,
            target=self.target_multi,
            covariate=self.covariate_multi,
            result=result_with_cov,
            merge_covariates=False,
        )

        # test train encoding without covariates
        self.helper_test_encoder_single_train(
            encoder=encoder,
            target=self.target_multi,
            covariate=[None] * len(self.target_multi),
            result=result_no_cov,
            merge_covariates=False,
        )
        # test inference encoding with covariates and n <= output_chunk_length
        self.helper_test_encoder_single_inference(
            encoder=encoder,
            n=self.n_short,
            target=self.target_multi,
            covariate=self.covariate_multi,
            result=result_with_cov,
            merge_covariates=False,
        )
        # test inference encoding with covariates and n > output_chunk_length
        self.helper_test_encoder_single_inference(
            encoder=encoder,
            n=self.n_long,
            target=self.target_multi,
            covariate=self.covariate_multi,
            result=result_with_cov,
            merge_covariates=False,
        )
        # test inference encoding without covariates and n <= output_chunk_length
        self.helper_test_encoder_single_inference(
            encoder=encoder,
            n=self.n_short,
            target=self.target_multi,
            covariate=[None] * len(self.target_multi),
            result=result_no_cov_inf_short,
            merge_covariates=False,
        )
        # test inference encoding without covariates and n > output_chunk_length
        self.helper_test_encoder_single_inference(
            encoder=encoder,
            n=self.n_long,
            target=self.target_multi,
            covariate=[None] * len(self.target_multi),
            result=result_no_cov_inf_long,
            merge_covariates=False,
        )

    def helper_test_encoder_single_train(
        self,
        encoder: SingleEncoder,
        target: Sequence[TimeSeries],
        covariate: Sequence[Optional[TimeSeries]],
        result: Sequence[TimeSeries],
        merge_covariates: bool = True,
    ):
        """Test `SingleEncoder.encode_train()`"""

        encoded = []
        for ts, cov in zip(target, covariate):
            encoded.append(
                encoder.encode_train(ts, cov, merge_covariate=merge_covariates)
            )
        self.assertTrue(encoded == result)

    def helper_test_encoder_single_inference(
        self,
        encoder: SingleEncoder,
        n: int,
        target: Sequence[TimeSeries],
        covariate: Sequence[Optional[TimeSeries]],
        result: Sequence[TimeSeries],
        merge_covariates: bool = True,
    ):
        """Test `SingleEncoder.encode_inference()`"""

        encoded = []
        for ts, cov in zip(target, covariate):
            encoded.append(
                encoder.encode_inference(n, ts, cov, merge_covariate=merge_covariates)
            )
        self.assertTrue(encoded == result)
