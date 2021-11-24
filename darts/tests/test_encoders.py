import pandas as pd
import numpy as np

from typing import Sequence, Optional
from .base_test_class import DartsBaseTestClass
from ..models import TFTModel
from ..utils import timeseries_generation as tg
from ..timeseries import TimeSeries
from ..utils.data.encoders import (Encoder,
                                   SingleEncoder,
                                   CyclicPastEncoder,
                                   CyclicFutureEncoder,
                                   DatetimeAttributePastEncoder,
                                   DatetimeAttributeFutureEncoder,
                                   PositionalPastEncoder,
                                   PositionalFutureEncoder,
                                   SequenceEncoder)

from ..logging import get_logger
logger = get_logger(__name__)


class EncoderTestCase(DartsBaseTestClass):
    n_target_1 = 12
    n_target_2 = 24
    shift = 50
    target_1 = tg.linear_timeseries(length=n_target_1,
                                    freq='MS')
    target_2 = tg.linear_timeseries(start=target_1.end_time() + shift * target_1.freq,
                                    length=n_target_2,
                                    freq='MS')
    covariate_1 = tg.linear_timeseries(length=2 * n_target_1,
                                       freq='MS')
    covariate_2 = tg.linear_timeseries(start=target_1.end_time() + shift * target_1.freq,
                                       length=2 * n_target_2,
                                       freq='MS')

    target_multi = [target_1, target_2]
    covariate_multi = [covariate_1, covariate_2]

    input_chunk_length = 12
    output_chunk_length = 6
    n_short = 6
    n_long = 8

    # for the given input_chunk_length, ..., n_long from above, the time_index of the expected encoded covariate
    # multi-TS at prediction should be as follows
    inf_ts_short_future = [TimeSeries.from_times_and_values(
        tg._generate_index(start=ts.end_time() + (1 - 12) * ts.freq,
                           length=12 + 6,
                           freq=ts.freq),
        np.arange(12 + 6)) for ts in target_multi]

    inf_ts_long_future = [TimeSeries.from_times_and_values(
        tg._generate_index(start=ts.end_time() + (1 - 12) * ts.freq,
                           length=12 + 8,
                           freq=ts.freq),
        np.arange(12 + 8)) for ts in target_multi]

    inf_ts_short_past = [TimeSeries.from_times_and_values(
        tg._generate_index(start=ts.end_time() + (1 - 12) * ts.freq,
                           length=12,
                           freq=ts.freq),
        np.arange(12)) for ts in target_multi]

    inf_ts_long_past = [TimeSeries.from_times_and_values(
        tg._generate_index(start=ts.end_time() + (1 - 12) * ts.freq,
                           length=12 + (8 - 6),
                           freq=ts.freq),
        np.arange(12 + (8 - 6))) for ts in target_multi]

    def test_sequence_encoder_from_model_params(self):
        """test if sequence encoder is initialized properly from model params"""
        # valid encoder model parameters are ('past', 'future') for the main key and datetime attribute for sub keys
        valid_encoder_args = {
            'cyclic': {'past': ['month'], 'future': ['dayofyear', 'dayofweek']}
        }
        encoders = self.helper_encoder_from_model(add_encoder_dict=valid_encoder_args)

        self.assertTrue(len(encoders.past_encoders) == 1)
        self.assertTrue(len(encoders.future_encoders) == 2)

        # test if encoders have the correct attributes
        self.assertTrue(encoders.past_encoders[0].attribute == 'month')
        self.assertTrue([enc.attribute for enc in encoders.future_encoders] == ['dayofyear', 'dayofweek'])

        valid_encoder_args = {'cyclic': {'past': ['month']}}
        encoders = self.helper_encoder_from_model(add_encoder_dict=valid_encoder_args, takes_future_covariates=False)
        self.assertTrue(len(encoders.past_encoders) == 1)
        self.assertTrue(len(encoders.future_encoders) == 0)

        # test invalid encoder kwarg at model creation
        bad_encoder = {'no_encoder': {'past': ['month']}}
        encoders = self.helper_encoder_from_model(add_encoder_dict=bad_encoder)
        self.assertTrue(len(encoders.past_encoders) == 0)
        self.assertTrue(len(encoders.future_encoders) == 0)

        # test invalid kwargs at model creation
        bad_time = {'cyclic': {'ppast': ['month']}}
        with self.assertRaises(ValueError):
            _ = self.helper_encoder_from_model(add_encoder_dict=bad_time)

        bad_attribute = {'cyclic': {'past': ['year']}}
        with self.assertRaises(ValueError):
            _ = self.helper_encoder_from_model(add_encoder_dict=bad_attribute)

        bad_type = {'cyclic': {'past': 1}}
        with self.assertRaises(ValueError):
            _ = self.helper_encoder_from_model(add_encoder_dict=bad_type)

    def test_encoder_sequence_train(self):
        """Test `SequenceEncoder.encode_train()` output"""
        # ====> Sequential Cyclic Encoder Tests <====
        encoder_args = {'cyclic': {'past': ['month'], 'future': ['month', 'month']}}
        encoders = self.helper_encoder_from_model(add_encoder_dict=encoder_args)

        # ==> test training <==
        past_covs_train, future_covs_train = encoders.encode_train(target=self.target_multi,
                                                                   past_covariate=self.covariate_multi,
                                                                   future_covariate=self.covariate_multi)

        # encoded multi TS covariates should have same number as input covariates
        self.assertEqual(len(past_covs_train), 2)
        self.assertEqual(len(future_covs_train), 2)

        # each attribute (i.e., 'month', ...) generates 2 output variables (+ 1 covariate from input covariates)
        self.assertEqual(past_covs_train[0].n_components, 3)
        self.assertEqual(future_covs_train[0].n_components, 5)

        # check with different inputs
        encoder_args = {'cyclic': {'past': ['month'], 'future': ['month']}}
        encoders = self.helper_encoder_from_model(add_encoder_dict=encoder_args)

        # ==> test training <==
        past_covs_train, future_covs_train = encoders.encode_train(target=self.target_multi,
                                                                   past_covariate=self.covariate_multi,
                                                                   future_covariate=self.covariate_multi)

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

    def test_encoder_sequence_inference(self):
        """Test `SequenceEncoder.encode_inference()` output"""
        # ==> test prediction <==
        encoder_args = {'cyclic': {'past': ['month'], 'future': ['month']}}
        encoders = self.helper_encoder_from_model(add_encoder_dict=encoder_args)

        # tests with n <= output_chunk_length
        self.helper_sequence_encode_inference(encoders=encoders,
                                              n=self.n_short,
                                              past_covariates=self.covariate_multi,
                                              future_covariates=self.covariate_multi,
                                              expected_past_idx_ts=self.covariate_multi,
                                              expected_future_idx_ts=self.covariate_multi)

        self.helper_sequence_encode_inference(encoders=encoders,
                                              n=self.n_short,
                                              past_covariates=None,
                                              future_covariates=None,
                                              expected_past_idx_ts=self.inf_ts_short_past,
                                              expected_future_idx_ts=self.inf_ts_short_future)
        # tests with n > output_chunk_length
        self.helper_sequence_encode_inference(encoders=encoders,
                                              n=self.n_long,
                                              past_covariates=self.covariate_multi,
                                              future_covariates=None,
                                              expected_past_idx_ts=self.covariate_multi,
                                              expected_future_idx_ts=self.inf_ts_long_future)

        self.helper_sequence_encode_inference(encoders=encoders,
                                              n=self.n_long,
                                              past_covariates=None,
                                              future_covariates=self.covariate_multi,
                                              expected_past_idx_ts=self.inf_ts_long_past,
                                              expected_future_idx_ts=self.covariate_multi)

    def helper_sequence_encode_inference(self,
                                         encoders,
                                         n,
                                         past_covariates,
                                         future_covariates,
                                         expected_past_idx_ts,
                                         expected_future_idx_ts):
        """test comparisons for `SequenceEncoder.encode_inference()`"""

        # generate encodings
        past_covs_pred, future_covs_pred = encoders.encode_inference(n=n,
                                                                     target=self.target_multi,
                                                                     past_covariate=past_covariates,
                                                                     future_covariate=future_covariates)
        # encoded past and future covariates must have equal index as expected past and future
        for pc, pc_in in zip(past_covs_pred, expected_past_idx_ts):
            self.assertTrue(pc.time_index.equals(pc_in.time_index))
        for fc, fc_in in zip(future_covs_pred, expected_future_idx_ts):
            self.assertTrue(fc.time_index.equals(fc_in.time_index))

    def helper_encoder_from_model(self, add_encoder_dict, takes_past_covariates=True, takes_future_covariates=True):
        """Extract encoders from model creation"""
        model = TFTModel(input_chunk_length=self.input_chunk_length,
                         output_chunk_length=self.output_chunk_length,
                         add_encoders=add_encoder_dict)

        encoders = model.initialize_encoders(model_params=model._model_params,
                                             input_chunk_length=model.input_chunk_length,
                                             output_chunk_length=model.output_chunk_length,
                                             shift=model.input_chunk_length,
                                             takes_past_covariates=takes_past_covariates,
                                             takes_future_covariates=takes_future_covariates)
        # see if encoding works
        _ = encoders.encode_train(self.target_multi, self.covariate_multi, self.covariate_multi)
        _ = encoders.encode_inference(3, self.target_multi, self.covariate_multi, self.covariate_multi)
        return encoders

    def test_cyclic_encoder(self):
        """Test past and future `CyclicTemporalEncoder`"""
        attribute = 'month'

        month_series = TimeSeries.from_times_and_values(
            times=tg._generate_index(start=pd.to_datetime('2000-01-01'), length=24, freq='MS'),
            values=np.arange(24)
        )

        encoder = CyclicFutureEncoder(input_chunk_length=1, output_chunk_length=1, attribute='month')
        first_halve = encoder.encode_train(target=month_series[:12],
                                           covariate=month_series[:12],
                                           merge_covariate=False)
        second_halve = encoder.encode_train(target=month_series[12:],
                                            covariate=month_series[12:],
                                            merge_covariate=False)
        # check if encoded values for first 12 months are equal to values of last 12 months
        self.assertTrue((first_halve.values() == second_halve.values()).all())

        # test past cyclic encoder
        self.helper_test_cyclic_encoder(CyclicPastEncoder,
                                        attribute=attribute,
                                        inf_ts_short=self.inf_ts_short_past,
                                        inf_ts_long=self.inf_ts_long_past,
                                        cyclic=True)

        # test future cyclic encoder
        self.helper_test_cyclic_encoder(CyclicFutureEncoder,
                                        attribute=attribute,
                                        inf_ts_short=self.inf_ts_short_future,
                                        inf_ts_long=self.inf_ts_long_future,
                                        cyclic=True)

    def test_datetime_attribute_encoder(self):
        """Test past and future `DatetimeAttributeEncoder`"""
        attribute = 'month'

        month_series = TimeSeries.from_times_and_values(
            times=tg._generate_index(start=pd.to_datetime('2000-01-01'), length=24, freq='MS'),
            values=np.arange(24)
        )

        encoder = DatetimeAttributeFutureEncoder(input_chunk_length=1, output_chunk_length=1, attribute='month')
        first_halve = encoder.encode_train(target=month_series[:12],
                                           covariate=month_series[:12],
                                           merge_covariate=False)
        second_halve = encoder.encode_train(target=month_series[12:],
                                            covariate=month_series[12:],
                                            merge_covariate=False)
        # check if encoded values for first 12 months are equal to values of last 12 months
        self.assertTrue((first_halve.values() == second_halve.values()).all())

        # test past cyclic encoder
        self.helper_test_cyclic_encoder(DatetimeAttributePastEncoder,
                                        attribute=attribute,
                                        inf_ts_short=self.inf_ts_short_past,
                                        inf_ts_long=self.inf_ts_long_past,
                                        cyclic=False)

        # test future cyclic encoder
        self.helper_test_cyclic_encoder(DatetimeAttributeFutureEncoder,
                                        attribute=attribute,
                                        inf_ts_short=self.inf_ts_short_future,
                                        inf_ts_long=self.inf_ts_long_future,
                                        cyclic=False)

    def helper_test_cyclic_encoder(self, Encoder, attribute, inf_ts_short, inf_ts_long, cyclic):
        """Test cases for both `CyclicPastEncoder` and `CyclicFutureEncoder`"""
        encoder = Encoder(input_chunk_length=self.input_chunk_length,
                          output_chunk_length=self.output_chunk_length,
                          attribute=attribute)
        result_with_cov = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=cyclic) for ts in self.covariate_multi
        ]
        result_no_cov = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=cyclic) for ts in self.target_multi
        ]
        result_no_cov_inf_short = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=cyclic) for ts in inf_ts_short
        ]
        result_no_cov_inf_long = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=cyclic) for ts in inf_ts_long
        ]

        # test train encoding with covariates
        self.helper_test_encoder_single_train(encoder=encoder,
                                              target=self.target_multi,
                                              covariate=self.covariate_multi,
                                              result=result_with_cov,
                                              merge_covariates=False)

        # test train encoding without covariates
        self.helper_test_encoder_single_train(encoder=encoder,
                                              target=self.target_multi,
                                              covariate=[None]*len(self.target_multi),
                                              result=result_no_cov,
                                              merge_covariates=False)

        # test inference encoding with covariates and n <= output_chunk_length
        self.helper_test_encoder_single_inference(encoder=encoder,
                                                  n=self.n_short,
                                                  target=self.target_multi,
                                                  covariate=self.covariate_multi,
                                                  result=result_with_cov,
                                                  merge_covariates=False)
        # test inference encoding with covariates and n > output_chunk_length
        self.helper_test_encoder_single_inference(encoder=encoder,
                                                  n=self.n_long,
                                                  target=self.target_multi,
                                                  covariate=self.covariate_multi,
                                                  result=result_with_cov,
                                                  merge_covariates=False)

        # test inference encoding without covariates and n <= output_chunk_length
        self.helper_test_encoder_single_inference(encoder=encoder,
                                                  n=self.n_short,
                                                  target=self.target_multi,
                                                  covariate=[None]*len(self.target_multi),
                                                  result=result_no_cov_inf_short,
                                                  merge_covariates=False)
        # test inference encoding without covariates and n > output_chunk_length
        self.helper_test_encoder_single_inference(encoder=encoder,
                                                  n=self.n_long,
                                                  target=self.target_multi,
                                                  covariate=[None]*len(self.target_multi),
                                                  result=result_no_cov_inf_long,
                                                  merge_covariates=False)

    def helper_test_encoder_single_train(self,
                                         encoder: SingleEncoder,
                                         target: Sequence[TimeSeries],
                                         covariate: Sequence[Optional[TimeSeries]],
                                         result: Sequence[TimeSeries],
                                         merge_covariates: bool = True):
        """ test `SingleEncoder.encode_train()`"""
        encoded = []
        for ts, cov in zip(target, covariate):
            encoded.append(encoder.encode_train(ts, cov, merge_covariate=merge_covariates))
        self.assertTrue(encoded == result)

    def helper_test_encoder_single_inference(self,
                                             encoder: SingleEncoder,
                                             n: int,
                                             target: Sequence[TimeSeries],
                                             covariate: Sequence[Optional[TimeSeries]],
                                             result: Sequence[TimeSeries],
                                             merge_covariates: bool = True):
        """ test `SingleEncoder.encode_inference()`"""
        encoded = []
        for ts, cov in zip(target, covariate):
            encoded.append(encoder.encode_inference(n, ts, cov, merge_covariate=merge_covariates))
        self.assertTrue(encoded == result)


