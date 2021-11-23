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

    def test_encoder_from_model_params(self):
        # valid encoder model parameters are ('past', 'future') for the main key and datetime attribute for sub keys
        valid_encoder_args = {'past': ['month'], 'future': ['dayofyear', 'dayofweek']}
        encoders = self.helper_encoder_from_model(add_encoder_dict=valid_encoder_args)

        self.assertTrue(len(encoders.past_encoders) == 1)
        self.assertTrue(len(encoders.future_encoders) == 2)
        self.assertTrue(encoders.past_encoders[0].attribute == 'month')
        self.assertTrue([enc.attribute for enc in encoders.future_encoders] == ['dayofyear', 'dayofweek'])

        valid_encoder_args = {'past': ['month'], 'future': ['dayofyear', 'dayofweek']}
        encoders = self.helper_encoder_from_model(add_encoder_dict=valid_encoder_args, takes_future_covariates=False)
        self.assertTrue(len(encoders.past_encoders) == 1)
        self.assertTrue(len(encoders.future_encoders) == 0)

        # test invalid kwargs at model creation
        bad_time = {'ppast': ['month']}
        with self.assertRaises(ValueError):
            _ = self.helper_encoder_from_model(add_encoder_dict=bad_time)

        bad_attribute = {'past': ['year']}
        with self.assertRaises(ValueError):
            _ = self.helper_encoder_from_model(add_encoder_dict=bad_attribute)

        bad_type = {'past': 1}
        with self.assertRaises(ValueError):
            _ = self.helper_encoder_from_model(add_encoder_dict=bad_type)

    def helper_encoder_from_model(self, add_encoder_dict, takes_past_covariates=True, takes_future_covariates=True):
        model = TFTModel(input_chunk_length=self.input_chunk_length,
                         output_chunk_length=self.output_chunk_length,
                         add_cyclic_encoder=add_encoder_dict)

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

    def test_encoder_sequence(self):
        pass

    def test_cyclic_encoder(self):
        attribute = 'month'

        # test past cyclic encoder
        self.helper_test_cyclic_encoder(CyclicPastEncoder,
                                        attribute=attribute,
                                        inf_ts_short=self.inf_ts_short_past,
                                        inf_ts_long=self.inf_ts_long_past)

        # test future cyclic encoder
        self.helper_test_cyclic_encoder(CyclicFutureEncoder,
                                        attribute=attribute,
                                        inf_ts_short=self.inf_ts_short_future,
                                        inf_ts_long=self.inf_ts_long_future)

    def helper_test_cyclic_encoder(self, CyclicEncoder, attribute, inf_ts_short, inf_ts_long):
        encoder = CyclicEncoder(input_chunk_length=self.input_chunk_length,
                                output_chunk_length=self.output_chunk_length,
                                attribute=attribute)
        result_with_cov = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=True) for ts in self.covariate_multi
        ]
        result_no_cov = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=True) for ts in self.target_multi
        ]
        result_no_cov_inf_short = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=True) for ts in inf_ts_short
        ]
        result_no_cov_inf_long = [
            tg.datetime_attribute_timeseries(ts, attribute=attribute, cyclic=True) for ts in inf_ts_long
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
        encoded = []
        for ts, cov in zip(target, covariate):
            encoded.append(encoder.encode_inference(n, ts, cov, merge_covariate=merge_covariates))
        self.assertTrue(encoded == result)


