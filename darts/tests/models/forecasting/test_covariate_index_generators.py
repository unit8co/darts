import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg
from darts.utils.data.encoder_base import (
    CovariateIndexGenerator,
    FutureCovariateIndexGenerator,
    PastCovariateIndexGenerator,
)

logger = get_logger(__name__)


class CovariateIndexGeneratorTestCase(DartsBaseTestClass):
    n_target = 24
    target_time = tg.linear_timeseries(length=n_target, freq="MS")
    cov_time_train = tg.datetime_attribute_timeseries(
        target_time, attribute="month", cyclic=True
    )
    cov_time_train_short = cov_time_train[1:]

    target_int = tg.linear_timeseries(length=n_target, start=2)
    cov_int_train = target_int
    cov_int_train_short = cov_int_train[1:]

    input_chunk_length = 12
    output_chunk_length = 6
    n_short = 6
    n_long = 8

    # pd.DatetimeIndex
    # target covariate for inference dataset for n <= output_chunk_length
    cov_time_inf_short = TimeSeries.from_times_and_values(
        tg._generate_index(
            start=target_time.start_time(),
            length=n_target + n_short,
            freq=target_time.freq,
        ),
        np.arange(n_target + n_short),
    )
    # target covariate for inference dataset for n > output_chunk_length
    cov_time_inf_long = TimeSeries.from_times_and_values(
        tg._generate_index(
            start=target_time.start_time(),
            length=n_target + n_long,
            freq=target_time.freq,
        ),
        np.arange(n_target + n_long),
    )

    # integer index
    # target covariate for inference dataset for n <= output_chunk_length
    cov_int_inf_short = TimeSeries.from_times_and_values(
        tg._generate_index(
            start=target_int.start_time(),
            length=n_target + n_short,
            freq=target_int.freq,
        ),
        np.arange(n_target + n_short),
    )
    # target covariate for inference dataset for n > output_chunk_length
    cov_int_inf_long = TimeSeries.from_times_and_values(
        tg._generate_index(
            start=target_int.start_time(),
            length=n_target + n_long,
            freq=target_int.freq,
        ),
        np.arange(n_target + n_long),
    )

    def helper_test_index_types(self, ig: CovariateIndexGenerator):
        """test the index type of generated index"""
        # pd.DatetimeIndex
        idx = ig.generate_train_series(self.target_time, self.cov_time_train)
        self.assertTrue(isinstance(idx, pd.DatetimeIndex))
        idx = ig.generate_inference_series(
            self.n_short, self.target_time, self.cov_time_inf_short
        )
        self.assertTrue(isinstance(idx, pd.DatetimeIndex))
        idx = ig.generate_train_series(self.target_time, None)
        self.assertTrue(isinstance(idx, pd.DatetimeIndex))

        # pd.RangeIndex
        idx = ig.generate_train_series(self.target_int, self.cov_int_train)
        self.assertTrue(isinstance(idx, pd.RangeIndex))
        idx = ig.generate_inference_series(
            self.n_short, self.target_int, self.cov_int_inf_short
        )
        self.assertTrue(isinstance(idx, pd.RangeIndex))
        idx = ig.generate_train_series(self.target_int, None)
        self.assertTrue(isinstance(idx, pd.RangeIndex))

    def helper_test_index_generator_train(self, ig: CovariateIndexGenerator):
        """
        If covariates are given, the index generators should return the covariate series' index.
        If covariates are not given, the index generators should return the target series' index.
        """
        # pd.DatetimeIndex
        # generated index must be equal to input covariate index
        idx = ig.generate_train_series(self.target_time, self.cov_time_train)
        self.assertTrue(idx.equals(self.cov_time_train.time_index))
        # generated index must be equal to input covariate index
        idx = ig.generate_train_series(self.target_time, self.cov_time_train_short)
        self.assertTrue(idx.equals(self.cov_time_train_short.time_index))
        # generated index must be equal to input target index when no covariates are defined
        idx = ig.generate_train_series(self.target_time, None)
        self.assertTrue(idx.equals(self.cov_time_train.time_index))

        # integer index
        # generated index must be equal to input covariate index
        idx = ig.generate_train_series(self.target_int, self.cov_int_train)
        self.assertTrue(idx.equals(self.cov_int_train.time_index))
        # generated index must be equal to input covariate index
        idx = ig.generate_train_series(self.target_time, self.cov_int_train_short)
        self.assertTrue(idx.equals(self.cov_int_train_short.time_index))
        # generated index must be equal to input target index when no covariates are defined
        idx = ig.generate_train_series(self.target_int, None)
        self.assertTrue(idx.equals(self.cov_int_train.time_index))

    def helper_test_index_generator_inference(self, ig, is_past=False):
        """
        For prediction (`n` is given) with past covariates we have to distinguish between two cases:
        1)  if past covariates are given, we can use them as reference
        2)  if past covariates are missing, we need to generate a time index that starts `input_chunk_length`
            before the end of `target` and ends `max(0, n - output_chunk_length)` after the end of `target`

        For prediction (`n` is given) with future covariates we have to distinguish between two cases:
        1)  if future covariates are given, we can use them as reference
        2)  if future covariates are missing, we need to generate a time index that starts `input_chunk_length`
            before the end of `target` and ends `max(n, output_chunk_length)` after the end of `target`
        """

        # check generated inference index without passing covariates when n <= output_chunk_length
        idx = ig.generate_inference_series(self.n_short, self.target_time, None)
        if is_past:
            n_out = self.input_chunk_length
            last_idx = self.target_time.end_time()
        else:
            n_out = self.input_chunk_length + self.output_chunk_length
            last_idx = self.cov_time_inf_short.end_time()

        self.assertTrue(len(idx) == n_out)
        self.assertTrue(idx[-1] == last_idx)

        # check generated inference index without passing covariates when n > output_chunk_length
        idx = ig.generate_inference_series(self.n_long, self.target_time, None)
        if is_past:
            n_out = self.input_chunk_length + self.n_long - self.output_chunk_length
            last_idx = (
                self.target_time.end_time()
                + (self.n_long - self.output_chunk_length) * self.target_time.freq
            )
        else:
            n_out = self.input_chunk_length + self.n_long
            last_idx = self.cov_time_inf_long.end_time()

        self.assertTrue(len(idx) == n_out)
        self.assertTrue(idx[-1] == last_idx)

        idx = ig.generate_inference_series(
            self.n_short, self.target_time, self.cov_time_inf_short
        )
        self.assertTrue(idx.equals(self.cov_time_inf_short.time_index))
        idx = ig.generate_inference_series(
            self.n_long, self.target_time, self.cov_time_inf_long
        )
        self.assertTrue(idx.equals(self.cov_time_inf_long.time_index))
        idx = ig.generate_inference_series(
            self.n_short, self.target_int, self.cov_int_inf_short
        )
        self.assertTrue(idx.equals(self.cov_int_inf_short.time_index))
        idx = ig.generate_inference_series(
            self.n_long, self.target_int, self.cov_int_inf_long
        )
        self.assertTrue(idx.equals(self.cov_int_inf_long.time_index))

    def test_past_index_generator(self):
        ig = PastCovariateIndexGenerator(
            self.input_chunk_length, self.output_chunk_length
        )
        self.helper_test_index_types(ig)
        self.helper_test_index_generator_train(ig)
        self.helper_test_index_generator_inference(ig, is_past=True)

    def test_future_index_generator(self):
        ig = FutureCovariateIndexGenerator(
            self.input_chunk_length, self.output_chunk_length
        )
        self.helper_test_index_types(ig)
        self.helper_test_index_generator_train(ig)
        self.helper_test_index_generator_inference(ig, is_past=False)
