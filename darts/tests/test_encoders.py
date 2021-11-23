import pandas as pd
import numpy as np

from typing import Sequence, Optional
from .base_test_class import DartsBaseTestClass
from ..utils import timeseries_generation as tg
from ..timeseries import TimeSeries
from ..utils.data.covariate_index_generators import (CovariateIndexGenerator,
                                                     PastCovariateIndexGenerator,
                                                     FutureCovariateIndexGenerator)

from ..logging import get_logger
logger = get_logger(__name__)


def _add_cyclic_encoder(self,
                        target: Sequence[TimeSeries],
                        future_covariates: Optional[Sequence[TimeSeries]] = None,
                        n: Optional[int] = None) -> Sequence[TimeSeries]:
    """adds cyclic encoding of time index to future covariates.
    For training (when `n` is `None`) we can simply use the future covariates (if available) or target as
    reference to extract the time index.
    For prediction (`n` is given) we have to distinguish between two cases:
        1)
            if future covariates are given, we can use them as reference
        2)
            if future covariates are missing, we need to generate a time index that starts `input_chunk_length`
            before the end of `target` and ends `max(n, output_chunk_length)` after the end of `target`

    Parameters
    ----------
    target
        past target TimeSeries
    future_covariates
        future covariates TimeSeries
    n
        prediciton length (only given for predictions)

    Returns
    -------
    Sequence[TimeSeries]
        future covariates including cyclic encoded time index
    """

    if n is None:  # training
        encode_ts = future_covariates if future_covariates is not None else target
    else:  # prediction
        if future_covariates is not None:
            encode_ts = future_covariates
        else:
            encode_ts = [tg._generate_index(start=ts.end_time() - ts.freq * (self.input_chunk_length - 1),
                                            length=self.input_chunk_length + max(n, self.output_chunk_length),
                                            freq=ts.freq) for ts in target]

    encoded_times = [
        tg.datetime_attribute_timeseries(ts,
                                         attribute=self.add_cyclic_encoder,
                                         cyclic=True,
                                         dtype=target[0].dtype)
        for ts in encode_ts
    ]

    if future_covariates is None:
        future_covariates = encoded_times
    else:
        future_covariates = [fc.stack(et) for fc, et in zip(future_covariates, encoded_times)]

    return future_covariates


class Encoders(DartsBaseTestClass):
    n_target = 24
    target_time = tg.linear_timeseries(length=n_target, freq='MS')
    cov_time_train = tg.datetime_attribute_timeseries(target_time, attribute='month', cyclic=True)
    cov_time_train_short = cov_time_train[1:]

    target_int = tg.linear_timeseries(length=n_target, start=2)
    cov_int_train = target_int
    cov_int_train_short = cov_int_train[1:]


    input_chunk_length = 12
    output_chunk_length = 6
    n_short = 6
    n_long = 8

    cov_time_inf_short = TimeSeries.from_times_and_values(tg._generate_index(start=target_time.start_time(),
                                                                             length=n_target + n_short,
                                                                             freq=target_time.freq),
                                                          np.arange(n_target + n_short))

    cov_time_inf_long = TimeSeries.from_times_and_values(tg._generate_index(start=target_time.start_time(),
                                                                            length=n_target + n_long,
                                                                            freq=target_time.freq),
                                                         np.arange(n_target + n_long))
    cov_int_inf_short = TimeSeries.from_times_and_values(tg._generate_index(start=target_int.start_time(),
                                                                            length=n_target + n_short,
                                                                            freq=target_int.freq),
                                                         np.arange(n_target + n_short))
    cov_int_inf_long = TimeSeries.from_times_and_values(tg._generate_index(start=target_int.start_time(),
                                                                           length=n_target + n_long,
                                                                           freq=target_int.freq),
                                                        np.arange(n_target + n_long))
