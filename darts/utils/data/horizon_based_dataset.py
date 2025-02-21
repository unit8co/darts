"""
Horizon-Based Training Dataset
------------------------------
"""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.training_dataset import PastCovariatesTrainingDataset
from darts.utils.data.utils import CovariateType, _process_sample_weight
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


class HorizonBasedDataset(PastCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        output_chunk_length: int = 12,
        lh: tuple[int, int] = (1, 3),
        lookback: int = 3,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ) -> None:
        """
        A time series dataset containing tuples of (past_target, past_covariates, static_covariates, sample weights,
        future_target)
        arrays,
        in a way inspired by the N-BEATS way of training on the M4 dataset: https://arxiv.org/abs/1905.10437.

        The "past" series have length `lookback * output_chunk_length`, and the "future" series has length
        `output_chunk_length`.

        Given the horizon `output_chunk_length` of a model, this dataset will compute some "past/future"
        splits as follows:
        First a "forecast point" is selected in the the range of the last
        `(min_lh * output_chunk_length, max_lh * output_chunk_length)` points before the end of the time series.
        The "future" then consists in the following `output_chunk_length` points, and the "past" will be the preceding
        `lookback * output_chunk_length` points.

        All the series in the provided sequence must be long enough; i.e. have length at least
        `(lookback + max_lh) * output_chunk_length`, and `min_lh` must be at least 1
        (to have targets of length exactly `1 * output_chunk_length`).
        The target and covariates time series are sliced together using their time indexes for alignment.

        The sampling is uniform both over the number of time series and the number of samples per series;
        i.e. the i-th sample of this dataset has 1/(N*M) chance of coming from any of the M samples in any of the N
        time series in the sequence.

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        output_chunk_length
            The length of the "output" series emitted by the model
        lh
            A `(min_lh, max_lh)` interval for the forecast point, starting from the end of the series.
            For example, `(1, 3)` will select forecast points uniformly between `1*H` and `3*H` points
            before the end of the series. It is required that `min_lh >= 1`.
        lookback:
            A integer interval for the length of the input in the emitted input and output splits, expressed as a
            multiple of `output_chunk_length`. For instance, `lookback=3` will emit "inputs" of lengths
            `3 * output_chunk_length`.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels. They are applied per observation,
            per label (each step in `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed globally based on the length of the longest series in `series`. Then for each series, the weights
            are extracted from the end of the global weights. This gives a common time weighting across all series.
        """
        super().__init__()

        self.target_series = series2seq(target_series)
        self.covariates = series2seq(covariates)
        self.covariate_type = CovariateType.PAST

        if covariates is not None and len(self.target_series) != len(self.covariates):
            raise_log(
                ValueError(
                    "The provided sequence of target series must have the same length as "
                    "the provided sequence of covariate series."
                ),
                logger=logger,
            )
        self.sample_weight = _process_sample_weight(sample_weight, self.target_series)

        self.output_chunk_length = output_chunk_length
        self.min_lh, self.max_lh = lh
        self.lookback = lookback

        # Checks
        if not (self.max_lh >= self.min_lh >= 1):
            raise_log(
                ValueError(
                    "The lh parameter should be an int tuple (min_lh, max_lh), "
                    "with 1 <= min_lh <= max_lh"
                ),
                logger=logger,
            )
        self.nr_samples_per_ts = (self.max_lh - self.min_lh) * self.output_chunk_length
        self.total_nr_samples = len(self.target_series) * self.nr_samples_per_ts
        self.use_static_covariates = use_static_covariates

    def __len__(self):
        """
        Returns the total number of possible (input, target) splits.
        """
        return self.total_nr_samples

    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        # determine the index of the time series.
        target_idx = idx // self.nr_samples_per_ts
        target_series = self.target_series[target_idx]
        target_vals = target_series.random_component_values(copy=False)

        if len(target_vals) < (self.lookback + self.max_lh) * self.output_chunk_length:
            raise_log(
                ValueError(
                    "The dataset contains some input/target series that are shorter than "
                    f"`(lookback + max_lh) * H` ({target_idx}-th series)"
                ),
                logger=logger,
            )

        # determine the index lh_idx of the forecasting point (the last point of the input series, before the target)
        # lh_idx should be in [0, self.nr_samples_per_ts)
        lh_idx = idx - (target_idx * self.nr_samples_per_ts)

        # determine the index at the end of the output chunk
        end_of_output_idx = len(target_series) - (
            (self.min_lh - 1) * self.output_chunk_length + lh_idx
        )

        # optionally, load covariates
        covariate_series = (
            self.covariates[target_idx] if self.covariates is not None else None
        )
        main_covariate_type = (
            CovariateType.NONE if self.covariates is None else CovariateType.PAST
        )

        # optionally, load sample weight
        sample_weight_series = (
            self.sample_weight[target_idx] if self.sample_weight is not None else None
        )

        shift = self.lookback * self.output_chunk_length
        input_chunk_length = shift

        # get all indices for the current sample
        (
            past_start,
            past_end,
            future_start,
            future_end,
            cov_start,
            cov_end,
            sample_weight_start,
            sample_weight_end,
        ) = self._memory_indexer(
            target_idx=target_idx,
            target_series=target_series,
            shift=shift,
            input_chunk_length=input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            end_of_output_idx=end_of_output_idx,
            covariate_series=covariate_series,
            covariate_type=main_covariate_type,
            sample_weight_series=sample_weight_series,
        )

        # extract sample target
        future_target = target_vals[future_start:future_end]
        past_target = target_vals[past_start:past_end]

        # extract sample covariates
        covariate = None
        if self.covariates is not None:
            if cov_end > len(covariate_series):
                raise_log(
                    ValueError(
                        f"The dataset contains past covariates that don't extend far enough into the future. "
                        f"({idx}-th sample)"
                    ),
                    logger=logger,
                )
            covariate = covariate_series.random_component_values(copy=False)[
                cov_start:cov_end
            ]
            if len(covariate) != len(past_target):
                raise_log(
                    ValueError(
                        "The dataset contains past covariates whose time axis doesn't allow to obtain the "
                        "input (or output) chunk relative to the target series."
                    ),
                    logger=logger,
                )

        # extract sample weights
        sample_weight = None
        if self.sample_weight is not None:
            if sample_weight_end > len(sample_weight_series):
                raise_log(
                    ValueError(
                        f"The dataset contains sample weights "
                        f"that don't extend far enough into the future. ({idx}-th sample)"
                    ),
                    logger=logger,
                )

            sample_weight = sample_weight_series.random_component_values(copy=False)[
                sample_weight_start:sample_weight_end
            ]

            if len(sample_weight) != self.output_chunk_length:
                raise_log(
                    ValueError(
                        "The dataset contains sample weights whose time axis doesn't allow to obtain "
                        "the input (or output) chunk relative to the target series."
                    ),
                    logger=logger,
                )

        # extract sample static covariates
        if self.use_static_covariates:
            static_covariate = target_series.static_covariates_values(copy=False)
        else:
            static_covariate = None
        return past_target, covariate, static_covariate, sample_weight, future_target
