"""
Horizon-Based Training Dataset
------------------------------
"""

from collections.abc import Sequence
from typing import Optional, Union

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.training_dataset import DatasetOutputType, TrainingDataset
from darts.utils.data.utils import FeatureType, _process_sample_weight
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


class HorizonBasedDataset(TrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
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
        First a "forecast point" is selected in the range of the last
        `(min_lh * output_chunk_length, max_lh * output_chunk_length)` points before the end of the time series.
        The "future" then consists in the following `output_chunk_length` points, and the "past" will be the preceding
        `lookback * output_chunk_length` points.

        All the series in the provided sequence must be long enough; i.e. have length at least
        `(lookback + max_lh) * output_chunk_length`, and `min_lh` must be at least 1
        (to have targets of length exactly `1 * output_chunk_length`).
        The target and past_covariates time series are sliced together using their time indexes for alignment.

        The sampling is uniform both over the number of time series and the number of samples per series;
        i.e. the i-th sample of this dataset has 1/(N*M) chance of coming from any of the M samples in any of the N
        time series in the sequence.

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        past_covariates
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

        # setup target and sequence
        target_series = series2seq(target_series)
        past_covariates = series2seq(past_covariates)

        if past_covariates is not None and len(target_series) != len(past_covariates):
            raise_log(
                ValueError(
                    "The provided sequence of target `series` must have the same length as "
                    f"the provided sequence of `{FeatureType.PAST_COVARIATES.value}`."
                ),
                logger=logger,
            )

        self.target_series = target_series
        self.past_covariates = past_covariates
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

    def __getitem__(self, idx: int) -> DatasetOutputType:
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
        past_covariates = (
            self.past_covariates[target_idx]
            if self.past_covariates is not None
            else None
        )

        # optionally, load sample weight
        sample_weight = (
            self.sample_weight[target_idx] if self.sample_weight is not None else None
        )

        shift = self.lookback * self.output_chunk_length
        input_chunk_length = shift

        # get start and end indices (positions) of all feature types for the current sample
        idx_bounds = self._memory_indexer(
            target_idx=target_idx,
            target_series=target_series,
            shift=shift,
            input_chunk_length=input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            end_of_output_idx=end_of_output_idx,
            past_covariates=past_covariates,
            future_covariates=None,
            sample_weight=sample_weight,
        )

        # extract past target
        start, end = idx_bounds[FeatureType.PAST_TARGET]
        pt = target_vals[start:end]

        # extract future target
        start, end = idx_bounds[FeatureType.FUTURE_TARGET]
        ft = target_vals[start:end]

        # past cov, historic future cov, future cov, static cov, sample weight
        pc, hfc, fc, sc, sw = None, None, None, None, None

        # extract sample covariates
        if self.past_covariates is not None:
            start, end = idx_bounds[FeatureType.PAST_COVARIATES]
            if end > len(past_covariates):
                raise_log(
                    ValueError(
                        f"The dataset contains `{FeatureType.PAST_COVARIATES.value}` that "
                        f"don't extend far enough into the future ({idx}-th sample)."
                    ),
                    logger=logger,
                )
            pc = past_covariates.random_component_values(copy=False)[start:end]
            if len(pc) != len(pt):
                raise_log(
                    ValueError(
                        f"The dataset contains `{FeatureType.PAST_COVARIATES.value}` whose "
                        f"time axis doesn't allow to obtain the input (or output) chunk "
                        f"relative to the target series."
                    ),
                    logger=logger,
                )

        # extract sample weights
        if self.sample_weight is not None:
            start, end = idx_bounds[FeatureType.SAMPLE_WEIGHT]
            if end > len(sample_weight):
                raise_log(
                    ValueError(
                        f"The dataset contains `{FeatureType.SAMPLE_WEIGHT.value}` series "
                        f"that don't extend far enough into the future. ({idx}-th sample)"
                    ),
                    logger=logger,
                )

            sw = sample_weight.random_component_values(copy=False)[start:end]

            if len(sw) != self.output_chunk_length:
                raise_log(
                    ValueError(
                        f"The dataset contains `{FeatureType.SAMPLE_WEIGHT.value}` series "
                        f"whose time axis don't allow to obtain the input (or output) "
                        f"chunk relative to the target series."
                    ),
                    logger=logger,
                )

        # extract sample static covariates
        if self.use_static_covariates:
            sc = target_series.static_covariates_values(copy=False)

        return (
            pt,
            pc,
            hfc,
            fc,
            sc,
            sw,
            ft,
        )
