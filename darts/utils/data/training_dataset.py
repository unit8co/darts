"""
Training Datasets
-----------------

- :class:`~darts.utils.data.training_dataset.TrainingDataset`
- :class:`~darts.utils.data.training_dataset.ShiftedTrainingDataset`
- :class:`~darts.utils.data.training_dataset.SequentialTrainingDataset`
- :class:`~darts.utils.data.training_dataset.HorizonBasedTrainingDataset`
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

from torch.utils.data import Dataset

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.utils import (
    FeatureType,
    TrainingDatasetOutput,
    _process_sample_weight,
)
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)

_SampleIndexType = dict[FeatureType, tuple[Optional[int], Optional[int]]]

_SERIES_TYPES = [
    FeatureType.PAST_TARGET,
    FeatureType.FUTURE_TARGET,
    FeatureType.PAST_COVARIATES,
    FeatureType.HISTORIC_FUTURE_COVARIATES,
    FeatureType.FUTURE_COVARIATES,
    FeatureType.SAMPLE_WEIGHT,
]


class TrainingDataset(ABC, Dataset):
    def __init__(self):
        """
        Abstract class for all training datasets that can be used with Darts' `TorchForecastingModel`.

        Each sample drawn from this dataset must be a seven-element tuple extracted from a specific time window and
        set of single input `TimeSeries`. The elements are:

        - past_target: target `series` values in the input chunk
        - past_covariates: Optional `past_covariates` values in the input chunk
        - historic_future_covariates: Optional `future_covariates` values in the input chunk
        - future_covariates: Optional `future_covariates` values in the output chunk
        - static_covariates: Optional `static_covariates` values of the `series`
        - sample_weight: Optional `sample_weight` values in the output chunk
        - future_target: `series` values in the output chunk

        Darts `TorchForecastingModel` can be fit from instances of `TrainingDataset` using the `fit_from_dataset()`
        method.

        `TrainingDataset` inherits torch `Dataset`; meaning that the implementations have to provide the
        `__getitem__()` method.

        It contains `np.ndarray` (and not `TimeSeries`), because training requires the values only,
        and so we can get big performance gains when slicing by returning only numpy views of the data
        underlying the `TimeSeries`.
        """

        self._index_memory: dict = {}

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> TrainingDatasetOutput:
        pass

    def _memory_indexer(
        self,
        series_idx: int,
        series: TimeSeries,
        shift: int,
        input_chunk_length: int,
        output_chunk_length: int,
        end_of_output_idx: int,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        sample_weight: Optional[TimeSeries] = None,
    ) -> _SampleIndexType:
        """Returns the (start, end) indices for each feature type (past target, future target, past covariates,
        historic future covariates, future covariates, and sample weight) of the current sample `i` from `series_idx`.

        Works for all TimeSeries index types: pd.DatetimeIndex, pd.RangeIndex (and the deprecated Int64Index)

        When `series_idx` is observed for the first time, it stores the position of the sample `0` within the full
        target time series and the (start, end) indices of all sub sets.
        This allows to calculate the sub set indices for all future samples `i` by simply adjusting for the difference
        between the positions of sample `i` and sample `0`.

        Parameters
        ----------
        series_idx
            index of the current target TimeSeries.
        series
            current target TimeSeries.
        shift
            The number of time steps by which to shift the output chunks relative to the input chunks.
        input_chunk_length
            The length of the lookback / past window the model takes as input.
        output_chunk_length
            The length of the lookahead / future window that the model emits as output (for the target) and takes as
            input (for future covariates).
        end_of_output_idx
            the index where the output chunk of the current sample ends in `series`.
        past_covariates
            current `past_covariates` TimeSeries.
        future_covariates
            current `future_covariates` TimeSeries.
        sample_weight
            current sample weight TimeSeries.
        """
        # store the start and end index (positions) for series and covariates
        idx_bounds = {}

        # the first time series_idx is observed
        if series_idx not in self._index_memory:
            start_of_output_idx = end_of_output_idx - output_chunk_length
            start_of_input_idx = start_of_output_idx - shift

            # select forecast point and target period, using the previously computed indexes
            future_start, future_end = (
                start_of_output_idx,
                start_of_output_idx + output_chunk_length,
            )

            # select input period; look at the `input_chunk_length` points after start of input
            past_start, past_end = (
                start_of_input_idx,
                start_of_input_idx + input_chunk_length,
            )
            idx_bounds[FeatureType.PAST_TARGET] = (past_start, past_end)
            idx_bounds[FeatureType.FUTURE_TARGET] = (future_start, future_end)

            series_times = series._time_index

            for cov, cov_type, start, end in zip(
                [past_covariates, future_covariates, future_covariates],
                [
                    FeatureType.PAST_COVARIATES,
                    FeatureType.HISTORIC_FUTURE_COVARIATES,
                    FeatureType.FUTURE_COVARIATES,
                ],
                [past_start, past_start, future_start],
                [past_end, past_end, future_end],
            ):
                if cov is None:
                    idx_bounds[cov_type] = (None, None)
                    continue

                # we need to be careful with getting ranges and indexes:
                # to get entire range, full_range = ts[:len(ts)]; to get last index: last_idx = ts[len(ts) - 1]
                # extract actual index value (respects datetime- and integer-based indexes; also from non-zero
                # start)
                cov_times = cov._time_index
                start_time = series_times[start]
                end_time = series_times[end - 1]

                if start_time not in cov_times or end_time not in cov_times:
                    raise_log(
                        ValueError(
                            f"Missing covariates; could not find `{cov_type.value}` in index "
                            f"value range: {start_time} - {end_time}."
                        ),
                        logger=logger,
                    )

                # extract the index position (index) from index value
                cov_start = cov_times.get_loc(start_time)
                cov_end = cov_times.get_loc(end_time) + 1
                idx_bounds[cov_type] = (cov_start, cov_end)

            # sample weight
            if sample_weight is not None:
                # extract the index position (index) from index value
                weight_times = sample_weight._time_index

                start_time = series_times[future_start]
                end_time = series_times[future_end - 1]

                if start_time not in weight_times or end_time not in weight_times:
                    raise_log(
                        ValueError(
                            f"Invalid `{FeatureType.SAMPLE_WEIGHT.value}`; could not find "
                            f"sample weights in index value range: {start_time} - {end_time}."
                        ),
                        logger=logger,
                    )

                sample_weight_start = weight_times.get_loc(start_time)
                sample_weight_end = weight_times.get_loc(end_time) + 1
                idx_bounds[FeatureType.SAMPLE_WEIGHT] = (
                    sample_weight_start,
                    sample_weight_end,
                )
            else:
                idx_bounds[FeatureType.SAMPLE_WEIGHT] = (None, None)

            # store position of initial sample and all relevant sub set indices
            self._index_memory[series_idx] = {
                "end_of_output_idx": end_of_output_idx,
                **idx_bounds,
            }
        else:
            # load position of initial sample and its sub set indices
            end_of_output_idx_last = self._index_memory[series_idx]["end_of_output_idx"]
            # evaluate how much the new sample needs to be shifted, and shift all indexes
            idx_shift = end_of_output_idx - end_of_output_idx_last

            for series_type in _SERIES_TYPES:
                start, end = self._index_memory[series_idx][series_type]

                idx_bounds[series_type] = (
                    start + idx_shift if start is not None else start,
                    end + idx_shift if end is not None else end,
                )

        return idx_bounds


class ShiftedTrainingDataset(TrainingDataset):
    def __init__(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        shift: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """Shifted Training Dataset

        Each sample drawn from this dataset is a seven-element tuple extracted from a specific time window and
        set of single input `TimeSeries`. The elements are:

        - past_target: target `series` values in the input chunk
        - past_covariates: `past_covariates` values in the input chunk (`None` if `past_covariates=None`)
        - historic_future_covariates: `future_covariates` values in the input chunk (`None` if `future_covariates=None`)
        - future_covariates: `future_covariates` values in the output chunk (`None` if `future_covariates=None`)
        - static_covariates: `static_covariates` values of the `series` (`None` if `use_static_covariates=False`)
        - sample_weight: `sample_weight` values in the output chunk (`None` if `sample_weight=None`)
        - future_target: `series` values in the output chunk

        The output chunk starts `shift` after the input chunk's start.

        The sample index determines:

        - the position / time of the extracted chunks relative to the end of a single target `series`
        - the index (which series and covariates) to use in case `series` (and covariates) are
          passed as a sequence of series.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        .. note::
            Each series in the provided sequence must have a minimum length of
            `max(input_chunk_length, shift + output_chunk_length)`.

        Parameters
        ----------
        series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past covariates.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates.
        input_chunk_length
            The length of the lookback / past window the model takes as input.
        output_chunk_length
            The length of the lookahead / future window that the model emits as output (for the target) and takes as
            input (for future covariates).
        shift
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of samples that can be produced per time series. It can be used to
            limit the total size of the dataset and ensure proper sampling. If `None`, will read all individual time
            series in advance (at dataset creation) to check their sizes. This might be expensive on big datasets.
            If not `None`, will only keep a maximum of `max_samples_per_ts` samples per series, extracted from the most
            recent past.
        use_static_covariates
            Whether to use/include static covariate data from the target `series`.
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
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        static_covariates = (
            series[0].static_covariates if use_static_covariates else None
        )

        for cov, cov_type in zip(
            [past_covariates, future_covariates],
            [FeatureType.PAST_COVARIATES, FeatureType.FUTURE_COVARIATES],
        ):
            if cov is not None and len(series) != len(cov):
                name = cov_type.value
                raise_log(
                    ValueError(
                        f"The sequence of `{name}` must have the same length as "
                        f"the sequence of target `series`."
                    ),
                    logger=logger,
                )

        self.series = series
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates

        self.uses_past_covariates = past_covariates is not None
        self.uses_future_covariates = future_covariates is not None
        self.uses_static_covariates_covariates = static_covariates is not None

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.shift = shift
        self.max_samples_per_ts = max_samples_per_ts
        self.size_of_both_chunks = max(
            self.input_chunk_length, self.shift + self.output_chunk_length
        )

        # setup sample weights; ignore weights when `ocl==0`
        if sample_weight is not None and output_chunk_length > 0:
            self.sample_weight = _process_sample_weight(sample_weight, self.series)
        else:
            self.sample_weight = None

        # setup samples
        if self.max_samples_per_ts is None:
            # read all time series to get the maximum size
            self.max_samples_per_ts = (
                max(len(ts) for ts in self.series) - self.size_of_both_chunks + 1
            )
        self.ideal_nr_samples = len(self.series) * self.max_samples_per_ts

    def __len__(self):
        return self.ideal_nr_samples

    def __getitem__(self, idx) -> TrainingDatasetOutput:
        # determine the index of the time series.
        series_idx = idx // self.max_samples_per_ts
        series = self.series[series_idx]
        series_vals = series.random_component_values(copy=False)

        # determine the index at the end of the output chunk
        end_of_output_idx = self._get_end_of_output_idx(series, series_idx, idx)

        # load covariates
        past_covariates = (
            self.past_covariates[series_idx] if self.uses_past_covariates else None
        )
        future_covariates = (
            self.future_covariates[series_idx] if self.uses_future_covariates else None
        )

        # optionally, load sample weight
        sample_weight = None
        if self.sample_weight is not None:
            sample_weight = self.sample_weight[series_idx]
            weight_n_comp = sample_weight.n_components
            if weight_n_comp > 1 and weight_n_comp != series.n_components:
                raise_log(
                    ValueError(
                        f"The number of components in `{FeatureType.SAMPLE_WEIGHT.value}` must "
                        f"either be `1` or match the number of target series components "
                        f"`{series.n_components}` ({series_idx}-th series)."
                    ),
                    logger=logger,
                )

        # get start and end indices (positions) of all feature types for the current sample
        idx_bounds = self._memory_indexer(
            series_idx=series_idx,
            series=series,
            shift=self.shift,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            end_of_output_idx=end_of_output_idx,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            sample_weight=sample_weight,
        )

        # extract past target series
        start, end = idx_bounds[FeatureType.PAST_TARGET]
        pt = series_vals[start:end]

        # extract future target series
        start, end = idx_bounds[FeatureType.FUTURE_TARGET]
        ft = series_vals[start:end]

        # past cov, historic future cov, future cov, static cov, sample weight
        pc, hfc, fc, sc, sw = None, None, None, None, None

        # extract past covariates
        if self.uses_past_covariates:
            start, end = idx_bounds[FeatureType.PAST_COVARIATES]
            if end > len(past_covariates):
                raise_log(
                    ValueError(
                        f"The dataset contains `{FeatureType.PAST_COVARIATES.value}` "
                        f"that don't extend far enough into the future. ({idx}-th sample)"
                    ),
                    logger=logger,
                )

            pc = past_covariates.random_component_values(copy=False)[start:end]

            if len(pc) != self.input_chunk_length:
                raise_log(
                    ValueError(
                        f"The dataset contains `{FeatureType.PAST_COVARIATES.value}` "
                        f"whose time axis doesn't allow to obtain the input (and / or output) chunk relative to the "
                        f"target `series`."
                    ),
                    logger=logger,
                )

        # extract future covariates
        if self.uses_future_covariates:
            # future part of future covariates
            start, end = idx_bounds[FeatureType.FUTURE_COVARIATES]
            if end > len(future_covariates):
                raise_log(
                    ValueError(
                        f"The dataset contains `{FeatureType.FUTURE_COVARIATES.value}` "
                        f"that don't extend far enough into the future. ({idx}-th sample)"
                    ),
                    logger=logger,
                )
            vals = future_covariates.random_component_values(copy=False)
            fc = vals[start:end]

            # historic part of future covariates
            hfc_start, hfc_end = idx_bounds[FeatureType.HISTORIC_FUTURE_COVARIATES]
            hfc = vals[hfc_start:hfc_end]

            if (
                len(hfc) != self.input_chunk_length
                or len(fc) != self.output_chunk_length
            ):
                raise_log(
                    ValueError(
                        f"The dataset contains `{FeatureType.FUTURE_COVARIATES.value}` "
                        "whose time axis doesn't allow to obtain the input (and / or output) chunk relative to the "
                        "target `series`."
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
                        f"chunk relative to the target `series`."
                    ),
                    logger=logger,
                )

        # extract static covariates
        if self.uses_static_covariates_covariates:
            sc = series.static_covariates_values(copy=False)

        # (
        #     past target,
        #     past cov,
        #     historic future cov,
        #     future cov,
        #     static cov,
        #     sample weight,
        #     future target
        # )
        return pt, pc, hfc, fc, sc, sw, ft

    def _get_end_of_output_idx(self, series, series_idx, idx):
        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(series) - self.size_of_both_chunks + 1

        if n_samples_in_ts < 1:
            raise_log(
                ValueError(
                    "The dataset contains some target `series` that are shorter than "
                    "`max(self.input_chunk_length, self.shift + self.output_chunk_length)` "
                    f"({series_idx}-th series)."
                ),
                logger=logger,
            )

        # determine the index at the end of the output chunk
        # it is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        return (
            len(series)
            - (idx - (series_idx * self.max_samples_per_ts)) % n_samples_in_ts
        )


class SequentialTrainingDataset(ShiftedTrainingDataset):
    def __init__(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """Sequential Training Dataset

        Each sample drawn from this dataset is a seven-element tuple extracted from a specific time window and
        set of single input `TimeSeries`. The elements are:

        - past_target: target `series` values in the input chunk
        - past_covariates: `past_covariates` values in the input chunk (`None` if `past_covariates=None`)
        - historic_future_covariates: `future_covariates` values in the input chunk (`None` if `future_covariates=None`)
        - future_covariates: `future_covariates` values in the output chunk (`None` if `future_covariates=None`)
        - static_covariates: `static_covariates` values of the `series` (`None` if `use_static_covariates=False`)
        - sample_weight: `sample_weight` values in the output chunk (`None` if `sample_weight=None`)
        - future_target: `series` values in the output chunk

        The output chunk starts `input_chunk_length + output_chunk_shift` after the input chunk's start.

        The sample index determines:

        - the position / time of the extracted chunks relative to the end of a single target `series`
        - the index (which series and covariates) to use in case `series` (and covariates) are
          passed as a sequence of series.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        .. note::
            Each series in the provided sequence must have a minimum length of
            `input_chunk_length + output_chunk_shift + output_chunk_length`.

        Parameters
        ----------
        series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past covariates.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates.
        input_chunk_length
            The length of the lookback / past window the model takes as input.
        output_chunk_length
            The length of the lookahead / future window that the model emits as output (for the target) and takes as
            input (for future covariates).
        output_chunk_shift
            The number of steps to shift the start of the output chunk into the future.
        max_samples_per_ts
            This is an upper bound on the number of samples that can be produced per time series. It can be used to
            limit the total size of the dataset and ensure proper sampling. If `None`, will read all individual time
            series in advance (at dataset creation) to check their sizes. This might be expensive on big datasets.
            If not `None`, will only keep a maximum of `max_samples_per_ts` samples per series, extracted from the most
            recent past.
        use_static_covariates
            Whether to use/include static covariate data from the target `series`.
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
        shift = input_chunk_length + output_chunk_shift
        super().__init__(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            shift=shift,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=use_static_covariates,
            sample_weight=sample_weight,
        )


class HorizonBasedTrainingDataset(SequentialTrainingDataset):
    def __init__(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        output_chunk_length: int = 12,
        output_chunk_shift: int = 0,
        lh: tuple[int, int] = (1, 3),
        lookback: int = 3,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ) -> None:
        """Horizon Based Training Dataset

        A dataset inspired by the N-BEATS way of training on the M4 dataset: https://arxiv.org/abs/1905.10437.

        Each sample drawn from this dataset is a seven-element tuple extracted from a specific time window and
        set of single input `TimeSeries`. The elements are:

        - past_target: target `series` values in the input chunk
        - past_covariates: `past_covariates` values in the input chunk (`None` if `past_covariates=None`)
        - historic_future_covariates: `future_covariates` values in the input chunk (`None` if `future_covariates=None`)
        - future_covariates: `future_covariates` values in the output chunk (`None` if `future_covariates=None`)
        - static_covariates: `static_covariates` values of the `series` (`None` if `use_static_covariates=False`)
        - sample_weight: `sample_weight` values in the output chunk (`None` if `sample_weight=None`)
        - future_target: `series` values in the output chunk

        Given the horizon `output_chunk_length` of a model, this dataset will compute some "past / future" input and
        output chunks as follows: First a "forecast point" is selected in the range of the last `(min_lh *
        output_chunk_length, max_lh * output_chunk_length)` points before the end of the time series.
        The "future" output chunk then consists in the following `output_chunk_length` points, and the "past" input
        chunk will be the preceding `lookback * output_chunk_length` points.

        The sample index determines:

        - the position / time of the extracted chunks relative to the end of a single target `series`
        - the index (which series and covariates) to use in case `series` (and covariates) are
          passed as a sequence of series.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        .. note::
            Each series in the provided sequence must have a minimum length of
            `(lookback + max_lh) * output_chunk_length`, and `min_lh` must be `>=1`.

        Parameters
        ----------
        series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past covariates.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates.
        output_chunk_length
            The length of the lookahead / future window that the model emits as output (for the target) and takes as
            input (for future covariates).
        output_chunk_shift
            The number of steps to shift the start of the output chunk into the future.
        lh
            A `(min_lh, max_lh)` interval for the forecast point, starting from the end of the series.
            For example, `(1, 3)` will select forecast points uniformly between `1*H` and `3*H` points
            before the end of the series. It is required that `min_lh >= 1`.
        lookback:
            A integer interval for the length of the input in the emitted input and output splits, expressed as a
            multiple of `output_chunk_length`. For instance, `lookback=3` will emit "inputs" of lengths
            `3 * output_chunk_length`.
        use_static_covariates
            Whether to use/include static covariate data from the target `series`.
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
        # Checks
        min_lh, max_lh = lh
        if not (max_lh >= min_lh >= 1):
            raise_log(
                ValueError(
                    "The lh parameter should be an int tuple (min_lh, max_lh), "
                    "with 1 <= min_lh <= max_lh"
                ),
                logger=logger,
            )
        max_samples_per_ts = (max_lh - min_lh) * output_chunk_length

        super().__init__(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=lookback * output_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=use_static_covariates,
            sample_weight=sample_weight,
        )

        self.min_lh, self.max_lh = min_lh, max_lh
        self.lookback = lookback

    def _get_end_of_output_idx(self, series, series_idx, idx):
        # determine the actual number of possible samples in this time series
        if len(series) < (self.lookback + self.max_lh) * self.output_chunk_length:
            raise_log(
                ValueError(
                    "The dataset contains some target `series` that are shorter than "
                    f"`(lookback + max_lh) * H` ({series_idx}-th series)."
                ),
                logger=logger,
            )

        # determine the index lh_idx of the forecasting point (the last point of the input series, before the target)
        # lh_idx should be in [0, self.max_samples_per_ts)
        lh_idx = idx - (series_idx * self.max_samples_per_ts)

        # determine the index at the end of the output chunk
        return len(series) - ((self.min_lh - 1) * self.output_chunk_length + lh_idx)
