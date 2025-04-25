"""
Shifted Training Dataset
------------------------
"""

from collections.abc import Sequence
from typing import Optional, Union

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.training_dataset import TrainingDataset, TrainingSample
from darts.utils.data.utils import FeatureType, _process_sample_weight
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


class GenericShiftedDataset(TrainingDataset):
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
        """Generic Shifted Dataset

        Each sample drawn from this dataset is an eight-element tuple extracted from a specific time window and
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

        Parameters
        ----------
        series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past covariates.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates.
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits as output.
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
        self.sample_weight = None
        if sample_weight is not None:
            if output_chunk_length > 0:
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

    def __getitem__(self, idx) -> TrainingSample:
        # determine the index of the time series.
        series_idx = idx // self.max_samples_per_ts
        series = self.series[series_idx]
        series_vals = series.random_component_values(copy=False)

        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(series_vals) - self.size_of_both_chunks + 1

        if n_samples_in_ts < 1:
            raise_log(
                ValueError(
                    "The dataset contains some target `series` that are too short to contain "
                    "`max(self.input_chunk_length, self.shift + self.output_chunk_length)` "
                    f"({series_idx}-th series)"
                ),
                logger=logger,
            )

        # determine the index at the end of the output chunk
        # it is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        end_of_output_idx = (
            len(series)
            - (idx - (series_idx * self.max_samples_per_ts)) % n_samples_in_ts
        )

        # load covariates
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

        # (past target, past cov, historic future cov, future cov, static cov, sample weight, future target)
        return pt, pc, hfc, fc, sc, sw, ft


class PastCovariatesShiftedDataset(GenericShiftedDataset):
    def __init__(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        shift: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of (past_target, past_covariates, static_covariates, sample weights,
        future_target)
        arrays, which all have length `length`.
        The "future_target" is the "past_target" target shifted by `shift` time steps forward.
        So if an emitted "past_target" (and "past_covariates") goes from position `i` to `i+length`,
        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        Parameters
        ----------
        series
            One or a sequence of target `TimeSeries`.
        covariates
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        length
            The length of the emitted past and future series.
        shift
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
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
        super().__init__(
            series=series,
            past_covariates=past_covariates,
            future_covariates=None,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            shift=shift,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=use_static_covariates,
            sample_weight=sample_weight,
        )


class FutureCovariatesShiftedDataset(GenericShiftedDataset):
    def __init__(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        shift: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of (past_target, future_covariates, static_covariates, sample weights,
        future_target)
        arrays, which all have length `length`.
        The "future_target" is the "past_target" target shifted by `shift` time steps forward.
        So if an emitted "past_target" goes from position `i` to `i+length`,
        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.
        The slicing future covariates matches that of future targets. The slicing
        itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        Parameters
        ----------
        series
            One or a sequence of target `TimeSeries`.
        covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. If this parameter is set,
            the provided sequence must have the same length as that of `series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        length
            The length of the emitted past and future series.
        shift
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
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
        super().__init__(
            series=series,
            past_covariates=None,
            future_covariates=future_covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            shift=shift,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=use_static_covariates,
            sample_weight=sample_weight,
        )


class DualCovariatesShiftedDataset(GenericShiftedDataset):
    def __init__(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        shift: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of
        (past_target, historic_future_covariates, future_covariates, static_covariates, sample weights,
        future_target)
        arrays, which all have length `length`.
        The "future_target" is the "past_target" target shifted by `shift` time steps forward.
        So if an emitted "past_target" goes from position `i` to `i+length`,
        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.
        The slicing "future_covariates" matches that of "futuretarget" and the slicing of "historic_future_covariates"
        matches that of "past_target". The slicing itself relies on time indexes to align the series if they have
        unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        Parameters
        ----------
        series
            One or a sequence of target `TimeSeries`.
        covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. If this parameter is set,
            the provided sequence must have the same length as that of `series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        length
            The length of the emitted past and future series.
        shift
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
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
        super().__init__(
            series=series,
            past_covariates=None,
            future_covariates=future_covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            shift=shift,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=use_static_covariates,
            sample_weight=sample_weight,
        )


class MixedCovariatesShiftedDataset(GenericShiftedDataset):
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
        """
        A time series dataset containing tuples of (past_target, past_covariates, historic_future_covariates,
        future_covariates, static_covariates, sample weights, future_target) arrays, which all have length `length`.
        The "future_target" is the "past_target" target shifted by `shift` time steps forward.
        So if an emitted "past_target" goes from position `i` to `i+length`,
        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.
        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing
        itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        Parameters
        ----------
        series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. This has to follow
            the same constraints as `past_covariates`.
        length
            The length of the emitted past and future series.
        shift
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
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


class SplitCovariatesShiftedDataset(GenericShiftedDataset):
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
        """
        A time series dataset containing tuples of (past_target, past_covariates, future_covariates, static_covariates,
        sample weights, future_target) arrays, which all have length `length`.
        The "future_target" is the "past_target" target shifted by `shift` time steps forward.
        So if an emitted "past_target" goes from position `i` to `i+length`,
        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.
        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing
        itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        Parameters
        ----------
        series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. This has to follow
            the same constraints as `past_covariates`.
        shift
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
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
