"""
Training Datasets
-----------------

- :class:`~darts.utils.data.training_dataset.TorchTrainingDataset`
- :class:`~darts.utils.data.training_dataset.ShiftedTorchTrainingDataset`
- :class:`~darts.utils.data.training_dataset.SequentialTorchTrainingDataset`
- :class:`~darts.utils.data.training_dataset.HorizonBasedTorchTrainingDataset`
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import ceil

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.torch_datasets.dataset import TorchDataset
from darts.utils.data.torch_datasets.utils import TorchTrainingDatasetOutput
from darts.utils.data.utils import (
    FeatureType,
    _process_sample_weight,
)
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


class TorchTrainingDataset(TorchDataset, ABC):
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

        Darts `TorchForecastingModel` can be fit from instances of `TorchTrainingDataset` using the `fit_from_dataset()`
        method.

        `TorchTrainingDataset` inherits from torch `Dataset`; meaning that all subclasses must implement the
        `__getitem__()` method. All returned elements must be of type `np.ndarray` (or `None` for optional covariates
        and sample weight).
        """
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx: int) -> TorchTrainingDatasetOutput:
        """Returns a sample drawn from this dataset."""


class ShiftedTorchTrainingDataset(TorchTrainingDataset):
    def __init__(
        self,
        series: TimeSeries | Sequence[TimeSeries],
        past_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        future_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        shift: int = 1,
        stride: int = 1,
        max_samples_per_ts: int | None = None,
        use_static_covariates: bool = True,
        sample_weight: TimeSeries | Sequence[TimeSeries] | str | None = None,
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
        stride
            The number of time steps between consecutive samples (windows of lagged values extracted from the target
            series), applied starting from the end of the series. This should be used with caution as it might
            introduce bias in the forecasts.
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

        if not (isinstance(stride, int) and stride > 0):
            raise_log(
                ValueError("`stride` must be a positive integer greater than 0."),
                logger=logger,
            )

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

        size_of_both_chunks = max(input_chunk_length, shift + output_chunk_length)

        # compute the maximum available samples over all series
        max_available_indices = max(len(ts) for ts in series) - size_of_both_chunks + 1
        max_available_samples = ceil(max_available_indices / stride)

        if max_available_indices <= 0:
            raise_log(
                ValueError(
                    f"The input `series` are too short to extract even a single sample. "
                    f"Expected min length: `{size_of_both_chunks}`, received max length: "
                    f"`{max(len(ts) for ts in series)}`."
                )
            )

        if max_samples_per_ts is None:
            max_samples_per_ts = max_available_samples
        else:
            # upper bound maximum available samples by max_samples_per_ts
            max_samples_per_ts = min(max_samples_per_ts, max_available_samples)

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.size_of_both_chunks = size_of_both_chunks
        self.shift = shift
        self.stride = stride
        self.max_samples_per_ts = max_samples_per_ts
        self.ideal_nr_samples = len(series) * self.max_samples_per_ts

        self.series = series
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates

        # setup sample weights; ignore weights when `ocl==0`
        if sample_weight is not None and output_chunk_length > 0:
            self.sample_weight = _process_sample_weight(sample_weight, self.series)
        else:
            self.sample_weight = None

        self.uses_past_covariates = past_covariates is not None
        self.uses_future_covariates = future_covariates is not None
        self.uses_static_covariates_covariates = static_covariates is not None

    def __len__(self):
        return self.ideal_nr_samples

    def __getitem__(self, idx) -> TorchTrainingDatasetOutput:
        # determine the index of the time series.
        series_idx = idx // self.max_samples_per_ts
        series = self.series[series_idx]

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
                        f"`{series.n_components}` (at series sequence idx `{series_idx}`)."
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
            n=None,
        )

        series_vals = series.random_component_values(copy=False)
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
            pc = past_covariates.random_component_values(copy=False)[start:end]

        # extract future covariates
        if self.uses_future_covariates:
            # future part of future covariates
            start, end = idx_bounds[FeatureType.FUTURE_COVARIATES]
            vals = future_covariates.random_component_values(copy=False)
            fc = vals[start:end]

            # historic part of future covariates
            hfc_start, hfc_end = idx_bounds[FeatureType.HISTORIC_FUTURE_COVARIATES]
            hfc = vals[hfc_start:hfc_end]

        # extract sample weights
        if self.sample_weight is not None:
            start, end = idx_bounds[FeatureType.SAMPLE_WEIGHT]
            sw = sample_weight.random_component_values(copy=False)[start:end]

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
        n_samples_in_ts = ceil(
            (len(series) - self.size_of_both_chunks + 1) / self.stride
        )

        if n_samples_in_ts < 1:
            raise_log(
                ValueError(
                    f"The dataset contains target `series` that are too short to extract "
                    f"even a single example. Expected min length: `{self.size_of_both_chunks}`, "
                    f"received length `{len(series)}` (at series sequence idx `{series_idx}`)."
                ),
                logger=logger,
            )

        # determine the index at the end of the output chunk
        # it is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        return (
            len(series)
            - (idx - (series_idx * self.max_samples_per_ts))
            % n_samples_in_ts
            * self.stride
        )


class SequentialTorchTrainingDataset(ShiftedTorchTrainingDataset):
    def __init__(
        self,
        series: TimeSeries | Sequence[TimeSeries],
        past_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        future_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        stride: int = 1,
        max_samples_per_ts: int | None = None,
        use_static_covariates: bool = True,
        sample_weight: TimeSeries | Sequence[TimeSeries] | str | None = None,
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
        stride
            The number of time steps between consecutive samples (windows of lagged values extracted from the target
            series), applied starting from the end of the series. This should be used with caution as it might
            introduce bias in the forecasts.
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
            stride=stride,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=use_static_covariates,
            sample_weight=sample_weight,
        )


class HorizonBasedTorchTrainingDataset(SequentialTorchTrainingDataset):
    def __init__(
        self,
        series: TimeSeries | Sequence[TimeSeries],
        past_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        future_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        output_chunk_length: int = 12,
        output_chunk_shift: int = 0,
        stride: int = 1,
        lh: tuple[int, int] = (1, 3),
        lookback: int = 3,
        use_static_covariates: bool = True,
        sample_weight: TimeSeries | Sequence[TimeSeries] | str | None = None,
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
        stride
            The number of time steps between consecutive samples (windows of lagged values extracted from the target
            series), applied starting from the end of the series. This should be used with caution as it might
            introduce bias in the forecasts.
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
                    f"Invalid `lh={lh}`. `lh` must be a tuple `(min_lh, max_lh)`, "
                    f"with `1 <= min_lh <= max_lh`."
                ),
                logger=logger,
            )
        max_samples_per_ts = (max_lh - min_lh) * output_chunk_length + 1
        max_samples_per_ts = ceil(max_samples_per_ts / stride)

        super().__init__(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=lookback * output_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            stride=stride,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=use_static_covariates,
            sample_weight=sample_weight,
        )

        self.min_lh, self.max_lh = min_lh, max_lh
        self.lookback = lookback

    def _get_end_of_output_idx(self, series, series_idx, idx):
        # determine the actual number of possible samples in this time series
        min_length = (self.lookback + self.max_lh) * self.output_chunk_length
        if len(series) < min_length:
            raise_log(
                ValueError(
                    f"The dataset contains target `series` that are too short to extract "
                    f"even a single example. Expected min length: `{min_length}`, received "
                    f"length `{len(series)}` (at series sequence idx `{series_idx}`)."
                ),
                logger=logger,
            )

        # determine the index lh_idx of the forecasting point (the last point of the input series, before the target)
        # lh_idx should be in [0, self.max_samples_per_ts)
        lh_idx = (idx - (series_idx * self.max_samples_per_ts)) * self.stride

        # determine the index at the end of the output chunk
        return len(series) - ((self.min_lh - 1) * self.output_chunk_length + lh_idx)
