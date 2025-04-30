"""
Inference Datasets
------------------

- :class:`~darts.utils.data.inference_dataset.TorchInferenceDataset`
- :class:`~darts.utils.data.inference_dataset.SequentialTorchInferenceDataset`
"""

import bisect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.torch_datasets.dataset import TorchDataset
from darts.utils.data.torch_datasets.utils import TorchInferenceDatasetOutput
from darts.utils.data.utils import FeatureType
from darts.utils.historical_forecasts.utils import _process_predict_start_points_bounds
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


class TorchInferenceDataset(TorchDataset, ABC):
    def __init__(self):
        """
        Abstract class for all inference datasets that can be used with Darts' `TorchForecastingModel`.

        Provides samples to compute forecasts using a `TorchForecastingModel`.

        Each sample drawn from this dataset is an eight-element tuple extracted from a specific time window and
        set of single input `TimeSeries`. The elements are:

        - past_target: target `series` values in the input chunk
        - past_covariates: Optional `past_covariates` values in the input chunk
        - future_past_covariates: Optional `past_covariates` values in the forecast horizon (for auto-regression with
          `n>output_chunk_length`)
        - historic_future_covariates: Optional `future_covariates` values in the input chunk
        - future_covariates: Optional `future_covariates` values in the output chunk and forecast horizon
        - static_covariates: Optional `static_covariates` values of the `series`
        - target_series: the target `TimeSeries`
        - pred_time: the time of the first point in the forecast horizon

        Darts `TorchForecastingModel` can predict from instances of `TorchInferenceDataset` using the
        `predict_from_dataset()` method.

        `TorchInferenceDataset` inherits from torch `Dataset`; meaning that all subclasses must implement the
        `__getitem__()` method. All returned elements except `target_series` (`TimeSeries`) and `pred_time`
        (`pd.Timestamp` or `int`) must be of type `np.ndarray` (or `None` for optional covariates).
        """
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx: int) -> TorchInferenceDatasetOutput:
        """Returns a sample drawn from this dataset."""


class SequentialTorchInferenceDataset(TorchInferenceDataset):
    def __init__(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        stride: int = 0,
        bounds: Optional[np.ndarray] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        use_static_covariates: bool = True,
    ):
        """Sequential Inference Dataset

        Each sample drawn from this dataset is an eight-element tuple extracted from a specific time window and
        set of single input `TimeSeries`. The elements are:

        - past_target: target `series` values in the input chunk
        - past_covariates: `past_covariates` values in the input chunk (`None` if `past_covariates=None`)
        - future_past_covariates: `past_covariates` values in the forecast horizon (`None` if `past_covariates=None`
          or `n<=output_chunk_length` / non-auto-regressive forecasting)
        - historic_future_covariates: `future_covariates` values in the input chunk (`None` if `future_covariates=None`)
        - future_covariates: `future_covariates` values in the forecast horizon (`None` if `future_covariates=None`)
        - static_covariates: `static_covariates` values of the `series` (`None` if `use_static_covariates=False`)
        - target_series: the target `TimeSeries`
        - pred_time: the time of the first point in the forecast horizon

        The output chunk / forecast horizon starts `output_chunk_length + output_chunk_shift` after the input chunk's
        start.

        The sample index determines:

        - the position / time of the extracted chunks relative to the end of a single target `series`
        - the index (which series and covariates) to use in case `series` (and covariates) are
          passed as a sequence of series.

        With `bounds=None`, all samples will be extracted relative to the end of the target `series` (input chunk's end
        time is the same as the target series' end time). Otherwise, samples will be extracted from the given
        boundaries `bounds` with a stride of `stride`.

        .. note::
            "historic_future_covariates" are the values of the future-known covariates that fall into the sample's
            input chunk (the past window / history in the view of the sample).

        .. note::
            "future_past_covariates" are past covariates that happen to be also known in the future - those
            are needed for forecasting with `n > output_chunk_length` (auto-regression) by any model relying on past
            covariates. For this reason, this dataset may also emit the "future past_covariates".

        Parameters
        ----------
        series
            One or a sequence of target `TimeSeries` that are to be predicted into the future.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past covariates. If past covariates
            were used during training, they must be supplied at prediction.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. If future covariates
            were used during training, they must be supplied at prediction.
        n
            Forecast horizon: The number of time steps to predict after the end of the target series.
        stride
            Optionally, the number of time steps between two consecutive predictions. Can only be used together
            with `bounds`.
        bounds
            Optionally, an array of shape `(n series, 2)`, with the left and right prediction start point boundaries
            per series. The boundaries must represent the positional index of the series (0, len(series)).
            If provided, `stride` must be `>=1`.
        input_chunk_length
            The length of the lookback / past window the model takes as input.
        output_chunk_length
            The length of the lookahead / future window that the model emits as output (for the target) and takes as
            input (for future covariates).
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
        use_static_covariates
            Whether to use/include static covariate data from the target `series`.
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
            name = cov_type.value
            if cov is not None and len(series) != len(cov):
                raise_log(
                    ValueError(
                        f"The sequence of `{name}` must have the same length as "
                        f"the sequence of target `series`."
                    ),
                    logger=logger,
                )

        if (bounds is not None and stride == 0) or (bounds is None and stride > 0):
            raise_log(
                ValueError(
                    "Must supply either both `stride` and `bounds`, or none of them."
                ),
                logger=logger,
            )

        if output_chunk_shift and n > output_chunk_length:
            raise_log(
                ValueError(
                    "Cannot perform auto-regression `(n > output_chunk_length)` with a model that uses a "
                    "shifted output chunk `(output_chunk_shift > 0)`."
                ),
                logger=logger,
            )

        self.series = series
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates

        self.uses_past_covariates = past_covariates is not None
        self.uses_future_covariates = future_covariates is not None
        self.uses_static_covariates_covariates = static_covariates is not None

        self.n = n
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.output_chunk_shift = output_chunk_shift
        self.use_static_covariates = use_static_covariates

        self.stride = stride
        if bounds is None:
            self.bounds = bounds
            self.cum_lengths = None
            self.len_preds = len(self.series)
        else:
            self.bounds, self.cum_lengths = _process_predict_start_points_bounds(
                series=series,
                bounds=bounds,
                stride=stride,
            )
            self.len_preds = self.cum_lengths[-1]

    def __len__(self):
        return self.len_preds

    @staticmethod
    def _find_list_index(index, cumulative_lengths, bounds, stride):
        list_index = bisect.bisect_right(cumulative_lengths, index)
        bound_left = bounds[list_index, 0]
        if list_index == 0:
            stride_idx = index * stride
        else:
            stride_idx = (index - cumulative_lengths[list_index - 1]) * stride
        return list_index, bound_left + stride_idx

    def __getitem__(self, idx: int) -> TorchInferenceDatasetOutput:
        # determine the series index, and the index + 1 (exclusive range) of the output chunk end within that series
        if self.bounds is None:
            series_idx = idx
            series_end_idx = len(self.series[idx])
        else:
            series_idx, series_end_idx = self._find_list_index(
                idx,
                self.cum_lengths,
                self.bounds,
                self.stride,
            )
        end_of_output_idx = (
            series_end_idx + self.output_chunk_shift + self.output_chunk_length
        )

        series = self.series[series_idx]
        if not len(series) >= self.input_chunk_length:
            raise_log(
                ValueError(
                    f"The dataset contains target `series` that are too short to extract "
                    f"the model input for prediction . Expected min length: `{self.input_chunk_length}`, "
                    f"received length `{len(series)}` (at series sequence idx `{series_idx}`)."
                ),
                logger=logger,
            )

        # load covariates
        past_covariates = (
            self.past_covariates[series_idx] if self.uses_past_covariates else None
        )
        future_covariates = (
            self.future_covariates[series_idx] if self.uses_future_covariates else None
        )

        idx_bounds = self._memory_indexer(
            series_idx=series_idx,
            series=series,
            shift=self.input_chunk_length + self.output_chunk_shift,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            end_of_output_idx=end_of_output_idx,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            sample_weight=None,
            n=self.n,
        )

        series_vals = series.random_component_values(copy=False)
        # extract past target series
        start, end = idx_bounds[FeatureType.PAST_TARGET]
        pt = series_vals[start:end]

        # extract prediction start
        start, _ = idx_bounds[FeatureType.FUTURE_TARGET]
        if start < len(series):
            pred_start = series._time_index[start]
        else:
            pred_start = (
                series._time_index[-1] + ((start + 1) - len(series)) * series.freq
            )

        # past cov, future past cov, historic future cov, future cov, static cov
        pc, fpc, hfc, fc, sc = None, None, None, None, None

        # extract past covariates
        if self.uses_past_covariates:
            # past part of past covariates
            start, end = idx_bounds[FeatureType.PAST_COVARIATES]
            vals = past_covariates.random_component_values(copy=False)
            pc = vals[start:end]

            # future part of past covariates (`None` if not performing auto-regression)
            fpc_start, fpc_end = idx_bounds[FeatureType.FUTURE_PAST_COVARIATES]
            fpc = vals[fpc_start:fpc_end] if fpc_start is not None else None

        # extract future covariates
        if self.uses_future_covariates:
            # future part of future covariates
            start, end = idx_bounds[FeatureType.FUTURE_COVARIATES]
            vals = future_covariates.random_component_values(copy=False)
            fc = vals[start:end]

            # historic part of future covariates
            hfc_start, hfc_end = idx_bounds[FeatureType.HISTORIC_FUTURE_COVARIATES]
            hfc = vals[hfc_start:hfc_end]

        # extract static covariates
        if self.uses_static_covariates_covariates:
            sc = series.static_covariates_values(copy=False)

        # (
        #     past target,
        #     past cov,
        #     future past cov,
        #     historic future cov,
        #     future cov,
        #     static cov,
        #     target series,
        #     prediction start time,
        # )
        return (
            pt,
            pc,
            fpc,
            hfc,
            fc,
            sc,
            series,
            pred_start,
        )
