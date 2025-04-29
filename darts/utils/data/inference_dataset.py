"""
Inference Datasets
------------------

- :class:`~darts.utils.data.inference_dataset.InferenceDataset`
- :class:`~darts.utils.data.inference_dataset.SequentialInferenceDataset`
"""

import bisect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.utils import FeatureType, InferenceDatasetOutput
from darts.utils.historical_forecasts.utils import _process_predict_start_points_bounds
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


class InferenceDataset(ABC, Dataset):
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

        Darts `TorchForecastingModel` can predict from instances of `InferenceDataset` using the
        `predict_from_dataset()` method.

        `InferenceDataset` inherits from torch `Dataset`; meaning that all subclasses must implement the
        `__getitem__()` method. All returned elements except `target_series` (`TimeSeries`) and `pred_time`
        (`pd.Timestamp` or `int`) must be of type `np.ndarray` (or `None` for optional covariates).
        """

    @abstractmethod
    def __len__(self) -> int:
        """The total number of samples that can be extracted."""

    @abstractmethod
    def __getitem__(self, idx: int) -> InferenceDatasetOutput:
        """Returns a sample drawn from this dataset."""

    @staticmethod
    def _covariate_indexer(
        series_idx: int,
        past_start: Union[pd.Timestamp, int],
        past_end: Union[pd.Timestamp, int],
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
        output_chunk_length: int,
        output_chunk_shift: int,
        n: int,
    ) -> dict[FeatureType, tuple[Optional[int], Optional[int]]]:
        """returns tuple of (past_start, past_end, future_start, future_end)"""
        # we need to use the time index (datetime or integer) here to match the index with the covariate series
        cov_times = {}

        for cov, cov_type in zip(
            [past_covariates, future_covariates],
            [FeatureType.PAST_COVARIATES, FeatureType.FUTURE_COVARIATES],
        ):
            if cov is None:
                cov_times[cov_type] = (None, None)
                continue

            # past covariates and historic part of future covariates
            cov_start = past_start

            if cov_type == FeatureType.PAST_COVARIATES:
                # with auto-regression: model will consume future values up until forecasting point
                cov_end = past_end + max(0, n - output_chunk_length) * cov.freq
            else:
                # future part of future covariates; up until the end of the forecast including `output_chunk_shift`
                cov_end = (
                    past_end
                    + (max(n, output_chunk_length) + output_chunk_shift)
                    * future_covariates.freq
                )

            # check start requirements for past covariates and historic part of future covariates
            if cov.start_time() > cov_start:
                raise_log(
                    ValueError(
                        f"For the given forecasting case, the provided `{cov_type.value}` at "
                        f"dataset index `{series_idx}` do not extend far enough into the past. The "
                        f"`{cov_type.value}` must start at or before time step `{cov_start}`, whereas now "
                        f"they start at time step `{cov.start_time()}`."
                    ),
                    logger=logger,
                )
            # check end requirements for past covariates and future part of future covariates
            if cov.end_time() < cov_end:
                raise_log(
                    ValueError(
                        f"For the given forecasting horizon `n={n}`, the provided `{cov_type.value}` "
                        f"at dataset index `{series_idx}` do not extend far enough into the future. As `"
                        f"{'n > output_chunk_length' if n > output_chunk_length else 'n <= output_chunk_length'}"
                        f"` the `{cov_type.value}` must end at or after time step `{cov_end}`, "
                        f"whereas now they end at time step `{cov.end_time()}`."
                    ),
                    logger=logger,
                )

            # extract the index position (integer index) from time_index value
            cov_start = cov.time_index.get_loc(cov_start)
            cov_end = cov.time_index.get_loc(cov_end) + 1
            cov_times[cov_type] = (cov_start, cov_end)
        return cov_times


class SequentialInferenceDataset(InferenceDataset):
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
        """Generic Inference Dataset

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

    def __getitem__(self, idx: int) -> InferenceDatasetOutput:
        if self.bounds is None:
            series_idx, series_start_idx, series_end_idx = (
                idx,
                -self.input_chunk_length,
                None,
            )
        else:
            series_idx, series_end_idx = self._find_list_index(
                idx,
                self.cum_lengths,
                self.bounds,
                self.stride,
            )
            series_start_idx = series_end_idx - self.input_chunk_length

        series = self.series[series_idx]
        if not len(series) >= self.input_chunk_length:
            raise_log(
                ValueError(
                    f"All input series must have length >= `input_chunk_length` ({self.input_chunk_length})."
                ),
                logger=logger,
            )
        past_end = series._time_index[
            series_end_idx - 1 if series_end_idx is not None else -1
        ]

        # load covariates
        past_covariates = (
            self.past_covariates[series_idx] if self.uses_past_covariates else None
        )
        future_covariates = (
            self.future_covariates[series_idx] if self.uses_future_covariates else None
        )

        # extract past target series
        pt = series.random_component_values(copy=False)[series_start_idx:series_end_idx]

        # past cov, future past cov, historic future cov, future cov, static cov
        pc, fpc, hfc, fc, sc = None, None, None, None, None

        if self.uses_past_covariates or self.uses_future_covariates:
            # get start and end indices (integer) of the covariates including historic and future parts
            cov_times = self._covariate_indexer(
                series_idx=series_idx,
                past_start=series.time_index[series_start_idx],
                past_end=past_end,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                output_chunk_length=self.output_chunk_length,
                output_chunk_shift=self.output_chunk_shift,
                n=self.n,
            )

            # extract past covariates
            if self.uses_past_covariates:
                start, end = cov_times[FeatureType.PAST_COVARIATES]
                vals = past_covariates.random_component_values(copy=False)[start:end]
                # historic part of past covariates
                pc = vals[: self.input_chunk_length]
                # future part of past covariates
                fpc = vals[self.input_chunk_length :]

                # set to None if empty array
                if len(pc) == 0:
                    pc = None
                if len(fpc) == 0:
                    fpc = None

            # extract future covariates
            if self.uses_future_covariates:
                start, end = cov_times[FeatureType.FUTURE_COVARIATES]
                vals = future_covariates.random_component_values(copy=False)[start:end]
                # historic part of future covariates
                hfc = vals[: self.input_chunk_length]
                # future part of future covariates
                fc = vals[self.input_chunk_length + self.output_chunk_shift :]

                # set to None if empty array
                if len(hfc) == 0:
                    hfc = None
                if len(fc) == 0:
                    fc = None

        # extract static covariates
        if self.use_static_covariates:
            sc = series.static_covariates_values(copy=False)

        return (
            pt,
            pc,
            fpc,
            hfc,
            fc,
            sc,
            series,
            past_end + series.freq * (1 + self.output_chunk_shift),
        )
