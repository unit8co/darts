"""
Inference Dataset
-----------------
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
from darts.utils.data.utils import CovariateType
from darts.utils.historical_forecasts.utils import _process_predict_start_points_bounds

logger = get_logger(__name__)


class InferenceDataset(ABC, Dataset):
    def __init__(self):
        """
        Abstract class for all darts torch inference dataset.

        It can be used as models' inputs, to obtain simple forecasts on each `TimeSeries`
        (using covariates if specified).

        The first elements of the tuples it contains are numpy arrays (which will be translated to torch tensors
        by the torch DataLoader). The last elements of the tuples are the (past) target TimeSeries, which is
        needed in order to properly construct the time axis of the forecast series.
        """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @staticmethod
    def _covariate_indexer(
        target_idx: int,
        past_start: Union[pd.Timestamp, int],
        past_end: Union[pd.Timestamp, int],
        covariate_series: TimeSeries,
        covariate_type: CovariateType,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int,
        n: int,
    ):
        """returns tuple of (past_start, past_end, future_start, future_end)"""
        # get the main covariate type: CovariateType.PAST or CovariateType.FUTURE
        main_covariate_type = (
            CovariateType.PAST
            if covariate_type is CovariateType.PAST
            else CovariateType.FUTURE
        )

        if main_covariate_type not in [CovariateType.PAST, CovariateType.FUTURE]:
            raise_log(
                ValueError(
                    "`main_covariate_type` must be one of `(CovariateType.PAST, CovariateType.FUTURE)`"
                ),
                logger=logger,
            )

        # we need to use the time index (datetime or integer) here to match the index with the covariate series
        if main_covariate_type is CovariateType.PAST:
            future_end = (
                past_end + max(0, n - output_chunk_length) * covariate_series.freq
            )
        else:  # CovariateType.FUTURE
            # optionally, for future part of future covariates shift start and end by `output_chunk_shift`
            future_end = (
                past_end
                + (max(n, output_chunk_length) + output_chunk_shift)
                * covariate_series.freq
            )

        future_start = (
            past_end + covariate_series.freq * (1 + output_chunk_shift)
            if future_end != past_end
            else future_end
        )

        if input_chunk_length == 0:  # for regression ensemble models
            past_start, past_end = future_start, future_start

        # check if case specific indexes are available
        case_start = (
            future_start if covariate_type is CovariateType.FUTURE else past_start
        )
        if not covariate_series.start_time() <= case_start:
            raise_log(
                ValueError(
                    f"For the given forecasting case, the provided {main_covariate_type.value} covariates at "
                    f"dataset index `{target_idx}` do not extend far enough into the past. The "
                    f"{main_covariate_type.value} covariates must start at time step `{case_start}`, whereas now "
                    f"they start at time step `{covariate_series.start_time()}`."
                ),
                logger=logger,
            )
        if not covariate_series.end_time() >= future_end:
            raise_log(
                ValueError(
                    f"For the given forecasting horizon `n={n}`, the provided {main_covariate_type.value} covariates "
                    f"at dataset index `{target_idx}` do not extend far enough into the future. As `"
                    f"{'n > output_chunk_length' if n > output_chunk_length else 'n <= output_chunk_length'}"
                    f"` the {main_covariate_type.value} covariates must end at time step `{future_end}`, "
                    f"whereas now they end at time step `{covariate_series.end_time()}`."
                ),
                logger=logger,
            )

        # extract the index position (integer index) from time_index value
        covariate_start = covariate_series.time_index.get_loc(past_start)
        covariate_end = covariate_series.time_index.get_loc(future_end) + 1
        return covariate_start, covariate_end


class GenericInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        stride: int = 0,
        bounds: Optional[np.ndarray] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        covariate_type: CovariateType = CovariateType.PAST,
        use_static_covariates: bool = True,
    ):
        """
        Contains (past_target, past_covariates | historic_future_covariates, future_past_covariates | future_covariate,
        static_covariates).

        "future_past_covariates" are past covariates that happen to be also known in the future - those
        are needed for forecasting with n > output_chunk_length by any model relying on past covariates.
        For this reason, when n > output_chunk_length, this dataset will also emit the "future past_covariates".

        "historic_future_covariates" are historic future covariates that are given for the input_chunk in the past.

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        covariates
            Optionally, one or a sequence of `TimeSeries` containing either past or future covariates. If covariates
            were used during training, the same type of cavariates must be supplied at prediction.
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
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        self.target_series = (
            [target_series] if isinstance(target_series, TimeSeries) else target_series
        )
        self.covariates = (
            [covariates] if isinstance(covariates, TimeSeries) else covariates
        )

        if not (covariates is None or len(self.target_series) == len(self.covariates)):
            raise_log(
                ValueError(
                    "The number of target series must be equal to the number of covariates."
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

        self.covariate_type = covariate_type

        self.n = n
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.output_chunk_shift = output_chunk_shift
        self.use_static_covariates = use_static_covariates

        self.stride = stride
        if bounds is None:
            self.bounds = bounds
            self.cum_lengths = None
            self.len_preds = len(self.target_series)
        else:
            self.bounds, self.cum_lengths = _process_predict_start_points_bounds(
                series=target_series,
                bounds=bounds,
                stride=stride,
            )
            self.len_preds = self.cum_lengths[-1]

    def __len__(self):
        return self.len_preds

    @staticmethod
    def find_list_index(index, cumulative_lengths, bounds, stride):
        list_index = bisect.bisect_right(cumulative_lengths, index)
        bound_left = bounds[list_index, 0]
        if list_index == 0:
            stride_idx = index * stride
        else:
            stride_idx = (index - cumulative_lengths[list_index - 1]) * stride
        return list_index, bound_left + stride_idx

    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
        Union[pd.Timestamp, int],
    ]:
        if self.bounds is None:
            series_idx, target_start_idx, target_end_idx = (
                idx,
                -self.input_chunk_length,
                None,
            )
        else:
            series_idx, target_end_idx = self.find_list_index(
                idx,
                self.cum_lengths,
                self.bounds,
                self.stride,
            )
            target_start_idx = target_end_idx - self.input_chunk_length

        target_series = self.target_series[series_idx]
        if not len(target_series) >= self.input_chunk_length:
            raise_log(
                ValueError(
                    f"All input series must have length >= `input_chunk_length` ({self.input_chunk_length})."
                ),
                logger=logger,
            )

        # extract past target values
        past_end = target_series.time_index[
            target_end_idx - 1 if target_end_idx is not None else -1
        ]
        past_target = target_series.random_component_values(copy=False)[
            target_start_idx:target_end_idx
        ]

        # optionally, extract covariates
        past_covariate, future_covariate = None, None
        covariate_series = (
            None if self.covariates is None else self.covariates[series_idx]
        )
        if covariate_series is not None:
            # get start and end indices (integer) of the covariates including historic and future parts
            covariate_start, covariate_end = self._covariate_indexer(
                target_idx=series_idx,
                past_start=target_series.time_index[target_start_idx],
                past_end=past_end,
                covariate_series=covariate_series,
                covariate_type=self.covariate_type,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                output_chunk_shift=self.output_chunk_shift,
                n=self.n,
            )

            # extract covariate values and split into a past (historic) and future part
            covariate = covariate_series.random_component_values(copy=False)[
                covariate_start:covariate_end
            ]
            if self.input_chunk_length != 0:  # regular models
                past_covariate, future_covariate = (
                    covariate[: self.input_chunk_length],
                    covariate[self.input_chunk_length + self.output_chunk_shift :],
                )
            else:  # regression ensemble models have a input_chunk_length == 0 part for using predictions as input
                past_covariate, future_covariate = covariate, covariate

            # set to None if empty array
            past_covariate = (
                past_covariate
                if past_covariate is not None and len(past_covariate) > 0
                else None
            )
            future_covariate = (
                future_covariate
                if future_covariate is not None and len(future_covariate) > 0
                else None
            )

        if self.use_static_covariates:
            static_covariate = target_series.static_covariates_values(copy=False)
        else:
            static_covariate = None

        return (
            past_target,
            past_covariate,
            future_covariate,
            static_covariate,
            target_series,
            past_end + target_series.freq * (1 + self.output_chunk_shift),
        )


class PastCovariatesInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        stride: int = 0,
        bounds: Optional[np.ndarray] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        covariate_type: CovariateType = CovariateType.PAST,
        use_static_covariates: bool = True,
    ):
        """
        Contains (past_target, past_covariates, future_past_covariates, static_covariates).

        "future_past_covariates" are past covariates that happen to be also known in the future - those
        are needed for forecasting with n > output_chunk_length by any model relying on past covariates.

        For this reason, when n > output_chunk_length, this dataset will also emit the "future past_covariates".

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        covariates
            Optionally, some past-observed covariates that are used for predictions. This argument is required
            if the model was trained with past-observed covariates.
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
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """

        super().__init__()

        self.ds = GenericInferenceDataset(
            target_series=target_series,
            covariates=covariates,
            n=n,
            stride=stride,
            bounds=bounds,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            covariate_type=covariate_type,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
        Union[pd.Timestamp, int],
    ]:
        return self.ds[idx]


class FutureCovariatesInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        stride: int = 0,
        bounds: Optional[np.ndarray] = None,
        input_chunk_length: int = 12,
        output_chunk_length: Optional[int] = None,
        output_chunk_shift: int = 0,
        covariate_type: CovariateType = CovariateType.FUTURE,
        use_static_covariates: bool = True,
    ):
        """
        Contains (past_target, future_covariates, static_covariates) tuples

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        covariates
            Optionally, some future-known covariates that are used for predictions. This argument is required
            if the model was trained with future-known covariates.
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
            The length of the target series the model takes as input.
        output_chunk_length
            Optionally, the length of the target series the model emits in output. If `None`, will use the same value
            as `n`.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        self.ds = GenericInferenceDataset(
            target_series=target_series,
            covariates=covariates,
            n=n,
            stride=stride,
            bounds=bounds,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length or n,
            output_chunk_shift=output_chunk_shift,
            covariate_type=covariate_type,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
        Union[pd.Timestamp, int],
    ]:
        (
            past_target,
            _,
            future_covariate,
            static_covariate,
            target_series,
            pred_point,
        ) = self.ds[idx]
        return (
            past_target,
            future_covariate,
            static_covariate,
            target_series,
            pred_point,
        )


class DualCovariatesInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        stride: int = 0,
        bounds: Optional[np.ndarray] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        use_static_covariates: bool = True,
    ):
        """
        Contains (past_target, historic_future_covariates, future_covariates, static_covariates) tuples.

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        covariates
            Optionally, some future-known covariates that are used for predictions. This argument is required
            if the model was trained with future-known covariates.
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
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        # This dataset is in charge of serving historic future covariates
        self.ds_past = PastCovariatesInferenceDataset(
            target_series=target_series,
            covariates=covariates,
            n=n,
            stride=stride,
            bounds=bounds,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            covariate_type=CovariateType.HISTORIC_FUTURE,
            use_static_covariates=use_static_covariates,
        )

        # This dataset is in charge of serving future covariates
        self.ds_future = FutureCovariatesInferenceDataset(
            target_series=target_series,
            covariates=covariates,
            n=n,
            stride=stride,
            bounds=bounds,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            covariate_type=CovariateType.FUTURE,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
        Union[pd.Timestamp, int],
    ]:
        (
            past_target,
            historic_future_covariate,
            _,
            static_covariate,
            ts_target,
            pred_point,
        ) = self.ds_past[idx]
        _, future_covariate, _, _, _ = self.ds_future[idx]
        return (
            past_target,
            historic_future_covariate,
            future_covariate,
            static_covariate,
            ts_target,
            pred_point,
        )


class MixedCovariatesInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
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
        """
        Contains (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates,
        static_covariates)
        tuples. "future_past_covariates" are past covariates that happen to be also known in the future - those
        are needed for forecasting with n > output_chunk_length by any model relying on past covariates.

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        past_covariates
            Optionally, some past-observed covariates that are used for predictions. This argument is required
            if the model was trained with past-observed covariates.
        future_covariates
            Optionally, some future-known covariates that are used for predictions. This argument is required
            if the model was trained with future-known covariates.
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
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = PastCovariatesInferenceDataset(
            target_series=target_series,
            covariates=past_covariates,
            n=n,
            stride=stride,
            bounds=bounds,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            covariate_type=CovariateType.PAST,
            use_static_covariates=use_static_covariates,
        )

        # This dataset is in charge of serving historic and future covariates
        self.ds_future = DualCovariatesInferenceDataset(
            target_series=target_series,
            covariates=future_covariates,
            n=n,
            stride=stride,
            bounds=bounds,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
        Union[pd.Timestamp, int],
    ]:
        (
            past_target,
            past_covariate,
            future_past_covariate,
            static_covariate,
            ts_target,
            pred_point,
        ) = self.ds_past[idx]
        _, historic_future_covariate, future_covariate, _, _, _ = self.ds_future[idx]
        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            future_past_covariate,
            static_covariate,
            ts_target,
            pred_point,
        )


class SplitCovariatesInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
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
        """
        Contains (past_target, past_covariates, future_covariates, future_past_covariates, static_covariates) tuples.
        "future_past_covariates" are past covariates that happen to be also known in the future - those
        are needed for forecasting with n > output_chunk_length by any model relying on past covariates.

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        past_covariates
            Optionally, some past-observed covariates that are used for predictions. This argument is required
            if the model was trained with past-observed covariates.
        future_covariates
            Optionally, some future-known covariates that are used for predictions. This argument is required
            if the model was trained with future-known covariates.
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
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = PastCovariatesInferenceDataset(
            target_series=target_series,
            covariates=past_covariates,
            n=n,
            stride=stride,
            bounds=bounds,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            covariate_type=CovariateType.PAST,
            use_static_covariates=use_static_covariates,
        )

        # This dataset is in charge of serving future covariates
        self.ds_future = FutureCovariatesInferenceDataset(
            target_series=target_series,
            covariates=future_covariates,
            n=n,
            stride=stride,
            bounds=bounds,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            covariate_type=CovariateType.FUTURE,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
        Union[pd.Timestamp, int],
    ]:
        (
            past_target,
            past_covariate,
            future_past_covariate,
            static_covariate,
            ts_target,
            pred_point,
        ) = self.ds_past[idx]
        _, future_covariate, _, _, _ = self.ds_future[idx]
        return (
            past_target,
            past_covariate,
            future_covariate,
            future_past_covariate,
            static_covariate,
            ts_target,
            pred_point,
        )
