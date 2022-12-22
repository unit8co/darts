"""
Inference Dataset
-----------------
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from darts import TimeSeries
from darts.logging import raise_if_not

from .utils import CovariateType


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
        target_series: TimeSeries,
        covariate_series: TimeSeries,
        covariate_type: CovariateType,
        input_chunk_length: int,
        output_chunk_length: int,
        n: int,
    ):
        """returns tuple of (past_start, past_end, future_start, future_end)"""
        # get the main covariate type: CovariateType.PAST or CovariateType.FUTURE
        main_covariate_type = (
            CovariateType.PAST
            if covariate_type is CovariateType.PAST
            else CovariateType.FUTURE
        )

        raise_if_not(
            main_covariate_type in [CovariateType.PAST, CovariateType.FUTURE],
            "`main_covariate_type` must be one of `(CovariateType.PAST, CovariateType.FUTURE)`",
        )

        # we need to use the time index (datetime or integer) here to match the index with the covariate series
        past_start = target_series.time_index[-input_chunk_length]
        past_end = target_series.time_index[-1]
        if main_covariate_type is CovariateType.PAST:
            future_end = past_end + max(0, n - output_chunk_length) * target_series.freq
        else:  # CovariateType.FUTURE
            future_end = past_end + max(n, output_chunk_length) * target_series.freq

        future_start = (
            past_end + target_series.freq if future_end != past_end else future_end
        )

        if input_chunk_length == 0:  # for regression ensemble models
            past_start, past_end = future_start, future_start

        # check if case specific indexes are available
        case_start = (
            future_start if covariate_type is CovariateType.FUTURE else past_start
        )
        raise_if_not(
            covariate_series.start_time() <= case_start,
            f"For the given forecasting case, the provided {main_covariate_type.value} covariates at dataset index "
            f"`{target_idx}` do not extend far enough into the past. The {main_covariate_type.value} covariates "
            f"must start at time step `{case_start}`, whereas now they start at time step "
            f"`{covariate_series.start_time()}`.",
        )
        raise_if_not(
            covariate_series.end_time() >= future_end,
            f"For the given forecasting horizon `n={n}`, the provided {main_covariate_type.value} covariates "
            f"at dataset index `{target_idx}` do not extend far enough into the future. As `"
            f"{'n > output_chunk_length' if n > output_chunk_length else 'n <= output_chunk_length'}"
            f"` the {main_covariate_type.value} covariates must end at time step `{future_end}`, "
            f"whereas now they end at time step `{covariate_series.end_time()}`.",
        )

        # extract the index position (index) from time_index value
        covariate_start = covariate_series.time_index.get_loc(past_start)
        covariate_end = covariate_series.time_index.get_loc(future_end) + 1
        return covariate_start, covariate_end


class GenericInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
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
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
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

        self.covariate_type = covariate_type

        self.n = n
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.use_static_covariates = use_static_covariates

        raise_if_not(
            (covariates is None or len(self.target_series) == len(self.covariates)),
            "The number of target series must be equal to the number of covariates.",
        )

    def __len__(self):
        return len(self.target_series)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
    ]:
        target_series = self.target_series[idx]
        raise_if_not(
            len(target_series) >= self.input_chunk_length,
            f"All input series must have length >= `input_chunk_length` ({self.input_chunk_length}).",
        )

        # extract past target values
        past_target = target_series.random_component_values(copy=False)[
            -self.input_chunk_length :
        ]

        # optionally, extract covariates
        past_covariate, future_covariate = None, None
        covariate_series = None if self.covariates is None else self.covariates[idx]
        if covariate_series is not None:
            # get start and end indices (integer) of the covariates including historic and future parts
            covariate_start, covariate_end = self._covariate_indexer(
                target_idx=idx,
                target_series=target_series,
                covariate_series=covariate_series,
                covariate_type=self.covariate_type,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                n=self.n,
            )

            # extract covariate values and split into a past (historic) and future part
            covariate = covariate_series.random_component_values(copy=False)[
                covariate_start:covariate_end
            ]
            if self.input_chunk_length != 0:  # regular models
                past_covariate, future_covariate = (
                    covariate[: self.input_chunk_length],
                    covariate[self.input_chunk_length :],
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
        )


class PastCovariatesInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
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
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """

        super().__init__()

        self.ds = GenericInferenceDataset(
            target_series=target_series,
            covariates=covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            covariate_type=covariate_type,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
    ]:
        return self.ds[idx]


class FutureCovariatesInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        input_chunk_length: int = 12,
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
        input_chunk_length
            The length of the target series the model takes as input.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        self.ds = GenericInferenceDataset(
            target_series=target_series,
            covariates=covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            output_chunk_length=n,
            covariate_type=covariate_type,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], TimeSeries]:
        past_target, _, future_covariate, static_covariate, target_series = self.ds[idx]
        return past_target, future_covariate, static_covariate, target_series


class DualCovariatesInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
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
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        # This dataset is in charge of serving historic future covariates
        self.ds_past = PastCovariatesInferenceDataset(
            target_series=target_series,
            covariates=covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            covariate_type=CovariateType.HISTORIC_FUTURE,
            use_static_covariates=use_static_covariates,
        )

        # This dataset is in charge of serving future covariates
        self.ds_future = FutureCovariatesInferenceDataset(
            target_series=target_series,
            covariates=covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            covariate_type=CovariateType.FUTURE,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
    ]:
        (
            past_target,
            historic_future_covariate,
            _,
            static_covariate,
            ts_target,
        ) = self.ds_past[idx]
        _, future_covariate, _, _ = self.ds_future[idx]
        return (
            past_target,
            historic_future_covariate,
            future_covariate,
            static_covariate,
            ts_target,
        )


class MixedCovariatesInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
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
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = PastCovariatesInferenceDataset(
            target_series=target_series,
            covariates=past_covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            covariate_type=CovariateType.PAST,
            use_static_covariates=use_static_covariates,
        )

        # This dataset is in charge of serving historic and future covariates
        self.ds_future = DualCovariatesInferenceDataset(
            target_series=target_series,
            covariates=future_covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
    ]:

        (
            past_target,
            past_covariate,
            future_past_covariate,
            static_covariate,
            ts_target,
        ) = self.ds_past[idx]
        _, historic_future_covariate, future_covariate, _, _ = self.ds_future[idx]
        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            future_past_covariate,
            static_covariate,
            ts_target,
        )


class SplitCovariatesInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
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
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = PastCovariatesInferenceDataset(
            target_series=target_series,
            covariates=past_covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            covariate_type=CovariateType.PAST,
            use_static_covariates=use_static_covariates,
        )

        # This dataset is in charge of serving future covariates
        self.ds_future = FutureCovariatesInferenceDataset(
            target_series=target_series,
            covariates=future_covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            covariate_type=CovariateType.FUTURE,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
    ]:

        (
            past_target,
            past_covariate,
            future_past_covariate,
            static_covariate,
            ts_target,
        ) = self.ds_past[idx]
        _, future_covariate, _, _ = self.ds_future[idx]
        return (
            past_target,
            past_covariate,
            future_covariate,
            future_past_covariate,
            static_covariate,
            ts_target,
        )
