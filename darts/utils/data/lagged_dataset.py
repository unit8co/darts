"""
Lagged Training Dataset
---------------------------
"""

from typing import Union, Sequence, Optional, Tuple, List
import numpy as np
from darts.logging import raise_if_not, raise_if

from darts.timeseries import TimeSeries
from darts.utils.data.sequential_dataset import SequentialDataset
from darts.utils.data.simple_inference_dataset import SimpleInferenceDataset


def _process_lags(lags: Optional[Union[int, List[int]]] = None,
                  lags_covariates: Optional[Union[int, List[int]]] = None
                  ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Process lags and lags_covariate.

    Params
    ------
    lags
        Number of lagged target values used to predict the next time step. If an integer is given the last `lags` lags
        are used (inclusive). Otherwise a list of integers with lags is required (each lag must be > 0).
    lags_covariates
        Number of lagged covariates values used to predict the next time step. If an integer is given
        the last `lags_covariates` lags are used (inclusive, starting from lag 1). Otherwise a list of
        integers with lags >= 0 is required. The special index 0 is supported, in case the covariate at time `t` should
        be used. Note that the 0 index is not included when passing a single interger value > 0.

    Returns
    -------
    Optional[List[int]]
        Processed `lags`, as a list of integers. If no lags are used, then `None` is returned.
    Optional[List[int]]
        Processed `lags_covariates` as a list of integers. If no lags covariates are used, then `None` is returned.

    Raises
    ------
    ValueError
        In case at least one of the required conditions is not met.
    """

    raise_if(
        (lags is None) and (lags_covariates is None),
        "At least one of `lags` or `lags_covariates` must be not None."
    )

    raise_if_not(
        isinstance(lags, (int, list)) or lags is None,
        f"`lags` must be of type int or list. Given: {type(lags)}."
    )

    raise_if_not(
        isinstance(lags_covariates, (int, list)) or lags_covariates is None,
        f"`lags_covariates` must be of type int or list. Given: {type(lags_covariates)}."
    )

    raise_if(
        isinstance(lags, bool) or isinstance(lags_covariates, bool),
        "`lags` and `lags_covariates` must be of type int or list, not bool.",
    )

    if isinstance(lags, int):
        raise_if_not(
            lags > 0,
            f"`lags` must be strictly positive. Given: {lags}.",
        )
        # selecting last `lags` lags, starting from position 1 (skipping current, pos 0, the one we want to predict)
        lags = list(range(1, lags + 1))

    elif isinstance(lags, list):
        for lag in lags:
            raise_if(
                not isinstance(lag, int) or (lag <= 0),
                f"Every element of `lags` must be a strictly positive integer. Given: {lags}.",
            )
    # using only the current current covariates, at position 0, which is the same timestamp as the prediction
    if isinstance(lags_covariates, int) and lags_covariates == 0:
        lags_covariates = [0]

    elif isinstance(lags_covariates, int):
        raise_if_not(
            lags_covariates > 0,
            f"`lags_covariates` must be an integer >= 0. Given: {lags_covariates}.",
        )
        lags_covariates = list(range(1, lags_covariates + 1))

    elif isinstance(lags_covariates, list):
        for lag in lags_covariates:
            raise_if(
                not isinstance(lag, int) or (lag < 0),
                f"Every element of `lags_covariates` must be an integer >= 0. Given: {lags_covariates}.",
            )

    return lags, lags_covariates


class LaggedTrainingDataset:
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        lags: Optional[Union[int, List[int]]] = None,
        lags_covariates: Optional[Union[int, List[int]]] = None,
        max_samples_per_ts: Optional[int] = None
    ):

        """Lagged Dataset
        A time series dataset wrapping around `SequentialDataset` containing tuples of (input_target, output_target,
        input_covariates) arrays, where "input_target" is #lags long, "input_covariates" is #lags_covariates long,
        and "output" has length 1.

        Params
        ------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates:
            Optionally, one or a sequence of `TimeSeries` containing covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`.
        lags
            Number of lagged target values used to predict the next time step. If an integer is given the last `lags`
            lags are used (inclusive). Otherwise a list of integers with lags is required (each lag must be > 0).
        lags_covariates
            Number of lagged covariates values used to predict the next time step. If an integer is given
            the last `lags_covariates` lags are used (inclusive, starting from lag 1). Otherwise a list of
            integers with lags >= 0 is required. The special index 0 is supported, in case the covariate at time `t`
            should be used. Note that the 0 index is not included when passing a single interger value > 0.
        """

        # the Sequential dataset will take care of handling series properly, and it is supporting
        # multiple TS

        super().__init__()

        self.lags, self.lags_covariates = _process_lags(lags, lags_covariates)

        if self.lags is not None and self.lags_covariates is not None:
            max_lags = max(max(self.lags), max(self.lags_covariates))
        elif self.lags_covariates is not None:
            max_lags = max(self.lags_covariates)
        else:
            max_lags = max(self.lags)

        if self.lags_covariates is not None and 0 in self.lags_covariates:
            # adding one for 0 covariate trick
            max_lags += 1

        self.sequential_dataset = SequentialDataset(
            target_series=target_series,
            covariates=covariates,
            input_chunk_length=max_lags,
            output_chunk_length=1,
            max_samples_per_ts=max_samples_per_ts,
        )

    def __len__(self):
        return len(self.sequential_dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        arrays = self.sequential_dataset[idx]

        input_target, output_target, input_covariates = [
            ar.copy() if ar is not None else None for ar in arrays
        ]

        if self.lags_covariates is not None and 0 in self.lags_covariates:
            """
            In case we need the 'time 0' covariate, we have to adjust the data with the following trick

            T5 T4 T3 T2 T1 T0 -> P | T5 T4 T3 T2 T1     -> T0~P'
            C5 C4 C3 C2 C1 C0      | C5 C4 C3 C2 C1 C0

            """
            # overwrite the prediction
            output_target = np.array(input_target[-1]).reshape(1, 1)
            # shortening the input_target by one
            input_target = input_target[:-1]

        # evaluating indexes from the end
        if self.lags is not None:
            lags_indices = np.array(self.lags) * (-1)
            input_target = input_target[lags_indices]
        else:
            input_target = None

        if self.lags_covariates is not None:
            if 0 in self.lags_covariates:
                cov_lags_indices = (np.array(self.lags_covariates) + 1) * (-1)
            else:
                cov_lags_indices = (np.array(self.lags_covariates)) * (-1)
            input_covariates = input_covariates[cov_lags_indices]
        else:
            input_covariates = None
        return input_target, output_target, input_covariates

    def get_data(self):
        """
        The function returns a training matrix X with shape (n_samples, lags + lags_covariates*covariates.width)
        and y with shape (n_sample,).
        """
        x = []
        y = []

        for input_target, output_target, input_covariates in self:
            row = []
            if input_target is not None:
                row.append(input_target.T)
            if input_covariates is not None:
                row.append(input_covariates.reshape(1, -1))

            x.append(np.concatenate(row, axis=1))
            y.append(output_target)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

        return x, y.ravel()


class LaggedInferenceDataset:
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        lags: Union[int, list] = None,
        lags_covariates: Union[int, list] = None,
        n: int = 1,
    ):
        """
        A time series dataset wrapping around `SimpleInferenceDataset`. The `input_chunk_length` is inferred through
        lags and lags_covariates.

        Params
        ------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates:
            Optionally, one or a sequence of `TimeSeries` containing covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`.
        lags
            Number of lagged target values used to predict the next time step. If an integer is given the last `lags`
            lags are used (inclusive). Otherwise a list of integers with lags is required (each lag must be > 0).
        lags_covariates
            Number of lagged covariates values used to predict the next time step. If an integer is given
            the last `lags_covariates` lags are used (inclusive, starting from lag 1). Otherwise a list of
            integers with lags >= 0 is required. The special index 0 is supported, in case the covariate at time `t`
            should be used. Note that the 0 index is not included when passing a single interger value > 0.
        n
            The number of time steps after the end of the training time series for which to produce predictions.
        """
        super().__init__()

        self.lags, self.lags_covariates = _process_lags(lags, lags_covariates)

        if self.lags is not None and self.lags_covariates is None:
            max_lag = max(self.lags)
        elif self.lags is None and self.lags_covariates is not None:
            max_lag = max(self.lags_covariates)
        else:
            max_lag = max([max(self.lags), max(self.lags_covariates)])

        input_chunk_length = max(max_lag, 1)

        self.inference_dataset = SimpleInferenceDataset(
            series=target_series,
            covariates=covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            model_is_recurrent=True if self.lags_covariates is not None and 0 in self.lags_covariates else False,
            add_prediction_covariate=True if self.lags_covariates is not None and 0 in self.lags_covariates else False
        )

    def __len__(self):
        return len(self.inference_dataset)

    def __getitem__(self, idx):
        return self.inference_dataset[idx]
