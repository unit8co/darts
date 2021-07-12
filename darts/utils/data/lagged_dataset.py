"""
Lagged Training Dataset
---------------------------
"""

from typing import Union, Sequence, Optional, Tuple
import numpy as np
import pandas as pd
from darts.logging import raise_if_not, raise_if

from darts.timeseries import TimeSeries
from darts.utils.data.matrix_dataset import (
    MatrixTrainingDataset,
    MatrixInferenceDataset,
)
from darts.utils.data.sequential_dataset import SequentialDataset


class LaggedDataset(MatrixTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        lags: Union[int, list] = None,
        lags_covariates: Union[int, list] = None,
        max_samples_per_ts: Optional[int] = None,
    ):
        """Lagged Dataset

        A timeseries dataset wrapping around a SequentialDataset, yielding tuples of (input, output, input_covariates)
        arrays, where "input" contains

        Params
        ------
        # TODO finish doc
        lags : Union[int, list]
            Number of lagged target values used to predict the next time step. If an integer is given
            the last `lags` lags are used (inclusive). Otherwise a list of integers with lags is required.
            The integers must be strictly positive (>0).
        lags_covariates : Union[int, list, bool]
            Number of lagged exogenous values used to predict the next time step. If an integer is given
            the last `lags_covariates` lags are used (inclusive, and starting from index 1 #TODO check/rewrite).
            Otherwise a list of integers with lags is required. The integers must be positive (>=0).
        """

        # the Sequential dataset will take care of handling series properly, and it is supporting
        # multiple TS

        super().__init__()
        self.target_series = target_series
        self.covariates = covariates
        self.lags = lags
        self.lags_covariates = lags_covariates
        self.max_samples_per_ts = max_samples_per_ts
        self.using_covariate_0 = False

        if isinstance(self.lags, int):
            raise_if_not(
                self.lags > 0,
                f"`lags` must be strictly positive. Given: {self.lags}.",
            )
            # selecting last `lags` lags, starting from position 1 (skipping current, pos 0, the one we want to predict)
            self.lags = list(range(1, self.lags + 1))
        elif isinstance(self.lags, list):
            for lag in self.lags:
                raise_if(
                    not isinstance(lag, int) or (lag <= 0),
                    f"Every element of `lags` must be a strictly positive integer. Given: {self.lags}.",
                )

        # using only the current current covariates, at position 0, which is the same timestamp as the prediction
        if self.lags_covariates == 0:
            self.lags_covariates = [0]
            self.using_covariate_0 = True
        elif isinstance(self.lags_covariates, int):
            raise_if_not(
                self.lags_covariates > 0,
                f"`lags_covariates` must be positive. Given: {self.lags_covariates}.",
            )
            self.lags_covariates = list(range(1, self.lags_covariates + 1))
        elif isinstance(self.lags_covariates, list):
            for lag in self.lags_covariates:
                raise_if(
                    not isinstance(lag, int) or (lag < 0),
                    f"Every element of `lags_covariates` must be a positive integer. Given: {self.lags_covariates}.",
                )
                if lag == 0:
                    self.using_covariate_0 = True

        if self.lags is not None and self.lags_covariates is None:
            self.max_lag = max(self.lags)
        elif self.lags is None and self.lags_covariates is not None:
            self.max_lag = max(self.lags_covariates)
        else:
            self.max_lag = max([max(self.lags), max(self.lags_covariates)])

        if self.using_covariate_0:
            # in case we are using the covariate at position 0, we need an extra input chunk length, to be able to
            # access the prediction covariate. In this case, we'll need to discard the output chunk, and use the last
            # input target instead (in this way the covariate will be one element longer than the target input chunk,
            # which will be exactly the time 0 covariate value).
            self.max_lag += 1
            self.lags_covariates = np.array(self.lags_covariates) + 1

        self.sequential_dataset = SequentialDataset(
            target_series=target_series,
            covariates=covariates,
            input_chunk_length=self.max_lag,
            output_chunk_length=1,
            max_samples_per_ts=max_samples_per_ts,
        )

    def __len__(self):
        return len(self.sequential_dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        input_target, output_target, input_covariates = self.sequential_dataset[idx]

        if self.using_covariate_0:
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
        if self.lags_covariates is not None:
            cov_lags_indices = np.array(self.lags_covariates) * (-1)
            input_covariates = input_covariates[cov_lags_indices]

        return input_target, output_target, input_covariates

    def get_data(self):
        x = []
        y = []

        for idx in range(len(self.sequential_dataset)):
            input_target, output_target, input_covariates = self.__getitem__(idx)
            if input_covariates is not None:
                x.append(
                    pd.DataFrame(
                        np.concatenate((input_target, input_covariates), axis=None)
                    )
                )
            else:
                x.append(pd.DataFrame(input_target))
            y.append(pd.DataFrame(output_target))

        x = pd.concat(x, axis=1)
        y = pd.concat(y, axis=1)

        return x.T, y.T
