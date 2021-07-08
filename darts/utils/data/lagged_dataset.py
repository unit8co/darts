"""
Lagged Training Dataset
---------------------------
"""

from typing import Union, Sequence, Optional, Tuple
import numpy as np
import pandas as pd
from darts.timeseries import TimeSeries
from darts.utils.data.sequential_dataset import SequentialDataset
from darts.utils.data.timeseries_dataset import TrainingDataset
from darts.logging import raise_if_not, raise_if


class LaggedDataset(TrainingDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 lags: Union[int, list] = None,
                 lags_exog: Union[int, list] = None,
                 max_samples_per_ts: Optional[int] = None):
        """Lagged Dataset

        A timeseries dataset wrapping around a SequentialDataset, yielding tuples of (input, output, input_covariates)
        arrays, where "input" contains
        """

    
        # the Sequential dataset will take care of handling series properly, and it is supporting
        # multiple TS
        self.sequential_dataset = SequentialDataset(target_series=target_series,
                                                    covariates=covariates,
                                                    input_chunk_length=self.max_lag,
                                                    output_chunk_length=1,
                                                    max_samples_per_ts=max_samples_per_ts)

    def __len__(self):
        return len(self.sequential_dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

        input_target, output_target, input_covariates = self.sequential_dataset[idx]
        # filter input_target and input covariates to have the proper lag

        # evaluating indexes from the end
        if self.lags is not None:
            lags_indices = np.array(self.lags) * (-1)
            input_target = input_target[lags_indices]
        if self.lags_exog is not None:
            exog_lags_indices = np.array(self.lags_exog) * (-1)
            input_covariates = input_covariates[exog_lags_indices]

        return input_target, output_target, input_covariates

    def get_matrix_data(self):
        X = []
        y = []

        for input_target, output_target, input_covariates in self:
            if input_covariates is not None:
                X.append(pd.DataFrame(np.concatenate((input_target, input_covariates), axis=None)))
            else:
                X.append(pd.DataFrame(input_target))
            y.append(pd.DataFrame(output_target))

        X = pd.concat(X, axis=1)
        y = pd.concat(y, axis=1)

        print(X.T, y.T)
        return X.T, y.T
