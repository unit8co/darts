"""
Inference Dataset
-----------------
"""

from typing import Union, Sequence, Optional, Tuple

from .timeseries_dataset import TimeSeriesInferenceDataset
from ...timeseries import TimeSeries
from ...logging import raise_if_not


class SimpleInferenceDataset(TimeSeriesInferenceDataset):
    def __init__(self,
                 series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates_past: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 covariates_future: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None):
        super().__init__()
        self.series = [series] if isinstance(series, TimeSeries) else series
        self.covariates_past = [covariates_past] if isinstance(covariates_past, TimeSeries) else covariates_past
        self.covariates_future = [covariates_future] if isinstance(covariates_future, TimeSeries) else covariates_future

        raise_if_not((covariates_past is None or len(series) == len(covariates_past)) and
                     (covariates_future is None or len(series) == len(covariates_future)),
                     'The number of target series must be equal to the number of covariates.')

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx: int) -> Tuple[TimeSeries, Optional[TimeSeries], Optional[TimeSeries]]:
        cov_past_output = None if self.covariates_past is None else self.covariates_past[idx]
        cov_future_output = None if self.covariates_future is None else self.covariates_future[idx]

        return self.series[idx], cov_past_output, cov_future_output


