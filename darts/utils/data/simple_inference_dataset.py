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
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None):
        super().__init__()
        self.series = [series] if isinstance(series, TimeSeries) else series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates
        raise_if_not(covariates is None or len(series) == len(covariates),
                     'The number of target series must be equal to the number of covariates.')

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx: int) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        return (self.series[idx], None) if self.covariates is None else (self.series[idx], self.covariates[idx])
