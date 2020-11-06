from typing import Union, Sequence
from .timeseries_dataset import TimeSeriesDataset
from ...timeseries import TimeSeries


class SimpleTimeSeriesDataset(TimeSeriesDataset):
    def __init__(self, series: Union[TimeSeries, Sequence[TimeSeries]]):
        super().__init__()
        self.series = [series] if isinstance(series, TimeSeries) else series

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx: int) -> TimeSeries:
        return self.series[idx]
