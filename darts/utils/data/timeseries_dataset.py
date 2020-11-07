from typing import Sequence, Tuple, Union
from ...logging import get_logger
from ...timeseries import TimeSeries
from abc import ABC, abstractmethod

logger = get_logger(__name__)


class TimeSeriesDataset(ABC, Sequence):
    # TODO: include data processing
    def __init__(self):
        """
        Abstract class for a `TimeSeries` dataset. These datasets can contain one of these two types:
         * simple `TimeSeries`
         * tuples of two (input, target) `TimeSeries`.

        Typically, tuples of (input, target) are used to train and/or evaluate models, whereas
        simple `TimeSeries` can be used to obtain simple forecasts on each `TimeSeries`.

        These datasets are usually used for one these purposes:
          * Train a model on several time series (or several input/target series splits).
          * Differentiate between the time series components (dimensions) used as inputs (or features) and targets.
          * Get bulk predictions on a whole collection of time series.

        `TimeSeriesDataset` are inheriting from `Sequence`; meaning that the implementations have to
        provide the `__len__()` and `__getitem__()` methods.
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Union[TimeSeries, Tuple[TimeSeries, TimeSeries]]:
        pass
