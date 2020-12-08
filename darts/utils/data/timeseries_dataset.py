from typing import Sequence, Tuple, Union, Optional
from ...logging import get_logger
from ...timeseries import TimeSeries
from abc import ABC, abstractmethod

logger = get_logger(__name__)


class TimeSeriesDataset(ABC, Sequence):
    # TODO: include data processing
    def __init__(self):
        """
        Abstract class for a `TimeSeries` dataset. It emits 2-tuples of `(target, covariate)` `TimeSeries`.
        The emitted covariates are optional and can be `None`.

        It can be used as models' inputs, to obtain simple forecasts on each `TimeSeries`
        (using covariates if specified).

        `TimeSeriesDataset` are inheriting from `Sequence`; meaning that the implementations have to
        provide the `__len__()` and `__getitem__()` methods.
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        pass


class TimeSeriesTrainingDataset(ABC, Sequence):
    # TODO: include data processing
    def __init__(self):
        """
        Abstract class for a `TimeSeries` training dataset. It emits 4-tuples of
        `(input_target, input_covariate, output_target, output_covariate)` `TimeSeries`.
        The covariates are optional and can be `None`.

        This is meant to be used for training (or validation), where `input*` series represent model
        inputs and `output*` represent model outputs.

        `TimeSeriesTrainingDataset` are inheriting from `Sequence`; meaning that the implementations have to
        provide the `__len__()` and `__getitem__()` methods.
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[TimeSeries, Optional[TimeSeries], TimeSeries, Optional[TimeSeries]]:
        pass
