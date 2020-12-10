from typing import Sequence, Tuple, Union, Optional
from ...logging import get_logger
from ...timeseries import TimeSeries
from abc import ABC, abstractmethod

logger = get_logger(__name__)


class TimeSeriesInferenceDataset(ABC, Sequence):
    def __init__(self):
        """
        Abstract class for a `TimeSeries` inference dataset. It emits 2-tuples of
        `(input_target, input_covariate)` `TimeSeries`.
        The emitted covariates are optional and can be `None`.

        It can be used as models' inputs, to obtain simple forecasts on each `TimeSeries`
        (using covariates if specified).

        `TimeSeriesDataset` are inheriting from `Sequence`; meaning that the implementations have to
        provide the `__len__()` and `__getitem__()` methods.

        TODO: handle data processing
        TODO: handle optional "future" covariates
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        pass


class TimeSeriesTrainingDataset(ABC, Sequence):
    def __init__(self):
        """
        Abstract class for a `TimeSeries` training dataset. It emits 3-tuples of
        `(input_target, output_target, input_covariate)` `TimeSeries`.
        The covariates are optional and can be `None`.

        This is meant to be used for training (or validation), where `input*` series represent model
        inputs and `output*` represent model outputs.

        `TimeSeriesTrainingDataset` are inheriting from `Sequence`; meaning that the implementations have to
        provide the `__len__()` and `__getitem__()` methods.

        TODO: handle data processing
        TODO: handle optional "future" covariates
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[TimeSeries, TimeSeries, Optional[TimeSeries]]:
        pass
