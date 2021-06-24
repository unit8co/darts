"""
TimeSeries Dataset Base Classes
-------------------------------
"""

from abc import ABC, abstractmethod
import numpy as np

from typing import Sequence, Tuple, Union, Optional
from ...logging import get_logger
from ...timeseries import TimeSeries

logger = get_logger(__name__)


class TimeSeriesInferenceDataset(ABC, Sequence):
    def __init__(self):
        """
        Abstract class for a `TimeSeries` inference dataset. It contains 3-tuples of
        `(input_target, past_covariate, future_covariate)` `TimeSeries`.
        The emitted covariates are optional and can be `None`.

        It can be used as models' inputs, to obtain simple forecasts on each `TimeSeries`
        (using covariates if specified).

        `TimeSeriesInferenceDataset` inherits from `Sequence`; meaning that the implementations have to
        provide the `__len__()` and `__getitem__()` methods.

        It contains `TimeSeries` (and not e.g. `np.ndarray`), because inference requires the time axes,
        and typically the performance penalty should be lower than for training datasets because there's no slicing.
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[TimeSeries, Optional[TimeSeries], Optional[TimeSeries]]:
        pass


class TrainingDataset(ABC, Sequence):
    def __init__(self):
        """
        Abstract class for a training dataset. It contains 3-tuples of
        `(input_target, output_target, input_covariate)` `np.ndarray`.
        The covariates are optional and can be `None`.

        This is meant to be used for training (or validation), where `input*` series represent model
        inputs and `output_target` represent model outputs.

        Darts `GlobalForecastingModel`s can be fit from instances of `TrainingDataset` using the
        `fit_from_dataset()` method.

        `TrainingDataset` inherits from `Sequence`; meaning that the implementations have to
        provide the `__len__()` and `__getitem__()` methods.

        It contains `np.ndarray` (and not `TimeSeries`), because training requires the values only,
        and so we can get big performance gains when slicing by returning only numpy views of the data.
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        pass
