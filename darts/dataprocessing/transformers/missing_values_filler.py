"""
Missing Values Filler
---------------------
"""
from typing import Optional, List, Union

from darts.dataprocessing import Validator
from darts.dataprocessing.transformers import BaseDataTransformer
from darts.utils.missing_values import auto_fillna, fillna

from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not

logger = get_logger(__name__)


class MissingValuesFiller(BaseDataTransformer[TimeSeries]):
    def __init__(self,
                 fill: Union[str, float] = 'auto',
                 name: str = "MissingValuesFiller",
                 validators: Optional[List[Validator]] = None):
        """
        Data transformer to fill missing values from time series

        Parameters
        ----------
        fill
            The value used to replace the missing values.
            If set to 'auto', will auto-fill missing values using the `pandas.Dataframe.interpolate()` method.
        name
            A specific name for the transformer
        validators
            List of validators that will be called before fit(), transform() and inverse_transform()
        """
        raise_if_not(isinstance(fill, str) or isinstance(fill, float),
                     "`fill` should either be a string or a float",
                     logger)
        raise_if(isinstance(fill, str) and fill != 'auto',
                 "invalid string for `fill`: can only be set to 'auto'",
                 logger)

        super().__init__(name=name, validators=validators)
        self._fill = fill

    def transform(self, data: TimeSeries, **interpolate_kwargs) -> TimeSeries:
        super().transform(data)
        if self._fill == 'auto':
            return auto_fillna(data, **interpolate_kwargs)

        return fillna(data, self._fill)
