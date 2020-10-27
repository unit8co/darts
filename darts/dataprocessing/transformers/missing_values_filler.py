"""
Missing Values Filler
---------------------
"""
from typing import Union

from darts.dataprocessing.transformers import BaseDataTransformer
from darts.utils.missing_values import fill_missing_values

from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not

logger = get_logger(__name__)


class MissingValuesFiller(BaseDataTransformer[TimeSeries]):
    def __init__(self,
                 fill: Union[str, float] = 'auto',
                 name: str = "MissingValuesFiller"):
        """
        Data transformer to fill missing values from time series

        Parameters
        ----------
        fill
            The value used to replace the missing values.
            If set to 'auto', will auto-fill missing values using the `pandas.Dataframe.interpolate()` method.
        name
            A specific name for the transformer
        """
        raise_if_not(isinstance(fill, str) or isinstance(fill, float),
                     "`fill` should either be a string or a float",
                     logger)
        raise_if(isinstance(fill, str) and fill != 'auto',
                 "invalid string for `fill`: can only be set to 'auto'",
                 logger)

        super().__init__(name)
        self._fill = fill

    def transform(self, data: TimeSeries, **interpolate_kwargs) -> TimeSeries:
        super().transform(data)
        return fill_missing_values(data, self._fill, **interpolate_kwargs)
