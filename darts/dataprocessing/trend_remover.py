from typing import Optional, List

from darts.dataprocessing import Validator, FittableDataTransformer, InvertibleDataTransformer

from ..timeseries import TimeSeries
from ..logging import get_logger

logger = get_logger(__name__)


class TrendRemover(FittableDataTransformer[TimeSeries], InvertibleDataTransformer[TimeSeries]):
    def __init__(self, name: str = "TrendRemover", validators: Optional[List[Validator]] = None):
        """
        Data transformer to remove trend from time series

        Parameters
        ----------
        name
            A specific name for the transformer
        TODO: complete
        """
        super().__init__(name=name, validators=validators)
