"""
Box-Cox Transformer
-------------------
"""

from typing import Optional, Union, Sequence
from scipy.stats import boxcox_normmax, boxcox
from scipy.special import inv_boxcox

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import FittableDataTransformer, InvertibleDataTransformer
from darts.logging import get_logger, raise_if

logger = get_logger(__name__)


class BoxCox(FittableDataTransformer[TimeSeries], InvertibleDataTransformer[TimeSeries]):

    def __init__(self,
                 name: str = "BoxCox"):
        """
        Box-Cox data transformer.
        See https://otexts.com/fpp2/transformations.html#mathematical-transformations for more information

        Parameters
        ----------
        name
            A specific name for the transformer
        """
        super().__init__(name)
        self._lmbda = None

    def fit(self,
            data: TimeSeries,
            lmbda: Optional[Union[float, Sequence[float]]] = None,
            optim_method='mle') -> 'BoxCox':
        """
        Sets the `lmbda` parameter value.

        Parameters
        ----------
        data
            The time series to fit on
        lmbda
            If None given, will automatically find an optimal value of lmbda (for each dimension
            of the time series) using `scipy.stats.boxcox_normmax` with `method=optim_method`
            If a single float is given, the same lmbda value will be used for all dimensions of the series.
            Also allows to specify a different lmbda value for each dimension of the time series by passing
            a sequence of values.
        optim_method
            Specifies which method to use to find an optimal value for the lmbda parameter.
            Either 'mle' or 'pearsonr'.

        Returns
        -------
            Fitted transformer (self)
        """
        super().fit(data)

        raise_if(not isinstance(optim_method, str) or optim_method not in ['mle', 'pearsonr'],
                 "optim_method parameter must be either 'mle' or 'pearsonr'",
                 logger)

        if lmbda is None:
            # Compute optimal lmbda for each dimension of the time series
            lmbda = data._df.apply(boxcox_normmax, method=optim_method)
        elif isinstance(lmbda, Sequence):
            raise_if(len(lmbda) != data.width,
                     "lmbda should have one value per dimension (ie. column or variable) of the time series",
                     logger)
        else:
            # Replicate lmbda to match dimensions of the time series
            lmbda = [lmbda] * data.width

        self._lmbda = lmbda

        return self

    def transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
        super().transform(data, *args, **kwargs)

        def _boxcox_wrapper(col):
            idx = data._df.columns.get_loc(col.name)  # get index from col name
            return boxcox(col, self._lmbda[idx])

        return TimeSeries.from_dataframe(data._df.apply(_boxcox_wrapper))

    def inverse_transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
        super().inverse_transform(data, *args, *kwargs)

        def _inv_boxcox_wrapper(col):
            idx = data._df.columns.get_loc(col.name)  # get index from col name
            return inv_boxcox(col, self._lmbda[idx])

        return TimeSeries.from_dataframe(data._df.apply(_inv_boxcox_wrapper))
