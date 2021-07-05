"""
Box-Cox Transformer
-------------------
"""

from typing import Optional, Union, Sequence, Iterator, Tuple
from scipy.stats import boxcox_normmax, boxcox
from scipy.special import inv_boxcox
import pandas as pd

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import FittableDataTransformer, InvertibleDataTransformer
from darts.logging import get_logger, raise_if


logger = get_logger(__name__)

# TODO: extend to stochastic series


class BoxCox(FittableDataTransformer, InvertibleDataTransformer):

    def __init__(self,
                 name: str = "BoxCox",
                 lmbda: Optional[Union[float,
                                 Sequence[float],
                                 Sequence[Sequence[float]]]] = None,
                 optim_method='mle',
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Box-Cox data transformer.
        See https://otexts.com/fpp2/transformations.html#mathematical-transformations for more information.

        Parameters
        ----------
        name
            A specific name for the transformer
        lmbda
            If None given, will automatically find an optimal value of lmbda (for each dimension
            of the time series, for each time series) using `scipy.stats.boxcox_normmax` with `method=optim_method`
            If a single float is given, the same lmbda value will be used for all dimensions of the series, for all
            the series.
            Also allows to specify a different lmbda value for each dimension of the time series by passing
            a sequence of values (or a sequence of sequence of values in case of multiple time series).
        optim_method
            Specifies which method to use to find an optimal value for the lmbda parameter.
            Either 'mle' or 'pearsonr'. Ignored if lmbda != None.
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
            passed as input to a method, parallelising operations regarding different `TimeSeries`. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress
        """

        super().__init__(name=name,
                         n_jobs=n_jobs,
                         verbose=verbose)

        raise_if(not isinstance(optim_method, str) or optim_method not in ['mle', 'pearsonr'],
                 "optim_method parameter must be either 'mle' or 'pearsonr'",
                 logger)

        self._lmbda = lmbda
        self._optim_method = optim_method

    def _fit_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries,
                                                                            Optional[Union[Sequence[float], float]]]]:

        if isinstance(self._lmbda, Sequence) and isinstance(self._lmbda[0], Sequence):
            # CASE 0: Sequence[Sequence[float]]
            raise_if(len(self._lmbda) != len(series),
                     "with multiple time series the number of lmbdas sequences must equal the number of time \
                        series",
                     logger)
            return zip(series, self._lmbda)
        else:
            # CASE 1: Sequence[float], float, None. Replicating the same value for each TS
            lmbda_gen = (self._lmbda for _ in range(len(series)))
            return zip(series, lmbda_gen)

    def _transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple]:
        return zip(series, self._fitted_params)

    def _inverse_transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple]:
        return zip(series, self._fitted_params)

    @staticmethod
    def ts_fit(series: TimeSeries,
               lmbda: Optional[Union[float, Sequence[float]]],
               method) -> Union[Sequence[float],
                                pd.core.series.Series]:
        if lmbda is None:
            # Compute optimal lmbda for each dimension of the time series. In this case, the return type is
            # a pd.core.series.Series, which is not inhering from collections.abs.Sequence
            lmbda = series.pd_dataframe(copy=False).apply(boxcox_normmax, method=method)
        elif isinstance(lmbda, Sequence):
            raise_if(len(lmbda) != series.width,
                     "lmbda should have one value per dimension (ie. column or variable) of the time series",
                     logger)
        else:
            # Replicate lmbda to match dimensions of the time series
            lmbda = [lmbda] * series.width

        return lmbda

    @staticmethod
    def ts_transform(series: TimeSeries, lmbda: Union[Sequence[float], pd.core.series.Series]) -> TimeSeries:

        def _boxcox_wrapper(col):
            idx = series.pd_dataframe(copy=False).columns.get_loc(col.name)  # get index from col name
            return boxcox(col, lmbda[idx])

        return TimeSeries.from_dataframe(series.pd_dataframe(copy=False).apply(_boxcox_wrapper))

    @staticmethod
    def ts_inverse_transform(series: TimeSeries, lmbda: Union[Sequence[float], pd.core.series.Series]) -> TimeSeries:

        def _inv_boxcox_wrapper(col):
            idx = series.pd_dataframe(copy=False).columns.get_loc(col.name)  # get index from col name
            return inv_boxcox(col, lmbda[idx])

        return TimeSeries.from_dataframe(series.pd_dataframe(copy=False).apply(_inv_boxcox_wrapper))

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> 'FittableDataTransformer':
        # adding lmbda and optim_method params
        return super().fit(series, method=self._optim_method)
