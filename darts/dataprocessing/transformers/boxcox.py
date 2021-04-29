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
        See https://otexts.com/fpp2/transformations.html#mathematical-transformations for more information

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
            Either 'mle' or 'pearsonr'.
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors
        verbose
            Optionally, whether to print progress
        """
        
        def boxcox_ts_fit(series: TimeSeries, lmbda, *args, **kwargs):
            if lmbda is None:
                # Compute optimal lmbda for each dimension of the time series
                lmbda = series._df.apply(boxcox_normmax, method=optim_method, *args, **kwargs)
            elif isinstance(lmbda, Sequence):
                raise_if(len(lmbda) != series.width,
                         "lmbda should have one value per dimension (ie. column or variable) of the time series",
                         logger)
            else:
                # Replicate lmbda to match dimensions of the time series
                lmbda = [lmbda] * series.width

            return lmbda

        def boxcox_ts_transform(series: TimeSeries, lmbda, *args,
                                **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:

            def _boxcox_wrapper(col):
                idx = series._df.columns.get_loc(col.name)  # get index from col name
                return boxcox(col, lmbda[idx])

            return TimeSeries.from_dataframe(series._df.apply(_boxcox_wrapper, *args, **kwargs))

        def boxcox_ts_inverse_transform(series: TimeSeries, lmbda, *args,
                                        **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:

            def _inv_boxcox_wrapper(col):
                idx = series._df.columns.get_loc(col.name)  # get index from col name
                return inv_boxcox(col, lmbda[idx])

            return TimeSeries.from_dataframe(series._df.apply(_inv_boxcox_wrapper, *args, **kwargs))

        super().__init__(ts_transform=boxcox_ts_transform,
                         ts_inverse_transform=boxcox_ts_inverse_transform,
                         ts_fit=boxcox_ts_fit,
                         name=name,
                         n_jobs=n_jobs,
                         verbose=verbose)

        raise_if(not isinstance(optim_method, str) or optim_method not in ['mle', 'pearsonr'],
                 "optim_method parameter must be either 'mle' or 'pearsonr'",
                 logger)

        self._lmbda = lmbda
        self._optim_method = optim_method

    def _fit_iterator(self, series: TimeSeries, *args, **kwargs):
        if isinstance(self._lmbda, Sequence) and isinstance(self._lmbda[0], Sequence):
            # CASE 0: Sequence[Sequence[float]]
            raise_if(len(self._lmbda) != len(series),
                     "with multiple time series the number of lmbdas sequences must equal the number of time \
                        series",
                     logger)
            return zip(series, self._lmbda)
        else:
            # CASE 1: Sequence[flaot], float, None
            class LmbdaGen:
                def __init__(self, lmbda):
                    self._lmbda = lmbda

                def __iter__(self):
                    return self

                def __next__(self):
                    return self._lmbda

            return zip(series, LmbdaGen(self._lmbda))

    def _transform_iterator(self, series: Sequence[TimeSeries]):
        return zip(series, self._fitted_params)

    def _inverse_transform_iterator(self, series: Sequence[TimeSeries]):
        return zip(series, self._fitted_params)
