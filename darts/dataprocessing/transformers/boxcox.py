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

from darts.utils import _build_tqdm_iterator
from joblib import Parallel, delayed


logger = get_logger(__name__)


class BoxCox(FittableDataTransformer[TimeSeries], InvertibleDataTransformer[TimeSeries]):

    def __init__(self,
                 name: str = "BoxCox",
                 lmbda: Optional[Union[float,
                                 Sequence[float],
                                 Sequence[Sequence[float]]]] = None,
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
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors
        verbose
            Optionally, whether to print progress
        """
        super().__init__(name)
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._lmbda_original = lmbda  # keep the initial setting, the fitting function should refer to this
        self._lmbda = lmbda

    def fit(self,
            data: Union[TimeSeries, Sequence[TimeSeries]],
            optim_method='mle') -> 'BoxCox':
        """
        Sets the `lmbda` parameter value.

        Parameters
        ----------
        data
            The time series to fit on
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

        def _fit_single_series(data, lmbda):

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

            return lmbda

        if isinstance(data, TimeSeries):
            self._lmbda = _fit_single_series(data, self._lmbda_original)

        elif isinstance(data, Sequence):

            # in case of sequence of sequence of values, the outer sequence must be as long as the number of time series
            if isinstance(self._lmbda_original, Sequence) and isinstance(self._lmbda_original[0], Sequence):
                raise_if(len(self._lmbda_original) != len(data),
                         "with multiple time series the number of lmbdas sequences must equal the number of time \
                         series",
                         logger)
            else:
                # we have either a single value or None
                iterator = _build_tqdm_iterator(data,
                                                verbose=self._verbose,
                                                desc="Fitting BoxCox")

                self._lmbda = Parallel(n_jobs=self._n_jobs, prefer="threads")(delayed(_fit_single_series)
                                                                              (ts, self._lmbda_original)
                                                                              for ts in iterator)

        return self

    def transform(self,
                  data: Union[TimeSeries, Sequence[TimeSeries]],
                  *args,
                  **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
        super().transform(data, *args, **kwargs)

        def _transform_single_series(data, lmbda):
            def _boxcox_wrapper(col):
                idx = data._df.columns.get_loc(col.name)  # get index from col name
                return boxcox(col, lmbda[idx])

            return TimeSeries.from_dataframe(data._df.apply(_boxcox_wrapper))

        if isinstance(data, TimeSeries):
            return _transform_single_series(data, self._lmbda)
        else:
            # multiple time series
            iterator = _build_tqdm_iterator(zip(data, self._lmbda),
                                            verbose=self._verbose,
                                            desc="Transforming BoxCox",
                                            total=len(data))

            transformed_data = Parallel(n_jobs=self._n_jobs)(delayed(_transform_single_series)(ts,
                                                                                                                 lmbda)
                                                                               for ts, lmbda in iterator)
            return transformed_data

    def inverse_transform(self,
                          data: Union[TimeSeries, Sequence[TimeSeries]],
                          *args,
                          **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
        super().inverse_transform(data, *args, *kwargs)

        def _inverse_single_series(data, lmbda):

            def _inv_boxcox_wrapper(col):
                idx = data._df.columns.get_loc(col.name)  # get index from col name
                return inv_boxcox(col, lmbda[idx])

            return TimeSeries.from_dataframe(data._df.apply(_inv_boxcox_wrapper))

        if isinstance(data, TimeSeries):
            return _inverse_single_series(data, self._lmbda)
        else:
            iterator = _build_tqdm_iterator(zip(data, self._lmbda),
                                            verbose=self._verbose,
                                            desc="Inverse BoxCox",
                                            total=len(data))

            inverse_data = Parallel(n_jobs=self._n_jobs, prefer="threads")(delayed(_inverse_single_series)(ts, lmbda)
                                                                           for ts, lmbda in iterator)
            return inverse_data
