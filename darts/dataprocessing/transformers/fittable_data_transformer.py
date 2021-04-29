"""
Fittable Data Transformer
-------------------------
"""

from abc import abstractmethod
from darts import TimeSeries
from typing import Union, Sequence, Callable
from darts.logging import get_logger
from darts.dataprocessing.transformers import BaseDataTransformer
from darts.utils import _parallel_apply, _build_tqdm_iterator

logger = get_logger(__name__)


class FittableDataTransformer(BaseDataTransformer):
    def __init__(self,
                 ts_transform: Callable,
                 ts_fit: Callable,
                 name: str = "FittableDataTransformer",
                 n_jobs: int = 1,
                 verbose: bool = False,
                 **kwargs):

        """
        Abstract class for data transformers implementing a fit method. All deriving classes must implement
        `fit()` and `transform()`.

        names
            The data transformer's name
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors
        verbose
            Optionally, whether to print operations progress
        """
        super().__init__(ts_transform=ts_transform,
                         name=name,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         **kwargs)
        self._fit_called = False
        self._fitted_params = None  # stores the fitted parameters/objects
        self._ts_fit = ts_fit

    def _fit_iterator(self, series: Sequence[TimeSeries]):
        return zip(series)

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs) -> 'FittableDataTransformer':
        """
        Fit the data transformer to data.
        Not implemented in base class and has to be implemented by deriving classes.

        Parameters
        ----------
        data
            Object on which data transformer will be fitted.

        Returns
        -------
        BaseDataTransformer
            Fitted data transformer (typically would be self)
        """
        self._fit_called = True

        desc = "Fitting ({})".format(self._name)

        if isinstance(series, TimeSeries):
            data = [series]
        else:
            data = series

        input_iterator = _build_tqdm_iterator(self._fit_iterator(data),
                                              verbose=self._verbose,
                                              desc=desc,
                                              total=len(data))

        self._fitted_params = _parallel_apply(input_iterator, self._ts_fit,
                                              self._n_jobs, args, kwargs)

        return self

    def fit_transform(self, series: TimeSeries, *args, **kwargs) -> TimeSeries:
        """
        Fit the transformer to data and then transform data.

        Parameters
        ----------
        data
            Object used to fit and transform.
        args
            Additional positional arguments for the `transform()` method
        kwargs
            Additional keyword arguments for the `transform()` method

        Returns
        -------
        T
            Transformed data.
        """
        return self.fit(series).transform(series, *args, **kwargs)
