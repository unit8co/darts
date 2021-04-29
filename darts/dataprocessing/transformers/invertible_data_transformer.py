"""
Invertible Data Transformer
---------------------------
"""
from typing import Union, Sequence, Callable
from abc import abstractmethod

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not
from darts.dataprocessing.transformers import BaseDataTransformer
from darts.utils import _parallel_apply, _build_tqdm_iterator

logger = get_logger(__name__)


class InvertibleDataTransformer(BaseDataTransformer):

    def __init__(self,
                 ts_transform: Callable,
                 ts_inverse_transform: Callable,
                 name: str = "InvertibleDataTransformer",
                 n_jobs: int = 1,
                 verbose: bool = False,
                 **kwargs):

        """
        Abstract class for data transformers implementing a fit method. All deriving classes must implement
        `transform()` and `inverse_transform()`.

        Parameters
        ----------
        names
            The data transformer's name
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors
        verbose
            Optionally, whether to print operations progress
        """
        super().__init__(ts_transform=ts_transform, name=name, n_jobs=n_jobs, verbose=verbose, **kwargs)
        self._ts_inverse_transform = ts_inverse_transform

    def _inverse_transform_iterator(self, series: Sequence[TimeSeries]):
        """
        Returns a generator for the input passed to the ts_inverse_transform function. This function could be
        redefined in case more data must be fed into the ts_inverse_transform function.
        """
        return zip(series)

    def inverse_transform(self,
                          series: Union[TimeSeries, Sequence[TimeSeries]],
                          *args,
                          **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        Perform inverse transformation of the data. Not implemented in base class.

        Parameters
        ----------
        series
            Object which will be inverse transformed.
        args
            Additional positional arguments for the `inverse_transform()` method
        kwargs
            Additional keyword arguments for the `inverse_transform()` method

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Inverse transformed data.
        """
        if hasattr(self, "_fit_called"):
            raise_if_not(self._fit_called, "fit() must have been called before inverse_transform()", logger)

        desc = "Inverse ({})".format(self._name)

        if isinstance(series, TimeSeries):
            data = [series]
        else:
            data = series

        input_iterator = _build_tqdm_iterator(self._inverse_transform_iterator(data),
                                              verbose=self._verbose,
                                              desc=desc,
                                              total=len(data))

        transformed_data = _parallel_apply(input_iterator, self._ts_inverse_transform,
                                           self._n_jobs, args, kwargs)

        return transformed_data[0] if isinstance(series, TimeSeries) else transformed_data
