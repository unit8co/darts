"""
Invertible Data Transformer
---------------------------
"""
from typing import Union, Sequence, Callable, Iterator, Tuple, List

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
        Base class for invertible transformers offering both `transform()` and `inverse_transform()` methods.
        The inverse transformation function must be passed during the transformer's initialization. This class
        takes care of parallelizing the transformation of multiple TimeSeries when possible.

        Parameters
        ----------
        ts_transform (Callable)
            Function that will be applied to each TimeSeries object once the `transform()` function is called. The
            function must take as first argument a TimeSeries object, and return the transformed TimeSeries object.
            Additional parameters can be added if necessary, but in this case, the `_transform_iterator()` should be
            redefined accordingly, to yield the necessary arguments to this function (See `_transform_iterator()` for
            further details)
        ts_inverse_transform (Callable)
            Function that will be applied to each TimeSeries object once the `inverse_transform()` function is called.
            The function must take as first argument a TimeSeries object, and return the transformed TimeSeries object.
            Additional parameters can be added if necessary, but in this case, the `_inverse_transform_iterator()`
            should be redefined accordingly, to yield the necessary arguments to this function (See
            `_inverse_transform_iterator()` for further details)
        name
            The data transformer's name
        n_jobs
            The number of jobs to run in parallel (in case the transformer is handling a Sequence[TimeSeries]).
            Defaults to `1` (sequential). `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress
        """
        super().__init__(ts_transform=ts_transform, name=name, n_jobs=n_jobs, verbose=verbose, **kwargs)
        self._ts_inverse_transform = ts_inverse_transform

    def _inverse_transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries]]:
        """
        Returns an `Iterator` object with tuples of inputs for each single call to `ts_inverse_transform()`.
        Additional `args` and `kwargs` from `inverse_transform()` (that don't change across the calls to
        `ts_inverse_transform()`) are already forwarded, and thus don't need to be included in this generator.

        The basic implementation of this method returns `zip(series)`, that is, a generator of single-valued tuples,
        each containing one TimeSeries object.

        Parameters
        ----------
        series (Sequence[TimeSeries])
            Sequence of TimeSeries received in input.

        Returns
        -------
        (Iterator[Tuple])
            An iterator containing tuples of inputs for the `ts_inverse_transform()` method.

        Examples
        ________

        def my_ts_transform(series: TimeSeries, n: int) -> TimeSeries:
            return series + n

        def my_ts_inverse_transform(series: TimeSeries, n: int) -> TimeSeries:
            return series - n


        class IncreasingAdder(InvertibleDataTransformer):
            def __init__(self):
                super().__init__(ts_transform=my_ts_transform,
                                 ts_inverse_transform=my_ts_inverse_transform)

            def _transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries, int]]:
                return zip(series, (i for i in range(len(series))))

            def _inverse_transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries, int]]:
                return zip(series, (i for i in range(len(series))))
        """
        return zip(series)

    def inverse_transform(self,
                          series: Union[TimeSeries, Sequence[TimeSeries]],
                          *args,
                          **kwargs) -> Union[TimeSeries, List[TimeSeries]]:
        """
        Inverse-transform the data. In case a Sequence is passed as input data, this function takes care of
        parallelising the transformation of multiple series in the sequence at the same time.

        Parameters
        ----------
        series
            TimeSeries or Sequence of TimeSeries which will be inverse-transformed.
        args
            Additional positional arguments for the `ts_inverse_transform()` method
        kwargs
            Additional keyword arguments for the `ts_inverse_transform()` method

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
