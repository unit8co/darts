"""
Pipeline
--------
"""
from copy import deepcopy
from collections import defaultdict
from typing import List, Optional, Union, Tuple, Iterator

from darts.logging import raise_if_not, raise_if, get_logger
from darts import TimeSeries
from darts.preprocessing.base_transformer import BaseTransformer

logger = get_logger(__name__)


class Pipeline:
    def __init__(self,
                 transformers: List[BaseTransformer[TimeSeries]],
                 names: Optional[List[str]] = None,
                 deep: bool = False):
        """
        Pipeline combines multiple transformers chaining them together.

        Parameters
        ----------
        transformers
            List of transformers.
        names
            Optional list of names for transformers must have same length as transformers and names should be unique.
            If no names are specified transformer name will be set to str value of his index in original list.
        deep
            If set prevents changes in original transformers made by pipeline.
        """

        raise_if_not(all((isinstance(t, BaseTransformer)) for t in transformers),
                     "transformers should be objects deriving from BaseTransformer", logger)
        if deep:
            self._transformers = deepcopy(transformers)
        else:
            self._transformers = transformers

        if self._transformers is None or len(self._transformers) == 0:
            self._transformers: List[BaseTransformer[TimeSeries]] = []
            logger.warning("Empty pipeline created")

        if names:
            raise_if_not(len(names) == len(self._transformers), "names should have same length as transformer", logger)
            raise_if_not(len(names) == len(set(names)), "names should be unique", logger)
        else:
            names = list(map(str, range(len(self._transformers))))

        self._name_2_idx = {
            str(name): idx for idx, name in enumerate(names)
        }
        self._names = names

        self._args_for_transformers = defaultdict(tuple)
        self._kwargs_for_transformers = defaultdict(dict)

        self._reversible = all((t.reversible for t in self._transformers))
        self._can_predict = len(self._transformers) and self._transformers[-1].can_predict

    def set_transformer_args_kwargs(self, key: Union[str, int], *args, **kwargs):
        """
        Set arguments for `transform` and `inverse_transform` functions got transformer identified by either index
        or name. Raises value error if transformer can't be find by key.

        Parameters
        ----------
        key
            Either int or key to identify transformer for which args and kwargs will be set.
        args
            Positional arguments for the `transform` or `inverse_transform` method
        kwargs
            Keyword arguments for the `transform` or `inverse_transform` method
        """
        if isinstance(key, str):
            raise_if_not(key in self._name_2_idx.keys(),
                         f"No transformer named '{key}' available transformers {self._name_2_idx.keys()}", logger)
        elif isinstance(key, int):
            key = self._names[key]
        else:
            raise ValueError("Key is neither int nor str")
        self._args_for_transformers[key] = args
        self._kwargs_for_transformers[key] = kwargs

    def fit(self, data: TimeSeries):
        """
        Fit data to all of transformers in pipeline that are capable of fitting.

        Parameters
        ----------
        data
            TimeSeries to fit on.
        """
        for transformer in filter(lambda t: t.fittable, self._transformers):
            transformer.fit(data)

    def fit_transform(self, data: TimeSeries) -> TimeSeries:
        """
        For each transformer in pipeline first fit the data if transformer is fittable and then transform data using
        fitted transformer. Then transformed data is passed to next transformer.

        Parameters
        ----------
        data
            TimeSeries to fit and transform on.

        Returns
        -------
        TimeSeries
            Transformed data.
        """
        x = data
        for name, transformer in zip(self._names, self._transformers):
            args = self._args_for_transformers[name]
            kwargs = self._kwargs_for_transformers[name]
            x = transformer(x, *args, fit=transformer.fittable, **kwargs)
        return x

    def transform(self, data: TimeSeries) -> TimeSeries:
        """
        For each transformer in pipeline transform data. Then transformed data is passed to next transformer.

        Parameters
        ----------
        data
            TimeSeries to be transformed.

        Returns
        -------
        TimeSeries
            Transformed data.
        """
        x = data
        for name, transformer in zip(self._names, self._transformers):
            args = self._args_for_transformers[name]
            kwargs = self._kwargs_for_transformers[name]
            x = transformer(x, *args, **kwargs)
        return x

    def predict(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
        """
        First transform data then on last transformer predict is run with transformed data.

        Parameters
        ----------
        data
            TimeSeries to be transformed and then predict on.
        args
            Positional arguments for the `predict` method
        kwargs
            Keyword arguments for the `predict` method

        Returns
        -------
        TimeSeries
            Predicted data.
        """
        raise_if_not(self._can_predict,
                     "last object of pipeline doesn't have method called predict", logger)
        x = self(data)
        return self._transformers[-1].predict(x, *args, **kwargs)

    def inverse_transform(self, data: TimeSeries) -> TimeSeries:
        """
        For each transformer in pipeline inverse_transform data. Then inverse transformed data is passed to next
        transformer. Transformers are traversed in reverse order. Raises value error if not all of the transformers are
        reversible.

        Parameters
        ----------
        data
            TimeSeries to be inverse transformed.

        Returns
        -------
        TimeSeries
            Inverse transformed data.
        """
        raise_if_not(self._reversible, "Not all transformers are able to perform inverse_transform", logger)
        x = data
        for name, transformer in zip(self._names, reversed(self._transformers)):
            args = self._args_for_transformers[name]
            kwargs = self._kwargs_for_transformers[name]
            x = transformer(x, *args, inverse=True, **kwargs)
        return x

    def __getitem__(self, key: Union[str, int, slice]) -> 'Pipeline':
        """
        Gets subset of Pipeline based either on index, name of transformer or slice with indexes. Transformers with
        names are picked based on the key. Resulting pipeline will deep copy transformers of original pipeline.

        Parameters
        ----------
        key
            Either int, str or slice.

        Returns
        -------
        Pipeline
            Subset of pipeline determined by key.
        """
        if isinstance(key, int):
            return Pipeline([self._transformers[key]], [self._names[key]], deep=True)
        elif isinstance(key, str):
            raise_if_not(key in self._name_2_idx.keys(),
                         f"No transformer named '{key}' available transformers {self._name_2_idx.keys()}", logger)
            return Pipeline([self._transformers[self._name_2_idx[key]]], [key], deep=True)
        elif isinstance(key, slice):
            return Pipeline(self._transformers[key], self._names[key], deep=True)
        raise_if(True, "Key must be int, str or slice", logger)

    def __call__(self, data: TimeSeries, inverse: bool = False) -> TimeSeries:
        """
        Calls `transform` or `inverse_transform` with data based on inverse flag.
        Parameters
        ----------
        data
            TimeSeries to be transformed.
        inverse
            Is set inverse transform will be run instead of transform.
        Returns
        -------
        TimeSeries
            Transformed (or inverse transformed) data.
        """
        if inverse:
            return self.inverse_transform(data)
        return self.transform(data)

    def __iter__(self) -> Iterator[Tuple[BaseTransformer[TimeSeries], str]]:
        """
        Returns
        -------
        Iterator
            Containing tuples of (transformer, transformer_name)
        """
        return zip(self._transformers, self._names)

    def __len__(self):
        return len(self._transformers)

    def __copy__(self, deep: bool = True):
        return Pipeline(self._transformers, self._names, deep=deep)

    def __deepcopy__(self):
        return self.__copy__(deep=True)
