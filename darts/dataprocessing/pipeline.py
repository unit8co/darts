"""
Pipeline
--------
"""
from copy import deepcopy
from typing import Sequence, Union, Iterator

from darts.logging import raise_if_not, get_logger
from darts import TimeSeries
from darts.dataprocessing.transformers import BaseDataTransformer, InvertibleDataTransformer, FittableDataTransformer

logger = get_logger(__name__)


class Pipeline:
    def __init__(self,
                 transformers: Sequence[BaseDataTransformer[TimeSeries]],
                 copy: bool = False):
        """
        Pipeline combines multiple data transformers chaining them together.

        Parameters
        ----------
        transformers
            Sequence of data transformers.
        copy
            If set makes a (deep) copy of each data transformer before adding them to the pipeline
        """
        raise_if_not(all((isinstance(t, BaseDataTransformer)) for t in transformers),
                     "transformers should be objects deriving from BaseDataTransformer", logger)

        if transformers is None or len(transformers) == 0:
            logger.warning("Empty pipeline created")
            self._transformers: Sequence[BaseDataTransformer[TimeSeries]] = []
        elif copy:
            self._transformers = deepcopy(transformers)
        else:
            self._transformers = transformers

        self._invertible = all((isinstance(t, InvertibleDataTransformer) for t in self._transformers))

    def fit(self, data: TimeSeries):
        """
        Fit all fittable transformers in pipeline.

        Parameters
        ----------
        data
            TimeSeries to fit on.
        """

        # Find the last fittable transformer index
        # No need to fit (and thus transform) after this index, possibly saving a fair bit of time
        last_fittable_idx = -1
        for idx, transformer in enumerate(self._transformers):
            if isinstance(transformer, FittableDataTransformer):
                last_fittable_idx = idx

        for idx, transformer in enumerate(self._transformers):
            if idx <= last_fittable_idx and isinstance(transformer, FittableDataTransformer):
                transformer.fit(data)

            if idx < last_fittable_idx:
                data = transformer.transform(data)

    def fit_transform(self, data: TimeSeries) -> TimeSeries:
        """
        For each data transformer in pipeline first fit the data if transformer is fittable then transform data using
        fitted transformer. The transformed data is then passed to next transformer.

        Parameters
        ----------
        data
            TimeSeries to fit and transform on.

        Returns
        -------
        TimeSeries
            Transformed data.
        """
        for transformer in self._transformers:
            if isinstance(transformer, FittableDataTransformer):
                transformer.fit(data)

            data = transformer.transform(data)
        return data

    def transform(self, data: TimeSeries) -> TimeSeries:
        """
        For each data transformer in pipeline transform data. Then transformed data is passed to next transformer.

        Parameters
        ----------
        data
            TimeSeries to be transformed.

        Returns
        -------
        TimeSeries
            Transformed data.
        """
        for transformer in self._transformers:
            data = transformer.transform(data)
        return data

    def inverse_transform(self, data: TimeSeries) -> TimeSeries:
        """
        For each data transformer in pipeline inverse_transform data. Then inverse transformed data is passed to next
        transformer. Transformers are traversed in reverse order. Raises value error if not all of the transformers are
        invertible.

        Parameters
        ----------
        data
            TimeSeries to be inverse transformed.

        Returns
        -------
        TimeSeries
            Inverse transformed data.
        """
        raise_if_not(self._invertible, "Not all transformers in the pipeline can perform inverse_transform", logger)

        for transformer in reversed(self._transformers):
            data = transformer.inverse_transform(data)
        return data

    def __getitem__(self, key: Union[int, slice]) -> 'Pipeline':
        """
        Gets subset of Pipeline based either on index or slice with indexes.
        Resulting pipeline will deep copy transformers of the original pipeline.

        Parameters
        ----------
        key
            Either int or slice indicating the subset of data transformers to keep.

        Returns
        -------
        Pipeline
            Subset of pipeline determined by key.
        """
        raise_if_not(isinstance(key, int) or isinstance(key, slice), "key must be either an int or a slice", logger)

        if isinstance(key, int):
            transformers = [self._transformers[key]]
        else:
            transformers = self._transformers[key]
        return Pipeline(transformers, copy=True)

    def __iter__(self) -> Iterator[BaseDataTransformer[TimeSeries]]:
        """
        Returns
        -------
        Iterator
            Iterator on sequence of data transformers
        """
        return self._transformers.__iter__()

    def __len__(self):
        return len(self._transformers)

    def __copy__(self, deep: bool = True):
        return Pipeline(self._transformers, copy=deep)

    def __deepcopy__(self, memo=None):
        return self.__copy__(deep=True)

    def __str__(self):
        string = "Pipeline: "
        arrow = " -> "
        for transformer in self._transformers:
            string += str(transformer) + arrow

        return string[: -len(arrow)]

    def __repr__(self):
        return self.__str__()
