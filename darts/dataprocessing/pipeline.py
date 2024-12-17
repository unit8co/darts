"""
Pipeline
--------
"""

from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import Optional, Union

from darts import TimeSeries
from darts.dataprocessing.transformers import (
    BaseDataTransformer,
    FittableDataTransformer,
    InvertibleDataTransformer,
)
from darts.logging import get_logger, raise_if_not

logger = get_logger(__name__)


class Pipeline:
    def __init__(
        self,
        transformers: Sequence[BaseDataTransformer],
        copy: bool = False,
        verbose: bool = None,
        n_jobs: int = None,
    ):
        """
        Pipeline to combine multiple data transformers, chaining them together.

        Parameters
        ----------
        transformers
            Sequence of data transformers.
        copy
            If set makes a (deep) copy of each data transformer before adding them to the pipeline
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input to a method, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
            Note: this parameter will overwrite the value set in each single transformer. Leave this parameter set to
            `None` for keeping the original transformers' configurations.
        verbose
            Whether to print progress of the operations.
            Note: this parameter will overwrite the value set in each single transformer. Leave this parameter set
            to `None` for keeping the transformers configurations.

        Examples
        --------
        >>> import numpy as np
        >>> from darts import TimeSeries
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
        >>> from darts.dataprocessing.pipeline import Pipeline
        >>> values = np.arange(start=0, stop=12.5, step=2.5)
        >>> values[1:3] = np.nan
        >>> series = series.from_values(values)
        >>> pipeline = Pipeline([MissingValuesFiller(), Scaler()])
        >>> series_transformed = pipeline.fit_transform(series)
        <TimeSeries (DataArray) (time: 5, component: 1, sample: 1)>
        array([[[0.  ]],
            [[0.25]],
            [[0.5 ]],
            [[0.75]],
            [[1.  ]]])
        Coordinates:
        * time       (time) int64 0 1 2 3 4
        * component  (component) object '0'
        Dimensions without coordinates: sample
        """

        raise_if_not(
            all((isinstance(t, BaseDataTransformer)) for t in transformers),
            "transformers should be objects deriving from BaseDataTransformer",
            logger,
        )

        if transformers is None or len(transformers) == 0:
            logger.warning("Empty pipeline created")
            self._transformers: Sequence[BaseDataTransformer[TimeSeries]] = []
        elif copy:
            self._transformers = deepcopy(transformers)
        else:
            self._transformers = transformers

        self._invertible = all(
            isinstance(t, InvertibleDataTransformer) for t in self._transformers
        )

        self._fittable = any(
            isinstance(t, FittableDataTransformer) for t in self._transformers
        )

        self._global_fit = all(
            t._global_fit
            for t in self._transformers
            if isinstance(t, FittableDataTransformer)
        )

        if verbose is not None:
            for transformer in self._transformers:
                transformer.set_verbose(verbose)

        if n_jobs is not None:
            for transformer in self._transformers:
                transformer.set_n_jobs(n_jobs)

    def fit(self, data: Union[TimeSeries, Sequence[TimeSeries]]):
        """
        Fit all fittable transformers in pipeline.

        Parameters
        ----------
        data
            (`Sequence` of) `TimeSeries` to fit on.
        """

        # Find the last fittable transformer index
        # No need to fit (and thus transform) after this index, possibly saving a fair bit of time
        last_fittable_idx = -1
        for idx, transformer in enumerate(self._transformers):
            if isinstance(transformer, FittableDataTransformer):
                last_fittable_idx = idx

        for idx, transformer in enumerate(self._transformers):
            if idx <= last_fittable_idx and isinstance(
                transformer, FittableDataTransformer
            ):
                transformer.fit(data)

            if idx < last_fittable_idx:
                data = transformer.transform(data)

    def fit_transform(
        self, data: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        For each data transformer in the pipeline, first fit the data if transformer is fittable then transform data
        using fitted transformer. The transformed data is then passed to next transformer.

        Parameters
        ----------
        data
            (`Sequence` of) `TimeSeries` to fit and transform on.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Transformed data.
        """
        for transformer in self._transformers:
            if isinstance(transformer, FittableDataTransformer):
                transformer.fit(data)

            data = transformer.transform(data)
        return data

    def transform(
        self,
        data: Union[TimeSeries, Sequence[TimeSeries]],
        series_idx: Optional[Union[int, Sequence[int]]] = None,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        For each data transformer in pipeline transform data. Then transformed data is passed to next transformer.

        Parameters
        ----------
        data
            (`Sequence` of) `TimeSeries` to be transformed.
        series_idx
            Optionally, the index(es) of each series corresponding to their positions within the series used to fit
            the transformer (to retrieve the appropriate transformer parameters).

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Transformed data.
        """
        for transformer in self._transformers:
            data = transformer.transform(data, series_idx=series_idx)
        return data

    def inverse_transform(
        self,
        data: Union[TimeSeries, Sequence[TimeSeries]],
        partial: bool = False,
        series_idx: Optional[Union[int, Sequence[int]]] = None,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        For each data transformer in the pipeline, inverse-transform data. Then inverse transformed data is passed to
        the next transformer. Transformers are traversed in reverse order. Raises value error if not all of the
        transformers are invertible and ``partial`` is set to `False`. Set ``partial`` to True for inverting only the
        InvertibleDataTransformer in the pipeline.

        Parameters
        ----------
        data
            (Sequence of) TimeSeries to be inverse transformed.
        partial
            If set to `True`, the inverse transformation is applied even if the pipeline is not fully invertible,
            calling `inverse_transform()` only on the `InvertibleDataTransformer`s
        series_idx
            Optionally, the index(es) of each series corresponding to their positions within the series used to fit
            the transformer (to retrieve the appropriate transformer parameters).

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Inverse transformed data.
        """
        if not partial:
            raise_if_not(
                self._invertible,
                "Not all transformers in the pipeline can perform inverse_transform",
                logger,
            )

            for transformer in reversed(self._transformers):
                data = transformer.inverse_transform(data, series_idx=series_idx)
            return data
        else:
            for transformer in reversed(self._transformers):
                if isinstance(transformer, InvertibleDataTransformer):
                    data = transformer.inverse_transform(
                        data,
                        series_idx=series_idx,
                    )
            return data

    @property
    def invertible(self) -> bool:
        """
        Returns whether the pipeline is invertible or not.
        A pipeline is invertible if all transformers in the pipeline are themselves invertible.

        Returns
        -------
        bool
            `True` if the pipeline is invertible, `False` otherwise
        """
        return self._invertible

    @property
    def fittable(self) -> bool:
        """
        Returns whether the pipeline is fittable or not.
        A pipeline is fittable if at least one of the transformers in the pipeline is fittable.

        Returns
        -------
        bool
            `True` if the pipeline is fittable, `False` otherwise
        """
        return self._fittable

    @property
    def _fit_called(self) -> bool:
        """
        Returns whether all the transformers in the pipeline were fitted (when applicable).

        Returns
        -------
        bool
            `True` if all the fittable transformers are fitted, `False` otherwise
        """
        return all(
            (not isinstance(t, FittableDataTransformer)) or t._fit_called
            for t in self._transformers
        )

    def __getitem__(self, key: Union[int, slice]) -> "Pipeline":
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
        raise_if_not(
            isinstance(key, int) or isinstance(key, slice),
            "key must be either an int or a slice",
            logger,
        )

        if isinstance(key, int):
            transformers = [self._transformers[key]]
        else:
            transformers = self._transformers[key]
        return Pipeline(transformers, copy=True)

    def __iter__(self) -> Iterator[BaseDataTransformer]:
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
