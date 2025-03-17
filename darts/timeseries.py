"""
Timeseries
----------

``TimeSeries`` is the main class in `darts`.
It represents a univariate or multivariate time series, deterministic or stochastic.

The values are stored in an array of shape `(time, dimensions, samples)`, where
`dimensions` are the dimensions (or "components", or "columns") of multivariate series,
and `samples` are samples of stochastic series.

Definitions:
    - A series with `dimensions = 1` is **univariate** and a series with `dimensions > 1` is **multivariate**.
    - | A series with `samples = 1` is **deterministic** and a series with `samples > 1` is
      | **stochastic** (or **probabilistic**).

Each series also stores a `time_index`, which contains either datetimes (:class:`pandas.DateTimeIndex`)
or integer indices (:class:`pandas.RangeIndex`).

``TimeSeries`` are guaranteed to:
    - Have a monotonically increasing time index, without holes (without missing dates)
    - Contain numeric types only
    - Have distinct components/columns names
    - Have a well-defined frequency (`date offset aliases
      <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
      for ``DateTimeIndex``, or step size for ``RangeIndex``)
    - Have static covariates consistent with their components, or no static covariates
    - Have a hierarchy consistent with their components, or no hierarchy

``TimeSeries`` can contain global or component-specific static covariate data. Static covariates in `darts` refers
to external time-invariant data that can be used by some models to help improve predictions.
Read our `user guide on covariates <https://unit8co.github.io/darts/userguide/covariates.html>`__ and the
``TimeSeries`` documentation for more information on covariates.
"""

import contextlib
import itertools
import pickle
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from inspect import signature
from io import StringIO
from types import ModuleType
from typing import Any, Callable, Literal, Optional, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import narwhals as nw
import numpy as np
import pandas as pd
import xarray as xr
from narwhals.typing import IntoDataFrame, IntoSeries
from narwhals.utils import Implementation
from pandas.tseries.frequencies import to_offset
from scipy.stats import kurtosis, skew

from darts.logging import (
    get_logger,
    raise_if,
    raise_if_not,
    raise_log,
)
from darts.utils import _build_tqdm_iterator, _parallel_apply
from darts.utils.utils import (
    SUPPORTED_RESAMPLE_METHODS,
    expand_arr,
    generate_index,
    n_steps_between,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

logger = get_logger(__name__)

# dimension names in the DataArray
# the "time" one can be different, if it has a name in the underlying Series/DataFrame.
DIMS = ("time", "component", "sample")
AXES = {"time": 0, "component": 1, "sample": 2}

VALID_INDEX_TYPES = (pd.DatetimeIndex, pd.RangeIndex)
STATIC_COV_TAG = "static_covariates"
DEFAULT_GLOBAL_STATIC_COV_NAME = "global_components"
HIERARCHY_TAG = "hierarchy"
METADATA_TAG = "metadata"


class TimeSeries:
    def __init__(self, xa: xr.DataArray, copy=True):
        """
        Create a TimeSeries from a (well-formed) DataArray.
        It is recommended to use the factory methods to create TimeSeries instead.

        See Also
        --------
        TimeSeries.from_dataframe : Create from a `DataFrame` (:class:`pandas.DataFrame`, :class:`polars.DataFrame`,
            and other backends).
        TimeSeries.from_group_dataframe : Create multiple TimeSeries by groups from a :class:`pandas.DataFrame`.
        TimeSeries.from_series : Create from a `Series` (:class:`pandas.Series`, :class:`polars.Series`, and other
            backends).
        TimeSeries.from_values : Create from a :class:`numpy.ndarray`.
        TimeSeries.from_times_and_values : Create from a time index and a :class:`numpy.ndarray`.
        TimeSeries.from_csv : Create from a CSV file.
        TimeSeries.from_json : Create from a JSON file.
        TimeSeries.from_xarray : Create from an :class:`xarray.DataArray`.
        """
        if not isinstance(xa, xr.DataArray):
            raise_log(
                ValueError(
                    "Data must be provided as an xarray DataArray instance. "
                    "If you need to create a TimeSeries from another type "
                    "(e.g. a DataFrame), look at TimeSeries factory methods "
                    "(e.g. TimeSeries.from_dataframe(), "
                    "TimeSeries.from_xarray(), TimeSeries.from_values()"
                    "TimeSeries.from_times_and_values(), etc...)."
                ),
                logger,
            )
        if len(xa.shape) != 3:
            raise_log(
                ValueError(
                    f"TimeSeries require DataArray of dimensionality 3 ({DIMS})."
                ),
                logger,
            )

        # Ideally values should be np.float, otherwise certain functionalities like diff()
        # relying on np.nan (which is a float) won't work very properly.
        if not np.issubdtype(xa.values.dtype, np.number):
            raise_log(
                ValueError("The time series must contain numeric values only."), logger
            )

        val_dtype = xa.values.dtype
        if not (
            np.issubdtype(val_dtype, np.float64) or np.issubdtype(val_dtype, np.float32)
        ):
            logger.warning(
                "TimeSeries is using a numeric type different from np.float32 or np.float64. "
                "Not all functionalities may work properly. It is recommended casting your data to floating "
                "point numbers before using TimeSeries."
            )

        if xa.dims[-2:] != DIMS[-2:]:
            # The first dimension represents the time and may be named differently.
            raise_log(
                ValueError(
                    f"The last two dimensions of the DataArray must be named {DIMS[-2:]}"
                ),
                logger,
            )

        # check that columns/component names are unique
        components = xa.get_index(DIMS[1])
        if not len(set(components)) == len(components):
            raise_log(
                ValueError(
                    f"The components (columns) names must be unique. Provided: {components}"
                ),
                logger,
            )

        # how the time dimension is named; we convert hashable to string
        self._time_dim = str(xa.dims[0])

        # The following sorting returns a copy, which we are relying on.
        # As of xarray 0.18.2, this sorting discards the freq of the index for some reason
        # https://github.com/pydata/xarray/issues/5466
        # We sort only if the time axis is not already sorted (monotonically increasing).
        self._xa = self._sort_index(xa, copy=copy)
        self._time_index = self._xa.get_index(self._time_dim)

        if not isinstance(self._time_index, VALID_INDEX_TYPES):
            raise_log(
                ValueError(
                    "The time dimension of the DataArray must be indexed either with a DatetimeIndex "
                    "or with an RangeIndex."
                ),
                logger,
            )

        self._has_datetime_index = isinstance(self._time_index, pd.DatetimeIndex)

        if self._has_datetime_index:
            # store original freq (see bug of sortby() above).
            freq_tmp = xa.get_index(self._time_dim).freq

            # if original frequency is known and positive (n > 0 -> increasing time index),
            # it is guaranteed that original array was sorted and new freq must be the same.
            # otherwise, infer the frequency from the sorted array
            if freq_tmp is not None and freq_tmp.n > 0:
                self._freq = freq_tmp
            else:
                self._freq = to_offset(self._xa.get_index(self._time_dim).inferred_freq)

            if self._freq is None:
                raise_log(
                    ValueError(
                        "The time index of the provided DataArray is missing the freq attribute, and the frequency "
                        "could not be directly inferred. This probably comes from inconsistent date frequencies with "
                        "missing dates. If you know the actual frequency, try setting `fill_missing_dates=True, "
                        "freq=actual_frequency`. If not, try setting `fill_missing_dates=True, freq=None` to see if a "
                        "frequency can be inferred."
                    ),
                    logger,
                )

            self._freq_str: str = self._freq.freqstr

            # reset freq inside the xarray index (see bug of sortby() above).
            self._xa.get_index(self._time_dim).freq = self._freq
        else:
            self._freq: int = self._time_index.step
            self._freq_str = None

        # check static covariates
        static_covariates = self._xa.attrs.get(STATIC_COV_TAG, None)
        if not (
            isinstance(static_covariates, (pd.Series, pd.DataFrame))
            or static_covariates is None
        ):
            raise_log(
                ValueError(
                    "`static_covariates` must be either a pandas Series, DataFrame or None"
                ),
                logger,
            )

        # check if valid static covariates for multivariate TimeSeries
        if isinstance(static_covariates, pd.DataFrame):
            n_components = len(static_covariates)
            if n_components > 1 and n_components != self.n_components:
                raise_log(
                    ValueError(
                        "When passing a multi-row pandas DataFrame, the number of rows must match the number of "
                        "components of the TimeSeries object (multi-component/multi-row static covariates "
                        "must map to each TimeSeries component)."
                    ),
                    logger,
                )
            if copy:
                static_covariates = static_covariates.copy()
        elif isinstance(static_covariates, pd.Series):
            static_covariates = static_covariates.to_frame().T
        else:  # None
            pass

        # prepare static covariates:
        if static_covariates is not None:
            static_covariates.index = (
                self.components
                if len(static_covariates) == self.n_components
                else [DEFAULT_GLOBAL_STATIC_COV_NAME]
            )
            static_covariates.columns.name = STATIC_COV_TAG
            # convert numerical columns to same dtype as series
            # we get all numerical columns, except those that have right dtype already
            cols_to_cast = static_covariates.select_dtypes(
                include=np.number, exclude=self.dtype
            ).columns

            # Calling astype is costly even when there's no change...
            if not cols_to_cast.empty:
                static_covariates = static_covariates.astype(
                    {col: self.dtype for col in cols_to_cast}, copy=False
                )

        # prepare metadata
        metadata = self._xa.attrs.get(METADATA_TAG, None)
        if metadata is not None and not isinstance(metadata, dict):
            raise_log(
                ValueError(
                    "`metadata` must be of type `dict` mapping metadata attributes to their values."
                ),
                logger,
            )

        # handle hierarchy
        hierarchy = self._xa.attrs.get(HIERARCHY_TAG, None)
        self._top_level_component = None
        self._bottom_level_components = None
        if hierarchy is not None:
            if not isinstance(hierarchy, dict):
                raise_log(
                    ValueError(
                        "The hierarchy must be a dict mapping (non-top) component names to their parent(s) "
                        "in the hierarchy."
                    ),
                    logger,
                )
            # pre-compute grouping information
            components_set = set(self.components)
            children = set(hierarchy.keys())

            # convert string ancestors to list of strings
            hierarchy = {
                k: ([v] if isinstance(v, str) else v) for k, v in hierarchy.items()
            }

            if not all(c in components_set for c in children):
                raise_log(
                    ValueError(
                        "The keys of the hierarchy must be time series components"
                    ),
                    logger,
                )
            ancestors = set().union(*hierarchy.values())
            if not all(a in components_set for a in ancestors):
                raise_log(
                    ValueError(
                        "The values of the hierarchy must only contain component names matching those "
                        "of the series."
                    ),
                    logger,
                )
            hierarchy_top = components_set - children
            if not len(hierarchy_top) == 1:
                raise_log(
                    ValueError(
                        "The hierarchy must be such that only one component does not appear as a key "
                        "(the top level component)."
                    ),
                    logger,
                )
            self._top_level_component = hierarchy_top.pop()
            if self._top_level_component not in ancestors:
                raise_log(
                    ValueError(
                        "Invalid hierarchy. Component {} appears as it should be top-level, but "
                        "does not appear as an ancestor in the hierarchy dict."
                    ),
                    logger,
                )
            bottom_level = components_set - ancestors

            # maintain the same order as the original components
            self._bottom_level_components = [
                c for c in self.components if c in bottom_level
            ]

        # store static covariates, hierarchy and metadata in attributes (potentially storing None)
        self._xa = _xarray_with_attrs(self._xa, static_covariates, hierarchy, metadata)

    """
    Factory Methods
    ===============
    """

    @classmethod
    def from_xarray(
        cls,
        xa: xr.DataArray,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        fillna_value: Optional[float] = None,
    ) -> Self:
        """
        Return a TimeSeries instance built from an xarray DataArray.
        The dimensions of the DataArray have to be (time, component, sample), in this order. The time
        dimension can have an arbitrary name, but component and sample must be named "component" and "sample",
        respectively.

        The first dimension (time), and second dimension (component) must be indexed (i.e., have coordinates).
        The time must be indexed either with a pandas DatetimeIndex, a pandas RangeIndex, or a pandas Index that can
        be converted to a RangeIndex. It is better if the index has no holes; alternatively setting
        `fill_missing_dates` can in some cases solve these issues (filling holes with NaN, or with the provided
        `fillna_value` numeric value, if any).

        If two components have the same name or are not strings, this method will disambiguate the components
        names by appending a suffix of the form "<name>_N" to the N-th column with name "name".
        The component names in the static covariates and hierarchy (if any) are *not* disambiguated.

        Parameters
        ----------
        xa
            The xarray DataArray
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates (or indices in case of integer index)
            with NaN values. This requires either a provided `freq` or the possibility to infer the frequency from the
            provided timestamps. See :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string or integer representing the frequency of the underlying index. This is useful in order
            to fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
            If a string, represents the frequency of the pandas DatetimeIndex (see `offset aliases
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.

        Returns
        -------
        TimeSeries
            A univariate or multivariate deterministic TimeSeries constructed from the inputs.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> from darts.timeseries import DIMS
        >>> from darts import TimeSeries
        >>> from darts.utils.utils import generate_index
        >>>
        >>> # create values with the required dimensions (time, component, sample)
        >>> vals = np.random.random((3, 1, 1))
        >>> # create time index with daily frequency
        >>> times = generate_index("2020-01-01", length=3, freq="D")
        >>> columns = ["vals"]
        >>>
        >>> # create xarray with the required dimensions and coordinates
        >>> xa = xr.DataArray(
        >>>     vals,
        >>>     dims=DIMS,
        >>>     coords={DIMS[0]: times, DIMS[1]: columns}
        >>> )
        >>> series = TimeSeries.from_xarray(xa)
        >>> series.shape
        (3, 1, 1)
        """
        xa_index = xa.get_index(xa.dims[0])

        has_datetime_index = isinstance(xa_index, pd.DatetimeIndex)
        has_range_index = isinstance(xa_index, pd.RangeIndex)
        has_integer_index = not (has_datetime_index or has_range_index)

        has_frequency = (
            has_datetime_index and xa_index.freq is not None
        ) or has_range_index

        # optionally fill missing dates; do it only when there is a DatetimeIndex (and not a RangeIndex)
        if fill_missing_dates:
            xa_ = cls._fill_missing_dates(xa, freq=freq)
        # provided index does not have a freq; using the provided freq
        elif (
            (has_datetime_index or has_integer_index)
            and freq is not None
            and not has_frequency
        ):
            xa_ = cls._restore_xarray_from_frequency(xa, freq=freq)
        # index is an integer index and no freq is provided; try convert it to pd.RangeIndex
        elif has_integer_index and freq is None:
            xa_ = cls._integer_to_range_indexed_xarray(xa)
        else:
            xa_ = xa
        if fillna_value is not None:
            xa_ = xa_.fillna(fillna_value)

        # clean components (columns) names if needed (if names are not unique, or not strings)
        components = xa_.get_index(DIMS[1])
        if len(set(components)) != len(components) or any([
            not isinstance(s, str) for s in components
        ]):

            def _clean_component_list(columns) -> list[str]:
                # return a list of string containing column names
                # make each column name unique in case some columns have the same names
                clist = columns.to_list()

                # convert everything to string if needed
                for i, column in enumerate(clist):
                    if not isinstance(column, str):
                        clist[i] = str(column)

                has_duplicate = len(set(clist)) != len(clist)
                while has_duplicate:
                    # we may have to loop several times (e.g. we could have columns ["0", "0_1", "0"] and not
                    # noticing when renaming the last "0" into "0_1" that "0_1" already exists...)
                    name_to_occurence = defaultdict(int)
                    for i, column in enumerate(clist):
                        name_to_occurence[clist[i]] += 1

                        if name_to_occurence[clist[i]] > 1:
                            clist[i] = clist[i] + f"_{name_to_occurence[clist[i]] - 1}"

                    has_duplicate = len(set(clist)) != len(clist)

                return clist

            time_index_name = xa_.dims[0]
            columns_list = _clean_component_list(components)

            # Note: an option here could be to also rename the component names in the static covariates
            # and/or hierarchy, if any. However, we decide not to do so as those are directly dependent on the
            # component names to work properly, so in case there's any name conflict it's better solved
            # by the user than handled by silent renaming, which can change the way things work.

            # TODO: is there a way to just update the component index without re-creating a new DataArray?
            # -> Answer: Yes, but it's slower: e.g.:
            # ```
            # xa_ = xa_.assign_coords(
            #     {
            #         time_index_name: xa_.get_index(time_index_name),
            #         DIMS[1]: columns_list
            #     }
            # )
            # ```
            xa_ = xr.DataArray(
                xa_.values,
                dims=xa_.dims,
                coords={
                    time_index_name: xa_.get_index(time_index_name),
                    DIMS[1]: columns_list,
                },
                attrs=xa_.attrs,
            )

        # We cast the array to float
        if np.issubdtype(xa_.values.dtype, np.float32) or np.issubdtype(
            xa_.values.dtype, np.float64
        ):
            return cls(xa_)
        else:
            return cls(xa_.astype(np.float64))

    @classmethod
    def from_csv(
        cls,
        filepath_or_buffer,
        time_col: Optional[str] = None,
        value_cols: Optional[Union[list[str], str]] = None,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        fillna_value: Optional[float] = None,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hierarchy: Optional[dict] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> Self:
        """
        Build a deterministic TimeSeries instance built from a single CSV file.
        One column can be used to represent the time (if not present, the time index will be a RangeIndex)
        and a list of columns `value_cols` can be used to indicate the values for this time series.

        Parameters
        ----------
        filepath_or_buffer
            The path to the CSV file, or the file object; consistent with the argument of `pandas.read_csv` function
        time_col
            The time column name. If set, the column will be cast to a pandas DatetimeIndex (if it contains
            timestamps) or a RangeIndex (if it contains integers).
            If not set, the pandas RangeIndex will be used.
        value_cols
            A string or list of strings representing the value column(s) to be extracted from the CSV file. If set to
            `None`, all columns from the CSV file will be used (except for the time_col, if specified)
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates (or indices in case of integer index)
            with NaN values. This requires either a provided `freq` or the possibility to infer the frequency from the
            provided timestamps. See :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string or integer representing the frequency of the underlying index. This is useful in order
            to fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
            If a string, represents the frequency of the pandas DatetimeIndex (see `offset aliases
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.
        static_covariates
            Optionally, a set of static covariates to be added to the TimeSeries. Either a pandas Series or a pandas
            DataFrame. If a Series, the index represents the static variables. The covariates are globally 'applied'
            to all components of the TimeSeries. If a DataFrame, the columns represent the static variables and the
            rows represent the components of the uni/multivariate TimeSeries. If a single-row DataFrame, the covariates
            are globally 'applied' to all components of the TimeSeries. If a multi-row DataFrame, the number of
            rows must match the number of components of the TimeSeries (in this case, the number of columns in the CSV
            file). This adds control for component-specific static covariates.
        hierarchy
            Optionally, a dictionary describing the grouping(s) of the time series. The keys are component names, and
            for a given component name `c`, the value is a list of component names that `c` "belongs" to. For instance,
            if there is a `total` component, split both in two divisions `d1` and `d2` and in two regions `r1` and `r2`,
            and four products `d1r1` (in division `d1` and region `r1`), `d2r1`, `d1r2` and `d2r2`, the hierarchy would
            be encoded as follows.

            .. highlight:: python
            .. code-block:: python

                hierarchy={
                    "d1r1": ["d1", "r1"],
                    "d1r2": ["d1", "r2"],
                    "d2r1": ["d2", "r1"],
                    "d2r2": ["d2", "r2"],
                    "d1": ["total"],
                    "d2": ["total"],
                    "r1": ["total"],
                    "r2": ["total"]
                }
            ..
            The hierarchy can be used to reconcile forecasts (so that the sums of the forecasts at
            different levels are consistent), see `hierarchical reconciliation
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`_.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.

        **kwargs
            Optional arguments to be passed to `pandas.read_csv` function

        Returns
        -------
        TimeSeries
            A univariate or multivariate deterministic TimeSeries constructed from the inputs.

        Examples
        --------
        >>> from darts import TimeSeries
        >>> TimeSeries.from_csv("data.csv", time_col="time")
        """

        df = pd.read_csv(filepath_or_buffer=filepath_or_buffer, **kwargs)
        return cls.from_dataframe(
            df=df,
            time_col=time_col,
            value_cols=value_cols,
            fill_missing_dates=fill_missing_dates,
            freq=freq,
            fillna_value=fillna_value,
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: IntoDataFrame,
        time_col: Optional[str] = None,
        value_cols: Optional[Union[list[str], str]] = None,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        fillna_value: Optional[float] = None,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hierarchy: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> Self:
        """
        Build a deterministic TimeSeries instance built from a selection of columns of a DataFrame.
        One column (or the DataFrame index) has to represent the time,
        and a list of columns `value_cols` has to represent the values for this time series.

        Parameters
        ----------
        df
            The DataFrame, or anything which can be converted to a narwhals DataFrame (e.g. pandas.DataFrame,
            polars.DataFrame, ...). See the `narwhals documentation
            <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_native>`_ for more
            information.
        time_col
            The time column name. If set, the column will be cast to a pandas DatetimeIndex (if it contains
            timestamps) or a RangeIndex (if it contains integers).
            If not set, the DataFrame index will be used. In this case the DataFrame must contain an index that is
            either a pandas DatetimeIndex, a pandas RangeIndex, or a pandas Index that can be converted to a
            RangeIndex. It is better if the index has no holes; alternatively setting `fill_missing_dates` can in some
            cases solve these issues (filling holes with NaN, or with the provided `fillna_value` numeric value, if
            any).
        value_cols
            A string or list of strings representing the value column(s) to be extracted from the DataFrame. If set to
            `None`, the whole DataFrame will be used.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates (or indices in case of integer index)
            with NaN values. This requires either a provided `freq` or the possibility to infer the frequency from the
            provided timestamps. See :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string or integer representing the frequency of the underlying index. This is useful in order
            to fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
            If a string, represents the frequency of the pandas DatetimeIndex (see `offset aliases
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.
        static_covariates
            Optionally, a set of static covariates to be added to the TimeSeries. Either a pandas Series or a pandas
            DataFrame. If a Series, the index represents the static variables. The covariates are globally 'applied'
            to all components of the TimeSeries. If a DataFrame, the columns represent the static variables and the
            rows represent the components of the uni/multivariate TimeSeries. If a single-row DataFrame, the covariates
            are globally 'applied' to all components of the TimeSeries. If a multi-row DataFrame, the number of
            rows must match the number of components of the TimeSeries (in this case, the number of columns in
            ``value_cols``). This adds control for component-specific static covariates.
        hierarchy
            Optionally, a dictionary describing the grouping(s) of the time series. The keys are component names, and
            for a given component name `c`, the value is a list of component names that `c` "belongs" to. For instance,
            if there is a `total` component, split both in two divisions `d1` and `d2` and in two regions `r1` and `r2`,
            and four products `d1r1` (in division `d1` and region `r1`), `d2r1`, `d1r2` and `d2r2`, the hierarchy would
            be encoded as follows.

            .. highlight:: python
            .. code-block:: python

                hierarchy={
                    "d1r1": ["d1", "r1"],
                    "d1r2": ["d1", "r2"],
                    "d2r1": ["d2", "r1"],
                    "d2r2": ["d2", "r2"],
                    "d1": ["total"],
                    "d2": ["total"],
                    "r1": ["total"],
                    "r2": ["total"]
                }
            ..
            The hierarchy can be used to reconcile forecasts (so that the sums of the forecasts at
            different levels are consistent), see `hierarchical reconciliation
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`_.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.

        Returns
        -------
        TimeSeries
            A univariate or multivariate deterministic TimeSeries constructed from the inputs.

        Examples
        --------
        >>> import pandas as pd
        >>> from darts import TimeSeries
        >>> from darts.utils.utils import generate_index
        >>> # create values and times with daily frequency
        >>> data = {"vals": range(3), "time": generate_index("2020-01-01", length=3, freq="D")}
        >>> # create from `pandas.DataFrame`
        >>> df = pd.DataFrame(data)
        >>> series = TimeSeries.from_dataframe(df, time_col="time")
        >>> # shape (n time steps, n components, n samples)
        >>> series.shape
        (3, 1, 1)

        >>> # or from `polars.DataFrame` (make sure Polars is installed)
        >>> import polars as pl
        >>> df = pl.DataFrame(data)
        >>> series = TimeSeries.from_dataframe(df, time_col="time")
        >>> series.shape
        (3, 1, 1)
        """
        df = nw.from_native(df, eager_only=True, pass_through=False)
        time_zone = None

        # get values
        if value_cols is None:
            series_df = df.drop(time_col) if time_col else df
        else:
            if isinstance(value_cols, (str, int)):
                value_cols = [value_cols]
            series_df = df[value_cols]

        # get time index
        if time_col:
            if time_col not in df.columns:
                raise_log(AttributeError(f"time_col='{time_col}' is not present."))

            time_col_vals = df.get_column(time_col)

            if time_col_vals.dtype == nw.String:
                # Try to convert to integers if needed
                with contextlib.suppress(Exception):
                    time_col_vals = time_col_vals.cast(nw.Int64)

            if time_col_vals.dtype.is_integer():
                if time_col_vals.is_duplicated().any():
                    raise_log(
                        ValueError(
                            "The provided integer time index column contains duplicate values."
                        )
                    )
                # Temporarily use an integer Index to sort the values, and replace by a
                # RangeIndex in `TimeSeries.from_xarray()`
                time_index = pd.Index(time_col_vals)

            elif isinstance(time_col_vals.dtype, nw.String):
                # The integer conversion failed; try datetimes
                try:
                    time_index = pd.DatetimeIndex(time_col_vals)
                except ValueError:
                    raise_log(
                        AttributeError(
                            "'time_col' is of 'String' dtype but doesn't contain valid timestamps"
                        )
                    )
            elif isinstance(time_col_vals.dtype, nw.Datetime):
                # remember time zone here as polars converts to UTC
                time_zone = time_col_vals.dtype.time_zone
                if time_zone is not None:
                    time_col_vals = time_col_vals.dt.replace_time_zone(None)
                time_index = pd.DatetimeIndex(time_col_vals)
            else:
                raise_log(
                    AttributeError(
                        "Invalid type of `time_col`: it needs to be of either 'String', 'Datetime' or 'Int' dtype."
                    )
                )
            time_index.name = time_col
        else:
            time_index = nw.maybe_get_index(df)
            if time_index is None:
                time_index = pd.RangeIndex(len(df))
                logger.warning(
                    "No time column specified (`time_col=None`) and no index found in the `DataFrame`. Defaulting to "
                    "`pandas.RangeIndex(len(df))`. If this is not desired consider adding a time column "
                    "to your `DataFrame` and defining `time_col`."
                )
            # if we are here, the dataframe was pandas
            elif not (
                isinstance(time_index, VALID_INDEX_TYPES)
                or np.issubdtype(time_index.dtype, np.integer)
            ):
                raise_log(
                    ValueError(
                        "If time_col is not specified, the DataFrame must be indexed either with "
                        "a DatetimeIndex, a RangeIndex, or an integer Index that can be converted into a RangeIndex"
                    ),
                    logger,
                )
            if isinstance(time_index, pd.DatetimeIndex):
                time_zone = time_index.tz
                if time_zone is not None:
                    # remove and remember time zone here as pandas converts to UTC
                    time_index = time_index.tz_localize(None)

        # BUGFIX : force time-index to be timezone naive as xarray doesn't support it
        if time_zone is not None:
            logger.warning(
                "The provided DatetimeIndex was associated with a timezone, which is currently not supported "
                "by xarray. To avoid unexpected behaviour, the tz information was removed. Consider calling "
                f"`ts.time_index.tz_localize({time_zone})` when exporting the results."
                "To plot the series with the right time steps, consider setting the matplotlib.pyplot "
                "`rcParams['timezone']` parameter to automatically convert the time axis back to the "
                "original timezone."
            )

        if not time_index.name:
            time_index.name = time_col if time_col else DIMS[0]

        xa = xr.DataArray(
            series_df.to_numpy()[:, :, np.newaxis],
            dims=(time_index.name,) + DIMS[-2:],
            coords={time_index.name: time_index, DIMS[1]: series_df.columns},
            attrs={
                STATIC_COV_TAG: static_covariates,
                HIERARCHY_TAG: hierarchy,
                METADATA_TAG: metadata,
            },
        )

        return cls.from_xarray(
            xa=xa,
            fill_missing_dates=fill_missing_dates,
            freq=freq,
            fillna_value=fillna_value,
        )

    @classmethod
    def from_group_dataframe(
        cls,
        df: pd.DataFrame,
        group_cols: Union[list[str], str],
        time_col: Optional[str] = None,
        value_cols: Optional[Union[list[str], str]] = None,
        static_cols: Optional[Union[list[str], str]] = None,
        metadata_cols: Optional[Union[list[str], str]] = None,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        fillna_value: Optional[float] = None,
        drop_group_cols: Optional[Union[list[str], str]] = None,
        n_jobs: Optional[int] = 1,
        verbose: Optional[bool] = False,
    ) -> list[Self]:
        """
        Build a list of TimeSeries instances grouped by a selection of columns from a DataFrame.
        One column (or the DataFrame index) has to represent the time,
        a list of columns `group_cols` must be used for extracting the individual TimeSeries by groups,
        and a list of columns `value_cols` has to represent the values for the individual time series.
        Values from columns ``group_cols`` and ``static_cols`` are added as static covariates to the resulting
        TimeSeries objects. These can be viewed with `my_series.static_covariates`. Different to `group_cols`,
        `static_cols` only adds the static values but are not used to extract the TimeSeries groups.

        Parameters
        ----------
        df
            The DataFrame
        group_cols
            A string or list of strings representing the columns from the DataFrame by which to extract the
            individual TimeSeries groups.
        time_col
            The time column name. If set, the column will be cast to a pandas DatetimeIndex (if it contains
            timestamps) or a RangeIndex (if it contains integers).
            If not set, the DataFrame index will be used. In this case the DataFrame must contain an index that is
            either a pandas DatetimeIndex, a pandas RangeIndex, or a pandas Index that can be converted to a
            RangeIndex. Be aware that the index must represents the actual index of each individual time series group
            (can contain non-unique values). It is better if the index has no holes; alternatively setting
            `fill_missing_dates` can in some cases solve these issues (filling holes with NaN, or with the provided
            `fillna_value` numeric value, if any).
        value_cols
            A string or list of strings representing the value column(s) to be extracted from the DataFrame. If set to
            `None`, the whole DataFrame will be used.
        static_cols
            A string or list of strings representing static variable columns from the DataFrame that should be
            appended as static covariates to the resulting TimeSeries groups. Different to `group_cols`, the
            DataFrame is not grouped by these columns. Uses the first encountered value per group and column
            (assumes that there is only one unique value). Static covariates can be used as input features to all
            Darts models that support it.
        metadata_cols
            Same as `static_cols` but appended as metadata to the resulting TimeSeries groups. Metadata will never be
            used by the underlying Darts models.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates (or indices in case of integer index)
            with NaN values. This requires either a provided `freq` or the possibility to infer the frequency from the
            provided timestamps. See :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string or integer representing the frequency of the underlying index. This is useful in order
            to fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
            If a string, represents the frequency of the pandas DatetimeIndex (see `offset aliases
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.
        drop_group_cols
            Optionally, a string or list of strings with `group_cols` column(s) to exclude from the static covariates.
        n_jobs
            Optionally, an integer representing the number of parallel jobs to run. Behavior is the same as in the
            `joblib.Parallel` class.
        verbose
            Optionally, a boolean value indicating whether to display a progress bar.

        Returns
        -------
        List[TimeSeries]
            A list containing a univariate or multivariate deterministic TimeSeries per group in the DataFrame.

        Examples
        --------
        >>> import pandas as pd
        >>> from darts import TimeSeries
        >>> from darts.utils.utils import generate_index
        >>>
        >>> # create a DataFrame with two series that have different ids,
        >>> # values, and frequencies
        >>> df_1 = pd.DataFrame({
        >>>     "ID": [0] * 3,
        >>>     "vals": range(3),
        >>>     "time": generate_index("2020-01-01", length=3, freq="D")}
        >>> )
        >>> df_2 = pd.DataFrame({
        >>>     "ID": [1] * 6,
        >>>     "vals": range(6),
        >>>     "time": generate_index("2020-01-01", length=6, freq="h")}
        >>> )
        >>> df = pd.concat([df_1, df_2], axis=0)
        >>>
        >>> # extract the series by "ID" groups from the DataFrame
        >>> series_multi = TimeSeries.from_group_dataframe(
        >>>     df,
        >>>     group_cols="ID",
        >>>     time_col="time"
        >>> )
        >>> len(series_multi), series_multi[0].shape, series_multi[1].shape
        (2, (3, 1, 1), (6, 1, 1))
        """
        if time_col is None and df.index.is_monotonic_increasing:
            logger.warning(
                "UserWarning: `time_col` was not set and `df` has a monotonically increasing (time) index. This "
                "results in time series groups with non-overlapping (time) index. You can ignore this warning if the "
                "index represents the actual index of each individual time series group."
            )

        # group cols: used to extract time series groups from `df`, will also be added as static covariates
        # (except `drop_group_cols`)
        group_cols = [group_cols] if not isinstance(group_cols, list) else group_cols
        if drop_group_cols:
            drop_group_cols = (
                [drop_group_cols]
                if not isinstance(drop_group_cols, list)
                else drop_group_cols
            )
            invalid_cols = set(drop_group_cols) - set(group_cols)
            if invalid_cols:
                raise_log(
                    ValueError(
                        f"Found invalid `drop_group_cols` columns. All columns must be in the passed `group_cols`. "
                        f"Expected any of: {group_cols}, received: {invalid_cols}."
                    ),
                    logger=logger,
                )
            drop_group_col_idx = [
                idx for idx, col in enumerate(group_cols) if col in drop_group_cols
            ]
        else:
            drop_group_cols = []
            drop_group_col_idx = []

        # static covariates: all `group_cols` (except `drop_group_cols`) and `static_cols`
        if static_cols is not None:
            static_cols = (
                [static_cols] if not isinstance(static_cols, list) else static_cols
            )
        else:
            static_cols = []
        static_cov_cols = group_cols + static_cols
        # columns that are used as static covariates but not for grouping
        extract_static_cov_cols = [
            col for col in static_cov_cols if col not in drop_group_cols
        ]

        # metadata: all `metadata_cols`
        if metadata_cols is not None:
            metadata_cols = (
                [metadata_cols]
                if not isinstance(metadata_cols, list)
                else metadata_cols
            )
        else:
            metadata_cols = []
        # columns that are used as metadata but not for grouping or static covariates
        extract_metadata_cols = [
            col for col in metadata_cols if col not in static_cov_cols
        ]

        extract_time_col = [] if time_col is None else [time_col]

        if value_cols is None:
            value_cols = df.columns.drop(
                static_cov_cols + extract_metadata_cols + extract_time_col
            ).tolist()
        extract_value_cols = [value_cols] if isinstance(value_cols, str) else value_cols

        df = df[
            static_cov_cols
            + extract_value_cols
            + extract_time_col
            + extract_metadata_cols
        ]

        if time_col:
            if np.issubdtype(df[time_col].dtype, object) or np.issubdtype(
                df[time_col].dtype, np.datetime64
            ):
                df.index = pd.DatetimeIndex(df[time_col])
                df = df.drop(columns=time_col)
            else:
                df = df.set_index(time_col)

        if df.index.is_monotonic_increasing:
            logger.warning(
                "UserWarning: The (time) index from `df` is monotonically increasing. This "
                "results in time series groups with non-overlapping (time) index. You can ignore this warning if the "
                "index represents the actual index of each individual time series group."
            )

        # sort on entire `df` to avoid having to sort individually later on
        else:
            df = df.sort_index()

        groups = df.groupby(group_cols[0] if len(group_cols) == 1 else group_cols)

        # build progress bar for iterator
        iterator = _build_tqdm_iterator(
            groups,
            verbose=verbose,
            total=len(groups),
            desc="Creating TimeSeries",
        )

        def from_group(static_cov_vals, group):
            split = group[extract_value_cols]

            static_cov_vals = (
                (static_cov_vals,)
                if not isinstance(static_cov_vals, tuple)
                else static_cov_vals
            )
            # optionally, exclude group columns from static covariates
            if drop_group_col_idx:
                if len(drop_group_col_idx) == len(group_cols):
                    static_cov_vals = tuple()
                else:
                    static_cov_vals = tuple(
                        val
                        for idx, val in enumerate(static_cov_vals)
                        if idx not in drop_group_col_idx
                    )

            if static_cols:
                # use first value as static covariate (assume only one unique per group)
                static_cov_vals += tuple(group[static_cols].values[0])

            metadata = None
            if metadata_cols:
                # use first value as metadata (assume only one unique per group)
                metadata = {
                    col: val
                    for col, val in zip(metadata_cols, group[metadata_cols].values[0])
                }

            return cls.from_dataframe(
                df=split,
                fill_missing_dates=fill_missing_dates,
                freq=freq,
                fillna_value=fillna_value,
                static_covariates=(
                    pd.DataFrame([static_cov_vals], columns=extract_static_cov_cols)
                    if extract_static_cov_cols
                    else None
                ),
                metadata=metadata,
            )

        return _parallel_apply(
            iterator,
            from_group,
            n_jobs,
            fn_args=dict(),
            fn_kwargs=dict(),
        )

    @classmethod
    def from_series(
        cls,
        pd_series: IntoSeries,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        fillna_value: Optional[float] = None,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        metadata: Optional[dict] = None,
    ) -> Self:
        """
        Build a univariate deterministic TimeSeries from a Series.

        The series must contain an index that is either a pandas DatetimeIndex, a pandas RangeIndex, or a pandas Index
        that can be converted into a RangeIndex. It is better if the index has no holes; alternatively setting
        `fill_missing_dates` can in some cases solve these issues (filling holes with NaN, or with the provided
        `fillna_value` numeric value, if any).

        Parameters
        ----------
        pd_series
            The Series, or anything which can be converted to a narwhals Series (e.g. pandas.Series, ...). See the
            `narwhals documentation
            <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_native>`_ for more
            information.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates (or indices in case of integer index)
            with NaN values. This requires either a provided `freq` or the possibility to infer the frequency from the
            provided timestamps. See :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string or integer representing the frequency of the underlying index. This is useful in order
            to fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
            If a string, represents the frequency of the pandas DatetimeIndex (see `offset aliases
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.
        static_covariates
            Optionally, a set of static covariates to be added to the TimeSeries. Either a pandas Series or a
            single-row pandas DataFrame. If a Series, the index represents the static variables. If a DataFrame, the
            columns represent the static variables and the single row represents the univariate TimeSeries component.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.

        Returns
        -------
        TimeSeries
            A univariate and deterministic TimeSeries constructed from the inputs.

        Examples
        --------
        >>> import pandas as pd
        >>> from darts import TimeSeries
        >>> from darts.utils.utils import generate_index
        >>> # create values and times with daily frequency
        >>> vals, times = range(3), generate_index("2020-01-01", length=3, freq="D")
        >>>
        >>> # create from `pandas.Series`
        >>> pd_series = pd.Series(vals, index=times)
        >>> series = TimeSeries.from_series(pd_series)
        >>> series.shape
        (3, 1, 1)
        """
        nw_series = nw.from_native(pd_series, series_only=True, pass_through=False)
        df = nw_series.to_frame()
        return cls.from_dataframe(
            df,
            time_col=None,
            value_cols=None,
            fill_missing_dates=fill_missing_dates,
            freq=freq,
            fillna_value=fillna_value,
            static_covariates=static_covariates,
            metadata=metadata,
        )

    @classmethod
    def from_times_and_values(
        cls,
        times: Union[pd.DatetimeIndex, pd.RangeIndex, pd.Index],
        values: np.ndarray,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        columns: Optional[pd._typing.Axes] = None,
        fillna_value: Optional[float] = None,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hierarchy: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> Self:
        """
        Build a series from a time index and value array.

        Parameters
        ----------
        times
            A pandas DateTimeIndex, RangeIndex, or Index that can be converted to a RangeIndex representing the time
            axis for the time series. It is better if the index has no holes; alternatively setting
            `fill_missing_dates` can in some cases solve these issues (filling holes with NaN, or with the provided
            `fillna_value` numeric value, if any).
        values
            A Numpy array of values for the TimeSeries. Both 2-dimensional arrays, for deterministic series,
            and 3-dimensional arrays, for probabilistic series, are accepted. In the former case the dimensions
            should be (time, component), and in the latter case (time, component, sample).
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates (or indices in case of integer index)
            with NaN values. This requires either a provided `freq` or the possibility to infer the frequency from the
            provided timestamps. See :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string or integer representing the frequency of the underlying index. This is useful in order
            to fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
            If a string, represents the frequency of the pandas DatetimeIndex (see `offset aliases
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        columns
            Columns to be used by the underlying pandas DataFrame.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.
        static_covariates
            Optionally, a set of static covariates to be added to the TimeSeries. Either a pandas Series or a pandas
            DataFrame. If a Series, the index represents the static variables. The covariates are globally 'applied'
            to all components of the TimeSeries. If a DataFrame, the columns represent the static variables and the
            rows represent the components of the uni/multivariate TimeSeries. If a single-row DataFrame, the covariates
            are globally 'applied' to all components of the TimeSeries. If a multi-row DataFrame, the number of
            rows must match the number of components of the TimeSeries (in this case, the number of columns in
            ``values``). This adds control for component-specific static covariates.
        hierarchy
            Optionally, a dictionary describing the grouping(s) of the time series. The keys are component names, and
            for a given component name `c`, the value is a list of component names that `c` "belongs" to. For instance,
            if there is a `total` component, split both in two divisions `d1` and `d2` and in two regions `r1` and `r2`,
            and four products `d1r1` (in division `d1` and region `r1`), `d2r1`, `d1r2` and `d2r2`, the hierarchy would
            be encoded as follows.

            .. highlight:: python
            .. code-block:: python

                hierarchy={
                    "d1r1": ["d1", "r1"],
                    "d1r2": ["d1", "r2"],
                    "d2r1": ["d2", "r1"],
                    "d2r2": ["d2", "r2"],
                    "d1": ["total"],
                    "d2": ["total"],
                    "r1": ["total"],
                    "r2": ["total"]
                }
            ..
            The hierarchy can be used to reconcile forecasts (so that the sums of the forecasts at
            different levels are consistent), see `hierarchical reconciliation
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`_.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.

        Returns
        -------
        TimeSeries
            A TimeSeries constructed from the inputs.

        Examples
        --------
        >>> import numpy as np
        >>> from darts import TimeSeries
        >>> from darts.utils.utils import generate_index
        >>> # create values and times with daily frequency
        >>> vals, times = np.arange(3), generate_index("2020-01-01", length=3, freq="D")
        >>> series = TimeSeries.from_times_and_values(times=times, values=vals)
        >>> series.shape
        (3, 1, 1)
        """
        raise_if_not(
            isinstance(times, VALID_INDEX_TYPES)
            or np.issubdtype(times.dtype, np.integer),
            "the `times` argument must be a RangeIndex, or a DateTimeIndex. Use "
            "TimeSeries.from_values() if you want to use an automatic RangeIndex.",
        )

        # BUGFIX : force time-index to be timezone naive as xarray doesn't support it
        if isinstance(times, pd.DatetimeIndex) and times.tz is not None:
            logger.warning(
                "The `times` argument was associated with a timezone, which is currently not supported "
                "by xarray. To avoid unexpected behaviour, the tz information was removed. Consider calling "
                f"`ts.time_index.tz_localize({times.tz})` when exporting the results."
                "To plot the series with the right time steps, consider setting the matplotlib.pyplot "
                "`rcParams['timezone']` parameter to automatically convert the time axis back to the "
                "original timezone."
            )
            times = times.tz_localize(None)

        times_name = DIMS[0] if not times.name else times.name

        # avoid copying if data is already np.ndarray:
        values = np.array(values) if not isinstance(values, np.ndarray) else values
        values = expand_arr(values, ndim=len(DIMS))
        coords = {times_name: times}
        if columns is not None:
            coords[DIMS[1]] = columns

        xa = xr.DataArray(
            values,
            dims=(times_name,) + DIMS[-2:],
            coords=coords,
            attrs={
                STATIC_COV_TAG: static_covariates,
                HIERARCHY_TAG: hierarchy,
                METADATA_TAG: metadata,
            },
        )
        return cls.from_xarray(
            xa=xa,
            fill_missing_dates=fill_missing_dates,
            freq=freq,
            fillna_value=fillna_value,
        )

    @classmethod
    def from_values(
        cls,
        values: np.ndarray,
        columns: Optional[pd._typing.Axes] = None,
        fillna_value: Optional[float] = None,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hierarchy: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> Self:
        """
        Build an integer-indexed series from an array of values.
        The series will have an integer index (RangeIndex).

        Parameters
        ----------
        values
            A Numpy array of values for the TimeSeries. Both 2-dimensional arrays, for deterministic series,
            and 3-dimensional arrays, for probabilistic series, are accepted. In the former case the dimensions
            should be (time, component), and in the latter case (time, component, sample).
        columns
            Columns to be used by the underlying pandas DataFrame.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.
        static_covariates
            Optionally, a set of static covariates to be added to the TimeSeries. Either a pandas Series or a pandas
            DataFrame. If a Series, the index represents the static variables. The covariates are globally 'applied'
            to all components of the TimeSeries. If a DataFrame, the columns represent the static variables and the
            rows represent the components of the uni/multivariate TimeSeries. If a single-row DataFrame, the covariates
            are globally 'applied' to all components of the TimeSeries. If a multi-row DataFrame, the number of
            rows must match the number of components of the TimeSeries (in this case, the number of columns in
            ``values``). This adds control for component-specific static covariates.
        hierarchy
            Optionally, a dictionary describing the grouping(s) of the time series. The keys are component names, and
            for a given component name `c`, the value is a list of component names that `c` "belongs" to. For instance,
            if there is a `total` component, split both in two divisions `d1` and `d2` and in two regions `r1` and `r2`,
            and four products `d1r1` (in division `d1` and region `r1`), `d2r1`, `d1r2` and `d2r2`, the hierarchy would
            be encoded as follows.

            .. highlight:: python
            .. code-block:: python

                hierarchy={
                    "d1r1": ["d1", "r1"],
                    "d1r2": ["d1", "r2"],
                    "d2r1": ["d2", "r1"],
                    "d2r2": ["d2", "r2"],
                    "d1": ["total"],
                    "d2": ["total"],
                    "r1": ["total"],
                    "r2": ["total"]
                }
            ..
            The hierarchy can be used to reconcile forecasts (so that the sums of the forecasts at
            different levels are consistent), see `hierarchical reconciliation
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`_.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.

        Returns
        -------
        TimeSeries
            A TimeSeries constructed from the inputs.

        Examples
        --------
        >>> import numpy as np
        >>> from darts import TimeSeries
        >>> from darts.utils.utils import generate_index
        >>> vals = np.arange(3)
        >>> series = TimeSeries.from_times_and_values(times=times, values=vals)
        >>> series.shape
        (3, 1, 1)
        """
        time_index = pd.RangeIndex(0, len(values), 1)
        values_ = (
            np.reshape(values, (len(values), 1)) if len(values.shape) == 1 else values
        )

        return cls.from_times_and_values(
            times=time_index,
            values=values_,
            fill_missing_dates=False,
            freq=None,
            columns=columns,
            fillna_value=fillna_value,
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
        )

    @classmethod
    def from_json(
        cls,
        json_str: str,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hierarchy: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> Self:
        """
        Build a series from the JSON String representation of a ``TimeSeries``
        (produced using :func:`TimeSeries.to_json()`).

        At the moment this only supports deterministic time series (i.e., made of 1 sample).

        Parameters
        ----------
        json_str
            The JSON String to convert
        static_covariates
            Optionally, a set of static covariates to be added to the TimeSeries. Either a pandas Series or a pandas
            DataFrame. If a Series, the index represents the static variables. The covariates are globally 'applied'
            to all components of the TimeSeries. If a DataFrame, the columns represent the static variables and the
            rows represent the components of the uni/multivariate TimeSeries. If a single-row DataFrame, the covariates
            are globally 'applied' to all components of the TimeSeries. If a multi-row DataFrame, the number of
            rows must match the number of components of the TimeSeries (in this case, the number of columns in
            ``value_cols``). This adds control for component-specific static covariates.
        hierarchy
            Optionally, a dictionary describing the grouping(s) of the time series. The keys are component names, and
            for a given component name `c`, the value is a list of component names that `c` "belongs" to. For instance,
            if there is a `total` component, split both in two divisions `d1` and `d2` and in two regions `r1` and `r2`,
            and four products `d1r1` (in division `d1` and region `r1`), `d2r1`, `d1r2` and `d2r2`, the hierarchy would
            be encoded as follows.

            .. highlight:: python
            .. code-block:: python

                hierarchy={
                    "d1r1": ["d1", "r1"],
                    "d1r2": ["d1", "r2"],
                    "d2r1": ["d2", "r1"],
                    "d2r2": ["d2", "r2"],
                    "d1": ["total"],
                    "d2": ["total"],
                    "r1": ["total"],
                    "r2": ["total"]
                }
            ..
            The hierarchy can be used to reconcile forecasts (so that the sums of the forecasts at
            different levels are consistent), see `hierarchical reconciliation
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`_.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.

        Returns
        -------
        TimeSeries
            The time series object converted from the JSON String

        Examples
        --------
        >>> from darts import TimeSeries
        >>> json_str = (
        >>>     '{"columns":["vals"],"index":["2020-01-01","2020-01-02","2020-01-03"],"data":[[0.0],[1.0],[2.0]]}'
        >>> )
        >>> series = TimeSeries.from_json("data.csv")
        >>> series.shape
        (3, 1, 1)
        """
        df = pd.read_json(StringIO(json_str), orient="split")
        return cls.from_dataframe(
            df,
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
        )

    @classmethod
    def from_pickle(cls, path: str) -> Self:
        """
        Read a pickled ``TimeSeries``.

        Parameters
        ----------
        path : string
            path pointing to a pickle file that will be loaded

        Returns
        -------
        TimeSeries
            timeseries object loaded from file

        Notes
        -----
        Xarray docs [1]_ suggest not using pickle as a long-term data storage.

        References
        ----------
        .. [1] http://xarray.pydata.org/en/stable/user-guide/io.html#pickle
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)

    """
    Properties
    ==========
    """

    @property
    def static_covariates(self) -> Optional[pd.DataFrame]:
        """
        Returns the static covariates contained in the series as a pandas DataFrame.
        The columns represent the static variables and the rows represent the components of the uni/multivariate
        series.
        """
        return self._xa.attrs.get(STATIC_COV_TAG, None)

    @property
    def hierarchy(self) -> Optional[dict]:
        """
        The hierarchy of this TimeSeries, if any.
        If set, the hierarchy is encoded as a dictionary, whose keys are individual components
        and values are the set of parent(s) of these components in the hierarchy.
        """
        return self._xa.attrs.get(HIERARCHY_TAG, None)

    @property
    def metadata(self) -> Optional[dict]:
        """
        The metadata of this TimeSeries, if any.
        """
        return self._xa.attrs.get(METADATA_TAG, None)

    @property
    def has_hierarchy(self) -> bool:
        """Whether this series is hierarchical or not."""
        return self.hierarchy is not None

    @property
    def top_level_component(self) -> Optional[str]:
        """
        The top level component name of this series, or None if the series has no hierarchy.
        """
        return self._top_level_component

    @property
    def bottom_level_components(self) -> Optional[list[str]]:
        """
        The bottom level component names of this series, or None if the series has no hierarchy.
        """
        return self._bottom_level_components

    @property
    def top_level_series(self) -> Optional[Self]:
        """
        The univariate series containing the single top-level component of this series,
        or None if the series has no hierarchy.
        """
        return self[self.top_level_component] if self.has_hierarchy else None

    @property
    def bottom_level_series(self) -> Optional[list[Self]]:
        """
        The series containing the bottom-level components of this series in the same
        order as they appear in the series, or None if the series has no hierarchy.

        The returned series is multivariate if there are multiple bottom components.
        """
        return (
            self[[c for c in self.components if c in self.bottom_level_components]]
            if self.has_hierarchy
            else None
        )

    @property
    def shape(self) -> tuple[int]:
        """The shape of the series (n_timesteps, n_components, n_samples)."""
        return self._xa.shape

    @property
    def n_samples(self) -> int:
        """Number of samples contained in the series."""
        return self.shape[AXES["sample"]]

    @property
    def n_components(self) -> int:
        """Number of components (dimensions) contained in the series."""
        return self.shape[AXES["component"]]

    @property
    def width(self) -> int:
        """ "Width" (= number of components) of the series."""
        return self.n_components

    @property
    def n_timesteps(self) -> int:
        """Number of time steps in the series."""
        return self.shape[AXES["time"]]

    @property
    def is_deterministic(self) -> bool:
        """Whether this series is deterministic."""
        return self.shape[AXES["sample"]] == 1

    @property
    def is_stochastic(self) -> bool:
        """Whether this series is stochastic."""
        return not self.is_deterministic

    @property
    def is_probabilistic(self) -> bool:
        """Whether this series is stochastic (= probabilistic)."""
        return self.is_stochastic

    @property
    def is_univariate(self) -> bool:
        """Whether this series is univariate."""
        return self.shape[AXES["component"]] == 1

    @property
    def freq(self) -> Union[pd.DateOffset, int]:
        """The frequency of the series.
        A `pd.DateOffset` if series is indexed with a `pd.DatetimeIndex`.
        An integer (step size) if series is indexed with a `pd.RangeIndex`.
        """
        return self._freq

    @property
    def freq_str(self) -> str:
        """The frequency string representation of the series."""
        return self._freq_str

    @property
    def dtype(self):
        """The dtype of the series' values."""
        return self._xa.values.dtype

    @property
    def components(self) -> pd.Index:
        """The names of the components, as a Pandas Index."""
        return self._xa.get_index(DIMS[1]).copy()

    @property
    def columns(self) -> pd.Index:
        """The names of the components, as a Pandas Index."""
        return self.components

    @property
    def time_index(self) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
        """The time index of this time series."""
        return self._time_index.copy()

    @property
    def time_dim(self) -> str:
        """The name of the time dimension for this time series."""
        return self._time_dim

    @property
    def has_datetime_index(self) -> bool:
        """Whether this series is indexed with a DatetimeIndex (otherwise it is indexed with an RangeIndex)."""
        return self._has_datetime_index

    @property
    def has_range_index(self) -> bool:
        """Whether this series is indexed with an RangeIndex (otherwise it is indexed with a DatetimeIndex)."""
        return not self._has_datetime_index

    @property
    def has_static_covariates(self) -> bool:
        """Whether this series contains static covariates."""
        return self.static_covariates is not None

    @property
    def has_metadata(self) -> bool:
        """Whether this series contains metadata."""
        return self.metadata is not None

    @property
    def duration(self) -> Union[pd.Timedelta, int]:
        """The duration of this time series (as a time delta or int)."""
        return self._time_index[-1] - self._time_index[0]

    """
    Some asserts
    =============
    """

    # TODO: put at the bottom

    def _assert_univariate(self):
        if not self.is_univariate:
            raise_log(
                AssertionError(
                    "Only univariate TimeSeries instances support this method"
                ),
                logger,
            )

    def _assert_deterministic(self):
        if not self.is_deterministic:
            raise_log(
                AssertionError(
                    "Only deterministic TimeSeries (with 1 sample) instances support this method"
                ),
                logger,
            )

    def _assert_stochastic(self):
        if not self.is_stochastic:
            raise_log(
                AssertionError(
                    "Only non-deterministic TimeSeries (with more than 1 samples) "
                    "instances support this method"
                ),
                logger,
            )

    def _raise_if_not_within(self, ts: Union[pd.Timestamp, int]):
        if isinstance(ts, pd.Timestamp):
            # Not that the converse doesn't apply (a time-indexed series can be called with an integer)
            raise_if_not(
                self._has_datetime_index,
                "Function called with a timestamp, but series not time-indexed.",
                logger,
            )
            is_inside = self.start_time() <= ts <= self.end_time()
        else:
            if self._has_datetime_index:
                is_inside = 0 <= ts <= len(self)
            else:
                is_inside = self.start_time() <= ts <= self.end_time()

        raise_if_not(
            is_inside,
            f"Timestamp must be between {self.start_time()} and {self.end_time()}",
            logger,
        )

    def _get_first_timestamp_after(self, ts: pd.Timestamp) -> Union[pd.Timestamp, int]:
        return next(filter(lambda t: t >= ts, self._time_index))

    def _get_last_timestamp_before(self, ts: pd.Timestamp) -> Union[pd.Timestamp, int]:
        return next(filter(lambda t: t <= ts, self._time_index[::-1]))

    """
    Export functions
    ================
    """

    def data_array(self, copy: bool = True) -> xr.DataArray:
        """
        Return the ``xarray.DataArray`` representation underlying this series.

        Parameters
        ----------
        copy
            Whether to return a copy of the series. Leave it to True unless you know what you are doing.

        Returns
        -------
        xarray.DataArray
            The xarray DataArray underlying this time series.
        """
        return self._xa.copy() if copy else self._xa

    def to_series(
        self,
        copy: bool = True,
        backend: Union[ModuleType, Implementation, str] = Implementation.PANDAS,
    ):
        """
        Return a Series representation of this time series in a given `backend`.

        Works only for univariate series that are deterministic (i.e., made of 1 sample).

        Parameters
        ----------
        copy
            Whether to return a copy of the series. Leave it to True unless you know what you are doing.
        backend
            The backend to which to export the `TimeSeries`. See the `narwhals documentation
            <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_dict>`_ for all supported
            backends.

        Returns
        -------
            A Series representation of this univariate time series in a given `backend`.
        """
        self._assert_univariate()
        self._assert_deterministic()

        backend = Implementation.from_backend(backend)
        if not backend.is_pandas():
            return self.to_dataframe(copy=copy, backend=backend, time_as_index=False)

        data = self._xa[:, 0, 0].values
        index = self._time_index
        name = self.components[0]

        if copy:
            data = data.copy()
            index = index.copy()

        return pd.Series(data=data, index=index, name=name)

    def to_dataframe(
        self,
        copy: bool = True,
        backend: Union[ModuleType, Implementation, str] = Implementation.PANDAS,
        time_as_index: bool = True,
        suppress_warnings: bool = False,
    ):
        """
        Return a DataFrame representation of this time series in a given `backend`.

        Each of the series components will appear as a column in the DataFrame.
        If the series is stochastic, the samples are returned as columns of the dataframe with column names
        as 'component_s#' (e.g. with two components and two samples:
        'comp0_s0', 'comp0_s1' 'comp1_s0' 'comp1_s1').

        Parameters
        ----------
        copy
            Whether to return a copy of the dataframe. Leave it to True unless you know what you are doing.
        backend
            The backend to which to export the `TimeSeries`. See the `narwhals documentation
            <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_dict>`_ for all supported
            backends.
        time_as_index
            Whether to set the time index as the index of the dataframe or in the left-most column.
            Only effective with the pandas `backend`.
        suppress_warnings
            Whether to suppress the warnings for the `DataFrame` creation.

        Returns
        -------
        DataFrame
            A DataFrame representation of this time series in the given `backend`.
        """

        backend = Implementation.from_backend(backend)
        if time_as_index and not backend.is_pandas():
            if not suppress_warnings:
                logger.warning(
                    '`time_as_index=True` is only supported with `backend="pandas"`, and will be ignored.'
                )
            time_as_index = False

        if not self.is_deterministic:
            if not suppress_warnings:
                logger.warning(
                    "You are transforming a stochastic TimeSeries (i.e., contains several samples). "
                    "The resulting DataFrame is a 2D object with all samples on the columns. "
                    "If this is not the expected behavior consider calling a function "
                    "adapted to stochastic TimeSeries like quantile_df()."
                )

            comp_name = list(self.components)
            samples = range(self.n_samples)
            columns = [
                "_s".join((comp_name, str(sample_id)))
                for comp_name, sample_id in itertools.product(comp_name, samples)
            ]
            data = self._xa.stack(data=(DIMS[1], DIMS[2])).values
        else:
            columns = self._xa.get_index(DIMS[1])
            data = self._xa[:, :, 0].values

        time_index = self._time_index

        if copy:
            data = data.copy()
            time_index = time_index.copy()

        if time_as_index:
            # special path for pandas with index
            return pd.DataFrame(data=data, index=time_index, columns=columns)

        data = {
            time_index.name: time_index,  # set time_index as left-most column
            **{col: data[:, idx] for idx, col in enumerate(columns)},
        }

        return nw.from_dict(data, backend=backend).to_native()

    def quantile_df(self, quantile=0.5) -> pd.DataFrame:
        """
        Return a Pandas DataFrame containing the single desired quantile of each component (over the samples).

        Each of the series components will appear as a column in the DataFrame. The column will be named
        "<component>_X", where "<component>" is the column name corresponding to this component, and "X"
        is the quantile value.
        The quantile columns represent the marginal distributions of the components of this series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        quantile
            The desired quantile value. The value must be represented as a fraction
            (between 0 and 1 inclusive). For instance, `0.5` will return a DataFrame
            containing the median of the (marginal) distribution of each component.

        Returns
        -------
        pandas.DataFrame
            The Pandas DataFrame containing the desired quantile for each component.
        """
        self._assert_stochastic()
        raise_if_not(
            0 <= quantile <= 1,
            "The quantile values must be expressed as fraction (between 0 and 1 inclusive).",
            logger,
        )

        # column names
        cnames = [s + f"_{quantile}" for s in self.columns]

        return pd.DataFrame(
            self._xa.quantile(q=quantile, dim=DIMS[2]),
            index=self._time_index,
            columns=cnames,
        )

    def quantile_timeseries(self, quantile=0.5, **kwargs) -> Self:
        """
        Return a deterministic ``TimeSeries`` containing the single desired quantile of each component
        (over the samples) of this stochastic ``TimeSeries``.

        The components in the new series are named "<component>_X", where "<component>"
        is the column name corresponding to this component, and "X" is the quantile value.
        The quantile columns represent the marginal distributions of the components of this series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        quantile
            The desired quantile value. The value must be represented as a fraction
            (between 0 and 1 inclusive). For instance, `0.5` will return a TimeSeries
            containing the median of the (marginal) distribution of each component.
        kwargs
            Other keyword arguments are passed down to `numpy.quantile()`

        Returns
        -------
        TimeSeries
            The TimeSeries containing the desired quantile for each component.
        """
        self._assert_stochastic()
        raise_if_not(
            0 <= quantile <= 1,
            "The quantile values must be expressed as fraction (between 0 and 1 inclusive).",
            logger,
        )

        # component names
        cnames = [f"{comp}_{quantile}" for comp in self.components]

        new_data = np.quantile(
            self._xa.values,
            q=quantile,
            axis=2,
            overwrite_input=False,
            keepdims=True,
            **kwargs,
        )
        new_xa = xr.DataArray(
            new_data,
            dims=self._xa.dims,
            coords={self._xa.dims[0]: self.time_index, DIMS[1]: pd.Index(cnames)},
            attrs=self._xa.attrs,
        )

        return self.__class__(new_xa)

    def quantiles_df(self, quantiles: tuple[float] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
        """
        Return a Pandas DataFrame containing the desired quantiles of each component (over the samples).

        Each of the series components will appear as a column in the DataFrame. The column will be named
        "<component>_X", where "<component>" is the column name corresponding to this component, and "X"
        is the quantile value.
        The quantiles represent the marginal distributions of the components of this series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        quantiles
            Tuple containing the desired quantiles. The values must be represented as fractions
            (between 0 and 1 inclusive). For instance, `(0.1, 0.5, 0.9)` will return a DataFrame
            containing the 10th-percentile, median and 90th-percentile of the (marginal) distribution of each component.

        Returns
        -------
        pandas.DataFrame
            The Pandas DataFrame containing the quantiles for each component.
        """
        return pd.concat(
            [
                self.quantile_timeseries(quantile).to_dataframe()
                for quantile in quantiles
            ],
            axis=1,
        )

    def astype(self, dtype: Union[str, np.dtype]) -> Self:
        """
        Converts this series to a new series with desired dtype.

        Parameters
        ----------
        dtype
            A NumPy dtype (np.float32 or np.float64)

        Returns
        -------
        TimeSeries
            A TimeSeries having the desired dtype.
        """
        return self.__class__(self._xa.astype(dtype))

    def start_time(self) -> Union[pd.Timestamp, int]:
        """
        Start time of the series.

        Returns
        -------
        Union[pandas.Timestamp, int]
            A timestamp containing the first time of the TimeSeries (if indexed by DatetimeIndex),
            or an integer (if indexed by RangeIndex)
        """
        return self._time_index[0]

    def end_time(self) -> Union[pd.Timestamp, int]:
        """
        End time of the series.

        Returns
        -------
        Union[pandas.Timestamp, int]
            A timestamp containing the last time of the TimeSeries (if indexed by DatetimeIndex),
            or an integer (if indexed by RangeIndex)
        """
        return self._time_index[-1]

    def first_value(self) -> float:
        """
        First value of this univariate series.

        Returns
        -------
        float
            The first value of this univariate deterministic time series
        """
        self._assert_univariate()
        self._assert_deterministic()
        return float(self._xa[0, 0, 0])

    def last_value(self) -> float:
        """
        Last value of this univariate series.

        Returns
        -------
        float
            The last value of this univariate deterministic time series
        """
        self._assert_univariate()
        self._assert_deterministic()
        return float(self._xa[-1, 0, 0])

    def first_values(self) -> np.ndarray:
        """
        First values of this potentially multivariate series.

        Returns
        -------
        np.ndarray
            The first values of every component of this deterministic time series
        """
        self._assert_deterministic()
        return self._xa.values[0, :, 0].copy()

    def last_values(self) -> np.ndarray:
        """
        Last values of this potentially multivariate series.

        Returns
        -------
        np.ndarray
            The last values of every component of this deterministic time series
        """
        self._assert_deterministic()
        return self._xa.values[-1, :, 0].copy()

    def values(self, copy: bool = True, sample: int = 0) -> np.ndarray:
        """
        Return a 2-D array of shape (time, component), containing this series' values for one `sample`.

        Parameters
        ----------
        copy
            Whether to return a copy of the values, otherwise returns a view.
            Leave it to True unless you know what you are doing.
        sample
            For stochastic series, the sample for which to return values. Default: 0 (first sample).

        Returns
        -------
        numpy.ndarray
            The values composing the time series.
        """
        raise_if(
            self.is_deterministic and sample != 0,
            "This series contains one sample only (deterministic),"
            "so only sample=0 is accepted.",
            logger,
        )
        if copy:
            return np.copy(self._xa.values[:, :, sample])
        else:
            return self._xa.values[:, :, sample]

    def random_component_values(self, copy: bool = True) -> np.array:
        """
        Return a 2-D array of shape (time, component), containing the values for
        one sample taken uniformly at random among this series' samples.

        Parameters
        ----------
        copy
            Whether to return a copy of the values, otherwise returns a view.
            Leave it to True unless you know what you are doing.

        Returns
        -------
        numpy.ndarray
            The values composing one sample taken at random from the time series.
        """
        sample = np.random.randint(low=0, high=self.n_samples)
        if copy:
            return np.copy(self._xa.values[:, :, sample])
        else:
            return self._xa.values[:, :, sample]

    def all_values(self, copy: bool = True) -> np.ndarray:
        """
        Return a 3-D array of dimension (time, component, sample),
        containing this series' values for all samples.

        Parameters
        ----------
        copy
            Whether to return a copy of the values, otherwise returns a view.
            Leave it to True unless you know what you are doing.

        Returns
        -------
        numpy.ndarray
            The values composing the time series.
        """
        if copy:
            return np.copy(self._xa.values)
        else:
            return self._xa.values

    def univariate_values(self, copy: bool = True, sample: int = 0) -> np.ndarray:
        """
        Return a 1-D Numpy array of shape (time,),
        containing this univariate series' values for one `sample`.

        Parameters
        ----------
        copy
            Whether to return a copy of the values. Leave it to True unless you know what you are doing.
        sample
            For stochastic series, the sample for which to return values. Default: 0 (first sample).

        Returns
        -------
        numpy.ndarray
            The values composing the time series guaranteed to be univariate.
        """

        self._assert_univariate()
        if copy:
            return np.copy(self._xa[:, 0, sample].values)
        else:
            return self._xa[:, 0, sample].values

    def static_covariates_values(self, copy: bool = True) -> Optional[np.ndarray]:
        """
        Return a 2-D array of dimension (component, static variable),
        containing the static covariate values of the TimeSeries.

        Parameters
        ----------
        copy
            Whether to return a copy of the values, otherwise returns a view.
            Can only return a view if all values have the same dtype.
            Leave it to True unless you know what you are doing.

        Returns
        -------
        Optional[numpy.ndarray]
            The static covariate values if the series has static covariates, else `None`.
        """
        return (
            self.static_covariates.to_numpy(copy=copy)
            if self.has_static_covariates
            else self.static_covariates
        )

    def head(
        self, size: Optional[int] = 5, axis: Optional[Union[int, str]] = 0
    ) -> Self:
        """
        Return a TimeSeries containing the first `size` points.

        Parameters
        ----------
        size : int, default 5
               number of points to retain
        axis : str or int, optional, default: 0
               axis along which to slice the series

        Returns
        -------
        TimeSeries
            The series made of the first `size` points along the desired `axis`.
        """

        axis_str = self._get_dim_name(axis)
        display_n = min(size, self._xa.sizes[axis_str])

        if axis_str == self._time_dim:
            return self[:display_n]
        else:
            return self.__class__(self._xa[{axis_str: range(display_n)}])

    def tail(
        self, size: Optional[int] = 5, axis: Optional[Union[int, str]] = 0
    ) -> Self:
        """
        Return last `size` points of the series.

        Parameters
        ----------
        size : int, default: 5
            number of points to retain
        axis : str or int, optional, default: 0 (time dimension)
            axis along which we intend to display records

        Returns
        -------
        TimeSeries
            The series made of the last `size` points along the desired `axis`.
        """

        axis_str = self._get_dim_name(axis)
        display_n = min(size, self._xa.sizes[axis_str])

        if axis_str == self._time_dim:
            return self[-display_n:]
        else:
            return self.__class__(self._xa[{axis_str: range(-display_n, 0)}])

    def concatenate(
        self,
        other: Self,
        axis: Optional[Union[str, int]] = 0,
        ignore_time_axis: Optional[bool] = False,
        ignore_static_covariates: bool = False,
        drop_hierarchy: bool = True,
        drop_metadata: bool = False,
    ) -> Self:
        """
        Concatenate another timeseries to the current one along given axis.

        Parameters
        ----------
        other : TimeSeries
            another timeseries to concatenate to this one
        axis : str or int
            axis along which timeseries will be concatenated. ['time', 'component' or 'sample'; Default: 0 (time)]
        ignore_time_axis : bool, default False
            Ignore errors when time axis varies for some timeseries. Note that this may yield unexpected results
        ignore_static_covariates : bool
            whether to ignore all requirements for static covariate concatenation and only transfer the
            static covariates of the current (`self`) timeseries to the concatenated timeseries.
            Only effective when `axis=1`.
        drop_hierarchy : bool
            When `axis=1`, whether to drop hierarchy information. True by default.
            When False, the hierarchies will be "concatenated" as well
            (by merging the hierarchy dictionaries), which may cause issues if the component
            names of the resulting series and that of the merged hierarchy do not match.
            When `axis=0` or `axis=2`, the hierarchy of the first series is always kept.
        drop_metadata : bool
            Whether to drop the metadata information of the concatenated timeseries. False by default.
            When False, the concatenated series will inherit the metadata from the current (`self`) timeseries.

        Returns
        -------
        TimeSeries
            concatenated timeseries

        See Also
        --------
        concatenate : a function to concatenate multiple series along a given axis.

        Notes
        -----
        When concatenating along the `time` dimension, the current series marks the start date of
        the resulting series, and the other series will have its time index ignored.
        """
        return concatenate(
            series=[self, other],
            axis=axis,
            ignore_time_axis=ignore_time_axis,
            ignore_static_covariates=ignore_static_covariates,
            drop_hierarchy=drop_hierarchy,
            drop_metadata=drop_metadata,
        )

    """
    Other methods
    =============
    """

    def gaps(self, mode: Literal["all", "any"] = "all") -> pd.DataFrame:
        """
        A function to compute and return gaps in the TimeSeries. Works only on deterministic time series (1 sample).

        Parameters
        ----------
        mode
            Only relevant for multivariate time series. The mode defines how gaps are defined. Set to
            'any' if a NaN value in any columns should be considered as as gaps. 'all' will only
            consider periods where all columns' values are NaN. Defaults to 'all'.

        Returns
        -------
        pd.DataFrame
            A pandas.DataFrame containing a row for every gap (rows with all-NaN values in underlying DataFrame)
            in this time series. The DataFrame contains three columns that include the start and end time stamps
            of the gap and the integer length of the gap (in `self.freq` units if the series is indexed
            by a DatetimeIndex).
        """

        df = self.to_dataframe()

        if mode == "all":
            is_nan_series = df.isna().all(axis=1).astype(int)
        elif mode == "any":
            is_nan_series = df.isna().any(axis=1).astype(int)
        else:
            raise_log(
                ValueError(
                    f"Keyword mode accepts only 'any' or 'all'. Provided {mode}"
                ),
                logger,
            )
        diff = pd.Series(np.diff(is_nan_series.values), index=is_nan_series.index[:-1])
        gap_starts = diff[diff == 1].index + self._freq
        gap_ends = diff[diff == -1].index

        if is_nan_series.iloc[0] == 1:
            gap_starts = gap_starts.insert(0, self.start_time())
        if is_nan_series.iloc[-1] == 1:
            gap_ends = gap_ends.insert(len(gap_ends), self.end_time())

        gap_df = pd.DataFrame(columns=["gap_start", "gap_end"])

        if gap_starts.size == 0:
            return gap_df
        else:

            def intvl(start, end):
                if self._has_datetime_index:
                    return pd.date_range(start=start, end=end, freq=self._freq).size
                else:
                    return int((end - start) / self._freq) + 1

            gap_df["gap_start"] = gap_starts
            gap_df["gap_end"] = gap_ends
            gap_df["gap_size"] = gap_df.apply(
                lambda row: intvl(start=row.gap_start, end=row.gap_end), axis=1
            )

            return gap_df

    def copy(self) -> Self:
        """
        Make a copy of this series.

        Returns
        -------
        TimeSeries
            A copy of this time series.
        """

        # the xarray will be copied in the TimeSeries constructor.
        return self.__class__(self._xa)

    def get_index_at_point(
        self, point: Union[pd.Timestamp, float, int], after=True
    ) -> int:
        """
        Converts a point along the time axis index into an integer index ranging in (0, len(series)-1).

        Parameters
        ----------
        point
            This parameter supports 3 different data types: ``pd.Timestamp``, ``float`` and ``int``.

            ``pd.Timestamp`` work only on series that are indexed with a ``pd.DatetimeIndex``. In such cases, the
            returned point will be the index of this timestamp if it is present in the series time index.
            If it's not present in the time index, the index of the next timestamp is returned if `after=True`
            (if it exists in the series), otherwise the index of the previous timestamp is returned
            (if it exists in the series).

            In case of a ``float``, the parameter will be treated as the proportion of the time series
            that should lie before the point.

            If an ``int`` and series is datetime-indexed, the value of `point` is returned.
            If an ``int`` and series is integer-indexed, the index position of `point` in the RangeIndex is returned
            (accounting for steps).
        after
            If the provided pandas Timestamp is not in the time series index, whether to return the index of the
            next timestamp or the index of the previous one.
        """
        point_index = -1
        if isinstance(point, float):
            raise_if_not(
                0.0 <= point <= 1.0,
                "point (float) should be between 0.0 and 1.0.",
                logger,
            )
            point_index = int((len(self) - 1) * point)
        elif isinstance(point, (int, np.int64)):
            if self.has_datetime_index or (self.start_time() == 0 and self.freq == 1):
                point_index = point
            else:
                point_index_float = (point - self.start_time()) / self.freq
                point_index = int(point_index_float)
                raise_if(
                    point_index != point_index_float,
                    "The provided point is not a valid index for this series.",
                )
            raise_if_not(
                0 <= point_index < len(self),
                f"The index corresponding to the provided point ({point}) should be a valid index in series",
                logger,
            )
        elif isinstance(point, pd.Timestamp):
            raise_if_not(
                self._has_datetime_index,
                "A Timestamp has been provided, but this series is not time-indexed.",
                logger,
            )
            self._raise_if_not_within(point)
            if point in self:
                point_index = self._time_index.get_loc(point)
            else:
                point_index = self._time_index.get_loc(
                    self._get_first_timestamp_after(point)
                    if after
                    else self._get_last_timestamp_before(point)
                )
        else:
            raise_log(
                TypeError(
                    "`point` needs to be either `float`, `int` or `pd.Timestamp`"
                ),
                logger,
            )
        return point_index

    def get_timestamp_at_point(
        self, point: Union[pd.Timestamp, float, int]
    ) -> Union[pd.Timestamp, int]:
        """
        Converts a point into a pandas.Timestamp (if Datetime-indexed) or into an integer (if Int64-indexed).

        Parameters
        ----------
        point
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In case of a `float`, the parameter will be treated as the proportion of the time series
            that should lie before the point.
            In case of `int`, the parameter will be treated as an integer index to the time index of
            `series`. Will raise a ValueError if not a valid index in `series`.
            In case of a `pandas.Timestamp`, point will be returned as is provided that the timestamp
            is present in the series time index, otherwise will raise a ValueError.
        """
        idx = self.get_index_at_point(point)
        return self._time_index[idx]

    def _split_at(
        self, split_point: Union[pd.Timestamp, float, int], after: bool = True
    ) -> tuple[Self, Self]:
        # Get index with not after in order to avoid moving twice if split_point is not in self
        point_index = self.get_index_at_point(split_point, not after)
        return (
            self[: point_index + (1 if after else 0)],
            self[point_index + (1 if after else 0) :],
        )

    def split_after(
        self, split_point: Union[pd.Timestamp, float, int]
    ) -> tuple[Self, Self]:
        """
        Splits the series in two, after a provided `split_point`.

        Parameters
        ----------
        split_point
            A timestamp, float or integer. If float, represents the proportion of the series to include in the
            first TimeSeries (must be between 0.0 and 1.0). If integer, represents the index position after
            which the split is performed. A pd.Timestamp can be provided for TimeSeries that are indexed by a
            pd.DatetimeIndex. In such cases, the timestamp will be contained in the first TimeSeries, but not
            in the second one. The timestamp itself does not have to appear in the original TimeSeries index.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries]
            A tuple of two time series. The first time series contains the first samples up to the `split_point`,
            and the second contains the remaining ones.
        """
        return self._split_at(split_point, after=True)

    def split_before(
        self, split_point: Union[pd.Timestamp, float, int]
    ) -> tuple[Self, Self]:
        """
        Splits the series in two, before a provided `split_point`.

        Parameters
        ----------
        split_point
            A timestamp, float or integer. If float, represents the proportion of the series to include in the
            first TimeSeries (must be between 0.0 and 1.0). If integer, represents the index position before
            which the split is performed. A pd.Timestamp can be provided for TimeSeries that are indexed by a
            pd.DatetimeIndex. In such cases, the timestamp will be contained in the second TimeSeries, but not
            in the first one. The timestamp itself does not have to appear in the original TimeSeries index.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries]
            A tuple of two time series. The first time series contains the first samples up to the `split_point`,
            and the second contains the remaining ones.
        """
        return self._split_at(split_point, after=False)

    def drop_after(self, split_point: Union[pd.Timestamp, float, int]):
        """
        Drops everything after the provided time `split_point`, included.
        The timestamp may not be in the series. If it is, the timestamp will be dropped.

        Parameters
        ----------
        split_point
            The timestamp that indicates cut-off time.

        Returns
        -------
        TimeSeries
            A new TimeSeries, after `ts`.
        """
        return self[: self.get_index_at_point(split_point, after=True)]

    def drop_before(self, split_point: Union[pd.Timestamp, float, int]):
        """
        Drops everything before the provided time `split_point`, included.
        The timestamp may not be in the series. If it is, the timestamp will be dropped.

        Parameters
        ----------
        split_point
            The timestamp that indicates cut-off time.

        Returns
        -------
        TimeSeries
            A new TimeSeries, after `ts`.
        """
        return self[self.get_index_at_point(split_point, after=False) + 1 :]

    def slice(
        self, start_ts: Union[pd.Timestamp, int], end_ts: Union[pd.Timestamp, int]
    ):
        """
        Return a new TimeSeries, starting later than `start_ts` and ending before `end_ts`.
        For series having DatetimeIndex, this is inclusive on both ends. For series having a RangeIndex,
        `end_ts` is exclusive.

        `start_ts` and `end_ts` don't have to be in the series.

        Parameters
        ----------
        start_ts
            The timestamp that indicates the left cut-off.
        end_ts
            The timestamp that indicates the right cut-off.

        Returns
        -------
        TimeSeries
            A new series, with indices greater or equal than `start_ts` and smaller or equal than `end_ts`.
        """
        raise_if_not(
            type(start_ts) is type(end_ts),
            "The two timestamps provided to slice() have to be of the same type.",
            logger,
        )
        if isinstance(start_ts, pd.Timestamp):
            raise_if_not(
                self._has_datetime_index,
                "Timestamps have been provided to slice(), but the series is "
                "indexed using an integer-based RangeIndex.",
                logger,
            )
            if start_ts in self._time_index and end_ts in self._time_index:
                return self[
                    start_ts:end_ts
                ]  # we assume this is faster than the filtering below
            else:
                idx = self._time_index[
                    (start_ts <= self._time_index) & (self._time_index <= end_ts)
                ]
                return self[idx]
        else:
            raise_if(
                self._has_datetime_index,
                "start and end times have been provided as integers to slice(), but "
                "the series is indexed with a DatetimeIndex.",
                logger,
            )
            # get closest timestamps if either start or end are not in the index
            effective_start_ts = (
                min(self._time_index, key=lambda t: abs(t - start_ts))
                if start_ts not in self._time_index
                else start_ts
            )
            if effective_start_ts < start_ts:
                # if the requested start_ts is smaller than the start argument,
                # we have to increase it to be consistent with the docstring
                effective_start_ts += self.freq

            effective_end_ts = (
                min(self._time_index, key=lambda t: abs(t - end_ts))
                if end_ts not in self._time_index
                else end_ts
            )
            if end_ts >= effective_end_ts + self.freq:
                # if the requested end_ts is further off from the end of the time series,
                # we have to increase effectiv_end_ts to make the last timestamp inclusive.
                effective_end_ts += self.freq
            idx = pd.RangeIndex(effective_start_ts, effective_end_ts, step=self.freq)
            return self[idx]

    def slice_n_points_after(self, start_ts: Union[pd.Timestamp, int], n: int) -> Self:
        """
        Return a new TimeSeries, starting a `start_ts` (inclusive) and having at most `n` points.

        The provided timestamps will be included in the series.

        Parameters
        ----------
        start_ts
            The timestamp or index that indicates the splitting time.
        n
            The maximal length of the new TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with length at most `n`, starting at `start_ts`
        """
        raise_if_not(n > 0, "n should be a positive integer.", logger)
        self._raise_if_not_within(start_ts)

        if isinstance(start_ts, (int, np.int64)):
            return self[pd.RangeIndex(start=start_ts, stop=start_ts + n)]
        elif isinstance(start_ts, pd.Timestamp):
            # get first timestamp greater or equal to start_ts
            tss = self._get_first_timestamp_after(start_ts)
            point_index = self.get_index_at_point(tss)
            return self[point_index : point_index + n]
        else:
            raise_log(
                ValueError("start_ts must be an int or a pandas Timestamp."), logger
            )

    def slice_n_points_before(self, end_ts: Union[pd.Timestamp, int], n: int) -> Self:
        """
        Return a new TimeSeries, ending at `end_ts` (inclusive) and having at most `n` points.

        The provided timestamps will be included in the series.

        Parameters
        ----------
        end_ts
            The timestamp or index that indicates the splitting time.
        n
            The maximal length of the new TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with length at most `n`, ending at `start_ts`
        """

        raise_if_not(n > 0, "n should be a positive integer.", logger)
        self._raise_if_not_within(end_ts)

        if isinstance(end_ts, (int, np.int64)):
            return self[pd.RangeIndex(start=end_ts - n + 1, stop=end_ts + 1)]
        elif isinstance(end_ts, pd.Timestamp):
            # get last timestamp smaller or equal to start_ts
            tss = self._get_last_timestamp_before(end_ts)
            point_index = self.get_index_at_point(tss)
            return self[max(0, point_index - n + 1) : point_index + 1]
        else:
            raise_log(
                ValueError("start_ts must be an int or a pandas Timestamp."), logger
            )

    def slice_intersect(self, other: Self) -> Self:
        """
        Return a ``TimeSeries`` slice of this series, where the time index has been intersected with the one
        of the `other` series.

        This method is in general *not* symmetric.

        Parameters
        ----------
        other
            the other time series

        Returns
        -------
        TimeSeries
            a new series, containing the values of this series, over the time-span common to both time series.
        """
        if other.has_same_time_as(self):
            return self.__class__(self._xa)
        elif other.freq == self.freq and len(self) and len(other):
            start, end = self._slice_intersect_bounds(other)
            return self[start:end]
        else:
            time_index = self.time_index.intersection(other.time_index)
            return self[time_index]

    def slice_intersect_values(self, other: Self, copy: bool = False) -> np.ndarray:
        """
        Return the sliced values of this series, where the time index has been intersected with the one
        of the `other` series.

        This method is in general *not* symmetric.

        Parameters
        ----------
        other
            The other time series
        copy
            Whether to return a copy of the values, otherwise returns a view.
            Leave it to True unless you know what you are doing.

        Returns
        -------
        np.ndarray
            The values of this series, over the time-span common to both time series.
        """
        vals = self.all_values(copy=copy)
        if other.has_same_time_as(self):
            return vals
        if other.freq == self.freq:
            start, end = self._slice_intersect_bounds(other)
            return vals[start:end]
        else:
            return vals[self._time_index.isin(other._time_index)]

    def slice_intersect_times(
        self, other: Self, copy: bool = True
    ) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
        """
        Return time index of this series, where the time index has been intersected with the one
        of the `other` series.

        This method is in general *not* symmetric.

        Parameters
        ----------
        other
            The other time series
        copy
            Whether to return a copy of the time index, otherwise returns a view. Leave it to True unless you know
            what you are doing.

        Returns
        -------
        Union[pd.DatetimeIndex, pd.RangeIndex]
            The time index of this series, over the time-span common to both time series.
        """

        time_index = self.time_index if copy else self._time_index
        if other.has_same_time_as(self):
            return time_index
        if other.freq == self.freq:
            start, end = self._slice_intersect_bounds(other)
            return time_index[start:end]
        else:
            return time_index[time_index.isin(other._time_index)]

    def _slice_intersect_bounds(self, other: Self) -> tuple[int, int]:
        """Find the start (absolute index) and end (index relative to the end) indices that represent the time
        intersection from `self` and `other`."""
        shift_start = n_steps_between(
            other.start_time(), self.start_time(), freq=self.freq
        )
        shift_end = len(other) - (len(self) - shift_start)

        shift_start = shift_start if shift_start >= 0 else 0
        shift_end = shift_end if shift_end < 0 else None
        return shift_start, shift_end

    def strip(self, how: str = "all") -> Self:
        """
        Return a ``TimeSeries`` slice of this deterministic time series, where NaN-containing entries at the beginning
        and the end of the series are removed. No entries after (and including) the first non-NaN entry and
        before (and including) the last non-NaN entry are removed.

        This method is only applicable to deterministic series (i.e., having 1 sample).

        Parameters
        ----------
        how
            Define if the entries containing `NaN` in all the components ('all') or in any of the components ('any')
            should be stripped. Default: 'all'

        Returns
        -------
        TimeSeries
            a new series based on the original where NaN-containing entries at start and end have been removed
        """
        raise_if(
            self.is_probabilistic,
            "`strip` cannot be applied to stochastic TimeSeries",
            logger,
        )

        first_finite_row, last_finite_row = _finite_rows_boundaries(
            self.values(), how=how
        )

        return self.__class__.from_times_and_values(
            times=self.time_index[first_finite_row : last_finite_row + 1],
            values=self.values()[first_finite_row : last_finite_row + 1],
            columns=self.components,
            static_covariates=self.static_covariates,
            hierarchy=self.hierarchy,
            metadata=self.metadata,
        )

    def longest_contiguous_slice(
        self, max_gap_size: int = 0, mode: str = "all"
    ) -> Self:
        """
        Return the largest TimeSeries slice of this deterministic series that contains no gaps
        (contiguous all-NaN values) larger than `max_gap_size`.

        This method is only applicable to deterministic series (i.e., having 1 sample).

        Parameters
        ----------
        max_gap_size
            Indicate the maximum gap size that the TimeSerie can contain
        mode
            Only relevant for multivariate time series. The mode defines how gaps are defined. Set to
            'any' if a NaN value in any columns should be considered as as gaps. 'all' will only
            consider periods where all columns' values are NaN. Defaults to 'all'.

        Returns
        -------
        TimeSeries
            a new series constituting the largest slice of the original with no or bounded gaps

        See Also
        --------
        TimeSeries.gaps : return the gaps in the TimeSeries
        """
        if not (np.isnan(self._xa)).any():
            return self.copy()
        stripped_series = self.strip()
        gaps = stripped_series.gaps(mode=mode)
        relevant_gaps = gaps[gaps["gap_size"] > max_gap_size]

        curr_slice_start = stripped_series.start_time()
        max_size = pd.Timedelta(days=0) if self._has_datetime_index else 0
        max_slice_start = None
        max_slice_end = None
        for index, row in relevant_gaps.iterrows():
            # evaluate size of the current slice. the slice ends one time step before row['gap_start']
            curr_slice_end = row["gap_start"] - self.freq
            size = curr_slice_end - curr_slice_start
            if size > max_size:
                max_size = size
                max_slice_start = curr_slice_start
                max_slice_end = row["gap_start"] - self._freq
            curr_slice_start = row["gap_end"] + self._freq

        if stripped_series.end_time() - curr_slice_start > max_size:
            max_slice_start = curr_slice_start
            max_slice_end = self.end_time()

        return stripped_series[max_slice_start:max_slice_end]

    def rescale_with_value(self, value_at_first_step: float) -> Self:
        """
        Return a new ``TimeSeries``, which is a multiple of this series such that
        the first value is `value_at_first_step`.
        (Note: numerical errors can appear with `value_at_first_step > 1e+24`).

        Parameters
        ----------
        value_at_first_step
            The new value for the first entry of the TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, where the first value is `value_at_first_step` and other values
            have been scaled accordingly.
        """

        raise_if_not(
            (self._xa[0, :, :] != 0).all(), "Cannot rescale with first value 0.", logger
        )
        coef = value_at_first_step / self._xa.isel({self._time_dim: [0]})
        coef = coef.values.reshape((self.n_components, self.n_samples))  # TODO: test
        new_series = coef * self._xa
        return self.__class__(new_series)

    def shift(self, n: int) -> Self:
        """
        Shifts the time axis of this TimeSeries by `n` time steps.

        If :math:`n > 0`, shifts in the future. If :math:`n < 0`, shifts in the past.

        For example, with :math:`n=2` and `freq='M'`, March 2013 becomes May 2013.
        With :math:`n=-2`, March 2013 becomes Jan 2013.

        Parameters
        ----------
        n
            The number of time steps (in self.freq unit) to shift by. Can be negative.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with a shifted index.
        """
        if not isinstance(n, (int, np.int64)):
            logger.warning(
                f"TimeSeries.shift(): converting n to int from {n} to {int(n)}"
            )
            n = int(n)

        try:
            self._time_index[-1] + n * self.freq
        except pd.errors.OutOfBoundsDatetime:
            raise_log(
                OverflowError(
                    f"the add operation between {n * self.freq} and {self.time_index[-1]} will "
                    "overflow"
                ),
                logger,
            )

        if self.has_range_index:
            new_time_index = self._time_index + n * self.freq
        else:
            new_time_index = self._time_index.map(lambda ts: ts + n * self.freq)
            if new_time_index.freq is None:
                new_time_index.freq = self.freq
        new_xa = self._xa.assign_coords({self._xa.dims[0]: new_time_index})
        return self.__class__(new_xa)

    def diff(
        self,
        n: Optional[int] = 1,
        periods: Optional[int] = 1,
        dropna: Optional[bool] = True,
    ) -> Self:
        """
        Return a differenced time series. This is often used to make a time series stationary.

        Parameters
        ----------
        n
            Optionally, a positive integer indicating the number of differencing steps (default = 1).
            For instance, n=2 computes the second order differences.
        periods
            Optionally, periods to shift for calculating difference. For instance, periods=12 computes the
            difference between values at time `t` and times `t-12`.
        dropna
            Whether to drop the missing values after each differencing steps. If set to `False`, the corresponding
            first `periods` time steps will be filled with NaNs.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with the differenced values.
        """
        if not isinstance(n, int) or n < 1:
            raise_log(ValueError("'n' must be a positive integer >= 1."), logger)
        if not isinstance(periods, int) or periods < 1:
            raise_log(ValueError("'periods' must be an integer >= 1."), logger)

        def _compute_diff(xa: xr.DataArray):
            # xarray doesn't support Pandas "period" so compute diff() ourselves
            if not dropna:
                # In this case the new DataArray will have the same size and filled with NaNs
                new_xa_ = xa.copy()
                new_xa_.values[:periods, :, :] = np.nan
                new_xa_.values[periods:, :, :] = (
                    xa.values[periods:, :, :] - xa.values[:-periods, :, :]
                )
            else:
                # In this case the new DataArray will be shorter
                new_xa_ = xa[periods:, :, :].copy()
                new_xa_.values = xa.values[periods:, :, :] - xa.values[:-periods, :, :]
            return new_xa_

        new_xa = _compute_diff(self._xa)
        for _ in range(n - 1):
            new_xa = _compute_diff(new_xa)
        return self.__class__(new_xa)

    def cumsum(self) -> Self:
        """
        Returns the cumulative sum of the time series along the time axis.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with the cumulatively summed values.
        """
        return self.__class__(self._xa.copy().cumsum(axis=0))

    def has_same_time_as(self, other: Self) -> bool:
        """
        Checks whether this series has the same time index as `other`.

        Parameters
        ----------
        other
            the other series

        Returns
        -------
        bool
            True if both TimeSeries have the same index, False otherwise.
        """
        if len(other) != len(self):
            return False
        elif other.freq != self.freq:
            return False
        elif other.start_time() != self.start_time():
            return False
        else:
            return True

    def append(self, other: Self) -> Self:
        """
        Appends another series to this series along the time axis.

        Parameters
        ----------
        other
            A second TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, obtained by appending the second TimeSeries to the first.

        See Also
        --------
        TimeSeries.concatenate : concatenate another series along a given axis.
        TimeSeries.prepend : prepend (i.e. add to the beginning) another series along the time axis.
        """
        raise_if_not(
            other.has_datetime_index == self.has_datetime_index,
            "Both series must have the same type of time index (either DatetimeIndex or RangeIndex).",
            logger,
        )
        raise_if_not(
            other.freq == self.freq,
            "Both series must have the same frequency.",
            logger,
        )
        raise_if_not(
            other.n_components == self.n_components,
            "Both series must have the same number of components.",
            logger,
        )
        raise_if_not(
            other.n_samples == self.n_samples,
            "Both series must have the same number of components.",
            logger,
        )
        if len(self) > 0 and len(other) > 0:
            raise_if_not(
                other.start_time() == self.end_time() + self.freq,
                "Appended TimeSeries must start one (time) step after current one.",
                logger,
            )

        other_xa = other.data_array()

        new_xa = xr.DataArray(
            np.concatenate((self._xa.values, other_xa.values), axis=0),
            dims=self._xa.dims,
            coords={
                self._time_dim: self._time_index.append(other.time_index),
                DIMS[1]: self.components,
            },
            attrs=self._xa.attrs,
        )

        return self.__class__.from_xarray(
            new_xa, fill_missing_dates=True, freq=self._freq_str
        )

    def append_values(self, values: np.ndarray) -> Self:
        """
        Appends new values to current TimeSeries, extending its time index.

        Parameters
        ----------
        values
            An array with the values to append.

        Returns
        -------
        TimeSeries
            A new TimeSeries with the new values appended
        """
        if len(values) == 0:
            return self.copy()

        values = np.array(values) if not isinstance(values, np.ndarray) else values
        values = expand_arr(values, ndim=len(DIMS))
        if not values.shape[1:] == self._xa.values.shape[1:]:
            raise_log(
                ValueError(
                    f"The (expanded) values must have the same number of components and samples "
                    f"(second and third dims) as the series to append to. "
                    f"Received shape: {values.shape}, expected: {self._xa.values.shape}"
                ),
                logger=logger,
            )

        idx = generate_index(
            start=self.end_time() + self.freq,
            length=len(values),
            freq=self.freq,
            name=self._time_index.name,
        )

        return self.append(
            self.__class__.from_times_and_values(
                values=values,
                times=idx,
                fill_missing_dates=False,
                static_covariates=self.static_covariates,
                metadata=self.metadata,
            )
        )

    def prepend(self, other: Self) -> Self:
        """
        Prepends (i.e. adds to the beginning) another series to this series along the time axis.

        Parameters
        ----------
        other
            A second TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, obtained by appending the second TimeSeries to the first.

        See Also
        --------
        Timeseries.append : append (i.e. add to the end) another series along the time axis.
        TimeSeries.concatenate : concatenate another series along a given axis.
        """
        raise_if_not(
            isinstance(other, self.__class__),
            f"`other` to prepend must be a {self.__class__.__name__} object.",
        )
        return other.append(self)

    def prepend_values(self, values: np.ndarray) -> Self:
        """
        Prepends (i.e. adds to the beginning) new values to current TimeSeries, extending its time index into the past.

        Parameters
        ----------
        values
            An array with the values to prepend to the start.

        Returns
        -------
        TimeSeries
            A new TimeSeries with the new values prepended.
        """
        if len(values) == 0:
            return self.copy()

        values = np.array(values) if not isinstance(values, np.ndarray) else values
        values = expand_arr(values, ndim=len(DIMS))
        if not values.shape[1:] == self._xa.values.shape[1:]:
            raise_log(
                ValueError(
                    f"The (expanded) values must have the same number of components and samples "
                    f"(second and third dims) as the series to prepend to. "
                    f"Received shape: {values.shape}, expected: {self._xa.values.shape}"
                ),
                logger=logger,
            )

        idx = generate_index(
            end=self.start_time() - self.freq,
            length=len(values),
            freq=self.freq,
            name=self._time_index.name,
        )

        return self.prepend(
            self.__class__.from_times_and_values(
                values=values,
                times=idx,
                fill_missing_dates=False,
                static_covariates=self.static_covariates,
                columns=self.columns,
                hierarchy=self.hierarchy,
                metadata=self.metadata,
            )
        )

    def with_times_and_values(
        self,
        times: Union[pd.DatetimeIndex, pd.RangeIndex, pd.Index],
        values: np.ndarray,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        fillna_value: Optional[float] = None,
    ) -> Self:
        """
        Return a new ``TimeSeries`` similar to this one but with new specified values.

        Parameters
        ----------
        times
            A pandas DateTimeIndex, RangeIndex (or Index that can be converted to a RangeIndex) representing the new
            time axis for the time series. It is better if the index has no holes; alternatively setting
            `fill_missing_dates` can in some cases solve these issues (filling holes with NaN, or with the provided
            `fillna_value` numeric value, if any).
        values
            A Numpy array with new values. It must have the dimensions for `times` and components, but may contain a
            different number of samples.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates (or indices in case of integer index)
            with NaN values. This requires either a provided `freq` or the possibility to infer the frequency from the
            provided timestamps. See :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string or integer representing the frequency of the underlying index. This is useful in order
            to fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
            If a string, represents the frequency of the pandas DatetimeIndex (see `offset aliases
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.

        Returns
        -------
        TimeSeries
            A new TimeSeries with the new values and same index, static covariates and hierarchy
        """
        values = np.array(values) if not isinstance(values, np.ndarray) else values
        values = expand_arr(values, ndim=len(DIMS))
        raise_if_not(
            values.shape[1] == self._xa.values.shape[1],
            "The new values must have the same number of components as the present series. "
            f"Received: {values.shape[1]}, expected: {self._xa.values.shape[1]}",
        )
        return self.from_times_and_values(
            times=times,
            values=values,
            fill_missing_dates=fill_missing_dates,
            freq=freq,
            columns=self.columns,
            fillna_value=fillna_value,
            static_covariates=self.static_covariates,
            hierarchy=self.hierarchy,
            metadata=self.metadata,
        )

    def with_values(self, values: np.ndarray) -> Self:
        """
        Return a new ``TimeSeries`` similar to this one but with new specified values.

        Parameters
        ----------
        values
            A Numpy array with new values. It must have the dimensions for time
            and componentns, but may contain a different number of samples.

        Returns
        -------
        TimeSeries
            A new TimeSeries with the new values and same index, static covariates and hierarchy
        """
        values = np.array(values) if not isinstance(values, np.ndarray) else values
        values = expand_arr(values, ndim=len(DIMS))
        raise_if_not(
            values.shape[:2] == self._xa.values.shape[:2],
            "The new values must have the same shape (time, components) as the present series. "
            f"Received: {values.shape[:2]}, expected: {self._xa.values.shape[:2]}",
        )

        new_xa = xr.DataArray(
            values,
            dims=self._xa.dims,
            coords=self._xa.coords,
            attrs=self._xa.attrs,
        )

        return self.__class__(new_xa)

    def with_static_covariates(
        self, covariates: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> Self:
        """Returns a new TimeSeries object with added static covariates.

        Static covariates contain data attached to the time series, but which are not varying with time.

        Parameters
        ----------
        covariates
            Optionally, a set of static covariates to be added to the TimeSeries. Either a pandas Series, a pandas
            DataFrame, or `None`. If `None`, will set the static covariates to `None`. If a Series, the index
            represents the static variables. The covariates are then globally 'applied' to all components of the
            TimeSeries. If a DataFrame, the columns represent the static variables and the rows represent the
            components of the uni/multivariate TimeSeries. If a single-row DataFrame, the covariates are globally
            'applied' to all components of the TimeSeries. If a multi-row DataFrame, the number of rows must match the
            number of components of the TimeSeries. This adds component-specific static covariates.

        Returns
        -------
        TimeSeries
            A new TimeSeries with the given static covariates.

        Notes
        -----
        If there are a large number of static covariates variables (i.e., the static covariates have a very large
        dimension), there might be a noticeable performance penalty for creating ``TimeSeries`` objects, unless
        the covariates already have the same ``dtype`` as the series data.

        Examples
        --------
        >>> import pandas as pd
        >>> from darts.utils.timeseries_generation import linear_timeseries
        >>> # add global static covariates
        >>> static_covs = pd.Series([0., 1.], index=["static_cov_1", "static_cov_2"])
        >>> series = linear_timeseries(length=3)
        >>> series_new1 = series.with_static_covariates(static_covs)
        >>> series_new1.static_covariates
                           static_cov_1  static_cov_2
        component
        linear              0.0           1.0

        >>> # add component specific static covariates
        >>> static_covs_multi = pd.DataFrame([[0., 1.], [2., 3.]], columns=["static_cov_1", "static_cov_2"])
        >>> series_multi = series.stack(series)
        >>> series_new2 = series_multi.with_static_covariates(static_covs_multi)
        >>> series_new2.static_covariates
                           static_cov_1  static_cov_2
        component
        linear              0.0           1.0
        linear_1            2.0           3.0
        """

        return self.__class__(
            xr.DataArray(
                self._xa.values,
                dims=self._xa.dims,
                coords=self._xa.coords,
                attrs={
                    STATIC_COV_TAG: covariates,
                    HIERARCHY_TAG: self.hierarchy,
                    METADATA_TAG: self.metadata,
                },
            )
        )

    def with_hierarchy(self, hierarchy: dict[str, Union[str, list[str]]]) -> Self:
        """
        Adds a hierarchy to the TimeSeries.

        Parameters
        ----------
        hierarchy
            A dictionary mapping components to a list of their parent(s) in the hierarchy.
            Single parents may be specified as string or list containing one string.
            For example, assume the series contains the components
            ``["total", "a", "b", "x", "y", "ax", "ay", "bx", "by"]``,
            the following dictionary would encode the groupings shown on
            `this figure <https://otexts.com/fpp3/hts.html#fig:GroupTree>`_:

            .. highlight:: python
            .. code-block:: python

                hierarchy = {'ax': ['a', 'x'],
                             'ay': ['a', 'y'],
                             'bx': ['b', 'x'],
                             'by': ['b', 'y'],
                             'a': ['total'],
                             'b': ['total'],
                             'x': 'total',  # or use a single string
                             'y': 'total'}
            ..

        Returns
        -------
        TimeSeries
            A new TimeSeries with the given hierarchy.
        """

        return self.__class__(
            xr.DataArray(
                self._xa.values,
                dims=self._xa.dims,
                coords=self._xa.coords,
                attrs={
                    STATIC_COV_TAG: self.static_covariates,
                    HIERARCHY_TAG: hierarchy,
                    METADATA_TAG: self.metadata,
                },
            )
        )

    def with_metadata(self, metadata: Optional[dict]) -> Self:
        """
        Adds metadata to the TimeSeries.

        Parameters
        ----------
        metadata
            A dictionary with metadata to be added to the TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries with the given metadata.

        Examples
        --------
        >>> from darts.utils.timeseries_generation import linear_timeseries
        >>> series = linear_timeseries(length=3)
        >>> # add metadata
        >>> metadata = {'name': 'my_series'}
        >>> series = series.with_metadata(metadata)
        >>> series.metadata
        {'name': 'my_series'}
        """

        return self.__class__(
            xr.DataArray(
                self._xa.values,
                dims=self._xa.dims,
                coords=self._xa.coords,
                attrs={
                    STATIC_COV_TAG: self.static_covariates,
                    HIERARCHY_TAG: self.hierarchy,
                    METADATA_TAG: metadata,
                },
            )
        )

    def stack(self, other: Self) -> Self:
        """
        Stacks another univariate or multivariate TimeSeries with the same time index on top of
        the current one (along the component axis).

        Return a new TimeSeries that includes all the components of `self` and of `other`.

        The resulting TimeSeries will have the same name for its time dimension as this TimeSeries, and the
        same number of samples.

        Parameters
        ----------
        other
            A TimeSeries instance with the same index and the same number of samples as the current one.

        Returns
        -------
        TimeSeries
            A new multivariate TimeSeries instance.
        """
        return concatenate([self, other], axis=1)

    def drop_columns(self, col_names: Union[list[str], str]) -> Self:
        """
        Return a new ``TimeSeries`` instance with dropped columns/components.

        Parameters
        -------
        col_names
            String or list of strings corresponding to the columns to be dropped.

        Returns
        -------
        TimeSeries
            A new TimeSeries instance with specified columns dropped.
        """
        if isinstance(col_names, str):
            col_names = [col_names]

        raise_if_not(
            all([(x in self.columns.to_list()) for x in col_names]),
            "Some column names in col_names don't exist in the time series.",
            logger,
        )

        new_xa = self._xa.drop_sel({"component": col_names})
        return self.__class__(new_xa)

    def univariate_component(self, index: Union[str, int]) -> Self:
        """
        Retrieve one of the components of the series
        and return it as new univariate ``TimeSeries`` instance.

        This drops the hierarchy (if any), and retains only the relevant static
        covariates column.

        Parameters
        ----------
        index
            An zero-indexed integer indicating which component to retrieve. If components have names,
            this can be a string with the component's name.

        Returns
        -------
        TimeSeries
            A new univariate TimeSeries instance.
        """

        return self[index if isinstance(index, str) else self.components[index]]

    def add_datetime_attribute(
        self,
        attribute,
        one_hot: bool = False,
        cyclic: bool = False,
        tz: Optional[str] = None,
    ) -> Self:
        """
        Build a new series with one (or more) additional component(s) that contain an attribute
        of the time index of the series.

        The additional components are specified with `attribute`, such as 'weekday', 'day' or 'month'.

        This works only for deterministic time series (i.e., made of 1 sample).

        Notes
        -----
        0-indexing is enforced across all the encodings, see
        :meth:`datetime_attribute_timeseries() <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        for more information.

        Parameters
        ----------
        attribute
            A pd.DatatimeIndex attribute which will serve as the basis of the new column(s).
        one_hot
            Boolean value indicating whether to add the specified attribute as a one hot encoding
            (results in more columns).
        cyclic
            Boolean value indicating whether to add the specified attribute as a cyclic encoding.
            Alternative to one_hot encoding, enable only one of the two.
            (adds 2 columns, corresponding to sin and cos transformation).
        tz
            Optionally, a time zone to convert the time index to before computing the attributes.

        Returns
        -------
        TimeSeries
            New TimeSeries instance enhanced by `attribute`.
        """
        self._assert_deterministic()
        from darts.utils import timeseries_generation as tg

        return self.stack(
            tg.datetime_attribute_timeseries(
                self.time_index,
                attribute=attribute,
                one_hot=one_hot,
                cyclic=cyclic,
                tz=tz,
            )
        )

    def add_holidays(
        self,
        country_code: str,
        prov: str = None,
        state: str = None,
        tz: Optional[str] = None,
    ) -> Self:
        """
        Adds a binary univariate component to the current series that equals 1 at every index that
        corresponds to selected country's holiday, and 0 otherwise.

        The frequency of the TimeSeries is daily.

        Available countries can be found `here <https://github.com/dr-prodigy/python-holidays#available-countries>`_.

        This works only for deterministic time series (i.e., made of 1 sample).

        Parameters
        ----------
        country_code
            The country ISO code
        prov
            The province
        state
            The state
        tz
            Optionally, a time zone to convert the time index to before computing the attributes.

        Returns
        -------
        TimeSeries
            A new TimeSeries instance, enhanced with binary holiday component.
        """
        self._assert_deterministic()
        from darts.utils import timeseries_generation as tg

        return self.stack(
            tg.holidays_timeseries(
                self.time_index,
                country_code=country_code,
                prov=prov,
                state=state,
                tz=tz,
            )
        )

    def resample(
        self,
        freq: Union[str, pd.DateOffset],
        method: str = "pad",
        method_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Self:
        """
        Build a reindexed ``TimeSeries`` with a given frequency.
        Provided method is used to aggregate/fill holes in the reindexed TimeSeries, by default 'pad'.

        Parameters
        ----------
        freq
            The new time difference between two adjacent entries in the returned TimeSeries.
            Expects a `pandas.DateOffset` or `DateOffset` alias.
        method
            Method to either aggregate grouped values (for down-sampling) or fill holes (for up-sampling)
            in the reindexed TimeSeries. For more information, see the `xarray DataArrayResample documentation
            <https://docs.xarray.dev/en/stable/generated/xarray.core.resample.DataArrayResample.html>`_.
            Supported methods: ["all", "any", "asfreq", "backfill", "bfill", "count", "ffill", "first", "interpolate",
            "last", "max", "mean", "median", "min", "nearest", "pad", "prod", "quantile", "reduce", "std", "sum",
            "var"].
        method_kwargs
            Additional keyword arguments for the specified `method`. Some methods require additional arguments.
            Xarray's errors will be raised on invalid keyword arguments.
        kwargs
            some keyword arguments for the `xarray.resample` method, notably `offset` or `base` to indicate where
            to start the resampling and avoid nan at the first value of the resampled TimeSeries
            For more information, see the `xarray resample() documentation
            <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.resample.html>`_.

        Returns
        -------
        TimeSeries
            A reindexed TimeSeries with given frequency.

        Examples
        --------
        >>> times = pd.date_range(start=pd.Timestamp("20200101233000"), periods=6, freq="15min")
        >>> pd_series = pd.Series(range(6), index=times)
        >>> ts = TimeSeries.from_series(pd_series)
        >>> print(ts.time_index)
        DatetimeIndex(['2020-01-01 23:30:00', '2020-01-01 23:45:00',
                       '2020-01-02 00:00:00', '2020-01-02 00:15:00',
                       '2020-01-02 00:30:00', '2020-01-02 00:45:00'],
                       dtype='datetime64[ns]', name='time', freq='15T')
        >>> resampled_nokwargs_ts = ts.resample(freq="1h")
        >>> print(resampled_nokwargs_ts.time_index)
        DatetimeIndex(['2020-01-01 23:00:00', '2020-01-02 00:00:00'],
                      dtype='datetime64[ns]', name='time', freq='H')
        >>> print(resampled_nokwargs_ts.values())
        [[nan]
        [ 2.]]
        >>> resampled_ts = ts.resample(freq="1h", offset=pd.Timedelta("30min"))
        >>> print(resampled_ts.time_index)
        DatetimeIndex(['2020-01-01 23:30:00', '2020-01-02 00:30:00'],
                      dtype='datetime64[ns]', name='time', freq='H')
        >>> print(resampled_ts.values())
        [[0.]
        [4.]]
        >>> resampled_ts = ts.resample(freq="1h", offset=pd.Timedelta("30min"))
        >>> downsampled_mean_ts = ts.resample(freq="30min", method="mean")
        >>> print(downsampled_mean_ts.values())
        [[0.5]
        [2.5]
        [4.5]]
        >>> downsampled_reduce_ts = ts.resample(freq="30min", method="reduce", method_args={"func":np.mean})
        >>> print(downsampled_reduce_ts.values())
        [[0.5]
        [2.5]
        [4.5]]
        """
        method_kwargs = method_kwargs or {}
        if isinstance(freq, pd.DateOffset):
            freq = freq.freqstr

        resample = self._xa.resample(
            indexer={self._time_dim: freq},
            **kwargs,
        )

        if method in SUPPORTED_RESAMPLE_METHODS:
            applied_method = getattr(xr.core.resample.DataArrayResample, method)
            new_xa = applied_method(resample, **method_kwargs)

            # Convert boolean to int as Timeseries must contain numeric values only
            # method: "all", "any"
            if new_xa.dtype == "bool":
                new_xa = new_xa.astype(int)
        else:
            raise_log(ValueError(f"Unknown method: {method}"), logger)
        return self.__class__(new_xa)

    def is_within_range(self, ts: Union[pd.Timestamp, int]) -> bool:
        """
        Check whether a given timestamp or integer is within the time interval of this time series.
        If a timestamp is provided, it does not need to be an element of the time index of the series.

        Parameters
        ----------
        ts
            The `pandas.Timestamp` (if indexed with DatetimeIndex) or integer (if indexed with RangeIndex) to check.

        Returns
        -------
        bool
            Whether `ts` is contained within the interval of this time series.
        """
        return self.time_index[0] <= ts <= self.time_index[-1]

    def map(
        self,
        fn: Union[
            Callable[[np.number], np.number],
            Callable[[Union[pd.Timestamp, int], np.number], np.number],
        ],
    ) -> Self:  # noqa: E501
        """
        Applies the function `fn` to the underlying NumPy array containing this series' values.

        Return a new TimeSeries instance. If `fn` takes 1 argument it is simply applied on the backing array
        of shape (time, n_components, n_samples).
        If it takes 2 arguments, it is applied repeatedly on the (ts, value[ts]) tuples, where
        "ts" denotes a timestamp value, and "value[ts]" denote the array of values at this timestamp, of shape
        (n_components, n_samples).

        Parameters
        ----------
        fn
            Either a function which takes a NumPy array and returns a NumPy array of same shape;
            e.g., `lambda x: x ** 2`, `lambda x: x / x.shape[0]` or `np.log`.
            It can also be a function which takes a timestamp and array, and returns a new array of same shape;
            e.g., `lambda ts, x: x / ts.days_in_month`.
            The type of `ts` is either `pd.Timestamp` (if the series is indexed with a DatetimeIndex),
            or an integer otherwise (if the series is indexed with an RangeIndex).

        Returns
        -------
        TimeSeries
            A new TimeSeries instance
        """
        if not isinstance(fn, Callable):
            raise_log(TypeError("fn should be callable"), logger)

        if isinstance(fn, np.ufunc):
            if fn.nin == 1 and fn.nout == 1:
                num_args = 1
            elif fn.nin == 2 and fn.nout == 1:
                num_args = 2
            else:
                raise_log(
                    ValueError(
                        "fn must have either one or two arguments and return a single value"
                    ),
                    logger,
                )
        else:
            try:
                num_args = len(signature(fn).parameters)
            except ValueError:
                raise_log(
                    ValueError(
                        "inspect.signature(fn) failed. Try wrapping fn in a lambda, e.g. lambda x: fn(x)"
                    ),
                    logger,
                )

        new_xa = self._xa.copy()
        if num_args == 1:  # apply fn on values directly
            new_xa.values = fn(self._xa.values)

        elif num_args == 2:  # map function uses timestamp f(timestamp, x)
            # go over shortest amount of iterations, either over time steps or components and samples
            if self.n_timesteps <= self.n_components * self.n_samples:
                new_vals = np.vstack([
                    np.expand_dims(fn(self.time_index[i], self._xa[i, :, :]), axis=0)
                    for i in range(self.n_timesteps)
                ])
            else:
                new_vals = np.stack(
                    [
                        np.column_stack([
                            fn(self.time_index, self._xa[:, i, j])
                            for j in range(self.n_samples)
                        ])
                        for i in range(self.n_components)
                    ],
                    axis=1,
                )
            new_xa.values = new_vals

        else:
            raise_log(ValueError("fn must have either one or two arguments"), logger)

        return self.__class__(new_xa)

    def window_transform(
        self,
        transforms: Union[dict, Sequence[dict]],
        treat_na: Optional[Union[str, Union[int, float]]] = None,
        forecasting_safe: Optional[bool] = True,
        keep_non_transformed: Optional[bool] = False,
        include_current: Optional[bool] = True,
        keep_names: Optional[bool] = False,
    ) -> Self:
        """
        Applies a moving/rolling, expanding or exponentially weighted window transformation over this ``TimeSeries``.

        Parameters
        ----------
        transforms
            A dictionary or a list of dictionaries.
            Each dictionary specifies a different window transform.

            The dictionaries can contain the following keys:

            :``"function"``: Mandatory. The name of one of the pandas builtin transformation functions,
                            or a callable function that can be applied to the input series.
                            Pandas' functions can be found in the
                            `documentation <https://pandas.pydata.org/docs/reference/window.html>`_.

            :``"mode"``: Optional. The name of the pandas windowing mode on which the ``"function"`` is going to be
                        applied. The options are "rolling", "expanding" and "ewm".
                        If not provided, Darts defaults to "expanding".
                        User defined functions can use either "rolling" or "expanding" modes.
                        More information on pandas windowing operations can be found in the `documentation
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html>`_.

            :``"components"``: Optional. A string or list of strings specifying the TimeSeries components on which the
                               transformation should be applied. If not specified, the transformation will be
                               applied on all components.

            :``"function_name"``: Optional. A string specifying the function name referenced as part of
                                  the transformation output name. For example, given a user-provided function
                                  transformation on rolling window size of 5 on the component "comp", the
                                  default transformation output name is "rolling_udf_5_comp" whereby "udf"
                                  refers to "user defined function". If specified, the ``"function_name"`` will
                                  replace the default name "udf". Similarly, the ``"function_name"`` will replace
                                  the name of the pandas builtin transformation function name in the output name.

            All other dictionary items provided will be treated as keyword arguments for the windowing mode
            (i.e., ``rolling/ewm/expanding``) or for the specific function
            in that mode (i.e., ``pandas.DataFrame.rolling.mean/std/max/min...`` or
            ``pandas.DataFrame.ewm.mean/std/sum``).
            This allows for more flexibility in configuring the transformation, by providing for
            example:

            * :``"window"``: Size of the moving window for the "rolling" mode.
                            If an integer, the fixed number of observations used for each window.
                            If an offset, the time period of each window with data type :class:`pandas.Timedelta`
                            representing a fixed duration.
            * :``"min_periods"``: The minimum number of observations in the window required to have a value (otherwise
                NaN). Darts reuses pandas defaults of 1 for "rolling" and "expanding" modes and of 0 for "ewm" mode.
            * :``"win_type"``: The type of weigthing to apply to the window elements.
                If provided, it should be one of `scipy.signal.windows
                <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`_.
            * :``"center"``: ``True``/``False`` to set the observation at the current timestep at the center of the
                window (when ``forecasting_safe`` is `True`, Darts enforces ``"center"`` to ``False``).
            * :``"closed"``: ``"right"``/``"left"``/``"both"``/``"neither"`` to specify whether the right,
                left or both ends of the window are included in the window, or neither of them.
                Darts defaults to pandas default of ``"right"``.

            More information on the available functions and their parameters can be found in the
            `Pandas documentation <https://pandas.pydata.org/docs/reference/window.html>`_.

            For user-provided functions, extra keyword arguments in the transformation dictionary are passed to the
            user-defined function.
            By default, Darts expects user-defined functions to receive numpy arrays as input.
            This can be modified by adding item ``"raw": False`` in the transformation dictionary.
            It is expected that the function returns a single
            value for each window. Other possible configurations can be found in the
            `pandas.DataFrame.rolling().apply()
            documentation <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html>`_
            and `pandas.DataFrame.expanding().apply()
            documentation <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.expanding.html>`_.

        treat_na
            Specifies how to treat missing values that were added by the window transformations
            at the beginning of the resulting TimeSeries. By default, Darts will leave NaNs in the resulting TimeSeries.
            This parameter can be one of the following:

            * :``"dropna"``: to truncate the TimeSeries and drop rows containing missing values.
                If multiple columns contain different numbers of missing values, only the minimum number
                of rows is dropped. This operation might reduce the length of the resulting TimeSeries.

            * :``"bfill"`` or ``"backfill"``: to specify that NaNs should be filled with the last transformed
                and valid observation. If the original TimeSeries starts with NaNs, those are kept.
                When ``forecasting_safe`` is ``True``, this option returns an exception to avoid future observation
                contaminating the past.

            * :an integer or float: in which case NaNs will be filled with this value.
                All columns will be filled with the same provided value.

        forecasting_safe
            If True, Darts enforces that the resulting TimeSeries is safe to be used in forecasting models as target
            or as feature. The window transformation will not allow future values to be included in the computations
            at their corresponding current timestep. Default is ``True``.
            "ewm" and "expanding" modes are forecasting safe by default.
            "rolling" mode is forecasting safe if ``"center": False`` is guaranteed.

        keep_non_transformed
            ``False`` to return the transformed components only, ``True`` to return all original components along
            the transformed ones. Default is ``False``. If the series has a hierarchy, must be set to ``False``.

        include_current
            ``True`` to include the current time step in the window, ``False`` to exclude it. Default is ``True``.

        keep_names
            Whether the transformed components should keep the original component names or. Must be set to ``False``
            if `keep_non_transformed = True` or the number of transformation is greater than 1.

        Returns
        -------
        TimeSeries
            Returns a new TimeSeries instance with the transformed components. If ``keep_non_transformed`` is ``True``,
            the resulting TimeSeries will contain the original non-transformed components along the transformed ones.
            If the input series is stochastic, all samples are identically transformed.
            The naming convention for the transformed components is as follows:
            [window_mode]_[function_name]_[window_size if provided]_[min_periods if not default]_[original_comp_name],
            e.g., rolling_sum_3_comp_0 (i.e., window_mode= rolling, function_name = sum, window_size=3,
            original_comp_name=comp_0) ;
            ewm_mean_comp_1 (i.e., window_mode= ewm, function_name = mean, original_comp_name=comp_1);
            expanding_sum_3_comp_2 (i.e., window_mode= expanding, function_name = sum, window_size=3,
            original_comp_name=comp_2). For user-defined functions, function_name = udf.
        """
        VALID_BFILL_NA = {"bfill", "backfill"}
        VALID_TREAT_NA = VALID_BFILL_NA.union({"dropna"})

        PD_WINDOW_OPERATIONS = {
            "rolling": pd.DataFrame.rolling,
            "expanding": pd.DataFrame.expanding,
            "ewm": pd.DataFrame.ewm,
        }

        # helper function to read and format kwargs
        def _get_kwargs(transformation, forecasting_safe):
            """
            Builds the kwargs dictionary for the transformation function.

            Parameters
            ----------
            transformation
                The transformation dictionary.
            builtins
                The built-in transformations read from the WindowTransformer class.

            Returns
            -------
            dict, dict
                The kwargs dictionaries for both the function group and the specific function.
            """

            # take expanding as the default window operation if not specified, safer than rolling
            mode = transformation.get("mode", "expanding")

            raise_if_not(
                mode in PD_WINDOW_OPERATIONS.keys(),
                f"Invalid window operation: '{mode}'. Must be one of {PD_WINDOW_OPERATIONS.keys()}.",
                logger,
            )
            window_mode = PD_WINDOW_OPERATIONS[mode]

            # minimum number of observations in window required to have a value (otherwise result in NaN)
            if "min_periods" not in transformation:
                transformation["min_periods"] = 0 if mode == "ewm" else 1

            if mode == "rolling":
                # pandas default for 'center' is False, no need to set it explicitly
                if "center" in transformation:
                    raise_if_not(
                        not (transformation["center"] and forecasting_safe),
                        "When `forecasting_safe` is True, `center` must be False.",
                        logger,
                    )

            if isinstance(transformation["function"], Callable):
                fn = "apply"
                udf = transformation["function"]
                # make sure that we provide a numpy array to the user function, "raw": True
                if "raw" not in transformation:
                    transformation["raw"] = True
            elif isinstance(transformation["function"], str):
                fn = transformation["function"]
            else:
                raise_log(
                    ValueError(
                        "Transformation function must be a string or a callable. "
                        "String can be the name of any function available for pandas window. "
                        "A list of those function can be found in the `documentation "
                        "<https://pandas.pydata.org/pandas-docs/stable/reference/window.html>`."
                    ),
                    logger,
                )

            available_keys = set(transformation.keys()) - {
                "function",
                "group",
                "components",
                "function_name",
            }

            window_mode_expected_args = set(window_mode.__code__.co_varnames)
            window_mode_available_keys = window_mode_expected_args.intersection(
                available_keys
            )

            window_mode_available_kwargs = {
                k: v
                for k, v in transformation.items()
                if k in window_mode_available_keys
            }

            available_keys -= window_mode_available_keys

            function_expected_args = set(
                getattr(
                    getattr(pd.DataFrame(), window_mode.__name__)(
                        **window_mode_available_kwargs
                    ),
                    fn,
                ).__code__.co_varnames
            )

            function_available_keys = function_expected_args.intersection(
                set(available_keys)
            )

            function_available_kwargs = {
                k: v for k, v in transformation.items() if k in function_available_keys
            }

            available_keys -= function_available_keys

            udf_expected_args = set(udf.__code__.co_varnames) if fn == "apply" else None
            udf_available_keys = (
                udf_expected_args.intersection(set(available_keys))
                if fn == "apply"
                else None
            )

            udf_kwargs = (
                {k: v for k, v in transformation.items() if k in udf_available_keys}
                if fn == "apply"
                else None
            )

            function_available_kwargs.update(
                {"func": udf, "kwargs": udf_kwargs} if fn == "apply" else {}
            )

            return (window_mode.__name__, window_mode_available_kwargs), (
                fn,
                function_available_kwargs,
            )

        # make sure we have a list in transforms
        if isinstance(transforms, dict):
            transforms = [transforms]

        # check if some transformations are applied to the same components
        overlapping_transforms = False
        transformed_components = set()
        for tr in transforms:
            if not isinstance(tr, dict):
                raise_log(
                    ValueError("Every entry in `transforms` must be a dictionary"),
                    logger,
                )
            tr_comps = set(tr["components"] if "components" in tr else self.components)
            if len(transformed_components.intersection(tr_comps)) > 0:
                overlapping_transforms = True
            transformed_components = transformed_components.union(tr_comps)

        if keep_names and overlapping_transforms:
            raise_log(
                ValueError(
                    "Cannot keep the original component names as some transforms are overlapping "
                    "(applied to the same components). Set `keep_names` to `False`."
                ),
                logger,
            )

        # actually, this could be allowed to allow transformation "in place"?
        # keep_non_transformed can be changed to False/ignored if the transforms are not partial
        if keep_names and keep_non_transformed:
            raise_log(
                ValueError(
                    "`keep_names = True` and `keep_non_transformed = True` cannot be used together."
                ),
                logger,
            )

        partial_transforms = transformed_components != set(self.components)
        new_hierarchy = None
        convert_hierarchy = False
        comp_names_map = dict()
        if self.hierarchy:
            # the partial_transform covers for scenario keep_non_transformed = True
            if len(transforms) > 1 or partial_transforms:
                logger.warning(
                    "The hierarchy cannot be retained, either because there is more than one transform or "
                    "because the transform is not applied to all the components of the series."
                )
            else:
                convert_hierarchy = True

        raise_if_not(
            all([isinstance(tr, dict) for tr in transforms]),
            "`transforms` must be a non-empty dictionary or a non-empty list of dictionaries.",
        )

        # read series dataframe
        ts_df = self.to_dataframe(copy=False, suppress_warnings=True)

        # store some original attributes of the series
        original_components = self.components
        n_samples = self.n_samples
        original_index = self._time_index

        resulting_transformations = pd.DataFrame()
        new_columns = []
        added_na = []

        # run through all transformations in transforms
        for transformation in transforms:
            if "components" in transformation:
                if isinstance(transformation["components"], str):
                    transformation["components"] = [transformation["components"]]
                comps_to_transform = transformation["components"]

            else:
                comps_to_transform = original_components

            df_cols = ts_df.columns

            if not self.is_deterministic:
                filter_df_columns = [
                    df_col
                    for df_col in df_cols
                    if re.sub("_s.*$", "", df_col) in comps_to_transform
                ]

            else:
                filter_df_columns = [df_col for df_col in comps_to_transform]

            (window_mode, window_mode_kwargs), (fn, function_kwargs) = _get_kwargs(
                transformation, forecasting_safe
            )

            closed = transformation.get("closed", None)
            if not include_current:
                if window_mode == "rolling":
                    shifts = 0 if closed == "left" else 1  # avoid shifting twice
                else:
                    shifts = 1
            else:
                shifts = 0

            resulting_transformations = pd.concat(
                [
                    resulting_transformations,
                    getattr(
                        getattr(ts_df[filter_df_columns], window_mode)(
                            **window_mode_kwargs
                        ),
                        fn,
                    )(**function_kwargs).shift(periods=shifts),
                ],
                axis=1,
            )
            min_periods = transformation["min_periods"]
            # set new columns names
            fn_name = transformation.get("function_name")
            if fn_name:
                function_name = fn_name
            else:
                function_name = fn if fn != "apply" else "udf"
            name_prefix = (
                f"{window_mode}_{function_name}"
                f"{'_' + str(transformation['window']) if 'window' in transformation else ''}"
                f"{'_' + str(min_periods) if min_periods > 1 else ''}"
            )

            if keep_names:
                new_columns.extend(comps_to_transform)
            else:
                names_w_prefix = [
                    f"{name_prefix}_{comp_name}" for comp_name in comps_to_transform
                ]
                new_columns.extend(names_w_prefix)
                if convert_hierarchy:
                    comp_names_map.update({
                        c_name: new_c_name
                        for c_name, new_c_name in zip(
                            comps_to_transform, names_w_prefix
                        )
                    })

            # track how many NaN rows are added by each transformation on each transformed column
            # NaNs would appear only if user changes "min_periods" to else than 1, if not,
            # by default there should be no NaNs unless the original series starts with NaNs (those would be maintained)
            total_na = min_periods + shifts + (closed == "left")
            added_na.extend([
                total_na - 1 if min_periods > 0 else total_na for _ in filter_df_columns
            ])

        # keep all original components
        if keep_non_transformed:
            resulting_transformations = pd.concat(
                [resulting_transformations, ts_df], axis=1
            )
            new_columns.extend(original_components)

        # Treat NaNs that were introduced by the transformations only
        # Default to leave NaNs
        if isinstance(treat_na, str):
            raise_if_not(
                treat_na in VALID_TREAT_NA,
                f"`treat_na` must be one of {VALID_TREAT_NA} or a scalar, but found {treat_na}",
                logger,
            )

            raise_if_not(
                not (treat_na in VALID_BFILL_NA and forecasting_safe),
                "when `forecasting_safe` is True, back filling NaNs is not allowed as "
                "it risks contaminating past time steps with future values.",
                logger,
            )

        if isinstance(treat_na, (int, float)) or (treat_na in VALID_BFILL_NA):
            for i in range(0, len(added_na), n_samples):
                s_idx = added_na[i : (i + n_samples)][0]
                value = (
                    treat_na
                    if isinstance(treat_na, (int, float))
                    else resulting_transformations.values[s_idx, i : (i + n_samples)]
                )
                resulting_transformations.iloc[:s_idx, i : (i + n_samples)] = value
        elif treat_na == "dropna":
            # can only drop the NaN rows that are common among the columns
            drop_before_index = original_index[np.min(added_na)]
            resulting_transformations = resulting_transformations.loc[
                drop_before_index:
            ]

        # revert dataframe to TimeSeries
        new_index = original_index.__class__(resulting_transformations.index)

        if convert_hierarchy:
            if keep_names:
                new_hierarchy = self.hierarchy
            else:
                new_hierarchy = {
                    comp_names_map[k]: [comp_names_map[old_name] for old_name in v]
                    for k, v in self.hierarchy.items()
                }

        transformed_time_series = TimeSeries.from_times_and_values(
            times=new_index,
            values=resulting_transformations.values.reshape(
                len(new_index), -1, n_samples
            ),
            columns=new_columns,
            static_covariates=self.static_covariates,
            hierarchy=new_hierarchy,
            metadata=self.metadata,
        )

        return transformed_time_series

    def to_json(self) -> str:
        """
        Return a JSON string representation of this deterministic series.

        At the moment this function works only on deterministic time series (i.e., made of 1 sample).

        Notes
        -----
        Static covariates are not returned in the JSON string. When using `TimeSeries.from_json()`, the static
        covariates can be added with input argument `static_covariates`.

        Returns
        -------
        str
            A JSON String representing the time series
        """
        return self.to_dataframe().to_json(orient="split", date_format="iso")

    def to_csv(self, *args, **kwargs):
        """
        Writes this deterministic series to a CSV file.
        For a list of parameters, refer to the documentation of :func:`pandas.DataFrame.to_csv()` [1]_.

        References
        ----------
        .. [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html?highlight=to_csv
        """
        if not self.is_deterministic:
            raise_log(
                ValueError(
                    "Writing to csv is only supported for deterministic time series "
                    "(a series with only one sample per time and component)."
                )
            )

        self.to_dataframe().to_csv(*args, **kwargs)

    def to_pickle(self, path: str, protocol: int = pickle.HIGHEST_PROTOCOL):
        """
        Save this series in pickle format.

        Parameters
        ----------
        path : string
            path to a file where current object will be pickled
        protocol : integer, default highest
            pickling protocol. The default is best in most cases, use it only if having backward compatibility issues

        Notes
        -----
        Xarray docs [1]_ suggest not using pickle as a long-term data storage.

        References
        ----------
        .. [1] http://xarray.pydata.org/en/stable/user-guide/io.html#pickle
        """

        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=protocol)

    def plot(
        self,
        new_plot: bool = False,
        central_quantile: Union[float, str] = 0.5,
        low_quantile: Optional[float] = 0.05,
        high_quantile: Optional[float] = 0.95,
        default_formatting: bool = True,
        title: Optional[str] = None,
        label: Optional[Union[str, Sequence[str]]] = "",
        max_nr_components: int = 10,
        ax: Optional[matplotlib.axes.Axes] = None,
        alpha: Optional[float] = None,
        color: Optional[Union[str, tuple, Sequence[str, tuple]]] = None,
        c: Optional[Union[str, tuple, Sequence[str, tuple]]] = None,
        *args,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the series.

        This is a wrapper method around :func:`xarray.DataArray.plot()`.

        Parameters
        ----------
        new_plot
            Whether to spawn a new axis to plot on. See also parameter `ax`.
        central_quantile
            The quantile (between 0 and 1) to plot as a "central" value, if the series is stochastic (i.e., if
            it has multiple samples). This will be applied on each component separately (i.e., to display quantiles
            of the components' marginal distributions). For instance, setting `central_quantile=0.5` will plot the
            median of each component. `central_quantile` can also be set to 'mean'.
        low_quantile
            The quantile to use for the lower bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `confidence_low_quantile` is None (default 0.05).
        high_quantile
            The quantile to use for the upper bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `high_quantile` is None (default 0.95).
        default_formatting
            Whether to use the darts default scheme.
        title
            Optionally, a custom plot title. If `None`, will use the name of the underlying `xarray.DataArray`.
        label
            Can either be a string or list of strings. If a string and the series only has a single component, it is
            used as the label for that component. If a string and the series has multiple components, it is used as
            a prefix for each component name. If a list of strings with length equal to the number of components in
            the series, the labels will be mapped to the components in order.
        max_nr_components
            The maximum number of components of a series to plot. -1 means all components will be plotted.
        ax
            Optionally, an axis to plot on. If `None`, and `new_plot=False`, will use the current axis. If
            `new_plot=True`, will create a new axis.
        alpha
            Optionally, set the line alpha for deterministic series, or the confidence interval alpha for
            probabilistic series.
        color
            Can either be a single color or list of colors. Any matplotlib color is accepted (string, hex string,
            RGB/RGBA tuple). If a single color and the series only has a single component, it is used as the color
            for that component. If a single color and the series has multiple components, it is used as the color
            for each component. If a list of colors with length equal to the number of components in the series, the
            colors will be mapped to the components in order.
        c
            An alias for `color`.
        args
            some positional arguments for the `plot()` method
        kwargs
            some keyword arguments for the `plot()` method

        Returns
        -------
        matplotlib.axes.Axes
            Either the passed `ax` axis, a newly created one if `new_plot=True`, or the existing one.
        """
        alpha_confidence_intvls = 0.25

        if central_quantile != "mean":
            raise_if_not(
                isinstance(central_quantile, float) and 0.0 <= central_quantile <= 1.0,
                'central_quantile must be either "mean", or a float between 0 and 1.',
                logger,
            )

        if high_quantile is not None and low_quantile is not None:
            raise_if_not(
                0.0 <= low_quantile <= 1.0 and 0.0 <= high_quantile <= 1.0,
                "confidence interval low and high quantiles must be between 0 and 1.",
                logger,
            )

        if max_nr_components == -1:
            n_components_to_plot = self.n_components
        else:
            n_components_to_plot = min(self.n_components, max_nr_components)

        if self.n_components > n_components_to_plot:
            logger.warning(
                f"Number of series components ({self.n_components}) is larger than the maximum number of "
                f"components to plot ({max_nr_components}). Plotting only the first `{max_nr_components}` "
                f"components. You can adjust the number of components to plot using `max_nr_components`."
            )

        if not isinstance(label, str) and isinstance(label, Sequence):
            if len(label) != self.n_components and len(label) != n_components_to_plot:
                raise_log(
                    ValueError(
                        f"The `label` sequence must have the same length as the number of series components "
                        f"({self.n_components}) or as the number of plotted components ({n_components_to_plot}). "
                        f"Received length `{len(label)}`."
                    ),
                    logger,
                )
            custom_labels = True
        else:
            custom_labels = False

        if color and c:
            raise_log(
                ValueError(
                    "`color` and `c` must not be used simultaneously, use one or the other."
                ),
                logger,
            )
        color = color or c
        if not isinstance(color, (str, tuple)) and isinstance(color, Sequence):
            if len(color) != self.n_components and len(color) != n_components_to_plot:
                raise_log(
                    ValueError(
                        f"The `color` sequence must have the same length as the number of series components "
                        f"({self.n_components}) or as the number of plotted components ({n_components_to_plot}). "
                        f"Received length `{len(label)}`."
                    ),
                    logger,
                )
            custom_colors = True
        else:
            custom_colors = False

        kwargs["alpha"] = alpha
        if not any(lw in kwargs for lw in ["lw", "linewidth"]):
            kwargs["lw"] = 2

        if new_plot:
            fig, ax = plt.subplots()
        else:
            if ax is None:
                ax = plt.gca()

        for i, c in enumerate(self._xa.component[:n_components_to_plot]):
            comp_name = str(c.values)
            comp = self._xa.sel(component=c)

            if comp.sample.size > 1:
                if central_quantile == "mean":
                    central_series = comp.mean(dim=DIMS[2])
                else:
                    central_series = comp.quantile(q=central_quantile, dim=DIMS[2])
            else:
                central_series = comp.mean(dim=DIMS[2])

            if custom_labels:
                label_to_use = label[i]
            else:
                if label == "":
                    label_to_use = comp_name
                elif len(self.components) == 1:
                    label_to_use = label
                else:
                    label_to_use = f"{label}_{comp_name}"
            kwargs["label"] = label_to_use
            kwargs["c"] = color[i] if custom_colors else color

            kwargs_central = deepcopy(kwargs)
            if not self.is_deterministic:
                kwargs_central["alpha"] = 1
            if central_series.shape[0] > 1:
                p = central_series.plot(*args, ax=ax, **kwargs_central)
            # empty TimeSeries
            elif central_series.shape[0] == 0:
                p = ax.plot(
                    [],
                    [],
                    *args,
                    **kwargs_central,
                )
                ax.set_xlabel(self.time_index.name)
            else:
                p = ax.plot(
                    [self.start_time()],
                    central_series.values[0],
                    "o",
                    *args,
                    **kwargs_central,
                )
            color_used = p[0].get_color() if default_formatting else None

            # Optionally show confidence intervals
            if (
                comp.sample.size > 1
                and low_quantile is not None
                and high_quantile is not None
            ):
                low_series = comp.quantile(q=low_quantile, dim=DIMS[2])
                high_series = comp.quantile(q=high_quantile, dim=DIMS[2])
                if low_series.shape[0] > 1:
                    ax.fill_between(
                        self.time_index,
                        low_series,
                        high_series,
                        color=color_used,
                        alpha=(alpha if alpha is not None else alpha_confidence_intvls),
                    )
                else:
                    ax.plot(
                        [self.start_time(), self.start_time()],
                        [low_series.values[0], high_series.values[0]],
                        "-+",
                        color=color_used,
                        lw=2,
                    )

        ax.legend()
        ax.set_title(title if title is not None else self._xa.name)
        return ax

    def with_columns_renamed(
        self, col_names: Union[list[str], str], col_names_new: Union[list[str], str]
    ) -> Self:
        """
        Return a new ``TimeSeries`` instance with new columns/components names. It also
        adapts the names in the hierarchy, if any.

        Parameters
        -------
        col_names
            String or list of strings corresponding the the column names to be changed.
        col_names_new
            String or list of strings corresponding to the new column names. Must be the same length as col_names.

        Returns
        -------
        TimeSeries
            A new TimeSeries instance.
        """

        if isinstance(col_names, str):
            col_names = [col_names]
        if isinstance(col_names_new, str):
            col_names_new = [col_names_new]

        raise_if_not(
            all([(x in self.columns.to_list()) for x in col_names]),
            "Some column names in col_names don't exist in the time series.",
            logger,
        )

        raise_if_not(
            len(col_names) == len(col_names_new),
            "Length of col_names_new list should be"
            " equal to the length of col_names list.",
            logger,
        )

        old2new = {old: new for (old, new) in zip(col_names, col_names_new)}

        # update component names
        cols = [old2new[old] if old in old2new else old for old in self.components]

        # update hierarchy names
        if self.hierarchy is not None:
            hierarchy = {
                (old2new[key] if key in old2new else key): [
                    old2new[old] if old in old2new else old
                    for old in self.hierarchy[key]
                ]
                for key in self.hierarchy
            }
        else:
            hierarchy = None
        new_attrs = self._xa.attrs
        new_attrs[HIERARCHY_TAG] = hierarchy

        new_xa = xr.DataArray(
            self._xa.values,
            dims=self._xa.dims,
            coords={self._xa.dims[0]: self.time_index, DIMS[1]: pd.Index(cols)},
            attrs=new_attrs,
        )

        return self.__class__(new_xa)

    """
    Simple statistic and aggregation functions. Calculate various statistics over the samples of stochastic time series
    or aggregate over components/time for deterministic series.
    """

    def _get_agg_coords(self, new_cname: str, axis: int) -> dict:
        """Helper function to rename reduced axis. Returns a dictionary containing the new coordinates"""
        if axis == 0:  # set time_index to first day
            return {self._xa.dims[0]: self.time_index[0:1], DIMS[1]: self.components}
        elif axis == 1:  # rename components
            return {self._xa.dims[0]: self.time_index, DIMS[1]: pd.Index([new_cname])}
        elif axis == 2:  # do nothing
            return {self._xa.dims[0]: self.time_index, DIMS[1]: self.components}

    def mean(self, axis: int = 2) -> Self:
        """
        Return a ``TimeSeries`` containing the mean calculated over the specified axis.

        If we reduce over time (``axis=0``), the resulting ``TimeSeries`` will have length one and will use the first
        entry of the original ``time_index``. If we perform the calculation over the components (``axis=1``), the
        resulting single component will be renamed to "components_mean".  When applied to the samples (``axis=2``),
        a deterministic ``TimeSeries`` is returned.

        If ``axis=1``, the static covariates and the hierarchy are discarded from the series.

        Parameters
        ----------
        axis
            The axis to reduce over. The default is to calculate over samples, i.e. axis=2.

        Returns
        -------
        TimeSeries
            A new TimeSeries with mean applied to the indicated axis.
        """
        new_data = self._xa.values.mean(axis=axis, keepdims=True)

        new_coords = self._get_agg_coords("components_mean", axis)

        new_xa = xr.DataArray(
            new_data,
            dims=self._xa.dims,
            coords=new_coords,
            attrs=(self._xa.attrs if axis != 1 else dict()),
        )
        return self.__class__(new_xa)

    def median(self, axis: int = 2) -> Self:
        """
        Return a ``TimeSeries`` containing the median calculated over the specified axis.

        If we reduce over time (``axis=0``), the resulting ``TimeSeries`` will have length one and will use the first
        entry of the original ``time_index``. If we perform the calculation over the components (``axis=1``), the
        resulting single component will be renamed to "components_median".  When applied to the samples (``axis=2``),
        a deterministic ``TimeSeries`` is returned.

        If ``axis=1``, the static covariates and the hierarchy are discarded from the series.

        Parameters
        ----------
        axis
            The axis to reduce over. The default is to calculate over samples, i.e. axis=2.

        Returns
        -------
        TimeSeries
            A new TimeSeries with median applied to the indicated axis.
        """
        new_data = np.median(
            self._xa.values, axis=axis, overwrite_input=False, keepdims=True
        )
        new_coords = self._get_agg_coords("components_median", axis)

        new_xa = xr.DataArray(
            new_data,
            dims=self._xa.dims,
            coords=new_coords,
            attrs=(self._xa.attrs if axis != 1 else dict()),
        )
        return self.__class__(new_xa)

    def sum(self, axis: int = 2) -> Self:
        """
        Return a ``TimeSeries`` containing the sum calculated over the specified axis.

        If we reduce over time (``axis=0``), the resulting ``TimeSeries`` will have length one and will use the first
        entry of the original ``time_index``. If we perform the calculation over the components (``axis=1``), the
        resulting single component will be renamed to "components_sum".  When applied to the samples (``axis=2``),
        a deterministic ``TimeSeries`` is returned.

        If ``axis=1``, the static covariates and the hierarchy are discarded from the series.

        Parameters
        ----------
        axis
            The axis to reduce over. The default is to calculate over samples, i.e. axis=2.

        Returns
        -------
        TimeSeries
            A new TimeSeries with sum applied to the indicated axis.
        """
        new_data = self._xa.values.sum(axis=axis, keepdims=True)

        new_coords = self._get_agg_coords("components_sum", axis)

        new_xa = xr.DataArray(
            new_data,
            dims=self._xa.dims,
            coords=new_coords,
            attrs=(self._xa.attrs if axis != 1 else dict()),
        )
        return self.__class__(new_xa)

    def min(self, axis: int = 2) -> Self:
        """
        Return a ``TimeSeries`` containing the min calculated over the specified axis.

        If we reduce over time (``axis=0``), the resulting ``TimeSeries`` will have length one and will use the first
        entry of the original ``time_index``. If we perform the calculation over the components (``axis=1``), the
        resulting single component will be renamed to "components_min".  When applied to the samples (``axis=2``),
        a deterministic ``TimeSeries`` is returned.

        If ``axis=1``, the static covariates and the hierarchy are discarded from the series.

        Parameters
        ----------
        axis
            The axis to reduce over. The default is to calculate over samples, i.e. axis=2.

        Returns
        -------
        TimeSeries
            A new TimeSeries with min applied to the indicated axis.
        """

        new_data = self._xa.values.min(axis=axis, keepdims=True)
        new_coords = self._get_agg_coords("components_min", axis)

        new_xa = xr.DataArray(
            new_data,
            dims=self._xa.dims,
            coords=new_coords,
            attrs=(self._xa.attrs if axis != 1 else dict()),
        )
        return self.__class__(new_xa)

    def max(self, axis: int = 2) -> Self:
        """
        Return a ``TimeSeries`` containing the max calculated over the specified axis.

        If we reduce over time (``axis=0``), the resulting ``TimeSeries`` will have length one and will use the first
        entry of the original ``time_index``. If we perform the calculation over the components (``axis=1``), the
        resulting single component will be renamed to "components_max".  When applied to the samples (``axis=2``),
        a deterministic ``TimeSeries`` is returned.

        If ``axis=1``, the static covariates and the hierarchy are discarded from the series.

        Parameters
        ----------
        axis
            The axis to reduce over. The default is to calculate over samples, i.e. axis=2.

        Returns
        -------
        TimeSeries
            A new TimeSeries with max applied to the indicated axis.
        """
        new_data = self._xa.values.max(axis=axis, keepdims=True)
        new_coords = self._get_agg_coords("components_max", axis)

        new_xa = xr.DataArray(
            new_data,
            dims=self._xa.dims,
            coords=new_coords,
            attrs=(self._xa.attrs if axis != 1 else dict()),
        )
        return self.__class__(new_xa)

    def var(self, ddof: int = 1) -> Self:
        """
        Return a deterministic ``TimeSeries`` containing the variance of each component
        (over the samples) of this stochastic ``TimeSeries``.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        ddof
            "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof where N represents the
            number of elements. By default, ddof is 1.

        Returns
        -------
        TimeSeries
            The TimeSeries containing the variance for each component.
        """
        self._assert_stochastic()
        new_data = self._xa.values.var(axis=2, ddof=ddof, keepdims=True)
        new_xa = xr.DataArray(
            new_data, dims=self._xa.dims, coords=self._xa.coords, attrs=self._xa.attrs
        )
        return self.__class__(new_xa)

    def std(self, ddof: int = 1) -> Self:
        """
        Return a deterministic ``TimeSeries`` containing the standard deviation of each component
        (over the samples) of this stochastic ``TimeSeries``.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        ddof
            "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof where N represents the
            number of elements. By default, ddof is 1.

        Returns
        -------
        TimeSeries
            The TimeSeries containing the standard deviation for each component.
        """
        self._assert_stochastic()
        new_data = self._xa.values.std(axis=2, ddof=ddof, keepdims=True)
        new_xa = xr.DataArray(
            new_data, dims=self._xa.dims, coords=self._xa.coords, attrs=self._xa.attrs
        )
        return self.__class__(new_xa)

    def skew(self, **kwargs) -> Self:
        """
        Return a deterministic ``TimeSeries`` containing the skew of each component
        (over the samples) of this stochastic ``TimeSeries``.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        kwargs
            Other keyword arguments are passed down to `scipy.stats.skew()`

        Returns
        -------
        TimeSeries
            The TimeSeries containing the skew for each component.
        """
        self._assert_stochastic()
        new_data = np.expand_dims(skew(self._xa.values, axis=2, **kwargs), axis=2)
        new_xa = xr.DataArray(
            new_data, dims=self._xa.dims, coords=self._xa.coords, attrs=self._xa.attrs
        )
        return self.__class__(new_xa)

    def kurtosis(self, **kwargs) -> Self:
        """
        Return a deterministic ``TimeSeries`` containing the kurtosis of each component
        (over the samples) of this stochastic ``TimeSeries``.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        kwargs
            Other keyword arguments are passed down to `scipy.stats.kurtosis()`

        Returns
        -------
        TimeSeries
            The TimeSeries containing the kurtosis for each component.
        """
        self._assert_stochastic()
        new_data = np.expand_dims(kurtosis(self._xa.values, axis=2, **kwargs), axis=2)
        new_xa = xr.DataArray(
            new_data, dims=self._xa.dims, coords=self._xa.coords, attrs=self._xa.attrs
        )
        return self.__class__(new_xa)

    def quantile(self, quantile: float, **kwargs) -> Self:
        """
        Return a deterministic ``TimeSeries`` containing the single desired quantile of each component
        (over the samples) of this stochastic ``TimeSeries``.

        The components in the new series are named "<component>_X", where "<component>"
        is the column name corresponding to this component, and "X" is the quantile value.
        The quantile columns represent the marginal distributions of the components of this series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        quantile
            The desired quantile value. The value must be represented as a fraction
            (between 0 and 1 inclusive). For instance, `0.5` will return a TimeSeries
            containing the median of the (marginal) distribution of each component.
        kwargs
            Other keyword arguments are passed down to `numpy.quantile()`

        Returns
        -------
        TimeSeries
            The TimeSeries containing the desired quantile for each component.
        """
        return self.quantile_timeseries(quantile, **kwargs)

    """
    Dunder methods
    """

    def _combine_arrays(
        self,
        other: Union[Self, xr.DataArray, np.ndarray],
        combine_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Self:
        """
        This is a helper function that allows us to combine this series with another one,
        directly applying an operation on their underlying numpy arrays.
        """

        if isinstance(other, TimeSeries):
            other_vals = other.data_array(copy=False).values
        elif isinstance(other, xr.DataArray):
            other_vals = other.values
        else:
            other_vals = other

        t, c, s = self._xa.shape
        other_shape = other_vals.shape
        if not (
            # can combine arrays if shapes are equal (t, c, s)
            other_shape == (t, c, s)
            # or broadcast [t, 1, 1] onto [t, c, s]
            or other_shape == (t, 1, 1)
            # or broadcast [t, c, 1] onto [t, c, s]
            or other_shape == (t, c, 1)
            # or broadcast [t, 1, s] onto [t, c, s]
            or other_shape == (t, 1, s),
        ):
            raise_log(
                ValueError(
                    "Attempted to perform operation on two TimeSeries of unequal shapes."
                ),
                logger=logger,
            )
        new_xa = self._xa.copy()
        new_xa.values = combine_fn(new_xa.values, other_vals)
        return self.__class__(new_xa)

    @classmethod
    def _fill_missing_dates(
        cls, xa: xr.DataArray, freq: Optional[Union[str, int]] = None
    ) -> xr.DataArray:
        """Return an xarray DataArray instance with missing dates inserted from an input xarray DataArray.
        The first dimension of the input DataArray `xa` has to be the time dimension.

        This requires either a provided `freq` or the possibility to infer a unique frequency (see
        `offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
        for more info on supported frequencies) from the provided timestamps.

        Parameters
        ----------
        xa
            The xarray DataArray
        freq
            Optionally, a string representing the frequency of the Pandas DateTimeIndex to fill in the missing dates.

        Raises
        -------
        ValueError
            If `xa`'s DateTimeIndex contains less than 3 elements;
            if no unique frequency can be inferred from `xa`'s DateTimeIndex;
            if the resampled DateTimeIndex does not contain all dates from `xa` (see
                :meth:`_restore_xarray_from_frequency() <TimeSeries._restore_xarray_from_frequency>`)

        Returns
        -------
        xarray DataArray
            xarray DataArray with filled missing dates from `xa`
        """

        if freq is not None:
            return cls._restore_xarray_from_frequency(xa, freq)

        raise_if(
            len(xa) <= 2,
            "Input time series must be of (length>=3) when fill_missing_dates=True and freq=None.",
            logger,
        )

        time_dim = xa.dims[0]
        sorted_xa = cls._sort_index(xa, copy=False)
        time_index: Union[pd.Index, pd.RangeIndex, pd.DatetimeIndex] = (
            sorted_xa.get_index(time_dim)
        )

        if isinstance(time_index, pd.DatetimeIndex):
            has_datetime_index = True
            observed_freqs = cls._observed_freq_datetime_index(time_index)
        else:  # integer index (non RangeIndex)
            has_datetime_index = False
            observed_freqs = cls._observed_freq_integer_index(time_index)

        offset_alias_info = (
            (
                " For more information about frequency aliases, read "
                "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases"
            )
            if has_datetime_index
            else ""
        )
        if not len(observed_freqs) == 1:
            raise_log(
                ValueError(
                    f"Could not observe an inferred frequency. An explicit frequency must be evident over a span of "
                    f"at least 3 consecutive time stamps in the input data. {offset_alias_info}"
                    if not len(observed_freqs)
                    else f"Could not find a unique inferred frequency (not constant). Observed frequencies: "
                    f"{observed_freqs}. If any of those is the actual frequency, try passing it with "
                    f"`fill_missing_dates=True` and `freq=your_frequency`.{offset_alias_info}"
                ),
                logger,
            )

        freq = observed_freqs.pop()

        return cls._restore_xarray_from_frequency(sorted_xa, freq)

    @staticmethod
    def _sort_index(xa: xr.DataArray, copy: bool = True) -> xr.DataArray:
        """Sorts an xarray by its time dimension index (only if it is not already monotonically increasing)."""
        time_dim = xa.dims[0]
        return (
            (xa.copy() if copy else xa)
            if xa.get_index(time_dim).is_monotonic_increasing
            else xa.sortby(time_dim)
        )

    @staticmethod
    def _observed_freq_datetime_index(index: pd.DatetimeIndex) -> set:
        """Returns all observed/inferred frequencies of a pandas DatetimeIndex. The frequencies are inferred from all
        combinations of three adjacent time steps
        """
        # find unique time deltas indices from three consecutive time stamps
        _, unq_td_index = np.unique(
            np.stack([(index[1:-1] - index[:-2]), (index[2:] - index[1:-1])]),
            return_index=True,
            axis=1,
        )

        # for each unique index, take one example including the left time stamp, and one including the right
        steps = np.column_stack([index[unq_td_index + i] for i in range(3)])

        # find all unique inferred frequencies
        observed_freqs = {pd.infer_freq(step) for step in steps}
        observed_freqs.discard(None)
        return observed_freqs

    @staticmethod
    def _observed_freq_integer_index(index: pd.Index) -> set:
        """Returns all observed/inferred frequencies of a pandas Index (integer index). The frequencies are inferred
        from all differences between two adjacent indices.
        """
        return set(index[1:] - index[:-1])

    @classmethod
    def _integer_to_range_indexed_xarray(cls, xa: xr.DataArray) -> xr.DataArray:
        """If possible, converts an integer indexed xarray DataArray to a range indexed (pd.RangeIndex) DataArray.
        Otherwise, raises an error. An integer Index can be converted to a pd.RangeIndex, if the sorted integer index
        has a constant step size.
        """
        time_dim = xa.dims[0]
        sorted_xa = cls._sort_index(xa, copy=False)
        time_index = sorted_xa.get_index(time_dim)
        observed_freqs = cls._observed_freq_integer_index(time_index)
        raise_if_not(
            len(observed_freqs) == 1,
            f"Could not convert integer index to a pd.RangeIndex. Found non-unique step sizes/frequencies: "
            f"{observed_freqs}. If any of those is the actual frequency, try passing it with fill_missing_dates=True "
            f"and freq=your_frequency.",
            logger,
        )
        freq = observed_freqs.pop()
        idx = pd.RangeIndex(
            start=min(time_index),
            stop=max(time_index) + freq,
            step=freq,
            name=time_index.name,
        )
        coords = {
            str(xa.dims[0]): idx,
            str(xa.dims[1]): xa.coords[DIMS[1]],
        }
        return xr.DataArray(
            data=sorted_xa.data,
            dims=xa.dims,
            coords=coords,
            attrs=xa.attrs,
        )

    @classmethod
    def _restore_xarray_from_frequency(
        cls, xa: xr.DataArray, freq: Union[str, int]
    ) -> xr.DataArray:
        """Return an xarray DataArray instance that is resampled from an input xarray DataArray `xa` with frequency
        `freq`. `freq` should be the inferred or actual frequency of `xa`. All data from `xa` is maintained in the
        output DataArray at the corresponding dates. Any missing dates from `xa` will be inserted into the returned
        DataArray with np.nan values.

        The first dimension of the input DataArray `xa` has to be the time dimension.

        This requires a provided frequency/step size `freq`

        Parameters
        ----------
        xa
            The xarray DataArray
        freq
            If a string, represents the actual or inferred frequency of the pandas DatetimeIndex from `xa` (see
            `offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            for more info on supported frequencies).
            If an integer, represents the actual or inferred step size of the pandas Index or pandas RangeIndex from
            `xa`.

        Raises
        -------
        ValueError
            If the resampled/reindexed DateTimeIndex/RangeIndex does not contain all dates from `xa`

        Returns
        -------
        xarray DataArray
            xarray DataArray resampled from `xa` with `freq` including all data from `xa` and inserted missing dates
        """

        time_dim = xa.dims[0]
        sorted_xa = cls._sort_index(xa, copy=False)

        time_index = sorted_xa.get_index(time_dim)
        resampled_time_index = pd.Series(index=time_index, dtype="object")
        if isinstance(time_index, pd.DatetimeIndex):
            has_datetime_index = True
            resampled_time_index = resampled_time_index.asfreq(freq)
        else:  # integer index (non RangeIndex) -> resampled to RangeIndex
            has_datetime_index = False
            resampled_time_index = resampled_time_index.reindex(
                range(min(time_index), max(time_index) + freq, freq)
            )
        # check if new time index with inferred frequency contains all input data
        contains_all_data = time_index.isin(resampled_time_index.index).all()

        offset_alias_info = (
            (
                " For more information about frequency aliases, read "
                "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases"
            )
            if has_datetime_index
            else ""
        )
        if not contains_all_data:
            raise_log(
                ValueError(
                    f"Could not correctly fill missing {'dates' if has_datetime_index else 'indices'} with the "
                    f"observed/passed {'frequency' if has_datetime_index else 'step size'} `freq='{freq}'`. "
                    f"Not all input {'time stamps' if has_datetime_index else 'indices'} contained in the newly "
                    f"created TimeSeries.{offset_alias_info}"
                ),
                logger,
            )

        coords = {
            str(xa.dims[0]): resampled_time_index.index,
            str(xa.dims[1]): xa.coords[DIMS[1]],
        }

        # convert to float as for instance integer arrays cannot accept nans
        dtype = (
            xa.dtype
            if (
                np.issubdtype(xa.values.dtype, np.float32)
                or np.issubdtype(xa.values.dtype, np.float64)
            )
            else np.float64
        )
        resampled_xa = xr.DataArray(
            data=np.empty(
                shape=((len(resampled_time_index),) + xa.shape[1:]), dtype=dtype
            ),
            dims=xa.dims,
            coords=coords,
            attrs=xa.attrs,
        )
        resampled_xa[:] = np.nan
        resampled_xa[resampled_time_index.index.isin(time_index)] = sorted_xa.data
        return resampled_xa

    def _get_dim_name(self, axis: Union[int, str]) -> str:
        if isinstance(axis, int):
            if axis == 0:
                return self._time_dim
            elif axis == 1 or axis == 2:
                return DIMS[axis]
            else:
                raise_if(True, "If `axis` is an integer it must be between 0 and 2.")
        else:
            known_dims = (self._time_dim,) + DIMS[1:]
            raise_if_not(
                axis in known_dims,
                f"`axis` must be a known dimension of this series: {known_dims}",
            )
            return axis

    def _get_dim(self, axis: Union[int, str]) -> int:
        if isinstance(axis, int):
            raise_if_not(
                0 <= axis <= 2, "If `axis` is an integer it must be between 0 and 2."
            )
            return axis
        else:
            known_dims = (self._time_dim,) + DIMS[1:]
            raise_if_not(
                axis in known_dims,
                f"`axis` must be a known dimension of this series: {known_dims}",
            )
            return known_dims.index(axis)

    def __eq__(self, other):
        if isinstance(other, TimeSeries):
            return self._xa.equals(other.data_array(copy=False))
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._xa)

    def __add__(self, other):
        if isinstance(other, (int, float, np.integer)):
            xa_ = _xarray_with_attrs(
                self._xa + other, self.static_covariates, self.hierarchy, self.metadata
            )
            return self.__class__(xa_)
        elif isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            return self._combine_arrays(other, lambda s1, s2: s1 + s2)
        else:
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for + or add(): '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float, np.integer)):
            xa_ = _xarray_with_attrs(
                self._xa - other, self.static_covariates, self.hierarchy, self.metadata
            )
            return self.__class__(xa_)
        elif isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            return self._combine_arrays(other, lambda s1, s2: s1 - s2)
        else:
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for - or sub(): '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer)):
            xa_ = _xarray_with_attrs(
                self._xa * other, self.static_covariates, self.hierarchy, self.metadata
            )
            return self.__class__(xa_)
        elif isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            return self._combine_arrays(other, lambda s1, s2: s1 * s2)
        else:
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for * or mul(): '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )

    def __rmul__(self, other):
        return self * other

    def __pow__(self, n):
        if isinstance(n, (int, float, np.integer)):
            raise_if(n < 0, "Attempted to raise a series to a negative power.", logger)
            xa_ = _xarray_with_attrs(
                self._xa ** float(n),
                self.static_covariates,
                self.hierarchy,
                self.metadata,
            )
            return self.__class__(xa_)
        if isinstance(n, (TimeSeries, xr.DataArray, np.ndarray)):
            return self._combine_arrays(n, lambda s1, s2: s1**s2)  # elementwise power
        else:
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for ** or pow(): '{type(self).__name__}' and '{type(n).__name__}'."
                ),
                logger,
            )

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.integer)):
            if other == 0:
                raise_log(ZeroDivisionError("Cannot divide by 0."), logger)
            xa_ = _xarray_with_attrs(
                self._xa / other, self.static_covariates, self.hierarchy, self.metadata
            )
            return self.__class__(xa_)
        elif isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            if isinstance(other, TimeSeries):
                other_vals = other.data_array(copy=False).values
            elif isinstance(other, xr.DataArray):
                other_vals = other.values
            else:
                other_vals = other
            if not (other_vals != 0).all():
                raise_log(
                    ZeroDivisionError("Cannot divide by a TimeSeries with a value 0."),
                    logger,
                )
            return self._combine_arrays(other_vals, lambda s1, s2: s1 / s2)
        else:
            raise_log(
                TypeError(
                    "unsupported operand type(s) for / or truediv():"
                    f" '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )

    def __rtruediv__(self, n):
        return n * (self ** (-1))

    def __abs__(self):
        return self.__class__(abs(self._xa))

    def __neg__(self):
        return self.__class__(-self._xa)

    def __contains__(self, ts: Union[int, pd.Timestamp]) -> bool:
        return ts in self.time_index

    def __round__(self, n=None):
        return self.__class__(self._xa.round(n))

    def __lt__(self, other) -> xr.DataArray:
        if isinstance(other, (int, float, np.integer, np.ndarray, xr.DataArray)):
            return _xarray_with_attrs(
                self._xa < other, self.static_covariates, self.hierarchy, self.metadata
            )
        elif isinstance(other, TimeSeries):
            return _xarray_with_attrs(
                self._xa < other.data_array(copy=False),
                self.static_covariates,
                self.hierarchy,
                self.metadata,
            )
        else:
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for < : '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )

    def __gt__(self, other) -> xr.DataArray:
        if isinstance(other, (int, float, np.integer, np.ndarray, xr.DataArray)):
            return _xarray_with_attrs(
                self._xa > other, self.static_covariates, self.hierarchy, self.metadata
            )
        elif isinstance(other, TimeSeries):
            return _xarray_with_attrs(
                self._xa > other.data_array(copy=False),
                self.static_covariates,
                self.hierarchy,
                self.metadata,
            )
        else:
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for < : '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )

    def __le__(self, other) -> xr.DataArray:
        if isinstance(other, (int, float, np.integer, np.ndarray, xr.DataArray)):
            return _xarray_with_attrs(
                self._xa <= other, self.static_covariates, self.hierarchy, self.metadata
            )
        elif isinstance(other, TimeSeries):
            return _xarray_with_attrs(
                self._xa <= other.data_array(copy=False),
                self.static_covariates,
                self.hierarchy,
                self.metadata,
            )
        else:
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for < : '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )

    def __ge__(self, other) -> xr.DataArray:
        if isinstance(other, (int, float, np.integer, np.ndarray, xr.DataArray)):
            return _xarray_with_attrs(
                self._xa >= other, self.static_covariates, self.hierarchy, self.metadata
            )
        elif isinstance(other, TimeSeries):
            return _xarray_with_attrs(
                self._xa >= other.data_array(copy=False),
                self.static_covariates,
                self.hierarchy,
                self.metadata,
            )
        else:
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for < : '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )

    def __str__(self):
        return str(self._xa).replace("xarray.DataArray", "TimeSeries (DataArray)")

    def __repr__(self):
        return self._xa.__repr__().replace("xarray.DataArray", "TimeSeries (DataArray)")

    def _repr_html_(self):
        return self._xa._repr_html_().replace(
            "xarray.DataArray", "TimeSeries (DataArray)"
        )

    def __copy__(self, deep: bool = True):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.__class__(deepcopy(self._xa, memo))

    def __getitem__(
        self,
        key: Union[
            pd.DatetimeIndex,
            pd.RangeIndex,
            list[str],
            list[int],
            list[pd.Timestamp],
            str,
            int,
            pd.Timestamp,
            Any,
        ],
    ) -> Self:
        """Allow indexing on darts TimeSeries.

        The supported index types are the following base types as a single value, a list or a slice:
        - pd.Timestamp -> return a TimeSeries corresponding to the value(s) at the given timestamp(s).
        - str -> return a TimeSeries including the column(s) (components) specified as str.
        - int -> return a TimeSeries with the value(s) at the given row (time) index.

        `pd.DatetimeIndex` and `pd.RangeIndex` are also supported and will return the corresponding value(s)
        at the provided time indices.

        .. warning::
            slices use pandas convention of including both ends of the slice.

        Notes
        -----
        For integer-indexed series, integers or slices of integer will return the result
        of ``isel()``. That is, if integer ``i`` is provided, it returns the ``i``-th value
        along the series, which is not necessarily the value where the time index is equal to ``i``
        (e.g., if the time index does not start at 0). In contrast, calling this method with a
        ``pd.RangeIndex`` returns the result of ``sel()`` - i.e., the values where the time
        index matches the provided range index.
        """

        def _check_dt():
            if not self._has_datetime_index:
                raise_log(
                    ValueError(
                        "Attempted indexing a series with a DatetimeIndex or a timestamp, "
                        "but the series uses a RangeIndex."
                    ),
                    logger,
                )

        def _check_range():
            if self._has_datetime_index:
                raise_log(
                    ValueError(
                        "Attempted indexing a series with a RangeIndex, "
                        "but the series uses a DatetimeIndex."
                    ),
                    logger,
                )

        def _set_freq_in_xa(xa_in: xr.DataArray, freq=None):
            # mutates the DataArray to make sure it contains the freq
            if isinstance(xa_in.get_index(self._time_dim), pd.DatetimeIndex):
                if freq is None:
                    freq = xa_in.get_index(self._time_dim).inferred_freq
                if freq is not None:
                    xa_in.get_index(self._time_dim).freq = freq
                else:
                    xa_in.get_index(self._time_dim).freq = self._freq

        def _get_freq(xa_in: xr.DataArray):
            if self._has_datetime_index:
                return xa_in.get_index(self._time_dim).freq
            else:
                return xa_in.get_index(self._time_dim).step

        adapt_covs_on_component = (
            True
            if self.has_static_covariates and len(self.static_covariates) > 1
            else False
        )

        # handle DatetimeIndex and RangeIndex:
        if isinstance(key, pd.DatetimeIndex):
            _check_dt()
            xa_ = self._xa.sel({self._time_dim: key})

            # indexing may discard the freq, so we restore it...
            # if the DateTimeIndex already has an associated freq, use it
            # otherwise key.freq is None and the freq will be inferred
            _set_freq_in_xa(xa_, key.freq)

            return self.__class__(xa_)
        elif isinstance(key, pd.RangeIndex):
            _check_range()
            idx_ = key
            if not len(key) and self.freq != key.step:
                # keep original step size in case of empty range index
                idx_ = pd.RangeIndex(step=self.freq)

            xa_ = self._xa.sel({self._time_dim: idx_})

            # sel() gives us an Int64Index. We have to set the RangeIndex.
            # see: https://github.com/pydata/xarray/issues/6256
            xa_ = xa_.assign_coords({self.time_dim: idx_})

            return self.__class__(xa_)

        # handle slices:
        elif isinstance(key, slice):
            if key.start is None and key.stop is None:
                if key.step is not None and key.step <= 0:
                    raise_log(
                        ValueError(
                            "Indexing a `TimeSeries` with a `slice` of `step<=0` (reverse) is not "
                            "possible since `TimeSeries` must have a monotonically increasing time index."
                        ),
                        logger=logger,
                    )
                else:
                    xa_ = self._xa.isel({self._time_dim: key})
                    if _get_freq(xa_) is None:
                        # indexing discarded the freq; we restore it
                        freq = key.step * self.freq if key.step else self.freq
                        _set_freq_in_xa(xa_, freq)
                    return self.__class__(xa_)
            elif isinstance(key.start, str) or isinstance(key.stop, str):
                xa_ = self._xa.sel({DIMS[1]: key})
                # selecting components discards the hierarchy, if any
                xa_ = _xarray_with_attrs(
                    xa_,
                    (
                        xa_.attrs[STATIC_COV_TAG][key.start : key.stop]
                        if adapt_covs_on_component
                        else xa_.attrs[STATIC_COV_TAG]
                    ),
                    None,
                    xa_.attrs[METADATA_TAG],
                )
                return self.__class__(xa_)
            elif isinstance(key.start, (int, np.int64)) or isinstance(
                key.stop, (int, np.int64)
            ):
                xa_ = self._xa.isel({self._time_dim: key})
                if _get_freq(xa_) is None:
                    # indexing discarded the freq; we restore it
                    freq = key.step * self.freq if key.step else self.freq
                    _set_freq_in_xa(xa_, freq)
                return self.__class__(xa_)
            elif isinstance(key.start, pd.Timestamp) or isinstance(
                key.stop, pd.Timestamp
            ):
                _check_dt()
                xa_ = self._xa.sel({self._time_dim: key})
                if _get_freq(xa_) is None:
                    # indexing discarded the freq; we restore it
                    freq = key.step * self.freq if key.step else self.freq
                    _set_freq_in_xa(xa_, freq)
                return self.__class__(xa_)

        # handle simple types:
        elif isinstance(key, str):
            # have to put key in a list not to drop the dimension
            xa_ = self._xa.sel({DIMS[1]: [key]})
            # selecting components discards the hierarchy, if any
            xa_ = _xarray_with_attrs(
                xa_,
                (
                    xa_.attrs[STATIC_COV_TAG].loc[[key]]
                    if adapt_covs_on_component
                    else xa_.attrs[STATIC_COV_TAG]
                ),
                None,
                xa_.attrs[METADATA_TAG],
            )
            return self.__class__(xa_)
        elif isinstance(key, (int, np.int64)):
            xa_ = self._xa.isel({self._time_dim: [key]})

            # restore a RangeIndex if needed:
            time_idx = xa_.get_index(self._time_dim)
            if pd.api.types.is_integer_dtype(time_idx) and not isinstance(
                time_idx, pd.RangeIndex
            ):
                xa_ = xa_.assign_coords({
                    self._time_dim: pd.RangeIndex(
                        start=time_idx[0],
                        stop=time_idx[0] + self.freq,
                        step=self.freq,
                    )
                })
            # indexing may discard the freq, so we restore it...
            _set_freq_in_xa(xa_, freq=self.freq)
            return self.__class__(xa_)
        elif isinstance(key, pd.Timestamp):
            _check_dt()

            # indexing may discard the freq, so we restore it...
            xa_ = self._xa.sel({self._time_dim: [key]})
            _set_freq_in_xa(xa_, self.freq)
            return self.__class__(xa_)

        # handle lists:
        if isinstance(key, list):
            if all(isinstance(s, str) for s in key):
                # when string(s) are provided, we consider it as (a list of) component(s)
                xa_ = self._xa.sel({DIMS[1]: key})
                xa_ = _xarray_with_attrs(
                    xa_,
                    (
                        xa_.attrs[STATIC_COV_TAG].loc[key]
                        if adapt_covs_on_component
                        else xa_.attrs[STATIC_COV_TAG]
                    ),
                    None,
                    xa_.attrs[METADATA_TAG],
                )
                return self.__class__(xa_)
            elif all(isinstance(i, (int, np.int64)) for i in key):
                xa_ = self._xa.isel({self._time_dim: key})

                # indexing may discard the freq, so we restore it...
                _set_freq_in_xa(xa_)

                orig_idx = self.time_index
                if isinstance(orig_idx, pd.RangeIndex):
                    # We have to restore a RangeIndex. But first we need to
                    # check the list is corresponding to a RangeIndex.
                    min_idx, max_idx = min(key), max(key)
                    if (
                        not key[0] == min_idx
                        and key[-1] == max_idx
                        and max_idx + 1 - min_idx == len(key)
                    ):
                        raise_log(
                            ValueError(
                                "Indexing a TimeSeries with a list requires the list to "
                                "contain monotonically increasing integers with no gap."
                            ),
                            logger=logger,
                        )
                    new_idx = orig_idx[min_idx : max_idx + 1]
                    xa_ = xa_.assign_coords({self._time_dim: new_idx})

                return self.__class__(xa_)

            elif all(isinstance(t, pd.Timestamp) for t in key):
                _check_dt()

                # indexing may discard the freq, so we restore it...
                xa_ = self._xa.sel({self._time_dim: key})
                _set_freq_in_xa(xa_)
                return self.__class__(xa_)

        raise_log(IndexError("The type of your index was not matched."), logger)


def _xarray_with_attrs(xa_, static_covariates, hierarchy, metadata):
    """Return an DataArray instance with static covariates and hierarchy stored in the array's attributes.
    Warning: This is an inplace operation (mutable) and should only be called from within TimeSeries construction
    or to restore static covariates, hierarchy and metadata after operations in which they did not get transferred.
    """
    xa_.attrs[STATIC_COV_TAG] = static_covariates
    xa_.attrs[HIERARCHY_TAG] = hierarchy
    xa_.attrs[METADATA_TAG] = metadata
    return xa_


def _concat_static_covs(series: Sequence[TimeSeries]) -> Optional[pd.DataFrame]:
    """Concatenates static covariates along component dimension (rows of static covariates). For stacking or
    concatenating TimeSeries along component dimension (axis=1).

    Some context for stacking or concatenating two or more TimeSeries with static covariates:
        Concat along axis=0 (time)
            Along time dimension, we only take the static covariates of the first series (as static covariates are
            time-independent).
        Concat along axis=1 (components) or stacking
            Along component dimension, we either concatenate or transfer the static covariates of the series if one
            of below cases applies:
            1)  concatenate along component dimension (rows of static covariates) when for each series the number of
                static covariate components is equal to the number of components in the series. The static variable
                names (columns in series.static_covariates) must be identical across all series
            2)  if only the first series contains static covariates transfer only those
            3)  if `ignore_static_covarites=True` (with `concatenate()`), case 1) is ignored and only the static
                covariates of the first series are transferred
        Concat along axis=2 (samples)
            Along sample dimension, we only take the static covariates of the first series (as we components and
            time don't change).
    """

    if not any([ts.has_static_covariates for ts in series]):
        return None

    only_first = series[0].has_static_covariates and not any([
        ts.has_static_covariates for ts in series[1:]
    ])
    all_have = all([ts.has_static_covariates for ts in series])

    raise_if_not(
        only_first or all_have,
        "Either none, only the first or all TimeSeries must have `static_covariates`.",
        logger,
    )

    if only_first:
        return series[0].static_covariates

    raise_if_not(
        all([len(ts.static_covariates) == ts.n_components for ts in series])
        and all([
            ts.static_covariates.columns.equals(series[0].static_covariates.columns)
            for ts in series
        ]),
        "Concatenation of multiple TimeSeries with static covariates requires all `static_covariates` "
        "DataFrames to have identical columns (static variable names), and the number of each TimeSeries' "
        "components must match the number of corresponding static covariate components (the number of rows "
        "in `series.static_covariates`).",
        logger,
    )

    return pd.concat(
        [ts.static_covariates for ts in series if ts.has_static_covariates], axis=0
    )


def _concat_hierarchy(series: Sequence[TimeSeries]):
    """
    Used to concatenate the hierarchies of multiple TimeSeries, when concatenating series
    along axis 1 (components). This simply merges the hierarchy dictionaries.
    """
    concat_hierarchy = dict()
    for s in series:
        if s.has_hierarchy:
            concat_hierarchy.update(s.hierarchy)
    return None if len(concat_hierarchy) == 0 else concat_hierarchy


def concatenate(
    series: Sequence[TimeSeries],
    axis: Union[str, int] = 0,
    ignore_time_axis: bool = False,
    ignore_static_covariates: bool = False,
    drop_hierarchy: bool = True,
    drop_metadata: bool = False,
):
    """Concatenates multiple ``TimeSeries`` along a given axis.

    ``axis`` can be an integer in (0, 1, 2) to denote (time, component, sample) or, alternatively,
    a string denoting the corresponding dimension of the underlying ``DataArray``.

    Parameters
    ----------
    series : Sequence[TimeSeries]
        Sequence of ``TimeSeries`` to concatenate
    axis : Union[str, int]
        Axis along which the series will be concatenated.
    ignore_time_axis : bool
        Allow concatenation even when some series do not have matching time axes.
        When done along component or sample dimensions, concatenation will work as long as the series
        have the same lengths (in this case the resulting series will have the time axis of the first
        provided series). When done along time dimension, concatenation will work even if the time axes
        are not contiguous (in this case, the resulting series will have a start time matching the start time
        of the first provided series). Default: False.
    ignore_static_covariates : bool
        Whether to ignore all requirements for static covariate concatenation and only transfer the static covariates
        of the first TimeSeries element in `series` to the concatenated TimeSeries. Only effective when `axis=1`.
    drop_hierarchy : bool
        When `axis=1`, whether to drop hierarchy information. True by default. When False, the hierarchies will be
        "concatenated" as well (by merging the hierarchy dictionaries), which may cause issues if the component
        names of the resulting series and that of the merged hierarchy do not match.
        When `axis=0` or `axis=2`, the hierarchy of the first series is always kept.
    drop_metadata : bool
        Whether to drop the metadata information of the concatenated series. False by default.
        When False, the concatenated series will inherit the metadata from the first TimeSeries element in `series`.

    Returns
    -------
    TimeSeries
        concatenated series
    """

    time_dims = [ts.time_dim for ts in series]
    if isinstance(axis, str):
        if axis == DIMS[1]:
            axis = 1
        elif axis == DIMS[2]:
            axis = 2
        else:
            raise_if_not(
                len(set(time_dims)) == 1 and axis == time_dims[0],
                "Unrecognised `axis` name. If `axis` denotes the time axis, all provided "
                "series must have the same time axis name (if that is not the case, try providing "
                "`axis=0` to concatenate along time dimension).",
            )
            axis = 0

    # At this point all series are supposed to have same time dim name
    time_dim_name = time_dims[0]

    da_sequence = [ts.data_array(copy=False) for ts in series]

    component_axis_equal = len({ts.width for ts in series}) == 1
    sample_axis_equal = len({ts.n_samples for ts in series}) == 1

    metadata = None if drop_metadata else series[0].metadata

    if axis == 0:
        # time
        raise_if(
            (not (component_axis_equal and sample_axis_equal)),
            "when concatenating along time dimension, the component and sample dimensions of all "
            "provided series must match.",
        )

        da_concat = xr.concat(da_sequence, dim=time_dim_name)

        # check, if timeseries are consecutive
        consecutive_time_axes = True
        for i in range(1, len(series)):
            if series[i - 1].end_time() + series[0].freq != series[i].start_time():
                consecutive_time_axes = False
                break

        if not consecutive_time_axes:
            raise_if_not(
                ignore_time_axis,
                "When concatenating over time axis, all series need to be contiguous "
                "in the time dimension. Use `ignore_time_axis=True` to override "
                "this behavior and concatenate the series by extending the time axis "
                "of the first series.",
            )

            tindex = generate_index(
                start=series[0].start_time(),
                freq=series[0].freq_str,
                length=da_concat.shape[0],
            )

            da_concat = da_concat.assign_coords({time_dim_name: tindex})
            da_concat = _xarray_with_attrs(
                da_concat, series[0].static_covariates, series[0].hierarchy, metadata
            )

    else:
        time_axes_equal = all(
            list(
                map(
                    lambda t: t[0].has_same_time_as(t[1]), zip(series[0:-1], series[1:])
                )
            )
        )
        time_axes_ok = (
            time_axes_equal
            if not ignore_time_axis
            else len({len(ts) for ts in series}) == 1
        )

        raise_if_not(
            (
                time_axes_ok
                and (
                    (axis == 1 and sample_axis_equal)
                    or (axis == 2 and component_axis_equal)
                )
            ),
            "When concatenating along component or sample dimensions, all the series must have the same time "
            "axes (unless `ignore_time_axis` is True), or time axes of same lengths (if `ignore_time_axis` is "
            "True), and all series must have the same number of samples (if concatenating along component "
            "dimension), or the same number of components (if concatenating along sample dimension).",
        )

        # we concatenate raw values using Numpy because not all series might have the same time axes
        # and joining using xarray.concatenate() won't work in some cases
        concat_vals = np.concatenate([da.values for da in da_sequence], axis=axis)

        if axis == 1:
            # When concatenating along component dimension, we have to re-create a component index
            # we rely on the factory method of TimeSeries to disambiguate names later on if needed.
            component_index = pd.Index([
                c for cl in [ts.components for ts in series] for c in cl
            ])
            static_covariates = (
                _concat_static_covs(series)
                if not ignore_static_covariates
                else series[0].static_covariates
            )
            hierarchy = None if drop_hierarchy else _concat_hierarchy(series)
        else:
            component_index = da_sequence[0].get_index(DIMS[1])
            static_covariates = series[0].static_covariates
            hierarchy = series[0].hierarchy

        da_concat = xr.DataArray(
            concat_vals,
            dims=(time_dim_name,) + DIMS[-2:],
            coords={time_dim_name: series[0].time_index, DIMS[1]: component_index},
            attrs={
                STATIC_COV_TAG: static_covariates,
                HIERARCHY_TAG: hierarchy,
                METADATA_TAG: metadata,
            },
        )

    return TimeSeries.from_xarray(da_concat, fill_missing_dates=False)


def slice_intersect(series: Sequence[TimeSeries]) -> list[TimeSeries]:
    """Returns a list of ``TimeSeries``, where all `series` have been intersected along the time index.

    Parameters
    ----------
    series : Sequence[TimeSeries]
        sequence of ``TimeSeries`` to intersect

    Returns
    -------
    Sequence[TimeSeries]
        Intersected series.
    """
    if not series:
        return []

    # find global intersection on first series
    intersection = series[0]
    for series_ in series[1:]:
        intersection = intersection.slice_intersect(series_)

    # intersect all other series
    series_intersected = [intersection]
    for series_ in series[1:]:
        series_intersected.append(series_.slice_intersect(intersection))

    return series_intersected


def _finite_rows_boundaries(
    values: np.ndarray, how: str = "all"
) -> tuple[Optional[int], Optional[int]]:
    """
    Return the indices of the first rows containing finite values starting from the start and the end of the first
    dimension of the ndarray.

    Parameters
    ----------
    values
        1D, 2D or 3D numpy array where the first dimension correspond to entries/rows, and the second to components/
        columns
    how
        Define if the entries containing `NaN` in all the components ('all') or in any of the components ('any')
        should be stripped. Default: 'all'
    """
    dims = values.shape

    raise_if(
        len(dims) > 3, f"Expected 1D to 3D array, received {len(dims)}D array", logger
    )

    finite_rows = ~np.isnan(values)

    if len(dims) == 3:
        finite_rows = finite_rows.all(axis=2)

    if len(dims) > 1 and dims[1] > 1:
        if how == "any":
            finite_rows = finite_rows.all(axis=1)
        elif how == "all":
            finite_rows = finite_rows.any(axis=1)
        else:
            raise_log(
                ValueError(
                    f"`how` parameter value not recognized, should be either 'all' or 'any', "
                    f"received {how}"
                )
            )

    first_finite_row = finite_rows.argmax()
    last_finite_row = len(finite_rows) - finite_rows[::-1].argmax() - 1

    return first_finite_row, last_finite_row
