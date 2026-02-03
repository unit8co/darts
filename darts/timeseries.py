"""
Timeseries
----------

``TimeSeries`` is `Darts` container for storing and handling time series data. It supports univariate or
multivariate time series that can be deterministic or stochastic.

The values are stored in an array of shape `(times, components, samples)`, where `times` are the number of
time steps, `components` are the number of columns, and `samples` are the number of samples in the series.

**Definitions:**

- A series with `components = 1` is **univariate**, and a series with `components > 1` is **multivariate**.
- A series with `samples = 1` is **deterministic** and a series with `samples > 1` is **stochastic** (or
  **probabilistic**).

Each series also stores a `time_index`, which contains either datetimes (:class:`pandas.DateTimeIndex`) or integer
indices (:class:`pandas.RangeIndex`).

Optionally, ``TimeSeries`` can store static covariates, a hierarchy, and / or metadata.

- **Static covariates** are time-invariant external data / information about the series and can be used by some models
  to help improve predictions. Find more info on covariates `here
  <https://unit8co.github.io/darts/userguide/covariates.html>`__.
- A **hierarchy** describes the hierarchical structure of the components which can be used to reconcile forecasts. For
  more info on hierarchical reconciliation `here
  <https://unit8co.github.io/darts/examples/16-hierarchical-reconciliation.html>`__.
- **Metadata** can be used to store any additional information about the series which will not be used by any model.

``TimeSeries`` **are guaranteed to:**

- Have a strictly monotonically increasing time index with a well-defined frequency (without holes / missing dates).
  For more info on available ``DateTimeIndex`` frequencies, see `date offset aliases
  <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__. For integer-indexed
  series the frequency corresponds to the constant step size between consecutive indices.
- Contain numeric data types only
- Have unique component / column names
- Have static covariates consistent with their components (global or component-specific), or no static covariates
- Have a hierarchy consistent with their components, or no hierarchy
"""

import itertools
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from inspect import signature
from io import StringIO
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import matplotlib.axes
import narwhals as nw
import numpy as np
import pandas as pd
import xarray as xr
from narwhals.utils import Implementation
from pandas.tseries.frequencies import to_offset
from scipy.stats import kurtosis, skew

from darts.config import get_option
from darts.logging import get_logger, raise_log
from darts.utils import _build_tqdm_iterator, _parallel_apply
from darts.utils._formatting import (
    format_bytes,
    format_dict,
    make_collapsible_section,
    make_paragraph,
)
from darts.utils._plotting import plot as _plot
from darts.utils._plotting import plotly as _plotly
from darts.utils.utils import (
    SUPPORTED_RESAMPLE_METHODS,
    dataframe_col_to_time_index,
    expand_arr,
    generate_index,
    n_steps_between,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = get_logger(__name__)

# dimension names in the array
# the "time" one can be different, if it has a name in the underlying Series/DataFrame.
DIMS = ("time", "component", "sample")
TIME_AX = 0
COMP_AX = 1
SMPL_AX = 2
AXES = {"time": TIME_AX, "component": COMP_AX, "sample": SMPL_AX}

VALID_INDEX_TYPES = (pd.DatetimeIndex, pd.RangeIndex)
STATIC_COV_TAG = "static_covariates"
DEFAULT_GLOBAL_STATIC_COV_NAME = "global_components"
HIERARCHY_TAG = "hierarchy"
METADATA_TAG = "metadata"


class TimeSeries:
    def __init__(
        self,
        times: Union[pd.DatetimeIndex, pd.RangeIndex, pd.Index],
        values: np.ndarray,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        components: Optional[Union[Sequence, str]] = None,
        fillna_value: Optional[float] = None,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hierarchy: Optional[dict] = None,
        metadata: Optional[dict] = None,
        copy: bool = True,
    ):
        """Create a ``TimeSeries`` from a time index `times` and values `values`.

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
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        components
            Optionally, some column names to use for the second `values` dimension.
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
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`__.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.
        copy
            Whether to copy the `times` and `values` objects. If `copy=False`, mutating the series data will affect the
            original data. Additionally, if `times` lack a frequency or step size, it will be assigned to the original
            object.

        Returns
        -------
        TimeSeries
            The resulting series.

        Examples
        --------
        >>> import numpy as np
        >>> from darts import TimeSeries
        >>> from darts.utils.utils import generate_index
        >>> # create values and times with daily frequency
        >>> vals, times = np.arange(3), generate_index("2020-01-01", length=3, freq="D")
        >>> series = TimeSeries(times=times, values=vals)
        >>> series.shape
        (3, 1, 1)
        """
        if not (
            isinstance(times, VALID_INDEX_TYPES)
            or np.issubdtype(times.dtype, np.integer)
        ):
            raise_log(
                ValueError(
                    "the `times` argument must be a `pandas.RangeIndex`, or `pandas.DateTimeIndex`. Use "
                    "TimeSeries.from_values() if you want to use an automatic `RangeIndex`.",
                ),
                logger=logger,
            )

        # avoid copying if data is already np.ndarray:
        values = np.array(values) if not isinstance(values, np.ndarray) else values

        # optionally, cast values to float
        if not np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float64)
        if not (
            np.issubdtype(values.dtype, np.float64)
            or np.issubdtype(values.dtype, np.float32)
        ):
            logger.warning(
                "TimeSeries is using a numeric type different from numpy.float32 or numpy.float64. "
                "Not all functionalities may work properly. It is recommended casting your data to floating "
                "point numbers before using TimeSeries."
            )

        values = expand_arr(values, ndim=len(DIMS))
        if len(values.shape) != 3:
            raise_log(
                ValueError(
                    f"TimeSeries require a `values` array that has or can be expanded to "
                    f"3 dimensions ({DIMS})."
                ),
                logger,
            )

        if not len(times) == len(values):
            raise_log(
                ValueError("The time index and values must have the same length."),
                logger,
            )

        if components is None:
            components = pd.Index([str(idx) for idx in range(values.shape[COMP_AX])])
        elif isinstance(components, str):
            components = pd.Index([components])
        elif not isinstance(components, pd.Index):
            components = pd.Index(components)

        if len(components) != values.shape[COMP_AX]:
            raise_log(
                ValueError(
                    "The number of provided components must match the number of "
                    "components from `values` (`values.shape[1]`). Expected "
                    f"number of components: `{values.shape[1]}`, received: `{len(components)}`."
                ),
                logger=logger,
            )

        if copy:
            # deepcopy the index as updating `times.freq` with a shallow `copy()` mutates the original index
            times = deepcopy(times)
            values = values.copy()

        # clean component (column) names if needed (when names are not unique, or not strings)
        if len(set(components)) != len(components) or any([
            not isinstance(s, str) for s in components
        ]):
            components = _clean_components(components)

        has_datetime_index = isinstance(times, pd.DatetimeIndex)
        has_range_index = isinstance(times, pd.RangeIndex)
        has_integer_index = not (has_datetime_index or has_range_index)

        # remove timezone information if present; drops the frequency
        # TODO: potential to use timezone-aware index since `TimeSeries` was refactored
        #  to use numpy as backend
        if has_datetime_index and times.tz is not None:
            logger.warning(
                f"The provided DatetimeIndex was associated with a timezone (tz), which is currently "
                f"not supported. To avoid unexpected behaviour, the tz information was removed. Consider "
                f"calling `ts.time_index.tz_localize({times.tz})` when exporting the results."
                f"To plot the series with the right time steps, consider setting the matplotlib.pyplot "
                f"`rcParams['timezone']` parameter to automatically convert the time axis back to the "
                f"original timezone."
            )
            times = times.tz_localize(None)

        has_frequency = (
            has_datetime_index and times.freq is not None
        ) or has_range_index
        if not has_frequency:
            # can only be `pd.DatetimeIndex` or int `pd.Index` (not `pd.RangeIndex`)
            if fill_missing_dates:
                # optionally fill missing dates
                times, values = self._fill_missing_dates(
                    times=times, values=values, freq=freq
                )
            elif freq is not None:
                # using the provided `freq`
                times, values = self._restore_from_frequency(
                    times=times, values=values, freq=freq
                )
            elif has_integer_index:
                # integer `pd.Index` and no `freq` is provided; try convert it to pd.RangeIndex
                times, values = self._restore_range_indexed(times=times, values=values)
            else:
                # `pd.DatetimeIndex`, and no `freq` provided; sort and see later if frequency can be inferred
                times, values = self._sort_index(times=times, values=values)
        elif (
            (has_datetime_index and times.freq.n < 0)
            or has_range_index
            and times.step < 0
        ):
            times = times[::-1]
            values = values[::-1]

        if fillna_value is not None:
            values[np.isnan(values)] = fillna_value

        if has_datetime_index:
            # frequency must be known or it can be inferred
            freq = times.freq
            if freq is None:
                freq = to_offset(times.inferred_freq)
                times.freq = freq

            if freq is None:
                raise_log(
                    ValueError(
                        "The time index is missing the `freq` attribute, and the frequency "
                        "could not be directly inferred. This probably comes from inconsistent date frequencies with "
                        "missing dates. If you know the actual frequency, try setting `fill_missing_dates=True, "
                        "freq=actual_frequency`. If not, try setting `fill_missing_dates=True, freq=None` to see if a "
                        "frequency can be inferred."
                    ),
                    logger,
                )

            freq_str: Optional[str] = freq.freqstr
        else:
            freq: int = times.step
            freq_str = str(freq)

        # how the dimensions are named; we convert hashable to string
        self._time_dim = str(times.name) if times.name is not None else DIMS[TIME_AX]
        self._time_index = times
        self._freq = freq
        self._freq_str = freq_str
        self._has_datetime_index = has_datetime_index
        self._values = values
        self._components = components

        # check static covariates
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
        if metadata is not None and not isinstance(metadata, dict):
            raise_log(
                ValueError(
                    "`metadata` must be of type `dict` mapping metadata attributes to their values."
                ),
                logger,
            )

        # handle hierarchy
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

        self._attrs = {
            STATIC_COV_TAG: static_covariates,
            HIERARCHY_TAG: hierarchy,
            METADATA_TAG: metadata,
        }

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
        copy: bool = True,
    ) -> Self:
        """Create a ``TimeSeries`` from an `xarray.DataArray`.

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
            The `xarray.DataArray`
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates (or indices in case of integer index)
            with NaN values. This requires either a provided `freq` or the possibility to infer the frequency from the
            provided timestamps. See :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string or integer representing the frequency of the underlying index. This is useful in order
            to fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
            If a string, represents the frequency of the pandas DatetimeIndex (see `offset aliases
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.
        copy
            Whether to copy the `times` (time index dimension) and `values` (data) objects. If `copy=False`, mutating
            the series data will affect the original data. Additionally, if `times` lack a frequency or step size, it
            will be assigned to the original object.

        Returns
        -------
        TimeSeries
            The resulting series.

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
        return cls(
            times=xa.get_index(xa.dims[TIME_AX]),
            values=xa.values,
            fill_missing_dates=fill_missing_dates,
            freq=freq,
            components=xa.get_index(xa.dims[COMP_AX]),
            fillna_value=fillna_value,
            static_covariates=xa.attrs.get(STATIC_COV_TAG),
            hierarchy=xa.attrs.get(HIERARCHY_TAG),
            metadata=xa.attrs.get(METADATA_TAG),
            copy=copy,
        )

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
        """Create a ``TimeSeries`` from a CSV file.

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
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__ for more info on
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
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`__.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.

        **kwargs
            Optional arguments to be passed to `pandas.read_csv` function

        Returns
        -------
        TimeSeries
            The resulting series.

        Examples
        --------
        >>> from darts import TimeSeries
        >>> TimeSeries.from_csv("data.csv", time_col="time")
        """
        return cls.from_dataframe(
            df=pd.read_csv(filepath_or_buffer=filepath_or_buffer, **kwargs),
            time_col=time_col,
            value_cols=value_cols,
            fill_missing_dates=fill_missing_dates,
            freq=freq,
            fillna_value=fillna_value,
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
            copy=False,
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        time_col: Optional[str] = None,
        value_cols: Optional[Union[list[str], str]] = None,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        fillna_value: Optional[float] = None,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hierarchy: Optional[dict] = None,
        metadata: Optional[dict] = None,
        copy: bool = True,
    ) -> Self:
        """Create a ``TimeSeries`` from a selection of columns of a `DataFrame`.

        One column (or the DataFrame index) has to represent the time, and a list of columns `value_cols` has to
        represent the values for this time series.

        Parameters
        ----------
        df
            The DataFrame, or anything which can be converted to a narwhals DataFrame (e.g. pandas.DataFrame,
            polars.DataFrame, ...). See the `narwhals documentation
            <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_native>`__ for more
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
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__ for more info on
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
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`__.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.
        copy
            Whether to copy the `times` (DataFrame index or the `time_col` column) and DataFrame `values`.
            If `copy=False`, mutating the series data will affect the original data. Additionally, if `times` lack a
            frequency or step size, it will be assigned to the original object.

        Returns
        -------
        TimeSeries
            The resulting series.

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

        # get time index
        if time_col:
            time_index = dataframe_col_to_time_index(df, time_col)
        else:
            time_index = nw.maybe_get_index(df)
            if time_index is None:
                time_index = pd.RangeIndex(len(df), name=DIMS[TIME_AX])
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

        # get values
        if value_cols is None:
            series_df = df.drop(time_col) if time_col else df
        else:
            if isinstance(value_cols, (str, int)):
                value_cols = [value_cols]
            series_df = df[value_cols]

        return cls(
            times=time_index,
            values=series_df.to_numpy(),
            fill_missing_dates=fill_missing_dates,
            freq=freq,
            components=series_df.columns,
            fillna_value=fillna_value,
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
            copy=copy,
        )

    @classmethod
    def from_group_dataframe(
        cls,
        df,
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
        copy: bool = True,
    ) -> list[Self]:
        """Create a list of ``TimeSeries`` grouped by a selection of columns from a `DataFrame`.

        One column (or the DataFrame index) has to represent the time, a list of columns `group_cols` must be used for
        extracting the individual TimeSeries by groups, and a list of columns `value_cols` has to represent the values
        for the individual time series. Values from columns ``group_cols`` and ``static_cols`` are added as static
        covariates to the resulting TimeSeries objects. These can be viewed with `my_series.static_covariates`.
        Different to `group_cols`, `static_cols` only adds the static values but are not used to extract the TimeSeries
        groups.

        Parameters
        ----------
        df
            The DataFrame, or anything which can be converted to a narwhals DataFrame (e.g. pandas.DataFrame,
            polars.DataFrame, ...). See the `narwhals documentation
            <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_native>`__ for more
            information.
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
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__ for more info on
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
        copy
            Whether to copy the `times` (DataFrame index or the `time_col` column) and DataFrame `values`.
            If `copy=False`, mutating the series data will affect the original data. Additionally, if `times` lack a
            frequency or step size, it will be assigned to the original object.

        Returns
        -------
        List[TimeSeries]
            A list of series, where each series represents one group from the DataFrame.

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
        df = nw.from_native(df, eager_only=True, pass_through=False)
        if time_col is None:
            if not df.implementation.is_pandas():
                raise_log(
                    ValueError(
                        "`time_col` is required when `df` is not a `pandas.DataFrame`."
                    ),
                    logger=logger,
                )
            is_sorted = nw.maybe_get_index(df).is_monotonic_increasing
        else:
            is_sorted = df.get_column(time_col).is_sorted()

            if df.implementation.is_pandas():
                # with pandas we can get a performance boost by converting the time_col to index
                time_index = dataframe_col_to_time_index(df, time_col)
                df: pd.DataFrame = df.drop(time_col).to_native().set_index(time_index)
                df = nw.from_native(df)
                time_col = None

        if is_sorted:
            logger.warning(
                "UserWarning: The (time) index from `df` is monotonically increasing. This may "
                "result in time series groups with non-overlapping (time) index. You can ignore this "
                "warning if the index represents the actual index of each individual time series group."
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
            value_cols = [
                col
                for col in df.columns
                if col
                not in set(static_cov_cols + extract_metadata_cols + extract_time_col)
            ]
        extract_value_cols = [value_cols] if isinstance(value_cols, str) else value_cols

        df = df[
            static_cov_cols
            + extract_value_cols
            + extract_time_col
            + extract_metadata_cols
        ]

        groups = df.group_by(group_cols[0] if len(group_cols) == 1 else group_cols)

        # not all backends maintain the order when grouping; need to sort the groups in the end for reproducibility
        unique_groups = df[group_cols].unique().sort(by=group_cols).to_numpy()
        sorted_group_idx = {
            tuple(group_): idx for idx, group_ in enumerate(unique_groups)
        }

        # build progress bar for iterator
        iterator = _build_tqdm_iterator(
            groups,
            verbose=verbose,
            total=len(unique_groups),
            desc="Creating TimeSeries",
        )

        def from_group(static_cov_vals, group):
            static_cov_vals = (
                (static_cov_vals,)
                if not isinstance(static_cov_vals, tuple)
                else static_cov_vals
            )
            group_idx = static_cov_vals
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
                static_cov_vals += group[static_cols].row(0)

            metadata = None
            if metadata_cols:
                # use first value as metadata (assume only one unique per group)
                metadata = {
                    col: val
                    for col, val in zip(metadata_cols, group[metadata_cols].row(0))
                }

            return (
                group_idx,
                cls.from_dataframe(
                    df=group,
                    time_col=time_col,
                    value_cols=extract_value_cols,
                    fill_missing_dates=fill_missing_dates,
                    freq=freq,
                    fillna_value=fillna_value,
                    static_covariates=(
                        pd.DataFrame([static_cov_vals], columns=extract_static_cov_cols)
                        if extract_static_cov_cols
                        else None
                    ),
                    metadata=metadata,
                    copy=copy,
                ),
            )

        series_groups = _parallel_apply(
            iterator,
            from_group,
            n_jobs,
            fn_args=dict(),
            fn_kwargs=dict(),
        )

        # re-order series to get reproducible results
        series = [None] * len(sorted_group_idx)
        for group_i, series_group in series_groups:
            series[sorted_group_idx[group_i]] = series_group
        return series

    @classmethod
    def from_series(
        cls,
        pd_series,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        fillna_value: Optional[float] = None,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        metadata: Optional[dict] = None,
        copy: bool = True,
    ) -> Self:
        """Create a ``TimeSeries`` from a `Series`.

        The series must contain an index that is either a pandas DatetimeIndex, a pandas RangeIndex, or a pandas Index
        that can be converted into a RangeIndex. It is better if the index has no holes; alternatively setting
        `fill_missing_dates` can in some cases solve these issues (filling holes with NaN, or with the provided
        `fillna_value` numeric value, if any).

        Parameters
        ----------
        pd_series
            The Series, or anything which can be converted to a narwhals Series (e.g. pandas.Series, ...). See the
            `narwhals documentation
            <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_native>`__ for more
            information.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates (or indices in case of integer index)
            with NaN values. This requires either a provided `freq` or the possibility to infer the frequency from the
            provided timestamps. See :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string or integer representing the frequency of the underlying index. This is useful in order
            to fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
            If a string, represents the frequency of the pandas DatetimeIndex (see `offset aliases
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__ for more info on
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
        copy
            Whether to copy the Series' `values`. If `copy=False`, mutating the series data will affect the original
            data.

        Returns
        -------
        TimeSeries
            The resulting series.

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
            copy=copy,
        )

    @classmethod
    def from_times_and_values(
        cls,
        times: Union[pd.DatetimeIndex, pd.RangeIndex, pd.Index],
        values: np.ndarray,
        fill_missing_dates: Optional[bool] = False,
        freq: Optional[Union[str, int]] = None,
        columns: Optional[Union[Sequence, str]] = None,
        fillna_value: Optional[float] = None,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hierarchy: Optional[dict] = None,
        metadata: Optional[dict] = None,
        copy: bool = True,
    ) -> Self:
        """Create a ``TimeSeries`` from a time index and value array.

        Parameters
        ----------
        times
            A pandas DateTimeIndex, RangeIndex, or Index that can be converted to a RangeIndex representing the time
            axis for the time series. It is better if the index has no holes; alternatively setting
            `fill_missing_dates` can in some cases solve these issues (filling holes with NaN, or with the provided
            `fillna_value` numeric value, if any).
        values
            A Numpy array, or array-like of values for the TimeSeries. Both 2-dimensional arrays, for deterministic
            series, and 3-dimensional arrays, for probabilistic series, are accepted. In the former case the dimensions
            should be (time, component), and in the latter case (time, component, sample).
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates (or indices in case of integer index)
            with NaN values. This requires either a provided `freq` or the possibility to infer the frequency from the
            provided timestamps. See :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string or integer representing the frequency of the underlying index. This is useful in order
            to fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
            If a string, represents the frequency of the pandas DatetimeIndex (see `offset aliases
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        columns
            Optionally, some column names to use for the second `values` dimension.
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
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`__.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.
        copy
            Whether to copy the `times` and `values` objects. If `copy=False`, mutating the series data will affect the
            original data. Additionally, if `times` lack a frequency or step size, it will be assigned to the original
            object.

        Returns
        -------
        TimeSeries
            The resulting series.

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
        return cls(
            times=times,
            values=values,
            fill_missing_dates=fill_missing_dates,
            freq=freq,
            components=columns,
            fillna_value=fillna_value,
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
            copy=copy,
        )

    @classmethod
    def from_values(
        cls,
        values: np.ndarray,
        columns: Optional[Union[Sequence, str]] = None,
        fillna_value: Optional[float] = None,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hierarchy: Optional[dict] = None,
        metadata: Optional[dict] = None,
        copy: bool = True,
    ) -> Self:
        """Create an ``TimeSeries`` from an array of values.

        The series will have an integer time index (RangeIndex).

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
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`__.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.
        copy
            Whether to copy the `values`. If `copy=False`, mutating the series data will affect the original data.

        Returns
        -------
        TimeSeries
            The resulting series.

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
        return cls(
            times=pd.RangeIndex(start=0, stop=len(values), step=1),
            values=values,
            fill_missing_dates=False,
            freq=None,
            components=columns,
            fillna_value=fillna_value,
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
            copy=copy,
        )

    @classmethod
    def from_json(
        cls,
        json_str: str,
        static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hierarchy: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> Self:
        """Create a ``TimeSeries`` from the JSON String representation of a ``TimeSeries``.

        The JSON String representation can be generated with :func:`TimeSeries.to_json()`.

        At the moment this only supports deterministic time series (i.e., made of 1 sample).

        If the JSON string contains static covariates, hierarchy, or metadata, they will be automatically
        loaded. The optional parameters `static_covariates`, `hierarchy`, and `metadata` can be used to
        override or provide these values if they are not present in the JSON string.

        Parameters
        ----------
        json_str
            The JSON String to convert.
        static_covariates
            Optionally, a set of static covariates to be added to the TimeSeries. Either a pandas Series or a pandas
            DataFrame. If a Series, the index represents the static variables. The covariates are globally 'applied'
            to all components of the TimeSeries. If a DataFrame, the columns represent the static variables and the
            rows represent the components of the uni/multivariate TimeSeries. If a single-row DataFrame, the covariates
            are globally 'applied' to all components of the TimeSeries. If a multi-row DataFrame, the number of
            rows must match the number of components of the TimeSeries (in this case, the number of columns in
            ``value_cols``). This adds control for component-specific static covariates.
            If the JSON string already contains static covariates, this parameter will override them.
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
            <https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.reconciliation.html>`__.
            If the JSON string already contains a hierarchy, this parameter will override it.
        metadata
            Optionally, a dictionary with metadata to be added to the TimeSeries.
            If the JSON string already contains metadata, this parameter will override it.

        Returns
        -------
        TimeSeries
            The resulting series.

        Examples
        --------
        >>> from darts import TimeSeries
        >>> json_str = (
        >>>     '{"columns":["vals"],"index":["2020-01-01","2020-01-02","2020-01-03"],"data":[[0.0],[1.0],[2.0]]}'
        >>> )
        >>> series = TimeSeries.from_json(json_str)
        >>> series.shape
        (3, 1, 1)
        """
        parsed = json.loads(json_str)

        static_covariates_ = parsed.pop("static_covariates", None)
        if static_covariates_ is not None and static_covariates is None:
            static_covariates = pd.read_json(
                StringIO(json.dumps(static_covariates_)), orient="split"
            )

        hierarchy_ = parsed.pop("hierarchy", None)
        if hierarchy is None:
            hierarchy = hierarchy_

        metadata_ = parsed.pop("metadata", None)
        if metadata is None:
            metadata = metadata_

        df = pd.read_json(StringIO(json.dumps(parsed)), orient="split")
        return cls.from_dataframe(
            df=df,
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
            copy=False,
        )

    @classmethod
    def from_pickle(cls, path: str) -> Self:
        """Read a pickled ``TimeSeries``.

        Parameters
        ----------
        path : string
            path pointing to a pickle file that will be loaded.

        Returns
        -------
        TimeSeries
            The resulting series.
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)

    """
    Properties
    ==========
    """

    @property
    def static_covariates(self) -> Optional[pd.DataFrame]:
        """The static covariates of this series.

        If defined, the static covariates are given as a `pandas.DataFrame`. The columns represent the static variables
        and the rows represent the components of the series.
        """
        return self._attrs.get(STATIC_COV_TAG, None)

    @property
    def hierarchy(self) -> Optional[dict]:
        """The hierarchy of this series.

        If defined, the hierarchy is given as a dictionary. The keys are the individual components and values are the
        set of parent(s) of these components in the hierarchy.
        """
        return self._attrs.get(HIERARCHY_TAG, None)

    @property
    def metadata(self) -> Optional[dict]:
        """The metadata of this series.

        If defined, the metadata is given as a dictionary.
        """
        return self._attrs.get(METADATA_TAG, None)

    @property
    def top_level_component(self) -> Optional[str]:
        """The top level component name of this series, or `None` if the series has no hierarchy."""
        return self._top_level_component

    @property
    def bottom_level_components(self) -> Optional[list[str]]:
        """The bottom level component names of this series, or `None` if the series has no hierarchy."""
        return self._bottom_level_components

    @property
    def top_level_series(self) -> Optional[Self]:
        """The univariate series containing the single top-level component of this series, or `None` if the series has
        no hierarchy.
        """
        return self[self.top_level_component] if self.has_hierarchy else None

    @property
    def bottom_level_series(self) -> Optional[list[Self]]:
        """The series containing the bottom-level components of this series in the same order as they appear in the
        series, or `None` if the series has no hierarchy.
        """
        return (
            self[[c for c in self.components if c in self.bottom_level_components]]
            if self.has_hierarchy
            else None
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the series `(n_timesteps, n_components, n_samples)`."""
        return self._values.shape

    @property
    def n_timesteps(self) -> int:
        """The number of time steps in the series."""
        return self.shape[TIME_AX]

    @property
    def n_samples(self) -> int:
        """The number of samples contained in the series."""
        return self.shape[SMPL_AX]

    @property
    def n_components(self) -> int:
        """The number of components (columns) in the series."""
        return self.shape[COMP_AX]

    @property
    def width(self) -> int:
        """The width (number of components) of the series."""
        return self.n_components

    @property
    def is_deterministic(self) -> bool:
        """Whether the series is deterministic."""
        return self.shape[SMPL_AX] == 1

    @property
    def is_stochastic(self) -> bool:
        """Whether the series is stochastic (probabilistic)."""
        return not self.is_deterministic

    @property
    def is_probabilistic(self) -> bool:
        """Whether the series is stochastic (probabilistic)."""
        return self.is_stochastic

    @property
    def is_univariate(self) -> bool:
        """Whether the series is univariate."""
        return self.shape[COMP_AX] == 1

    @property
    def freq(self) -> Union[pd.DateOffset, int]:
        """The frequency of the series.

        A ``pandas.DateOffset`` if the series is indexed with a ``pandas.DatetimeIndex``.
        An integer (step size) if the series is indexed with a ``pandas.RangeIndex``.
        """
        return self._freq

    @property
    def freq_str(self) -> str:
        """The string representation of the series' frequency."""
        return self._freq_str

    @property
    def dtype(self):
        """The dtype of the series' values."""
        return self._values.dtype

    @property
    def components(self) -> pd.Index:
        """The component (column) names of the series, as a ``pandas.Index``."""
        return self._components

    @property
    def columns(self) -> pd.Index:
        """The component (column) names of the series, as a ``pandas.Index``."""
        return self.components

    @property
    def time_index(self) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
        """The time index of the series."""
        return self._time_index.copy()

    @property
    def time_dim(self) -> str:
        """The time dimension name of the series."""
        return self._time_dim

    @property
    def has_datetime_index(self) -> bool:
        """Whether the series is indexed with a ``pandas.DatetimeIndex`` (otherwise it is indexed with an
        ``pandas.RangeIndex``)."""
        return self._has_datetime_index

    @property
    def has_range_index(self) -> bool:
        """Whether the series is indexed with an ``pandas.RangeIndex`` (otherwise it is indexed with a
        ``pandas.DatetimeIndex``).
        """
        return not self._has_datetime_index

    @property
    def has_hierarchy(self) -> bool:
        """Whether the series contains a hierarchy."""
        return self.hierarchy is not None

    @property
    def has_static_covariates(self) -> bool:
        """Whether the series contains static covariates."""
        return self.static_covariates is not None

    @property
    def has_metadata(self) -> bool:
        """Whether the series contains metadata."""
        return self.metadata is not None

    @property
    def duration(self) -> Union[pd.Timedelta, int]:
        """The duration of the series (as a ``pandas.Timedelta`` or `int`)."""
        return self._time_index[-1] - self._time_index[0]

    """
    Export functions
    ================
    """

    def data_array(self, copy: bool = True) -> xr.DataArray:
        """Return an ``xarray.DataArray`` representation of the series.

        Parameters
        ----------
        copy
            Whether to return a copy of the series. Leave it to True unless you know what you are doing.

        Returns
        -------
        xarray.DataArray
            An ``xarray.DataArray`` representation of  represents the time series.
        """
        xa = xr.DataArray(
            self._values,
            dims=(self._time_dim,) + DIMS[-2:],
            coords={self._time_dim: self._time_index, DIMS[COMP_AX]: self.components},
            attrs=self._attrs,
        )
        return xa.copy() if copy else xa

    def to_series(
        self,
        copy: bool = True,
        backend: Union[ModuleType, Implementation, str] = Implementation.PANDAS,
    ):
        """Return a `Series` representation of the series in a given `backend`.

        Works only for univariate series that are deterministic (i.e., made of 1 sample).

        Parameters
        ----------
        copy
            Whether to return a copy of the series. Leave it to True unless you know what you are doing.
        backend
            The backend to which to export the `TimeSeries`. See the `narwhals documentation
            <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_dict>`__ for all supported
            backends.

        Returns
        -------
            A Series representation of the series in a given `backend`.
        """
        self._assert_univariate()
        self._assert_deterministic()

        backend = Implementation.from_backend(backend)
        if not backend.is_pandas():
            return self.to_dataframe(copy=copy, backend=backend, time_as_index=False)

        data = self._values[:, 0, 0]
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
        add_static_cov: Optional[Union[list[str], str, bool]] = False,
        add_metadata: Optional[Union[list[str], str, bool]] = False,
    ):
        """Return a DataFrame representation of the series in a given `backend`.

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
            <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_dict>`__ for all supported
            backends.
        time_as_index
            Whether to set the time index as the index of the dataframe or in the left-most column.
            Only effective with the pandas `backend`.
        suppress_warnings
            Whether to suppress the warnings for the `DataFrame` creation.
        add_static_cov
            Whether to add static covariates from the time series as columns in the resulting dataframe (one column per
            component-static covariate pair). Can be a bool in case all the static covariates should be added, or a
            string/list of string in case only a subset are needed.
        add_metadata
            Whether to add metadata from the time series as columns in the resulting dataframe (one column per
            metadata). Can be a bool in case all the metadata should be added, or a string/list of string in case only
            a subset are needed.
        Returns
        -------
        DataFrame
            A DataFrame representation of the series in a given `backend`.
        """

        backend = Implementation.from_backend(backend)
        if time_as_index and not backend.is_pandas():
            if not suppress_warnings:
                logger.warning(
                    '`time_as_index=True` is only supported with `backend="pandas"`, and will be ignored.'
                )
            time_as_index = False

        values = self._values
        if not self.is_deterministic:
            if not suppress_warnings:
                logger.warning(
                    "You are transforming a stochastic TimeSeries (it contains several samples). "
                    "The resulting DataFrame is a 2D object with all samples on the columns. "
                    "If this is not the expected behavior, extract statistics from the TimeSeries "
                    "before calling `to_dataframe()` (e.g. with `TimeSeries.quantile()`, `mean()`, "
                    "...)."
                )

            comp_name = list(self.components)
            samples = range(self.n_samples)
            columns = [
                "_s".join((comp_name, str(sample_id)))
                for comp_name, sample_id in itertools.product(comp_name, samples)
            ]
            data = values.reshape(values.shape[0], len(columns))
        else:
            columns = self.components
            data = values[:, :, 0]
        data_dict = {col: data[:, idx] for idx, col in enumerate(columns)}

        # handle static covariates
        if self.has_static_covariates and add_static_cov:
            static_covs = self.static_covariates
            components = list(static_covs.index)

            if isinstance(add_static_cov, bool):
                # Add all the static cov cols
                static_cov_cols = static_covs.columns
            elif isinstance(add_static_cov, (str, list)):
                static_cov_cols = (
                    [add_static_cov]
                    if isinstance(add_static_cov, str)
                    else add_static_cov
                )
                if not all(isinstance(x, str) for x in static_cov_cols):
                    raise_log(
                        ValueError("All values in add_static_cov must be of type str"),
                        logger=logger,
                    )
                missing_cols = [
                    col for col in static_cov_cols if col not in static_covs.columns
                ]
                if missing_cols:
                    raise_log(
                        ValueError(
                            f"The following static covariates to add via `add_static_cov` do not exist: {missing_cols}."
                            f"Available static covariates are: {list(static_covs.columns)}"
                        ),
                        logger=logger,
                    )
            else:
                raise_log(
                    ValueError("add_static_cov must be of type bool, str or list[str]"),
                    logger=logger,
                )
            for static_cov_col in static_cov_cols:
                for comp in components:
                    value = static_covs.loc[comp, static_cov_col]
                    data_col = np.full(data.shape[0], value)
                    if len(components) > 1:
                        column = "_".join((comp, static_cov_col))
                    else:
                        column = static_cov_col
                    data_dict[column] = data_col

        # handle metadata
        if self.has_metadata and add_metadata:
            metadata = self.metadata
            if isinstance(add_metadata, bool):
                # Add all the metadata
                metadata_cols = metadata.keys()
            elif isinstance(add_metadata, (str, list)):
                metadata_cols = (
                    [add_metadata] if isinstance(add_metadata, str) else add_metadata
                )
                if not all(isinstance(x, str) for x in metadata_cols):
                    raise_log(
                        ValueError("All values in add_metadata must be of type str"),
                        logger=logger,
                    )
                missing_cols = [
                    col for col in metadata_cols if col not in metadata.keys()
                ]
                if missing_cols:
                    raise_log(
                        ValueError(
                            f"The following metadata to add via `add_metadata` do not exist: {missing_cols}."
                            f"Available static covariates are: {list(metadata.keys())}"
                        ),
                        logger=logger,
                    )
            else:
                raise_log(
                    ValueError("add_metadata must be of type bool, str or list[str]"),
                    logger=logger,
                )
            for metadata_col in metadata_cols:
                data_col = np.full(data.shape[0], metadata[metadata_col])
                data_dict[metadata_col] = data_col
        time_index = self._time_index

        if copy:
            data_dict = data_dict.copy()
            time_index = time_index.copy()

        if time_as_index:
            # special path for pandas with index
            output_df = pd.DataFrame.from_dict(data=data_dict)
            output_df.index = time_index
            return output_df

        data_dict = {time_index.name: time_index, **data_dict}

        return nw.from_dict(data_dict, backend=backend).to_native()

    def schema(self, copy: bool = True) -> dict[str, Any]:
        """Return the schema of the series as a dictionary.

        Can be used to create new `TimeSeries` with the same schema.

        The keys and values are:

        - "time_freq": the frequency (or step size) of the time (or range) index
        - "time_name": the name of the time index
        - "columns": the columns / components
        - "static_covariates": the static covariates
        - "hierarchy": the hierarchy
        - "metadata": the metadata
        """
        schema = {
            "time_freq": self._freq,
            "time_name": self._time_index.name,
            "columns": self.components,
            STATIC_COV_TAG: self.static_covariates,
            HIERARCHY_TAG: self.hierarchy,
            METADATA_TAG: self.metadata,
        }
        if copy:
            schema = {k: deepcopy(v) for k, v in schema.items()}
        return schema

    def astype(self, dtype: Union[str, np.dtype]) -> Self:
        """Return a new series with the values have been converted to the desired `dtype`.

        Parameters
        ----------
        dtype
            A NumPy dtype (numpy.float32 or numpy.float64)

        Returns
        -------
        TimeSeries
            A series having the desired dtype.
        """
        return self.__class__(
            times=self._time_index,
            values=self._values.astype(dtype),
            components=self.components,
            **self._attrs,
        )

    def start_time(self) -> Union[pd.Timestamp, int]:
        """Start time of the series.

        Returns
        -------
        Union[pandas.Timestamp, int]
            A timestamp containing the first time of the TimeSeries (if indexed by DatetimeIndex),
            or an integer (if indexed by RangeIndex)
        """
        return self._time_index[0]

    def end_time(self) -> Union[pd.Timestamp, int]:
        """End time of the series.

        Returns
        -------
        Union[pandas.Timestamp, int]
            A timestamp containing the last time of the TimeSeries (if indexed by DatetimeIndex),
            or an integer (if indexed by RangeIndex)
        """
        return self._time_index[-1]

    def first_value(self) -> float:
        """First value of the univariate series.

        Returns
        -------
        float
            The first value of this univariate deterministic time series
        """
        self._assert_univariate()
        self._assert_deterministic()
        return float(self._values[0, 0, 0])

    def last_value(self) -> float:
        """Last value of the univariate series.

        Returns
        -------
        float
            The last value of this univariate deterministic time series
        """
        self._assert_univariate()
        self._assert_deterministic()
        return float(self._values[-1, 0, 0])

    def first_values(self) -> np.ndarray:
        """First values of the potentially multivariate series.

        Returns
        -------
        numpy.ndarray
            The first values of every component of this deterministic time series
        """
        self._assert_deterministic()
        return self._values[0, :, 0].copy()

    def last_values(self) -> np.ndarray:
        """Last values of the potentially multivariate series.

        Returns
        -------
        numpy.ndarray
            The last values of every component of this deterministic time series
        """
        self._assert_deterministic()
        return self._values[-1, :, 0].copy()

    def values(self, copy: bool = True, sample: int = 0) -> np.ndarray:
        """Return a 2-D array of shape (time, component), containing the series' values for one `sample`.

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
        if self.is_deterministic and sample != 0:
            raise_log(
                ValueError(
                    "This series contains one sample only (deterministic),"
                    "so only sample=0 is accepted.",
                ),
                logger=logger,
            )
        values = self._values[:, :, sample]
        if copy:
            values = values.copy()
        return values

    def random_component_values(self, copy: bool = True) -> np.array:
        """Return a 2-D array of shape (time, component), containing the series' values for one sample taken uniformly
        at random from all samples.

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
        values = self._values[:, :, sample]
        if copy:
            values = values.copy()
        return values

    def all_values(self, copy: bool = True) -> np.ndarray:
        """Return a 3-D array of dimension (time, component, sample) containing the series' values for all samples.

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
        values = self._values
        if copy:
            values = values.copy()
        return values

    def univariate_values(self, copy: bool = True, sample: int = 0) -> np.ndarray:
        """Return a 1-D Numpy array of shape (time,) containing the univariate series' values for one `sample`.

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
        values = self._values[:, 0, sample]
        if copy:
            values = values.copy()
        return values

    def static_covariates_values(self, copy: bool = True) -> Optional[np.ndarray]:
        """Return a 2-D array of dimension (component, static variable) containing the series' static covariate values.

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
        """Return a new series with the first `size` points.

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

        axis = self._get_axis(axis)
        display_n = min(size, self.shape[axis])

        if axis == TIME_AX:
            return self[:display_n]
        elif axis == COMP_AX:
            return self[self.components.tolist()[:display_n]]
        else:
            return self.__class__(
                times=self._time_index,
                values=self._values[:, :, :display_n],
                components=self.components,
                **self._attrs,
            )

    def tail(
        self, size: Optional[int] = 5, axis: Optional[Union[int, str]] = 0
    ) -> Self:
        """Return a new series with the last `size` points.

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
        axis = self._get_axis(axis)
        display_n = min(size, self.shape[axis])

        if axis == TIME_AX:
            return self[-display_n:]
        elif axis == COMP_AX:
            return self[self.components.tolist()[-display_n:]]
        else:
            return self.__class__(
                times=self._time_index,
                values=self._values[:, :, -display_n:],
                components=self.components,
                **self._attrs,
            )

    def concatenate(
        self,
        other: Self,
        axis: Optional[Union[str, int]] = 0,
        ignore_time_axis: Optional[bool] = False,
        ignore_static_covariates: bool = False,
        drop_hierarchy: bool = True,
        drop_metadata: bool = False,
    ) -> Self:
        """Return a new series where this series is concatenated with the `other` series along a given `axis`.

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
            The concatenated series.

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
        """Compute and return gaps in the series.

        Works only on deterministic time series (1 sample).

        Parameters
        ----------
        mode
            Only relevant for multivariate time series. The mode defines how gaps are defined. Set to
            'any' if a NaN value in any columns should be considered as as gaps. 'all' will only
            consider periods where all columns' values are NaN. Defaults to 'all'.

        Returns
        -------
        pandas.DataFrame
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
        """Create a copy of the series.

        Returns
        -------
        TimeSeries
            A copy of the series.
        """

        # the data will be copied in the TimeSeries constructor.
        return self.__class__(
            times=self._time_index,
            values=self._values,
            components=self.components,
            copy=True,
            **self._attrs,
        )

    def get_index_at_point(
        self, point: Union[pd.Timestamp, float, int], after=True
    ) -> int:
        """Convert a point along the time index into an integer index ranging from (0, len(series)-1) inclusive.

        Parameters
        ----------
        point
            This parameter supports 3 different data types: ``pandas.Timestamp``, ``float`` and ``int``.

            ``pandas.Timestamp`` work only on series that are indexed with a ``pandas.DatetimeIndex``. In such cases,
            the returned point will be the index of this timestamp if it is present in the series time index.
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

        Returns
        -------
        int
            The index position corresponding to the provided point in the series.
        """
        point_index = -1
        if isinstance(point, float):
            if not 0.0 <= point <= 1.0:
                raise_log(
                    ValueError("point (float) should be between 0.0 and 1.0."), logger
                )
            point_index = int((len(self) - 1) * point)
        elif isinstance(point, (int, np.int64)):
            if self.has_datetime_index or (self.start_time() == 0 and self.freq == 1):
                point_index = point
            else:
                point_index_float = (point - self.start_time()) / self.freq
                point_index = int(point_index_float)
                if point_index != point_index_float:
                    raise_log(
                        ValueError(
                            "The provided point is not a valid index for this series."
                        ),
                        logger,
                    )
            if not 0 <= point_index < len(self):
                raise_log(
                    ValueError(
                        f"The index corresponding to the provided point ({point}) should be a valid index in series"
                    ),
                    logger,
                )
        elif isinstance(point, pd.Timestamp):
            if not self._has_datetime_index:
                raise_log(
                    ValueError(
                        "A Timestamp has been provided, but this series is not time-indexed."
                    ),
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
                    "`point` needs to be either `float`, `int` or `pandas.Timestamp`"
                ),
                logger,
            )
        return point_index

    def get_timestamp_at_point(
        self, point: Union[pd.Timestamp, float, int]
    ) -> Union[pd.Timestamp, int]:
        """Convert a point into a ``pandas.Timestamp`` (if datetime-indexed) or integer (if integer-indexed).

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

        Returns
        -------
        Union[pandas.Timestamp, int]
            The index value corresponding to the provided point in the series.
            If the series is indexed by a `pandas.DatetimeIndex`, returns a `pandas.Timestamp`.
            If the series is indexed by a `pandas.RangeIndex`, returns an integer.
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
        """Split the series in two, after a provided `split_point`.

        Parameters
        ----------
        split_point
            A timestamp, float or integer. If float, represents the proportion of the series to include in the
            first TimeSeries (must be between 0.0 and 1.0). If integer, represents the index position after
            which the split is performed. A pandas.Timestamp can be provided for TimeSeries that are indexed by a
            pandas.DatetimeIndex. In such cases, the timestamp will be contained in the first TimeSeries, but not
            in the second one. The timestamp itself does not have to appear in the original TimeSeries index.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries]
            A tuple of two series. The first time series contains the first entries up to the `split_point` (inclusive),
            and the second contains the remaining ones.
        """
        return self._split_at(split_point, after=True)

    def split_before(
        self, split_point: Union[pd.Timestamp, float, int]
    ) -> tuple[Self, Self]:
        """Split the series in two, before a provided `split_point`.

        Parameters
        ----------
        split_point
            A timestamp, float or integer. If float, represents the proportion of the series to include in the
            first TimeSeries (must be between 0.0 and 1.0). If integer, represents the index position before
            which the split is performed. A pandas.Timestamp can be provided for TimeSeries that are indexed by a
            pandas.DatetimeIndex. In such cases, the timestamp will be contained in the second TimeSeries, but not
            in the first one. The timestamp itself does not have to appear in the original TimeSeries index.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries]
            A tuple of two series. The first time series contains the first entries up to the `split_point` (exclusive),
            and the second contains the remaining ones.
        """
        return self._split_at(split_point, after=False)

    def drop_after(
        self,
        split_point: Union[pd.Timestamp, float, int],
        keep_point: bool = False,
    ):
        """Return a new series where everything after (and in-/excluding) the provided time `split_point` was dropped.

        The timestamp may not be in the series. If it is, the timestamp will be dropped.

        Parameters
        ----------
        split_point
            The timestamp that indicates cut-off time.
        keep_point
            Whether the provided `split_point` should be included in the returned series (if it exists in the series).

        Returns
        -------
        TimeSeries
            A series that contains all entries until `split_point` (exclusive).
        """
        return self[
            : self.get_index_at_point(split_point, after=not keep_point)
            + int(keep_point)
        ]

    def drop_before(
        self,
        split_point: Union[pd.Timestamp, float, int],
        keep_point: bool = False,
    ):
        """Return a new series where everything before (and in-/excluding) the provided time `split_point` was dropped.

        The timestamp may not be in the series. If it is, the timestamp will be dropped.

        Parameters
        ----------
        split_point
            The timestamp that indicates cut-off time.
        keep_point
            Whether the provided `split_point` should be included in the returned series (if it exists in the series).

        Returns
        -------
        TimeSeries
            A series that contains all entries starting after `split_point` (exclusive).
        """
        return self[
            self.get_index_at_point(split_point, after=keep_point)
            + int(not keep_point) :
        ]

    def slice(
        self, start_ts: Union[pd.Timestamp, int], end_ts: Union[pd.Timestamp, int]
    ):
        """Return a slice of the series starting at `start_ts` and ending before `end_ts`.

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
        if type(start_ts) is not type(end_ts):
            raise_log(
                ValueError(
                    "The two timestamps provided to slice() have to be of the same type."
                ),
                logger,
            )
        if isinstance(start_ts, pd.Timestamp):
            if not self._has_datetime_index:
                raise_log(
                    ValueError(
                        "Timestamps have been provided to slice(), but the series is "
                        "indexed using an integer-based RangeIndex."
                    ),
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
            if self._has_datetime_index:
                raise_log(
                    ValueError(
                        "start and end times have been provided as integers to slice(), but "
                        "the series is indexed with a DatetimeIndex."
                    ),
                    logger,
                )
            # get closest timestamp if either start or end are not in the index
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
                # we have to increase effective_end_ts to make the last timestamp inclusive.
                effective_end_ts += self.freq
            idx = pd.RangeIndex(effective_start_ts, effective_end_ts, step=self.freq)
            return self[idx]

    def slice_n_points_after(self, start_ts: Union[pd.Timestamp, int], n: int) -> Self:
        """Return a slice of the series starting at `start_ts` (inclusive) and having at most `n` points.

        Parameters
        ----------
        start_ts
            The timestamp or index that indicates the splitting time.
        n
            The maximal length of the new TimeSeries.

        Returns
        -------
        TimeSeries
            A new series, with length at most `n`, starting at `start_ts`.
        """
        if n <= 0:
            raise_log(ValueError("n should be a positive integer."), logger)
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
        """Return a slice of the series ending at `end_ts` (inclusive) and having at most `n` points.

        Parameters
        ----------
        end_ts
            The timestamp or index that indicates the splitting time.
        n
            The maximal length of the new TimeSeries.

        Returns
        -------
        TimeSeries
            A new series, with length at most `n`, ending at `end_ts`.
        """
        if n <= 0:
            raise_log(ValueError("n should be a positive integer."), logger)
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
        """Return a slice of the series where the time index was intersected with the `other` series.

        This method is in general *not* symmetric.

        Parameters
        ----------
        other
            the other time series

        Returns
        -------
        TimeSeries
            A new series, containing the values of this series, over the time-span common to both series.
        """
        if other.has_same_time_as(self):
            return self.copy()
        elif other.freq == self.freq and len(self) and len(other):
            start, end = self._slice_intersect_bounds(other)
            return self[start:end]
        else:
            time_index = self.time_index.intersection(other.time_index)
            return self[time_index]

    def slice_intersect_values(self, other: Self, copy: bool = False) -> np.ndarray:
        """Return the sliced values of the series where the time index was intersected with the `other` series.

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
        numpy.ndarray
            The values of this series, over the time-span common to both series.
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
        """Return the time index of the series where the time index was intersected with the `other` series.

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
        Union[pandas.DatetimeIndex, pandas.RangeIndex]
            The time index of this series, over the time-span common to both series.
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
        """Return a slice of the deterministic time series where NaN-containing entries at the beginning and the end
        were removed.

        No entries after (and including) the first non-NaN entry and before (and including) the last non-NaN entry are
        removed.

        This method is only applicable to deterministic series (i.e., having 1 sample).

        Parameters
        ----------
        how
            Define if the entries containing `NaN` in all the components ('all') or in any of the components ('any')
            should be stripped. Default: 'all'

        Returns
        -------
        TimeSeries
            A new series where NaN-containing entries at start and end were removed.
        """
        if self.is_probabilistic:
            raise_log(
                ValueError("`strip` cannot be applied to stochastic TimeSeries"), logger
            )

        first_finite_row, last_finite_row = _finite_rows_boundaries(
            self.values(copy=False), how=how
        )

        return self.__class__(
            times=self._time_index[first_finite_row : last_finite_row + 1],
            values=self._values[first_finite_row : last_finite_row + 1],
            components=self.components,
            **self._attrs,
        )

    def longest_contiguous_slice(
        self, max_gap_size: int = 0, mode: str = "all"
    ) -> Self:
        """Return the largest slice of the deterministic series without any gaps (contiguous all-NaN value entries)
        larger than `max_gap_size`.

        This method is only applicable to deterministic series (i.e., having 1 sample).

        Parameters
        ----------
        max_gap_size
            Indicate the maximum gap size that the series can contain.
        mode
            Only relevant for multivariate time series. The mode defines how gaps are defined. Set to
            'any' if a NaN value in any columns should be considered as as gaps. 'all' will only
            consider periods where all columns' values are NaN. Defaults to 'all'.

        Returns
        -------
        TimeSeries
            A new series with the largest slice of the original that has no gaps longer than `max_gap_size`.

        See Also
        --------
        TimeSeries.gaps : return the gaps in the TimeSeries
        """
        if not (np.isnan(self._values)).any():
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
        """Return a new series, which is a multiple of this series such that the first value is `value_at_first_step`.

        Note: Numerical errors can appear with `value_at_first_step > 1e+24`.

        Parameters
        ----------
        value_at_first_step
            The new value for the first entry of the TimeSeries.

        Returns
        -------
        TimeSeries
            A new series, where the first value is `value_at_first_step` and other values have been scaled accordingly.
        """
        if (self._values[0, :, :] == 0).any():
            raise_log(ValueError("Cannot rescale with first value `0`."), logger)
        coef = value_at_first_step / self._values[:1]
        return self.__class__(
            times=self._time_index,
            values=self._values * coef,
            components=self.components,
            **self._attrs,
        )

    def shift(self, n: int) -> Self:
        """Return a new series where the time index was shifted by `n` steps.

        If :math:`n > 0`, shifts into the future. If :math:`n < 0`, shifts into the past.

        For example, with :math:`n=2` and `freq='M'`, March 2013 becomes May 2013.
        With :math:`n=-2`, March 2013 becomes Jan 2013.

        Parameters
        ----------
        n
            The number of time steps (in self.freq unit) to shift by. Can be negative.

        Returns
        -------
        TimeSeries
            A new series, with a shifted time index.
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
        return self.__class__(
            times=new_time_index,
            values=self._values,
            components=self.components,
            **self._attrs,
        )

    def diff(
        self,
        n: Optional[int] = 1,
        periods: Optional[int] = 1,
        dropna: Optional[bool] = True,
    ) -> Self:
        """Return a new series with differenced values.

        This is often used to make a time series stationary.

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
            A new series, with the differenced values.
        """
        if not isinstance(n, int) or n < 1:
            raise_log(ValueError("'n' must be a positive integer >= 1."), logger)
        if not isinstance(periods, int) or periods < 1:
            raise_log(ValueError("'periods' must be an integer >= 1."), logger)

        def _compute_diff(values_: np.ndarray, times_):
            if not dropna:
                # In this case the new DataArray will have the same size and filled with NaNs
                values_diff = values_.copy()
                values_diff[:periods, :, :] = np.nan
                values_diff[periods:, :, :] = (
                    values_[periods:, :, :] - values_[:-periods, :, :]
                )
            else:
                # In this case the new DataArray will be shorter
                times_ = times_[periods:]
                values_diff = values_[periods:, :, :].copy()
                values_diff[:] = values_[periods:, :, :] - values_[:-periods, :, :]
            return values_diff, times_

        values, times = _compute_diff(self._values, self._time_index)
        for _ in range(n - 1):
            values, times = _compute_diff(values, times)
        return self.__class__(
            times=times,
            values=values,
            components=self.components,
            **self._attrs,
        )

    def cumsum(self) -> Self:
        """Return a new series with the cumulative sum along the time axis.

        Returns
        -------
        TimeSeries
            A new series, with the cumulatively summed values.
        """
        return self.__class__(
            times=self._time_index,
            values=self._values.cumsum(axis=0),
            components=self.components,
            **self._attrs,
        )

    def has_same_time_as(self, other: Self) -> bool:
        """Whether the series has the same time index as the `other` series.

        Parameters
        ----------
        other
            the other series

        Returns
        -------
        bool
            `True` if both series have the same index, `False` otherwise.
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
        """Return a new series with the `other` series appended to this series along the time axis (added to the end).

        Parameters
        ----------
        other
            A second TimeSeries.

        Returns
        -------
        TimeSeries
            A new series, obtained by appending the second series to the first.

        See Also
        --------
        TimeSeries.concatenate : concatenate another series along a given axis.
        TimeSeries.prepend : prepend another series along the time axis.
        """
        if other.has_datetime_index != self.has_datetime_index:
            raise_log(
                ValueError(
                    "Both series must have the same type of time index (either DatetimeIndex or RangeIndex)."
                ),
                logger,
            )
        if other.freq != self.freq:
            raise_log(ValueError("Both series must have the same frequency."), logger)
        if other.n_components != self.n_components:
            raise_log(
                ValueError("Both series must have the same number of components."),
                logger,
            )
        if other.n_samples != self.n_samples:
            raise_log(
                ValueError("Both series must have the same number of samples."), logger
            )
        if len(self) > 0 and len(other) > 0:
            if other.start_time() != self.end_time() + self.freq:
                raise_log(
                    ValueError(
                        "Appended TimeSeries must start one (time) step after current one."
                    ),
                    logger,
                )
        values = np.concatenate((self._values, other._values), axis=0)
        times = self._time_index.append(other._time_index)
        return self.__class__(
            times=times,
            values=values,
            components=self.components,
            **self._attrs,
        )

    def append_values(self, values: np.ndarray) -> Self:
        """Return a new series with `values` appended to this series along the time axis (added to the end).

        This adds time steps to the end of the new series.

        Parameters
        ----------
        values
            An array with the values to append.

        Returns
        -------
        TimeSeries
            A new series with the new values appended.

        See Also
        --------
        TimeSeries.prepend_values : prepend the values of another series along the time axis.
        """
        if len(values) == 0:
            return self.copy()

        values = np.array(values) if not isinstance(values, np.ndarray) else values
        values = expand_arr(values, ndim=len(DIMS))
        if not values.shape[1:] == self.shape[1:]:
            raise_log(
                ValueError(
                    f"The (expanded) values must have the same number of components and samples "
                    f"(second and third dims) as the series to append to. "
                    f"Received shape: {values.shape}, expected: {self.shape}"
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
            self.__class__(
                values=values, times=idx, components=self.components, **self._attrs
            )
        )

    def prepend(self, other: Self) -> Self:
        """Return a new series with the `other` series prepended to this series along the time axis (added to the
        beginning).

        Parameters
        ----------
        other
            A second TimeSeries.

        Returns
        -------
        TimeSeries
            A new series, obtained by prepending the second series to the first.

        See Also
        --------
        TimeSeries.concatenate : concatenate another series along a given axis.
        TimeSeries.append : append another series along the time axis.
        """
        if not isinstance(other, self.__class__):
            raise_log(
                ValueError(
                    f"`other` to prepend must be a {self.__class__.__name__} object."
                ),
                logger,
            )
        return other.append(self)

    def prepend_values(self, values: np.ndarray) -> Self:
        """Return a new series with `values` prepended to this series along the time axis (added to the beginning).

        This adds time steps to the beginning of the new series.

        Parameters
        ----------
        values
            An array with the values to prepend to the start.

        Returns
        -------
        TimeSeries
            A new series with the new values prepended.

        See Also
        --------
        TimeSeries.append_values : append the values of another series along the time axis.
        """
        if len(values) == 0:
            return self.copy()

        values = np.array(values) if not isinstance(values, np.ndarray) else values
        values = expand_arr(values, ndim=len(DIMS))
        if not values.shape[1:] == self._values.shape[1:]:
            raise_log(
                ValueError(
                    f"The (expanded) values must have the same number of components and samples "
                    f"(second and third dims) as the series to prepend to. "
                    f"Received shape: {values.shape}, expected: {self._values.shape}"
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
            self.__class__(
                times=idx,
                values=values,
                components=self.columns,
                **self._attrs,
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
        """Return a new series similar to this one but with new `times` and `values`.

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
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__ for more info on
            supported frequencies).
            If an integer, represents the step size of the pandas Index or pandas RangeIndex.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.

        Returns
        -------
        TimeSeries
            A new series with the new time index and values but identical static covariates and hierarchy.
        """
        values = np.array(values) if not isinstance(values, np.ndarray) else values
        values = expand_arr(values, ndim=len(DIMS))
        if values.shape[1] != self.shape[1]:
            raise_log(
                ValueError(
                    "The new values must have the same number of components as the present series. "
                    f"Received: {values.shape[1]}, expected: {self.shape[1]}"
                ),
                logger,
            )
        return self.__class__(
            times=times,
            values=values,
            fill_missing_dates=fill_missing_dates,
            freq=freq,
            components=self.components,
            fillna_value=fillna_value,
            **self._attrs,
        )

    def with_values(self, values: np.ndarray) -> Self:
        """Return a new series similar to this one but with new `values`.

        Parameters
        ----------
        values
            A Numpy array with new values. It must have the dimensions for time
            and components, but may contain a different number of samples.

        Returns
        -------
        TimeSeries
            A new series with the new values but same index, static covariates and hierarchy
        """
        values = np.array(values) if not isinstance(values, np.ndarray) else values
        values = expand_arr(values, ndim=len(DIMS))
        if values.shape[:2] != self.shape[:2]:
            raise_log(
                ValueError(
                    "The new values must have the same shape (time, components) as the present series. "
                    f"Received: {values.shape[:2]}, expected: {self.shape[:2]}"
                ),
                logger,
            )
        return self.__class__(
            times=self._time_index,
            values=values,
            components=self.components,
            **self._attrs,
        )

    def with_static_covariates(
        self, covariates: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> Self:
        """Return a new series with added static covariates.

        Static covariates hold information / data about the time series which does not vary over time.

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
            A new series with the given static covariates.

        Notes
        -----
        If there are a large number of static covariates variables (i.e., the static covariates have a very large
        dimension), there might be a noticeable performance penalty for creating ``TimeSeries``, unless the covariates
        already have the same ``dtype`` as the series data.

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
            times=self._time_index,
            values=self._values,
            components=self.components,
            static_covariates=covariates,
            hierarchy=self.hierarchy,
            metadata=self.metadata,
        )

    def with_hierarchy(self, hierarchy: dict[str, Union[str, list[str]]]) -> Self:
        """Return a new series with added hierarchy.

        Parameters
        ----------
        hierarchy
            A dictionary mapping components to a list of their parent(s) in the hierarchy.
            Single parents may be specified as string or list containing one string.
            For example, assume the series contains the components
            ``["total", "a", "b", "x", "y", "ax", "ay", "bx", "by"]``,
            the following dictionary would encode the groupings shown on
            `this figure <https://otexts.com/fpp3/hts.html#fig:GroupTree>`__:

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
            A new series with the given hierarchy.
        """
        return self.__class__(
            times=self._time_index,
            values=self._values,
            components=self.components,
            static_covariates=self.static_covariates,
            hierarchy=hierarchy,
            metadata=self.metadata,
        )

    def with_metadata(self, metadata: Optional[dict]) -> Self:
        """Return a new series with added metadata.

        Parameters
        ----------
        metadata
            A dictionary with metadata to be added to the TimeSeries.

        Returns
        -------
        TimeSeries
            A new series with the given metadata.

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
            times=self._time_index,
            values=self._values,
            components=self.components,
            static_covariates=self.static_covariates,
            hierarchy=self.hierarchy,
            metadata=metadata,
        )

    def stack(self, other: Self) -> Self:
        """Return a new series with the `other` series stacked to this series along the component axis.

        The resulting TimeSeries will have the same name for its time dimension as this TimeSeries, and the
        same number of samples.

        Parameters
        ----------
        other
            A TimeSeries instance with the same index and the same number of samples as the current one.

        Returns
        -------
        TimeSeries
            A new series with the components of the other series added to the original.
        """
        return concatenate([self, other], axis=1)

    def drop_columns(self, col_names: Union[list[str], str]) -> Self:
        """Return a new series with dropped components (columns).

        Parameters
        ----------
        col_names
            String or list of strings corresponding to the columns to be dropped.

        Returns
        -------
        TimeSeries
            A new series with the specified columns dropped.
        """
        if isinstance(col_names, str):
            col_names = [col_names]

        comp_list = self.components.tolist()
        if not all([x in comp_list for x in col_names]):
            raise_log(
                ValueError(
                    "Some column names in `col_names` don't exist in the time series."
                ),
                logger,
            )
        indexer = []
        for idx, col in enumerate(comp_list):
            if col not in col_names:
                indexer.append(idx)

        return self.__class__(
            times=self._time_index,
            values=self._values[:, indexer],
            components=self.components[indexer],
            static_covariates=(
                self.static_covariates.iloc[indexer]
                if self.static_covariates is not None
                else None
            ),
            hierarchy=None,
            metadata=self.metadata,
        )

    def univariate_component(self, index: Union[str, int]) -> Self:
        """Return a new univariate series with a selected component.

        This drops the hierarchy (if any), and retains only the relevant static covariates column.

        Parameters
        ----------
        index
            If a string, the name of the component to retrieve. If an integer, the positional index of the component.

        Returns
        -------
        TimeSeries
            A new series with a selected component.
        """

        return self[index if isinstance(index, str) else self.components[index]]

    def add_datetime_attribute(
        self,
        attribute,
        one_hot: bool = False,
        cyclic: bool = False,
        tz: Optional[str] = None,
    ) -> Self:
        """Return a new series with one (or more) additional component(s) that contain an attribute of the series' time
        index.

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
            A pandas.DatatimeIndex attribute which will serve as the basis of the new column(s).
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
            A new series with an added datetime attribute component(s).
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
        """Return a new series with an added holiday component.

        The holiday component is binary where `1` corresponds to a time step falling on a holiday.

        Available countries can be found `here <https://holidays.readthedocs.io/en/latest/#available-countries>`__.

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
            A new series with an added holiday component.
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
        """Return a new series where the time index and values were resampled with a given frequency.

        The provided `method` is used to aggregate/fill holes in the resampled series, by default 'pad'.

        Parameters
        ----------
        freq
            The new time difference between two adjacent entries in the returned TimeSeries.
            Expects a `pandas.DateOffset` or `DateOffset` alias.
        method
            A method to either aggregate grouped values (for down-sampling) or fill holes (for up-sampling)
            in the reindexed TimeSeries. For more information, see the `xarray DataArrayResample documentation
            <https://docs.xarray.dev/en/stable/generated/xarray.core.resample.DataArrayResample.html>`__.
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
            <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.resample.html>`__.

        Returns
        -------
        TimeSeries
            A resampled series with given frequency.

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
        >>> downsampled_reduce_ts = ts.resample(freq="30min", method="reduce", method_args={"func": np.mean})
        >>> print(downsampled_reduce_ts.values())
        [[0.5]
        [2.5]
        [4.5]]
        """
        method_kwargs = method_kwargs or {}
        if isinstance(freq, pd.DateOffset):
            freq = freq.freqstr

        resample = self.data_array(copy=False).resample(
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
        return self.__class__.from_xarray(new_xa)

    def is_within_range(self, ts: Union[pd.Timestamp, int]) -> bool:
        """Whether the given timestamp or integer is within the time interval of the series.

        `ts` does not need to be an element of the series' time index.

        Parameters
        ----------
        ts
            The `pandas.Timestamp` (if indexed with DatetimeIndex) or integer (if indexed with RangeIndex) to check.

        Returns
        -------
        bool
            Whether `ts` is contained within the interval of this series.
        """
        return self.time_index[0] <= ts <= self.time_index[-1]

    def map(
        self,
        fn: Union[
            Callable[[np.ndarray], np.ndarray],
            Callable[[Union[pd.DatetimeIndex, pd.RangeIndex], np.ndarray], np.ndarray],
        ],
    ) -> Self:  # noqa: E501
        """Return a new series with the function `fn` applied to the values of this series.

        If `fn` takes 1 argument it is simply applied on the values array of shape `(time, n_components, n_samples)`.
        If `fn` takes 2 arguments, it is applied on the `(ts, values)` tuple, where `ts` denotes the
        series' time index, and `values` denotes the series' array of values, of shape
        `(n_timestamps, n_components, n_samples)`. Timestamp index's shape should be `(n, 1, 1)`;

        Parameters
        ----------
        fn
            Either a function which takes a NumPy array and returns a NumPy array of same shape;
            e.g., `lambda x: x ** 2`, `lambda x: x / x.shape[0]` or `np.log`.
            It can also be a function which takes a timestamp and array, and returns a new array of same shape;
            e.g., `lambda ts, x: x / ts.days_in_month`.
            The type of `ts` is either `pandas.Timestamp` (if the series is indexed with a DatetimeIndex),
            or an integer otherwise (if the series is indexed with an RangeIndex).

        Returns
        -------
        TimeSeries
            A new series with the function `fn` applied to the values.

        Examples
        --------
        >>> from darts import TimeSeries
        >>> from darts.utils.utils import generate_index
        >>> # create a simple TimeSeries
        >>> series = TimeSeries.from_times_and_values(
        >>>     times=generate_index("2020-01-01", length=3, freq="D"),
        >>>     values=range(3),
        >>> )
        >>> # map function on values only
        >>> def fn1(values):
        >>>     return values / 3.
        >>>
        >>> series.map(fn1).values()
        array([[0.        ],
               [0.33333333],
               [0.66666667]])
        >>>
        >>> # map function on time index and values
        >>> def fn2(times, values):
        >>>     return values / times.days_in_month.values.reshape(-1, 1, 1)
        >>>
        >>> series.map(fn2).values()
        array([[0.        ],
               [0.03225806],
               [0.06451613]])
        """
        if not isinstance(fn, Callable):
            raise_log(TypeError("fn must be a callable"), logger)

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

        if num_args == 1:  # apply fn on values directly
            values = fn(self._values)
        elif num_args == 2:
            # apply function on (times, values)
            values = fn(self._time_index, self._values)
        else:
            raise_log(ValueError("fn must accept either one or two arguments"), logger)

        if values.shape != self.shape:
            raise_log(
                ValueError(
                    f"fn must return an array of shape `{self.shape}`. Received shape `{values.shape}`"
                )
            )

        return self.__class__(
            times=self._time_index,
            values=values,
            components=self.components,
            **self._attrs,
        )

    def window_transform(
        self,
        transforms: Union[dict, Sequence[dict]],
        treat_na: Optional[Union[str, Union[int, float]]] = None,
        forecasting_safe: Optional[bool] = True,
        keep_non_transformed: Optional[bool] = False,
        include_current: Optional[bool] = True,
        keep_names: Optional[bool] = False,
    ) -> Self:
        """Return a new series with the specified window transformations applied.

        Supports moving/rolling, expanding or exponentially weighted window transformations.

        Parameters
        ----------
        transforms
            A dictionary or a list of dictionaries.
            Each dictionary specifies a different window transform.

            The dictionaries can contain the following keys:

            :``"function"``: Mandatory. The name of one of the pandas builtin transformation functions,
                            or a callable function that can be applied to the input series.
                            Pandas' functions can be found in the
                            `documentation <https://pandas.pydata.org/docs/reference/window.html>`__.

            :``"mode"``: Optional. The name of the pandas windowing mode on which the ``"function"`` is going to be
                        applied. The options are "rolling", "expanding" and "ewm".
                        If not provided, Darts defaults to "expanding".
                        User defined functions can use either "rolling" or "expanding" modes.
                        More information on pandas windowing operations can be found in the `documentation
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html>`__.

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
                <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`__.
            * :``"center"``: ``True``/``False`` to set the observation at the current timestep at the center of the
                window (when ``forecasting_safe`` is `True`, Darts enforces ``"center"`` to ``False``).
            * :``"closed"``: ``"right"``/``"left"``/``"both"``/``"neither"`` to specify whether the right,
                left or both ends of the window are included in the window, or neither of them.
                Darts defaults to pandas default of ``"right"``.

            More information on the available functions and their parameters can be found in the
            `Pandas documentation <https://pandas.pydata.org/docs/reference/window.html>`__.

            For user-provided functions, extra keyword arguments in the transformation dictionary are passed to the
            user-defined function.
            By default, Darts expects user-defined functions to receive numpy arrays as input.
            This can be modified by adding item ``"raw": False`` in the transformation dictionary.
            It is expected that the function returns a single
            value for each window. Other possible configurations can be found in the
            `pandas.DataFrame.rolling().apply()
            documentation <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html>`__
            and `pandas.DataFrame.expanding().apply()
            documentation <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.expanding.html>`__.

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
            Returns a new series with the transformed components. If ``keep_non_transformed`` is ``True``,
            the series will contain the original non-transformed components along the transformed ones.
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
            if mode not in PD_WINDOW_OPERATIONS.keys():
                raise_log(
                    ValueError(
                        f"Invalid window operation: '{mode}'. Must be one of {PD_WINDOW_OPERATIONS.keys()}."
                    ),
                    logger,
                )
            window_mode = PD_WINDOW_OPERATIONS[mode]

            # minimum number of observations in window required to have a value (otherwise result in NaN)
            if "min_periods" not in transformation:
                transformation["min_periods"] = 0 if mode == "ewm" else 1

            if mode == "rolling":
                # pandas default for 'center' is False, no need to set it explicitly
                if "center" in transformation:
                    if transformation["center"] and forecasting_safe:
                        raise_log(
                            ValueError(
                                "When `forecasting_safe` is True, `center` must be False."
                            ),
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

        if not all([isinstance(tr, dict) for tr in transforms]):
            raise_log(
                ValueError(
                    "`transforms` must be a non-empty dictionary or a non-empty list of dictionaries."
                ),
                logger,
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
            if treat_na not in VALID_TREAT_NA:
                raise_log(
                    ValueError(
                        f"`treat_na` must be one of {VALID_TREAT_NA} or a scalar, but found {treat_na}",
                    ),
                    logger,
                )

            if treat_na in VALID_BFILL_NA and forecasting_safe:
                raise_log(
                    ValueError(
                        "when `forecasting_safe` is True, back filling NaNs is not allowed as "
                        "it risks contaminating past time steps with future values."
                    ),
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

        transformed_time_series = TimeSeries(
            times=new_index,
            values=resulting_transformations.values.reshape(
                len(new_index), -1, n_samples
            ),
            components=new_columns,
            static_covariates=self.static_covariates,
            hierarchy=new_hierarchy,
            metadata=self.metadata,
            copy=False,
        )

        return transformed_time_series

    def to_json(self) -> str:
        """Return a JSON string representation of the deterministic series.

        At the moment this function works only on deterministic time series (i.e., made of 1 sample).

        The JSON string includes the series values, time index, component names, as well as static covariates,
        hierarchy, and metadata (if any).

        Returns
        -------
        str
            A JSON String representing the series

        See Also
        --------
        TimeSeries.from_json : Create a TimeSeries from a JSON string.
        """
        result = json.loads(
            self.to_dataframe().to_json(orient="split", date_format="iso")
        )
        if self.static_covariates is not None:
            result["static_covariates"] = json.loads(
                self.static_covariates.to_json(orient="split")
            )
        if self.hierarchy is not None:
            result["hierarchy"] = self.hierarchy
        if self.metadata is not None:
            result["metadata"] = self.metadata

        return json.dumps(result)

    def to_csv(self, *args, **kwargs):
        """Write the deterministic series to a CSV file.

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
        """Save the series in pickle format.

        Parameters
        ----------
        path : string
            path to a file where current object will be pickled
        protocol : integer, default highest
            pickling protocol. The default is best in most cases, use it only if having backward compatibility issues
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
        """Plot the series using Matplotlib.

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
            interval is shown if `low_quantile` is None (default 0.05).
        high_quantile
            The quantile to use for the upper bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `high_quantile` is None (default 0.95).
        default_formatting
            Whether to use the darts default scheme.
        title
            Optionally, a plot title.
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
        return _plot(
            self,
            new_plot=new_plot,
            central_quantile=central_quantile,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            default_formatting=default_formatting,
            title=title,
            label=label,
            max_nr_components=max_nr_components,
            ax=ax,
            alpha=alpha,
            color=color,
            c=c,
            *args,
            **kwargs,
        )

    def plotly(
        self,
        fig: Optional["go.Figure"] = None,
        central_quantile: Union[float, str] = 0.5,
        low_quantile: Optional[float] = 0.05,
        high_quantile: Optional[float] = 0.95,
        title: Optional[str] = None,
        label: Optional[Union[str, Sequence[str]]] = "",
        max_nr_components: int = 10,
        alpha: Optional[float] = None,
        color: Optional[Union[str, Sequence[str]]] = None,
        c: Optional[Union[str, Sequence[str]]] = None,
        downsample_threshold: int = 100_000,
        **kwargs,
    ) -> "go.Figure":
        """Plot the series using Plotly.

        Parameters
        ----------
        fig
            Optionally, a Plotly `go.Figure` object to plot on. If provided, the series will be added to this
            figure. If None, a new figure will be created.
        central_quantile
            The quantile (between 0 and 1) to plot as a "central" value, if the series is stochastic (i.e., if
            it has multiple samples). This will be applied on each component separately (i.e., to display quantiles
            of the components' marginal distributions). For instance, setting `central_quantile=0.5` will plot the
            median of each component. `central_quantile` can also be set to 'mean'.
        low_quantile
            The quantile to use for the lower bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `low_quantile` is None (default 0.05).
        high_quantile
            The quantile to use for the upper bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `high_quantile` is None (default 0.95).
        title
            Optionally, a plot title.
        label
            Can either be a string or list of strings. If a string and the series only has a single component, it is
            used as the label for that component. If a string and the series has multiple components, it is used as
            a prefix for each component name. If a list of strings with length equal to the number of components in
            the series, the labels will be mapped to the components in order.
        max_nr_components
            The maximum number of components of a series to plot. -1 means all components will be plotted.
        alpha
            Optionally, set the line alpha for deterministic series, or the confidence interval alpha for
            probabilistic series.
        color
            Set the line color(s). Can be a single color string (name or hex), or a sequence of
            strings (one per component). If a sequence, it must match the number of components.
            By default, colors are pulled from the active Plotly template.
        c
            An alias for `color`.
        downsample_threshold
            The maximum number of total data points (time steps * components * traces) to plot.
            If exceeded, the series will be automatically downsampled using a constant step
            size to avoid rendering crashes. Set to -1 to disable downsampling. Defaults to 100,000.
        **kwargs
            Additional keyword arguments to pass to `plotly.graph_objects.Scatter()` for trace customization
            (e.g., `line_dash`, `line_width`, `marker_symbol`, `opacity`, or `hovertemplate`).

        Returns
        -------
        plotly.graph_objects.Figure
            The Plotly figure object containing the plot. Call `.show()` on the returned figure to display it.
        """
        return _plotly(
            self,
            fig=fig,
            central_quantile=central_quantile,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            title=title,
            label=label,
            max_nr_components=max_nr_components,
            alpha=alpha,
            color=color,
            c=c,
            downsample_threshold=downsample_threshold,
            **kwargs,
        )

    def with_columns_renamed(
        self, col_names: Union[list[str], str], col_names_new: Union[list[str], str]
    ) -> Self:
        """Return a new series with new columns/components names.

        It also adapts the names in the hierarchy, if any.

        Parameters
        ----------
        col_names
            String or list of strings corresponding the the column names to be changed.
        col_names_new
            String or list of strings corresponding to the new column names. Must be the same length as col_names.

        Returns
        -------
        TimeSeries
            A new series with renamed columns.
        """
        if isinstance(col_names, str):
            col_names = [col_names]
        if isinstance(col_names_new, str):
            col_names_new = [col_names_new]

        if not all([(x in self.components.to_list()) for x in col_names]):
            raise_log(
                ValueError(
                    "Some column names in col_names don't exist in the time series."
                ),
                logger,
            )

        if len(col_names) != len(col_names_new):
            raise_log(
                ValueError(
                    "Length of col_names_new list should be equal to the length of col_names list."
                ),
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

        return self.__class__(
            times=self._time_index,
            values=self._values,
            components=pd.Index(cols),
            static_covariates=self.static_covariates,
            hierarchy=hierarchy,
            metadata=self.metadata,
        )

    """
    Simple statistic and aggregation functions. Calculate various statistics over the samples of stochastic time series
    or aggregate over components/time for deterministic series.
    """

    def mean(self, axis: int = 2) -> Self:
        """Return a new series with the mean computed over the specified `axis`.

        If we reduce over time (``axis=0``), the series will have length one and will use the first entry of the
        original ``time_index``. If we perform the calculation over the components (``axis=1``), the resulting single
        component will be renamed to "components_mean".  When applied to the samples (``axis=2``), a deterministic
        series is returned.

        If ``axis=1``, the static covariates and the hierarchy are discarded from the series.

        Parameters
        ----------
        axis
            The axis to reduce over. The default is to calculate over samples, i.e. axis=2.

        Returns
        -------
        TimeSeries
            A new series with mean applied to the indicated axis.
        """
        values = self._values.mean(axis=axis, keepdims=True)
        times, components = self._get_agg_dims("components_mean", axis)
        return self.__class__(
            times=times,
            values=values,
            components=components,
            **(self._attrs if axis != 1 else dict()),
        )

    def median(self, axis: int = 2) -> Self:
        """Return a new series with the median computed over the specified `axis`.

        If we reduce over time (``axis=0``), the series will have length one and will use the first entry of the
        original ``time_index``. If we perform the calculation over the components (``axis=1``), the resulting single
        component will be renamed to "components_median".  When applied to the samples (``axis=2``), a deterministic
        series is returned.

        If ``axis=1``, the static covariates and the hierarchy are discarded from the series.

        Parameters
        ----------
        axis
            The axis to reduce over. The default is to calculate over samples, i.e. axis=2.

        Returns
        -------
        TimeSeries
            A new series with median applied to the indicated axis.
        """
        values = np.median(
            self._values, axis=axis, overwrite_input=False, keepdims=True
        )
        times, components = self._get_agg_dims("components_median", axis)
        return self.__class__(
            times=times,
            values=values,
            components=components,
            **(self._attrs if axis != 1 else dict()),
        )

    def sum(self, axis: int = 2) -> Self:
        """Return a new series with the sum computed over the specified `axis`.

        If we reduce over time (``axis=0``), the series will have length one and will use the first entry of the
        original ``time_index``. If we perform the calculation over the components (``axis=1``), the resulting single
        component will be renamed to "components_sum".  When applied to the samples (``axis=2``), a deterministic
        series is returned.

        If ``axis=1``, the static covariates and the hierarchy are discarded from the series.

        Parameters
        ----------
        axis
            The axis to reduce over. The default is to calculate over samples, i.e. axis=2.

        Returns
        -------
        TimeSeries
            A new series with sum applied to the indicated axis.
        """
        values = self._values.sum(axis=axis, keepdims=True)
        times, components = self._get_agg_dims("components_sum", axis)
        return self.__class__(
            times=times,
            values=values,
            components=components,
            **(self._attrs if axis != 1 else dict()),
        )

    def min(self, axis: int = 2) -> Self:
        """Return a new series with the minimum computed over the specified `axis`.

        If we reduce over time (``axis=0``), the series will have length one and will use the first entry of the
        original ``time_index``. If we perform the calculation over the components (``axis=1``), the resulting single
        component will be renamed to "components_min".  When applied to the samples (``axis=2``), a deterministic
        series is returned.

        If ``axis=1``, the static covariates and the hierarchy are discarded from the series.

        Parameters
        ----------
        axis
            The axis to reduce over. The default is to calculate over samples, i.e. axis=2.

        Returns
        -------
        TimeSeries
            A new series with min applied to the indicated axis.
        """
        values = self._values.min(axis=axis, keepdims=True)
        times, components = self._get_agg_dims("components_min", axis)
        return self.__class__(
            times=times,
            values=values,
            components=components,
            **(self._attrs if axis != 1 else dict()),
        )

    def max(self, axis: int = 2) -> Self:
        """Return a new series with the maximum computed over the specified `axis`.

        If we reduce over time (``axis=0``), the series will have length one and will use the first entry of the
        original ``time_index``. If we perform the calculation over the components (``axis=1``), the resulting single
        component will be renamed to "components_max".  When applied to the samples (``axis=2``), a deterministic
        series is returned.

        If ``axis=1``, the static covariates and the hierarchy are discarded from the series.

        Parameters
        ----------
        axis
            The axis to reduce over. The default is to calculate over samples, i.e. axis=2.

        Returns
        -------
        TimeSeries
            A new series with max applied to the indicated axis.
        """
        values = self._values.max(axis=axis, keepdims=True)
        times, components = self._get_agg_dims("components_max", axis)
        return self.__class__(
            times=times,
            values=values,
            components=components,
            **(self._attrs if axis != 1 else dict()),
        )

    def quantile(self, q: Union[float, Sequence[float]] = 0.5, **kwargs) -> Self:
        """Return a deterministic series with the desired quantile(s) `q` of each component computed over the samples
        of the stochastic series.

        The component quantiles in the new series are named "<component>_q<quantile>", where "<component>" is the
        column name, and "<quantile>" is the quantile value.

        The order of the component quantiles is: `[<c_1>_q<q_1>, ... <c_1>_q<q_2>, ..., <c_n>_q<q_n>]`.

        This works only on stochastic series (i.e., with more than 1 sample).

        Parameters
        ----------
        q
            The desired quantile value or sequence of quantile values. Each value must be between 0. and 1. inclusive.
            For instance, `0.5` will return a TimeSeries containing the median of the (marginal) distribution of each
            component.
        kwargs
            Other keyword arguments are passed down to `numpy.quantile()`.

        Returns
        -------
        TimeSeries
            A new series containing the desired quantile(s) of each component.
        """
        self._assert_stochastic()
        if isinstance(q, float):
            q = [q]

        if not all([0 <= q_i <= 1 for q_i in q]):
            raise_log(
                ValueError(
                    "The quantile values must be expressed as fraction (between 0 and 1 inclusive)."
                ),
                logger,
            )

        # component names
        cnames = [f"{comp}_q{q_i:.3f}" for comp in self.components for q_i in q]

        # get quantiles of shape (n quantiles, n times, n components)
        new_data = np.quantile(self._values, q=q, axis=2, **kwargs)
        # transpose and reshape into (n times, n components * n quantiles, 1)
        new_data = new_data.transpose((1, 2, 0)).reshape(len(self), len(cnames), 1)

        # only add static covariates and hierarchy if the number of output components matches the input components
        return self.__class__(
            times=self._time_index,
            values=new_data,
            components=cnames,
            static_covariates=self.static_covariates if len(q) == 1 else None,
            hierarchy=self.hierarchy if len(q) == 1 else None,
            metadata=self.metadata,
            copy=False,
        )

    def var(self, ddof: int = 1) -> Self:
        """Return a deterministic series with the variance of each component computed over the samples of the
        stochastic series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        ddof
            "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof where N represents the
            number of elements. By default, ddof is 1.

        Returns
        -------
        TimeSeries
            A new series containing the variance of each component.
        """
        self._assert_stochastic()
        vals = self._values.var(axis=2, ddof=ddof, keepdims=True)
        return self.with_values(vals)

    def std(self, ddof: int = 1) -> Self:
        """Return a deterministic series with the standard deviation of each component computed over the samples of the
        stochastic series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        ddof
            "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof where N represents the
            number of elements. By default, ddof is 1.

        Returns
        -------
        TimeSeries
            A new series containing the standard deviation of each component.
        """
        self._assert_stochastic()
        vals = self._values.std(axis=2, ddof=ddof, keepdims=True)
        return self.with_values(vals)

    def skew(self, **kwargs) -> Self:
        """Return a deterministic series with the skew of each component computed over the samples of the
        stochastic series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        kwargs
            Other keyword arguments are passed down to `scipy.stats.skew()`

        Returns
        -------
        TimeSeries
            A new series containing the skew of each component.
        """
        self._assert_stochastic()
        vals = np.expand_dims(skew(self._values, axis=2, **kwargs), axis=2)
        return self.with_values(vals)

    def kurtosis(self, **kwargs) -> Self:
        """Return a deterministic series with the kurtosis of each component computed over the samples of the
        stochastic series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        kwargs
            Other keyword arguments are passed down to `scipy.stats.kurtosis()`

        Returns
        -------
        TimeSeries
            A new series containing the kurtosis of each component.
        """
        self._assert_stochastic()
        vals = np.expand_dims(kurtosis(self._values, axis=2, **kwargs), axis=2)
        return self.with_values(vals)

    """
    Dunder methods
    """

    def _extract_values(
        self,
        other: Union[Self, xr.DataArray, np.ndarray],
    ) -> Self:
        """Extract values from another series or array and check for compatible shapes."""

        if isinstance(other, TimeSeries):
            other_vals = other._values
        elif isinstance(other, xr.DataArray):
            other_vals = other.values
        else:
            other_vals = other

        t, c, s = self.shape
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
        return other_vals

    @classmethod
    def _fill_missing_dates(
        cls,
        times: Union[pd.DatetimeIndex, pd.RangeIndex, pd.Index],
        values: np.ndarray,
        freq: Optional[Union[str, int]] = None,
    ) -> tuple[Union[pd.DatetimeIndex, pd.RangeIndex], np.ndarray]:
        """Return the time index and values with missing dates inserted.

        This requires either a provided `freq` or the possibility to infer a unique frequency from `times` (see
        `offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__
        for more info on supported frequencies).

        Parameters
        ----------
        times
            The time index.
        values
            The values.
        freq
            Optionally, the target frequency to fill in the missing times. A pandas frequency offset (string) if
            `times` is a `pandas.DatetimeIndex`, or a step step size if `times` is an integer index.
            It must represent a target frequency that allows to maintain all dates / integers from `times`.

        Raises
        -------
        ValueError
            If `times` contains less than 3 elements,
            if no unique frequency can be inferred from `times`,
            if the resampled index does not contain all dates from `times`.

        Returns
        -------
        tuple[Union[pandas.DatetimeIndex, pandas.RangeIndex], numpy.ndarray]
            The `times` with inserted missing dates and `values` with `numpy.nan` for the newly inserted dates.
        """

        if freq is not None:
            return cls._restore_from_frequency(times=times, values=values, freq=freq)

        if len(times) <= 2:
            raise_log(
                ValueError(
                    "Input time series must be of (length>=3) when fill_missing_dates=True and freq=None."
                ),
                logger,
            )

        times, values = cls._sort_index(times=times, values=values)

        if isinstance(times, pd.DatetimeIndex):
            has_datetime_index = True
            observed_freqs = cls._observed_freq_datetime_index(times)
        else:  # integer index (non RangeIndex)
            has_datetime_index = False
            observed_freqs = cls._observed_freq_integer_index(times)

        if not len(observed_freqs) == 1:
            offset_alias_info = (
                (
                    " For more information about frequency aliases, read "
                    "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases"
                )
                if has_datetime_index
                else ""
            )
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

        return cls._restore_from_frequency(times=times, values=values, freq=freq)

    @staticmethod
    def _sort_index(
        times: Union[pd.DatetimeIndex, pd.RangeIndex, pd.Index],
        values: np.ndarray,
    ) -> tuple[Union[pd.DatetimeIndex, pd.RangeIndex], np.ndarray]:
        """Sort `times` and `values` by ascending dates.

        Only performed if `times` is not already monotonically increasing.
        """
        if times.is_monotonic_increasing:
            return times, values
        times, idx_sorted = times.sort_values(return_indexer=True)
        return times, values[idx_sorted]

    @staticmethod
    def _observed_freq_datetime_index(index: pd.DatetimeIndex) -> set:
        """Return all observed/inferred frequencies of a `pandas.DatetimeIndex`.

        The frequencies are inferred from all combinations of three consecutive time steps.

        Assumes that `index` is sorted in ascending order.
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
        """Return all observed/inferred frequencies of a ``pandas.Index`` (an integer-valued index).

        The inferred frequencies are given by all unique differences between two consecutive elements.

        Assumes that `index` is sorted in ascending order.
        """
        return set(index[1:] - index[:-1])

    @classmethod
    def _restore_range_indexed(
        cls,
        times: pd.Index,
        values: np.ndarray,
    ) -> tuple[Union[pd.DatetimeIndex, pd.RangeIndex], np.ndarray]:
        """Return `times` re-indexed into a `pandas.RangeIndex` and `values` in the re-indexed order.

        An integer `pandas.Index` can be converted to a `pandas.RangeIndex`, if the sorted index has a constant step
        size. Raises a `ValueError` otherwise.
        """
        times, values = cls._sort_index(times=times, values=values)
        observed_freqs = cls._observed_freq_integer_index(times)
        if len(observed_freqs) != 1:
            raise_log(
                ValueError(
                    f"Could not convert integer index to a `pandas.RangeIndex`. Found non-unique step "
                    f"sizes/frequencies: `{observed_freqs}`. If any of those is the actual frequency, "
                    f"try passing it with `fill_missing_dates=True` and `freq=your_frequency`."
                ),
                logger=logger,
            )
        freq = observed_freqs.pop()
        times = pd.RangeIndex(
            start=min(times),
            stop=max(times) + freq,
            step=freq,
            name=times.name,
        )
        return times, values

    @classmethod
    def _restore_from_frequency(
        cls,
        times: Union[pd.DatetimeIndex, pd.RangeIndex, pd.Index],
        values: np.ndarray,
        freq: Union[str, int],
    ) -> tuple[Union[pd.DatetimeIndex, pd.RangeIndex], np.ndarray]:
        """Return `times` resampled with frequency `freq` and values with `np.nan` for the newly inserted dates.

        The frequency `freq` must represent a target frequency that allows to maintain all dates from `times`.

        Parameters
        ----------
        times
            The time index.
        values
            The values.
        freq
            The target frequency to fill in the missing times. A pandas frequency offset (string) if
            `times` is a `pandas.DatetimeIndex`, or a step size if `times` is an integer index.
            It must represent a target frequency that allows to maintain all dates / integers from `times`.

        Raises
        -------
        ValueError
            If the resampled/re-indexed DateTimeIndex/RangeIndex does not contain all dates from `times`.

        Returns
        -------
        tuple[Union[pandas.DatetimeIndex, pandas.RangeIndex], numpy.ndarray]
            The resampled `times` with frequency `freq` and `values` with `numpy.nan` for the newly inserted dates.
        """
        times, values = cls._sort_index(times=times, values=values)

        resampled_times = pd.Series(index=times, dtype="object")
        if isinstance(times, pd.DatetimeIndex):
            has_datetime_index = True
            resampled_times = resampled_times.asfreq(freq)
        else:  # integer index (non RangeIndex) -> resampled to RangeIndex
            has_datetime_index = False
            resampled_times = resampled_times.reindex(
                range(min(times), max(times) + freq, freq)
            )
        # check if new time index with inferred frequency contains all input data
        contains_all_data = times.isin(resampled_times.index).all()
        if not contains_all_data:
            offset_alias_info = (
                (
                    " For more information about frequency aliases, read "
                    "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases"
                )
                if has_datetime_index
                else ""
            )
            raise_log(
                ValueError(
                    f"Could not correctly fill missing {'dates' if has_datetime_index else 'indices'} with the "
                    f"observed/passed {'frequency' if has_datetime_index else 'step size'} `freq='{freq}'`. "
                    f"Not all input {'time stamps' if has_datetime_index else 'indices'} contained in the newly "
                    f"created TimeSeries.{offset_alias_info}"
                ),
                logger,
            )
        # convert to float as for instance integer arrays cannot accept nans
        dtype = (
            values.dtype
            if (
                np.issubdtype(values.dtype, np.float32)
                or np.issubdtype(values.dtype, np.float64)
            )
            else np.float64
        )
        resampled_values = np.empty(
            shape=((len(resampled_times),) + values.shape[1:]), dtype=dtype
        )
        resampled_values[:] = np.nan
        resampled_values[resampled_times.index.isin(times)] = values
        return resampled_times.index, resampled_values

    @staticmethod
    def _get_axis(axis: Union[int, str]) -> int:
        """Convert different `axis` types to an integer axis."""
        if isinstance(axis, int):
            if not 0 <= axis <= 2:
                raise_log(
                    ValueError("If `axis` is an integer it must be between 0 and 2."),
                    logger,
                )
            return axis
        else:
            if axis not in DIMS:
                raise_log(
                    ValueError(
                        f"`axis` must be a known dimension of this series: {DIMS}"
                    ),
                    logger,
                )
            return DIMS.index(axis)

    def _get_agg_dims(
        self, new_cname: str, axis: int
    ) -> tuple[Union[pd.DatetimeIndex, pd.RangeIndex], pd.Index]:
        """Get output time index and components based on a aggregation `axis` and potential new column name
        `new_cname`.
        """

        if axis == 0:  # set time_index to first day
            return self._time_index[0:1], self.components
        elif axis == 1:  # rename components
            return self._time_index, pd.Index([new_cname])
        elif axis == 2:  # do nothing
            return self._time_index, self.components
        else:
            raise_log(
                ValueError(f"Invalid `axis={axis}`. Must be one of `(1, 2, 3)`."),
                logger,
            )

    def _get_first_timestamp_after(self, ts: pd.Timestamp) -> Union[pd.Timestamp, int]:
        return next(filter(lambda t: t >= ts, self._time_index))

    def _get_last_timestamp_before(self, ts: pd.Timestamp) -> Union[pd.Timestamp, int]:
        return next(filter(lambda t: t <= ts, self._time_index[::-1]))

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
            if not self._has_datetime_index:
                raise_log(
                    ValueError(
                        "Function called with a timestamp, but series not time-indexed."
                    ),
                    logger,
                )
            is_inside = self.start_time() <= ts <= self.end_time()
        else:
            if self._has_datetime_index:
                is_inside = 0 <= ts <= len(self)
            else:
                is_inside = self.start_time() <= ts <= self.end_time()

        if not is_inside:
            raise_log(
                ValueError(
                    f"Timestamp must be between {self.start_time()} and {self.end_time()}"
                ),
                logger,
            )

    def __eq__(self, other):
        if not isinstance(other, TimeSeries):
            return False

        if self.shape != other.shape:
            return False

        if not self._time_index.equals(other._time_index):
            return False

        if not np.array_equal(self._values, other._values, equal_nan=True):
            return False

        if not self.components.equals(other.components):
            return False

        sc_self, sc_other = self.static_covariates, other.static_covariates
        if (sc_self is not None) != (sc_other is not None):
            return False
        elif isinstance(sc_self, pd.DataFrame) and not sc_self.equals(sc_other):
            return False

        hr_self, hr_other = self.hierarchy, other.hierarchy
        if (hr_self is not None) != (hr_other is not None):
            return False
        elif isinstance(hr_self, dict) and hr_self != hr_other:
            return False

        md_self, md_other = self.metadata, other.metadata
        if (md_self is not None) != (md_other is not None):
            return False
        elif isinstance(md_self, dict) and md_self != md_other:
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._values)

    def __add__(self, other):
        if isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            other = self._extract_values(other)
        elif not isinstance(other, (int, float, np.integer)):
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for + or add(): '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )
        ts = self.copy()
        np.add(ts._values, other, out=ts._values)
        return ts

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            other = self._extract_values(other)
        elif not isinstance(other, (int, float, np.integer)):
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for - or sub(): '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )
        ts = self.copy()
        np.subtract(ts._values, other, out=ts._values)
        return ts

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            other = self._extract_values(other)
        elif not isinstance(other, (int, float, np.integer)):
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for * or mul(): '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )
        ts = self.copy()
        np.multiply(ts._values, other, out=ts._values)
        return ts

    def __rmul__(self, other):
        return self * other

    def __pow__(self, n):
        if isinstance(n, (int, float, np.integer)):
            if n < 0:
                raise_log(
                    ValueError("Attempted to raise a series to a negative power."),
                    logger,
                )
            n = float(n)
        elif isinstance(n, (TimeSeries, xr.DataArray, np.ndarray)):
            n = self._extract_values(n)  # elementwise power
        else:
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for ** or pow(): '{type(self).__name__}' and '{type(n).__name__}'."
                ),
                logger,
            )
        ts = self.copy()
        np.power(ts._values, n, out=ts._values)
        return ts

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.integer)):
            if other == 0:
                raise_log(ZeroDivisionError("Cannot divide by 0."), logger)
        elif isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            other = self._extract_values(other)
            if (other == 0).any():
                raise_log(
                    ZeroDivisionError("Cannot divide by a TimeSeries with a value 0."),
                    logger,
                )
        else:
            raise_log(
                TypeError(
                    "unsupported operand type(s) for / or truediv():"
                    f" '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )
        ts = self.copy()
        np.divide(ts._values, other, out=ts._values)
        return ts

    def __rtruediv__(self, n):
        return n * (self ** (-1))

    def __abs__(self):
        ts = self.copy()
        np.absolute(ts._values, out=ts._values)
        return ts

    def __neg__(self):
        ts = self.copy()
        np.negative(ts._values, out=ts._values)
        return ts

    def __contains__(self, ts: Union[int, pd.Timestamp]) -> bool:
        return ts in self.time_index

    def __round__(self, n=None):
        ts = self.copy()
        np.round(ts._values, n, out=ts._values)
        return ts

    def __lt__(self, other) -> np.ndarray:
        if isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            other = self._extract_values(other)
        elif not isinstance(other, (int, float, np.integer)):
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for < : '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )
        return np.less(self._values, other)

    def __gt__(self, other) -> np.ndarray:
        if isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            other = self._extract_values(other)
        elif not isinstance(other, (int, float, np.integer)):
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for > : '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )
        return np.greater(self._values, other)

    def __le__(self, other) -> np.ndarray:
        if isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            other = self._extract_values(other)
        elif not isinstance(other, (int, float, np.integer)):
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for <= : '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )
        return np.less_equal(self._values, other)

    def __ge__(self, other) -> np.ndarray:
        if isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            other = self._extract_values(other)
        elif not isinstance(other, (int, float, np.integer)):
            raise_log(
                TypeError(
                    f"unsupported operand type(s) for >= : '{type(self).__name__}' and '{type(other).__name__}'."
                ),
                logger,
            )
        return np.greater_equal(self._values, other)

    def __str__(self):
        values_str, info_str = self._get_values_repr("string")

        representation = f"{values_str}\n\n{info_str}\n\n"

        if self.static_covariates is not None:
            static_cov_str = self.static_covariates.to_string(max_rows=10, max_cols=10)
            # indentation, first line needs to be manual
            static_cov_str = "    " + static_cov_str.replace("\n", "\n    ")
            representation += f"Static covariates:\n{static_cov_str}\n"
        if self.hierarchy is not None:
            representation += f"Hierarchy:\n{format_dict(self.hierarchy)}\n"
        if self.metadata is not None:
            representation += f"Metadata:\n{format_dict(self.metadata)}\n"

        return representation.rstrip()

    def __repr__(self):
        return str(self)

    def _repr_html_(self):
        values_str, info_str = self._get_values_repr("html")

        representation = make_paragraph(values_str, margin_left="0") + make_paragraph(
            info_str
        )
        if self.static_covariates is not None:
            representation += make_collapsible_section(
                "Static covariates",
                self.static_covariates.to_html(max_rows=10, max_cols=10)
                if self.static_covariates is not None
                else "&lt;empty&gt;",
                open_by_default=True,
            )
        if self.hierarchy is not None:
            representation += make_collapsible_section(
                "Hierarchy",
                f"{format_dict(self.hierarchy, render_html=True)}",
                open_by_default=True,
            )
        if self.metadata is not None:
            representation += make_collapsible_section(
                "Metadata",
                f"{format_dict(self.metadata, render_html=True)}",
                open_by_default=True,
            )
        return representation

    def _get_values_repr(self, repr_type: str) -> tuple[str, str]:
        """Create a representation of the TimeSeries values.

        The returned dimensions respect the maximum allowed items to be displayed

        Parameters
        ----------
        repr_type
            The type of representation to use ("html" or "string").
        """
        max_rows = get_option("display.max_rows")
        max_cols = get_option("display.max_cols")
        margin = 2
        values = self.all_values(copy=False)
        times = self.time_index
        columns = self.columns

        # limit the number of rows
        if self.n_timesteps > max_rows + 2 * margin:
            n_rows = math.ceil(max_rows / 2) + margin
            values = np.concatenate([values[:n_rows], values[-n_rows:]], axis=TIME_AX)
            times = times[:n_rows].append(times[-n_rows:])
        # limit the number of columns
        if self.n_components > max_cols + 2 * margin:
            n_cols = math.ceil(max_cols / 2) + margin
            values = np.concatenate(
                [values[:, :n_cols], values[:, -n_cols:]], axis=COMP_AX
            )
            columns = columns[:n_cols].append(columns[-n_cols:])
        # aggregate samples
        if self.n_samples > 1:
            values = np.median(values, axis=SMPL_AX)
        else:
            values = values[:, :, 0]

        df = pd.DataFrame(data=values, index=times, columns=columns, copy=False)
        values_repr = getattr(df, f"to_{repr_type}")(
            max_rows=max_rows, max_cols=max_cols
        )

        # additional information
        info_str = f"shape: {self.shape}, freq: {self.freq_str}, size: {format_bytes(self._values.nbytes)}"

        # notify when samples were aggregated
        if self.n_samples > 1:
            info_str += (
                "<br>" if repr_type == "html" else "\n"
            ) + "info: only sample median was displayed"
        return values_repr, info_str

    def __copy__(self, deep: bool = True):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.__class__(
            times=deepcopy(self._time_index, memo),
            values=deepcopy(self._values, memo),
            components=deepcopy(self.components, memo),
            copy=False,
            **deepcopy(self._attrs, memo),
        )

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
        """Return a new series with elements selected by `key`.

        The supported index types are the following base types as a single value, a list or a slice:

        - pandas.Timestamp -> return a TimeSeries corresponding to the value(s) at the given timestamp(s).
        - str -> return a TimeSeries including the column(s) (components) specified as str.
        - int -> return a TimeSeries with the value(s) at the given row (time) index.

        `pandas.DatetimeIndex` and `pandas.RangeIndex` are also supported and will return the corresponding value(s)
        at the provided time indices.

        .. warning::
            slices use pandas convention of including both ends of the slice.

        Notes
        -----
        For integer-indexed series; passing integers or slices of integers act as positional indices. For example,
        passing `series[i]` will return the ``i``-th value along the series, which is not necessarily the value where
        the time index is equal to ``i`` (if the time index does not start at 0 and / or has a step size > 1). In
        contrast, calling this method with a ``pandas.RangeIndex`` returns the values where the time index matches the
        provided range index.
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

        adapt_covs_on_component = (
            True
            if self.has_static_covariates and len(self.static_covariates) > 1
            else False
        )

        # handle DatetimeIndex and RangeIndex:
        if isinstance(key, (pd.DatetimeIndex, pd.RangeIndex)):
            is_dti = isinstance(key, pd.DatetimeIndex)
            _check_dt() if is_dti else _check_range()
            times = self._time_index

            if len(key) == 0:
                # keep original frequency in case of empty index
                times = self._time_index[:0]
                values = self._values[:0]
            else:
                idx = times.get_indexer(key)
                if (idx < 0).any():
                    raise_log(KeyError("Not all indices found in time index."), logger)
                times = self._time_index[idx]
                values = self._values[idx]

                # make sure the frequency is transferred
                if is_dti:
                    times.freq = key.freq
                else:
                    # `get_indexer()` converts `RangeIndex` into regular `Index`
                    times = pd.RangeIndex(
                        start=times[0], stop=times[-1] + key.step, step=key.step
                    )

            return self.__class__(
                times=times, values=values, components=self.components, **self._attrs
            )
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
                    return self.__class__(
                        times=self._time_index[key],
                        values=self._values[key],
                        components=self.components,
                        **self._attrs,
                    )
            elif isinstance(key.start, str) or isinstance(key.stop, str):
                # selecting components discards the hierarchy, if any
                idx = self.components.get_indexer(pd.Index([key.start, key.stop]))
                if (idx < 0).any():
                    raise_log(
                        KeyError("Not all components found in time index."), logger
                    )
                indexer = slice(idx[0], idx[-1] + 1)
                values = self._values[:, indexer]
                components = self.components[indexer]
                static_covariates = self.static_covariates
                return self.__class__(
                    times=self._time_index,
                    values=values,
                    components=components,
                    static_covariates=(
                        static_covariates[indexer]
                        if adapt_covs_on_component
                        else static_covariates
                    ),
                    hierarchy=None,
                    metadata=self.metadata,
                )
            elif isinstance(key.start, (int, np.int64)) or isinstance(
                key.stop, (int, np.int64)
            ):
                return self.__class__(
                    times=self._time_index[key],
                    values=self._values[key],
                    components=self.components,
                    **self._attrs,
                )
            elif isinstance(key.start, pd.Timestamp) or isinstance(
                key.stop, pd.Timestamp
            ):
                if key.step is not None and key.step <= 0:
                    raise_log(
                        ValueError(
                            "Indexing a `TimeSeries` with a `slice` of `step<=0` (reverse) is not "
                            "possible since `TimeSeries` must have a monotonically increasing time index."
                        ),
                        logger=logger,
                    )
                _check_dt()
                start_time = self.start_time()

                if key.start is not None:
                    start = n_steps_between(
                        end=key.start, start=start_time, freq=self.freq
                    )
                    if start < 0:
                        # shift start a round-multip of `step` ahead until it lies within the index
                        start = 0 if key.step is None else start % key.step
                else:
                    start = 0

                if key.stop is not None:
                    end = n_steps_between(
                        end=key.stop, start=start_time, freq=self.freq
                    )
                else:
                    end = len(self) - 1
                key = slice(start, end + 1, key.step)
                return self.__class__(
                    times=self._time_index[key],
                    values=self._values[key],
                    components=self.components,
                    **self._attrs,
                )

        # handle simple types:
        elif isinstance(key, str):
            col_idx = self.components.get_loc(key)
            static_covariates = self.static_covariates
            return self.__class__(
                times=self._time_index,
                values=self._values[:, col_idx : col_idx + 1],
                components=self.components[col_idx : col_idx + 1],
                static_covariates=(
                    static_covariates.loc[[key]]
                    if adapt_covs_on_component
                    else static_covariates
                ),
                hierarchy=None,
                metadata=self.metadata,
            )
        elif isinstance(key, (int, np.int64)):
            key = slice(key, key + 1 if key != -1 else None)
            ts = self.__class__(
                times=self._time_index[key],
                values=self._values[key],
                components=self.components,
                **self._attrs,
            )
            if len(ts) == 0:
                raise_log(IndexError("Integer index out of range."), logger)
            return ts
        elif isinstance(key, pd.Timestamp):
            _check_dt()
            key = self._time_index.get_loc(key)
            key = slice(key, key + 1)
            return self.__class__(
                times=self._time_index[key],
                values=self._values[key],
                components=self.components,
                **self._attrs,
            )

        # handle lists:
        if isinstance(key, list):
            if all(isinstance(s, str) for s in key):
                # when string(s) are provided, we consider it as (a list of) component(s)
                indexer = self.components.get_indexer(key)
                if (indexer < 0).any():
                    raise_log(
                        KeyError("Not all components found in time index."), logger
                    )
                values = self._values[:, indexer]
                components = self.components[indexer]
                static_covariates = self.static_covariates
                return self.__class__(
                    times=self._time_index,
                    values=values,
                    components=components,
                    static_covariates=(
                        static_covariates.iloc[indexer]
                        if adapt_covs_on_component
                        else static_covariates
                    ),
                    hierarchy=None,
                    metadata=self.metadata,
                )
            elif all(isinstance(i, (int, np.int64)) for i in key):
                # convert list of integers to slice (must have constant step size)
                step_sizes = set(right - left for left, right in zip(key[:-1], key[1:]))
                if len(step_sizes) > 1:
                    raise_log(
                        ValueError(
                            f"Cannot index a `TimeSeries` with a list of integers with non-constant step sizes. "
                            f"Observed step sizes: `{step_sizes}`."
                        ),
                        logger,
                    )
                elif len(step_sizes) == 1:
                    step_size = step_sizes.pop()
                else:
                    step_size = 1

                if step_size <= 0:
                    raise_log(
                        ValueError(
                            "Indexing a `TimeSeries` with a list of integers with `step<=0` is not "
                            "possible since `TimeSeries` must have a monotonically increasing time index."
                        ),
                        logger=logger,
                    )
                return self[key[0] : key[-1] + step_size : step_size]

            elif all(isinstance(t, pd.Timestamp) for t in key):
                _check_dt()
                key = self._time_index.get_indexer(key)
                return self.__class__(
                    times=self._time_index[key],
                    values=self._values[key],
                    components=self.components,
                    **self._attrs,
                )

        raise_log(IndexError("The type of your index was not matched."), logger)


def _concat_static_covs(series: Sequence[TimeSeries]) -> Optional[pd.DataFrame]:
    """Concatenate static covariates along the component axis (rows of static covariates). Use this for stacking or
    concatenating time series along component dimension (axis=1).

    Some context for stacking or concatenating two or more TimeSeries with static covariates:

    - Concat along axis=0 (time): Along the time dimension, we only take the static covariates of the first series (as
      static covariates are time-independent).
    - Concat along axis=1 (components) or stacking: Along the component dimension, we either concatenate or transfer
      the static covariates of the series if one of below cases applies:
      1) concatenate along component dimension (rows of static covariates) when for each series the number of static
         covariate components is equal to the number of components in the series. The static variable names (columns in
         series.static_covariates) must be identical across all series
      2) if only the first series contains static covariates transfer only those
      3) if `ignore_static_covariates=True` (with `concatenate()`), case 1) is ignored and only the static covariates
         of the first series are transferred
    - Concat along axis=2 (samples): Along the sample dimension, we only take the static covariates of the first series
      (as the components and time don't change).
    """

    if not any([ts.has_static_covariates for ts in series]):
        return None

    only_first = series[0].has_static_covariates and not any([
        ts.has_static_covariates for ts in series[1:]
    ])
    all_have = all([ts.has_static_covariates for ts in series])

    if not (only_first or all_have):
        raise_log(
            ValueError(
                "Either none, only the first or all TimeSeries must have `static_covariates`."
            ),
            logger,
        )

    if only_first:
        return series[0].static_covariates

    if not (
        all([len(ts.static_covariates) == ts.n_components for ts in series])
        and all([
            ts.static_covariates.columns.equals(series[0].static_covariates.columns)
            for ts in series
        ])
    ):
        raise_log(
            ValueError(
                "Concatenation of multiple TimeSeries with static covariates requires all `static_covariates` "
                "DataFrames to have identical columns (static variable names), and the number of each TimeSeries' "
                "components must match the number of corresponding static covariate components (the number of rows "
                "in `series.static_covariates`)."
            ),
            logger,
        )

    return pd.concat(
        [ts.static_covariates for ts in series if ts.has_static_covariates], axis=0
    )


def _concat_hierarchy(series: Sequence[TimeSeries]):
    """Concatenate the hierarchies of multiple series, when concatenating series along axis 1 (components). This simply
    merges the hierarchy dictionaries.
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
    """Concatenate multiple series along a given axis.

    ``axis`` can be an integer in (0, 1, 2) to denote (time, component, sample) or, alternatively, a string denoting
    the corresponding dimension of the underlying ``DataArray``.

    Parameters
    ----------
    series : Sequence[TimeSeries]
        Sequence of ``TimeSeries`` to concatenate.
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
        The concatenated series.
    """
    axis = TimeSeries._get_axis(axis)
    vals = [ts.all_values(copy=False) for ts in series]

    component_axis_equal = len({ts.shape[COMP_AX] for ts in series}) == 1
    sample_axis_equal = len({ts.shape[SMPL_AX] for ts in series}) == 1

    times = series[0]._time_index
    components = series[0].components
    static_covariates = series[0].static_covariates
    hierarchy = series[0].hierarchy
    metadata = None if drop_metadata else series[0].metadata

    vals = np.concatenate(vals, axis=axis)
    if axis == 0:
        # time
        if not (component_axis_equal and sample_axis_equal):
            raise_log(
                ValueError(
                    "when concatenating along time dimension, the component and sample dimensions of all "
                    "provided series must match."
                ),
                logger,
            )

        # check, if timeseries are consecutive
        consecutive_time_axes = True
        for i in range(1, len(series)):
            if series[i - 1].end_time() + series[0].freq != series[i].start_time():
                consecutive_time_axes = False
                break

        if not consecutive_time_axes:
            if not ignore_time_axis:
                raise_log(
                    ValueError(
                        "When concatenating over time axis, all series need to be contiguous "
                        "in the time dimension. Use `ignore_time_axis=True` to override "
                        "this behavior and concatenate the series by extending the time axis "
                        "of the first series."
                    ),
                    logger,
                )

        times = generate_index(
            start=series[0].start_time(),
            freq=series[0].freq,
            length=len(vals),
            name=times.name,
        )
    else:
        if ignore_time_axis:
            time_axes_ok = len({len(ts) for ts in series}) == 1
        else:
            time_axes_ok = all([
                ts.has_same_time_as(ts_next)
                for ts, ts_next in zip(series[0:-1], series[1:])
            ])

        if (
            (not time_axes_ok)
            or (axis == 1 and not sample_axis_equal)
            or (axis == 2 and not component_axis_equal)
        ):
            raise_log(
                ValueError(
                    "When concatenating along component or sample dimensions, all the series must have the same time "
                    "axes (unless `ignore_time_axis` is True), or time axes of same lengths (if `ignore_time_axis` is "
                    "True), and all series must have the same number of samples (if concatenating along component "
                    "dimension), or the same number of components (if concatenating along sample dimension)."
                ),
                logger,
            )

        if axis == 1:
            # When concatenating along component dimension, we have to re-create a component index
            # we rely on the factory method of TimeSeries to disambiguate names later on if needed.
            components = pd.Index([
                c for cl in [ts.components for ts in series] for c in cl
            ])
            static_covariates = (
                _concat_static_covs(series)
                if not ignore_static_covariates
                else static_covariates
            )
            hierarchy = None if drop_hierarchy else _concat_hierarchy(series)

    return series[0].__class__(
        times=times,
        values=vals,
        components=components,
        static_covariates=static_covariates,
        hierarchy=hierarchy,
        metadata=metadata,
    )


def slice_intersect(series: Sequence[TimeSeries]) -> list[TimeSeries]:
    """Return a list of series, where all series have been intersected along the time index.

    Parameters
    ----------
    series : Sequence[TimeSeries]
        sequence of ``TimeSeries`` to intersect

    Returns
    -------
    Sequence[TimeSeries]
        The intersected series.
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


def to_group_dataframe(
    series: Union[TimeSeries, Sequence[TimeSeries]],
    copy: bool = True,
    backend: Union[ModuleType, Implementation, str] = Implementation.PANDAS,
    time_as_index: bool = True,
    suppress_warnings: bool = False,
    add_static_cov: Union[bool, str, list[str], None] = True,
    add_metadata: Union[bool, str, list[str], None] = False,
    add_group_col: Optional[bool] = False,
):
    """
    Return a grouped DataFrame representation from one or multiple `TimeSeries`.

    This method converts a single `TimeSeries` or a sequence of `TimeSeries` into individual DataFrames
    using `TimeSeries.to_dataframe()` and concatenates them into a single DataFrame.
    This is particularly useful when working with collections of time series that share a common schema
    and need to be represented in a tabular format for downstream processing.

    Each series is converted independently, and the resulting DataFrames are concatenated row-wise
    using the specified backend.

    Parameters
    ----------
    series
        A single `TimeSeries` or a sequence of `TimeSeries` to convert into a grouped DataFrame.
    copy
        Whether to return a copy of the resulting DataFrame. Leave it to True unless you know what you are doing.
    backend
        The backend to which to export the `TimeSeries`. See the `narwhals documentation
        <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_dict>`__ for all supported
        backends.
    time_as_index
        Whether to set the time index as the index of the DataFrame or in the left-most column.
        Only effective with the pandas `backend`.
    suppress_warnings
        Whether to suppress warnings raised during the DataFrame creation.
    add_static_cov
        Whether to add static covariates from the time series as columns in the resulting DataFrame
        (one column per component–static covariate pair). Can be a bool in case all the static covariates
        should be added, or a string/list of strings in case only a subset is needed. True by default as static
        covariates should be provided to specify groups.
    add_metadata
        Whether to add metadata from the time series as columns in the resulting DataFrame
        (one column per metadata entry). Can be a bool in case all metadata should be added,
        or a string/list of strings in case only a subset is needed.
    add_group_col
        Whether to add a group column in the resulting long format dataframe. The values of that group column will go
        from 1 to the number of time series in the input list

    Returns
    -------
    DataFrame
        A grouped DataFrame representation of the input `TimeSeries`(s) in the specified `backend`.
        The DataFrame is obtained by concatenating the individual DataFrames generated from each series.
    """

    dfs = []
    backend = Implementation.from_backend(backend)

    if isinstance(series, TimeSeries):
        series = [series]

    for idx, serie in enumerate(series):
        _df = serie.to_dataframe(
            copy=copy,
            backend=backend,
            time_as_index=time_as_index,
            suppress_warnings=suppress_warnings,
            add_static_cov=add_static_cov,
            add_metadata=add_metadata,
        )
        _df = nw.from_native(_df)
        if add_group_col:
            _df = _df.with_columns(nw.lit(idx).alias("group"))
        dfs.append(_df)

    df = nw.concat(dfs)
    df = df.to_native()
    if backend.is_pandas() and not time_as_index:
        df.reset_index(inplace=True, drop=True)
    return df


def _finite_rows_boundaries(
    values: np.ndarray, how: str = "all"
) -> tuple[Optional[int], Optional[int]]:
    """Return the indices of the first rows containing finite values starting from the start and the end of the first
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

    if len(dims) > 3:
        raise_log(
            ValueError(f"Expected 1D to 3D array, received {len(dims)}D array"), logger
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


def _clean_components(components: pd.Index) -> pd.Index:
    """Return a `pandas.Index` with unique string component / column names"""
    # convert everything to string if needed
    clist = [(col if isinstance(col, str) else str(col)) for col in components]

    has_duplicate = len(set(clist)) != len(clist)
    while has_duplicate:
        # we may have to loop several times (e.g. we could have components ["0", "0_1", "0"] and not
        # noticing when renaming the last "0" into "0_1" that "0_1" already exists...)
        name_to_occurence = defaultdict(int)
        for i in range(len(clist)):
            name_to_occurence[clist[i]] += 1

            if name_to_occurence[clist[i]] > 1:
                clist[i] = clist[i] + f"_{name_to_occurence[clist[i]] - 1}"

        has_duplicate = len(set(clist)) != len(clist)

    return pd.Index(clist)
