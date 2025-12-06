"""
Utils for TimeSeries generation
-------------------------------
"""

import math
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import holidays
import numpy as np
import pandas as pd
from pandas.tseries.offsets import Tick

from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import (
    DIMS,
    HIERARCHY_TAG,
    METADATA_TAG,
    STATIC_COV_TAG,
    TIME_AX,
    TimeSeries,
)
from darts.utils.utils import generate_index

logger = get_logger(__name__)

ONE_INDEXED_FREQS = {
    "day",
    "month",
    "quarter",
    "dayofyear",
    "day_of_year",
    "week",
    "weekofyear",
    "week_of_year",
}
TIMES_NAME = DIMS[TIME_AX]
MAX_DATETIME_VALUES = {
    "month": 12,
    "day": 31,
    "weekday": 7,
    "dayofweek": 7,
    "day_of_week": 7,
    "hour": 24,
    "minute": 60,
    "second": 60,
    "microsecond": 1000000,
    "nanosecond": 1000,
    "quarter": 4,
    # leap years insert an additional day on the 29th of February
    "dayofyear": 365 + 1,
    "day_of_year": 365 + 1,
    # years contain an additional week if they are :
    # - a regular year starting on a thursday
    # - a leap year starting on a wednesday
    "week": 52 + 1,
    "weekofyear": 52 + 1,
    "week_of_year": 52 + 1,
}
FULL_CALENDAR_CYCLE = pd.Timedelta(days=365 * 28 + 7)  # ~28 years
"""The solar calendar cycle (https://en.wikipedia.org/wiki/Solar_cycle_(calendar)) of the Julian calendar."""

MAX_GENERATION_STEPS = 100000
"""Threshold to prevent generating too massive arrays when calculating unique datetime attribute values."""

ATTRIBUTE_PERIODS = {
    "microsecond": pd.Timedelta("1s"),
    "nanosecond": pd.Timedelta("1us"),
    "second": pd.Timedelta("1min"),
    "minute": pd.Timedelta("1h"),
    "hour": pd.Timedelta("1D"),
    "weekday": pd.Timedelta("1W"),
    "day_of_week": pd.Timedelta("1W"),
    "day": FULL_CALENDAR_CYCLE,
    "month": FULL_CALENDAR_CYCLE,
    "dayofyear": FULL_CALENDAR_CYCLE,
    "week": FULL_CALENDAR_CYCLE,
}
"""The time is takes for an attribute to naturally reset/wrap around.

For example, minutes wrap around every hour, hours wrap around every day, etc.
"""

DATETIME_ATT_WITH_VARIABLE_MAX = [
    "day",
    "dayofyear",
    "day_of_year",
    "week",
    "weekofyear",
    "week_of_year",
]
"""Time index attributes whose maximum value varies (e.g., day of month (´28, 30 or 31), week of year (52 or 53))."""


def constant_timeseries(
    value: float = 1,
    start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
    end: Optional[Union[pd.Timestamp, int]] = None,
    length: Optional[int] = None,
    freq: Union[str, int] = None,
    column_name: Optional[str] = "constant",
    dtype: np.dtype = np.float64,
) -> TimeSeries:
    """
    Creates a constant univariate TimeSeries with the given value, length (or end date), start date and frequency.

    Parameters
    ----------
    value
        The constant value that the TimeSeries object will assume at every index.
    start
        The start of the returned TimeSeries' index. If a pandas Timestamp is passed, the TimeSeries will have a pandas
        DatetimeIndex. If an integer is passed, the TimeSeries will have a pandas RangeIndex index. Works only with
        either `length` or `end`.
    end
        Optionally, the end of the returned index. Works only with either `start` or `length`. If `start` is
        set, `end` must be of same type as `start`. Else, it can be either a pandas Timestamp or an integer.
    length
        Optionally, the length of the returned index. Works only with either `start` or `end`.
    freq
        The time difference between two adjacent entries in the returned index. In case `start` is a timestamp,
        a DateOffset alias is expected; see
        `docs <https://pandas.pydata.org/pandas-docs/stable/user_guide/TimeSeries.html#dateoffset-objects>`__.
        By default, "D" (daily) is used.
        If `start` is an integer, `freq` will be interpreted as the step size in the underlying RangeIndex.
        The freq is optional for generating an integer index (if not specified, 1 is used).
    column_name
        Optionally, the name of the value column for the returned TimeSeries
    dtype
        The desired NumPy dtype (np.float32 or np.float64) for the resulting series

    Returns
    -------
    TimeSeries
        A constant TimeSeries with value 'value'.
    """

    index = generate_index(
        start=start, end=end, freq=freq, length=length, name=TIMES_NAME
    )
    values = np.full(len(index), value, dtype=dtype)
    return TimeSeries(
        times=index,
        values=values,
        components=pd.Index([column_name]),
        copy=False,
    )


def linear_timeseries(
    start_value: float = 0,
    end_value: float = 1,
    start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
    end: Optional[Union[pd.Timestamp, int]] = None,
    length: Optional[int] = None,
    freq: Union[str, int] = None,
    column_name: Optional[str] = "linear",
    dtype: np.dtype = np.float64,
) -> TimeSeries:
    """
    Creates a univariate TimeSeries with a starting value of `start_value` that increases linearly such that
    it takes on the value `end_value` at the last entry of the TimeSeries. This means that
    the difference between two adjacent entries will be equal to
    (`end_value` - `start_value`) / (`length` - 1).

    Parameters
    ----------
    start_value
        The value of the first entry in the TimeSeries.
    end_value
        The value of the last entry in the TimeSeries.
    start
        The start of the returned TimeSeries' index. If a pandas Timestamp is passed, the TimeSeries will have a pandas
        DatetimeIndex. If an integer is passed, the TimeSeries will have a pandas RangeIndex index. Works only with
        either `length` or `end`.
    end
        Optionally, the end of the returned index. Works only with either `start` or `length`. If `start` is
        set, `end` must be of same type as `start`. Else, it can be either a pandas Timestamp or an integer.
    length
        Optionally, the length of the returned index. Works only with either `start` or `end`.
    freq
        The time difference between two adjacent entries in the returned index. In case `start` is a timestamp,
        a DateOffset alias is expected; see
        `docs <https://pandas.pydata.org/pandas-docs/stable/user_guide/TimeSeries.html#dateoffset-objects>`__.
        By default, "D" (daily) is used.
        If `start` is an integer, `freq` will be interpreted as the step size in the underlying RangeIndex.
        The freq is optional for generating an integer index (if not specified, 1 is used).
    column_name
        Optionally, the name of the value column for the returned TimeSeries
    dtype
        The desired NumPy dtype (np.float32 or np.float64) for the resulting series

    Returns
    -------
    TimeSeries
        A linear TimeSeries created as indicated above.
    """

    index = generate_index(
        start=start, end=end, freq=freq, length=length, name=TIMES_NAME
    )
    values = np.linspace(start_value, end_value, len(index), dtype=dtype)
    return TimeSeries(
        times=index,
        values=values,
        components=pd.Index([column_name]),
        copy=False,
    )


def sine_timeseries(
    value_frequency: float = 0.1,
    value_amplitude: float = 1.0,
    value_phase: float = 0.0,
    value_y_offset: float = 0.0,
    start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
    end: Optional[Union[pd.Timestamp, int]] = None,
    length: Optional[int] = None,
    freq: Union[str, int] = None,
    column_name: Optional[str] = "sine",
    dtype: np.dtype = np.float64,
) -> TimeSeries:
    """
    Creates a univariate TimeSeries with a sinusoidal value progression with a given frequency, amplitude,
    phase and y offset.

    Parameters
    ----------
    value_frequency
        The number of periods that take place within one time unit given in `freq`.
    value_amplitude
        The maximum  difference between any value of the returned TimeSeries and `y_offset`.
    value_phase
        The relative position within one period of the first value of the returned TimeSeries (in radians).
    value_y_offset
        The shift of the sine function along the y axis.
    start
        The start of the returned TimeSeries' index. If a pandas Timestamp is passed, the TimeSeries will have a pandas
        DatetimeIndex. If an integer is passed, the TimeSeries will have a pandas RangeIndex index. Works only with
        either `length` or `end`.
    end
        Optionally, the end of the returned index. Works only with either `start` or `length`. If `start` is
        set, `end` must be of same type as `start`. Else, it can be either a pandas Timestamp or an integer.
    length
        Optionally, the length of the returned index. Works only with either `start` or `end`.
    freq
        The time difference between two adjacent entries in the returned index. In case `start` is a timestamp,
        a DateOffset alias is expected; see
        `docs <https://pandas.pydata.org/pandas-docs/stable/user_guide/TimeSeries.html#dateoffset-objects>`__.
        By default, "D" (daily) is used.
        If `start` is an integer, `freq` will be interpreted as the step size in the underlying RangeIndex.
        The freq is optional for generating an integer index (if not specified, 1 is used).
    column_name
        Optionally, the name of the value column for the returned TimeSeries
    dtype
        The desired NumPy dtype (np.float32 or np.float64) for the resulting series

    Returns
    -------
    TimeSeries
        A sinusoidal TimeSeries parametrized as indicated above.
    """

    index = generate_index(
        start=start, end=end, freq=freq, length=length, name=TIMES_NAME
    )
    values = np.array(range(len(index)), dtype=dtype)
    values = (
        value_amplitude * np.sin(2 * np.pi * value_frequency * values + value_phase)
        + value_y_offset
    )
    return TimeSeries(
        times=index,
        values=values,
        components=pd.Index([column_name]),
        copy=False,
    )


def gaussian_timeseries(
    mean: Union[float, np.ndarray] = 0.0,
    std: Union[float, np.ndarray] = 1.0,
    start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
    end: Optional[Union[pd.Timestamp, int]] = None,
    length: Optional[int] = None,
    freq: Union[str, int] = None,
    column_name: Optional[str] = "gaussian",
    dtype: np.dtype = np.float64,
) -> TimeSeries:
    """
    Creates a gaussian univariate TimeSeries by sampling all the series values independently,
    from a gaussian distribution with mean `mean` and standard deviation `std`.

    Parameters
    ----------
    mean
        The mean of the gaussian distribution that is sampled at each step.
        If a float value is given, the same mean is used at every step.
        If a numpy.ndarray of floats with the same length as `length` is
        given, a different mean is used at each time step.
    std
        The standard deviation of the gaussian distribution that is sampled at each step.
        If a float value is given, the same standard deviation is used at every step.
        If an array of dimension `(length, length)` is given, it will
        be used as covariance matrix for a multivariate gaussian distribution.
    start
        The start of the returned TimeSeries' index. If a pandas Timestamp is passed, the TimeSeries will have a pandas
        DatetimeIndex. If an integer is passed, the TimeSeries will have a pandas RangeIndex index. Works only with
        either `length` or `end`.
    end
        Optionally, the end of the returned index. Works only with either `start` or `length`. If `start` is
        set, `end` must be of same type as `start`. Else, it can be either a pandas Timestamp or an integer.
    length
        Optionally, the length of the returned index. Works only with either `start` or `end`.
    freq
        The time difference between two adjacent entries in the returned index. In case `start` is a timestamp,
        a DateOffset alias is expected; see
        `docs <https://pandas.pydata.org/pandas-docs/stable/user_guide/TimeSeries.html#dateoffset-objects>`__.
        By default, "D" (daily) is used.
        If `start` is an integer, `freq` will be interpreted as the step size in the underlying RangeIndex.
        The freq is optional for generating an integer index (if not specified, 1 is used).
    column_name
        Optionally, the name of the value column for the returned TimeSeries
    dtype
        The desired NumPy dtype (np.float32 or np.float64) for the resulting series

    Returns
    -------
    TimeSeries
        A white noise TimeSeries created as indicated above.
    """

    if isinstance(mean, np.ndarray):
        raise_if_not(
            mean.shape == (length,),
            "If a vector of means is provided, "
            "it requires the same length as the TimeSeries.",
            logger,
        )
    if isinstance(std, np.ndarray):
        raise_if_not(
            std.shape == (length, length),
            "If a matrix of standard deviations is provided, "
            "its shape has to match the length of the TimeSeries.",
            logger,
        )

    index = generate_index(
        start=start, end=end, freq=freq, length=length, name=TIMES_NAME
    )
    values = np.random.normal(mean, std, size=len(index)).astype(dtype)
    return TimeSeries(
        times=index,
        values=values,
        components=pd.Index([column_name]),
        copy=False,
    )


def random_walk_timeseries(
    mean: float = 0.0,
    std: float = 1.0,
    start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
    end: Optional[Union[pd.Timestamp, int]] = None,
    length: Optional[int] = None,
    freq: Union[str, int] = None,
    column_name: Optional[str] = "random_walk",
    dtype: np.dtype = np.float64,
) -> TimeSeries:
    """
    Creates a random walk univariate TimeSeries, where each step is obtained by sampling a gaussian distribution
    with mean `mean` and standard deviation `std`.

    Parameters
    ----------
    mean
        The mean of the gaussian distribution that is sampled at each step.
    std
        The standard deviation of the gaussian distribution that is sampled at each step.
    start
        The start of the returned TimeSeries' index. If a pandas Timestamp is passed, the TimeSeries will have a pandas
        DatetimeIndex. If an integer is passed, the TimeSeries will have a pandas RangeIndex index. Works only with
        either `length` or `end`.
    end
        Optionally, the end of the returned index. Works only with either `start` or `length`. If `start` is
        set, `end` must be of same type as `start`. Else, it can be either a pandas Timestamp or an integer.
    length
        Optionally, the length of the returned index. Works only with either `start` or `end`.
    freq
        The time difference between two adjacent entries in the returned index. In case `start` is a timestamp,
        a DateOffset alias is expected; see
        `docs <https://pandas.pydata.org/pandas-docs/stable/user_guide/TimeSeries.html#dateoffset-objects>`__.
        By default, "D" (daily) is used.
        If `start` is an integer, `freq` will be interpreted as the step size in the underlying RangeIndex.
        The freq is optional for generating an integer index (if not specified, 1 is used).
    column_name
        Optionally, the name of the value column for the returned TimeSeries
    dtype
        The desired NumPy dtype (np.float32 or np.float64) for the resulting series

    Returns
    -------
    TimeSeries
        A random walk TimeSeries created as indicated above.
    """

    index = generate_index(
        start=start, end=end, freq=freq, length=length, name=TIMES_NAME
    )
    values = np.cumsum(np.random.normal(mean, std, size=len(index)), dtype=dtype)
    return TimeSeries(
        times=index,
        values=values,
        components=pd.Index([column_name]),
        copy=False,
    )


def autoregressive_timeseries(
    coef: Sequence[float],
    start_values: Optional[Sequence[float]] = None,
    start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
    end: Optional[Union[pd.Timestamp, int]] = None,
    length: Optional[int] = None,
    freq: Union[str, int] = None,
    column_name: Optional[str] = "autoregressive",
    dtype: np.dtype = np.float64,
) -> TimeSeries:
    """
    Creates a univariate, autoregressive TimeSeries whose values are calculated using specified coefficients `coef` and
    starting values `start_values`.

    Parameters
    ----------
    coef
        The autoregressive coefficients used for calculating the next time step.
        series[t] = coef[-1] * series[t-1] + coef[-2] * series[t-2] + ... + coef[0] * series[t-len(coef)]
    start_values
        The starting values used for calculating the first few values for which no lags exist yet.
        series[0] = coef[-1] * starting_values[-1] + coef[-2] * starting_values[-2] + ... + coef[0] * starting_values[0]
    start
        The start of the returned TimeSeries' index. If a pandas Timestamp is passed, the TimeSeries will have a pandas
        DatetimeIndex. If an integer is passed, the TimeSeries will have a pandas RangeIndex index. Works only with
        either `length` or `end`.
    end
        Optionally, the end of the returned index. Works only with either `start` or `length`. If `start` is
        set, `end` must be of same type as `start`. Else, it can be either a pandas Timestamp or an integer.
    length
        Optionally, the length of the returned index. Works only with either `start` or `end`.
    freq
        The time difference between two adjacent entries in the returned index. In case `start` is a timestamp,
        a DateOffset alias is expected; see
        `docs <https://pandas.pydata.org/pandas-docs/stable/user_guide/TimeSeries.html#dateoffset-objects>`__.
        By default, "D" (daily) is used.
        If `start` is an integer, `freq` will be interpreted as the step size in the underlying RangeIndex.
        The freq is optional for generating an integer index (if not specified, 1 is used).
    column_name
        Optionally, the name of the value column for the returned TimeSeries
    dtype
        The desired NumPy dtype (np.float32 or np.float64) for the resulting series

    Returns
    -------
    TimeSeries
        An autoregressive TimeSeries created as indicated above.
    """

    # if no start values specified default to a list of 1s
    if start_values is None:
        start_values = np.ones(len(coef), dtype=dtype)
    else:
        raise_if_not(
            len(start_values) == len(coef),
            "start_values must have same length as coef.",
        )

    index = generate_index(
        start=start, end=end, freq=freq, length=length, name=TIMES_NAME
    )

    values = np.empty(len(coef) + len(index), dtype=dtype)
    values[: len(coef)] = start_values

    for i in range(len(coef), len(coef) + len(index)):
        # calculate next time step as dot product of coefs with previous len(coef) time steps
        values[i] = np.dot(values[i - len(coef) : i], coef)
    return TimeSeries(
        times=index,
        values=values[len(coef) :],
        components=pd.Index([column_name]),
        copy=False,
    )


def _extend_time_index_until(
    time_index: Union[pd.DatetimeIndex, pd.RangeIndex],
    until: Optional[Union[int, str, pd.Timestamp]],
    add_length: int,
    name,
) -> pd.DatetimeIndex:
    if not add_length and not until:
        return time_index

    raise_if(bool(add_length) and bool(until), "set only one of add_length and until")

    end = time_index[-1]
    freq = time_index.freq

    if add_length:
        raise_if_not(
            add_length >= 0,
            f"Expected add_length, by which to extend the time series by, "
            f"to be positive, got {add_length}",
        )

        try:
            end += add_length * freq
        except pd.errors.OutOfBoundsDatetime:
            raise_log(
                ValueError(
                    f"the add operation between {end} and {add_length * freq} will overflow"
                ),
                logger,
            )
    else:
        datetime_index = isinstance(time_index, pd.DatetimeIndex)

        if datetime_index:
            raise_if_not(
                isinstance(until, (str, pd.Timestamp)),
                "Expected valid timestamp for TimeSeries, "
                "indexed by DatetimeIndex, "
                f"for parameter until, got {type(end)}",
                logger,
            )
        else:
            raise_if_not(
                isinstance(until, int),
                "Expected integer for TimeSeries, indexed by RangeIndex, "
                f"for parameter until, got {type(end)}",
                logger,
            )

        timestamp = pd.Timestamp(until) if datetime_index else until

        raise_if_not(
            timestamp > end,
            f"Expected until, {timestamp} to lie past end of time index {end}",
        )

        ahead = timestamp - end
        raise_if_not(
            (ahead % freq) == pd.Timedelta(0),
            f"End date must correspond with frequency {freq} of the time axis",
            logger,
        )

        end = timestamp

    new_time_index = pd.date_range(start=time_index[0], end=end, freq=freq, name=name)
    return new_time_index


def holidays_timeseries(
    time_index: Union[TimeSeries, pd.DatetimeIndex],
    country_code: str,
    prov: str = None,
    state: str = None,
    column_name: Optional[str] = "holidays",
    until: Optional[Union[int, str, pd.Timestamp]] = None,
    add_length: int = 0,
    dtype: np.dtype = np.float64,
    tz: Optional[str] = None,
) -> TimeSeries:
    """
    Creates a binary univariate TimeSeries with index `time_index` that equals 1 at every index that lies within
    (or equals) a selected country's holiday, and 0 otherwise.

    Available countries can be found `here <https://github.com/dr-prodigy/python-holidays#available-countries>`__.

    Parameters
    ----------
    time_index
        Either a `pd.DatetimeIndex` or a `TimeSeries` for which to generate the holidays.
    country_code
        The country ISO code.
    prov
        The province.
    state
        The state.
    until
        Extend the time_index up until timestamp for datetime indexed series
        and int for range indexed series, should match or exceed forecasting window.
    add_length
        Extend the time_index by add_length, should match or exceed forecasting window.
        Set only one of until and add_length.
    column_name
        Optionally, the name of the value column for the returned TimeSeries.
    dtype
        The desired NumPy dtype (np.float32 or np.float64) for the resulting series.
    tz
        Optionally, a time zone to convert the time index to before generating the holidays.

    Returns
    -------
    TimeSeries
        A new binary holiday TimeSeries instance.
    """
    time_index_ts, time_index = _process_time_index(
        time_index=time_index,
        tz=tz,
        until=until,
        add_length=add_length,
    )

    scope = range(time_index[0].year, (time_index[-1] + pd.Timedelta(days=1)).year)
    country_holidays = holidays.country_holidays(
        country_code, prov=prov, state=state, years=scope
    )
    index_series = pd.Series(time_index, index=time_index)
    values = index_series.apply(lambda x: x in country_holidays).astype(dtype)
    return TimeSeries(
        times=time_index_ts,
        values=values,
        components=pd.Index([column_name]),
        copy=False,
    )


def _get_datetime_attribute_values(
    attribute: str, time_index: pd.DatetimeIndex
) -> pd.Index:
    if attribute not in ["week", "weekofyear", "week_of_year"]:
        values = getattr(time_index, attribute)
    else:
        values = (
            time_index.isocalendar()
            .set_index("week")
            .index.astype("int64")
            .rename("time")
        )
    # shift 1-indexed datetime attributes
    if attribute in ONE_INDEXED_FREQS:
        values -= 1
    return values


def _timedelta_lcm(td1: pd.Timedelta, td2: pd.Timedelta) -> pd.Timedelta:
    """Returns the least common multiple (LCM) of two pandas Timedelta objects.

    Raises a ValueError if no meaningful LCM exists (e.g., for zero or non-integer nanosecond values).

    Parameters
    ----------
    td1
        The first Timedelta.
    td2
        The second Timedelta.

    Returns
    -------
    pd.Timedelta
        The LCM of the two Timedelta objects.

    Raises
    ------
    ValueError
        If no meaningful LCM exists (e.g., for zero or non-integer nanosecond values).
    """
    ns1 = td1.value
    ns2 = td2.value

    # Check for zero timedelta
    if ns1 == 0 or ns2 == 0:
        raise ValueError("Timedelta values must be non-zero.")

    # Check for integer nanosecond representation
    if not isinstance(ns1, int) or not isinstance(ns2, int):
        raise ValueError("Timedelta values must be integer nanoseconds.")

    gcd = math.gcd(ns1, ns2)
    if gcd == 0:
        raise ValueError("No meaningful LCM possible (GCD is zero).")

    lcm_ns = abs(ns1 * ns2) // gcd

    # Check if LCM is a multiple of both inputs
    if lcm_ns % ns1 != 0 or lcm_ns % ns2 != 0:
        raise ValueError("No integer LCM exists for these Timedelta values.")

    return pd.Timedelta(lcm_ns, unit="ns")


def unique_datetime_value_freq_aware(
    attribute: str, freq: Union[str, pd.tseries.offsets.BaseOffset], start: pd.Timestamp
) -> np.ndarray[tuple[int], int]:
    """Returns a sorted array of unqiue values that the given datetime attribute can take, based on `freq` and `start`.

    Parameters
    ----------
    attribute
        An attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month", "weekday", "day",
        "hour", "minute", "second". See all available attributes in
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
    freq
        The frequency of the time index.
    start
        The start of the time index.

    Returns
    -------
        Sorted array of all the unique values that the given datetime attribute can take.

    See Also
    --------
    unique_datetime_values: When all possible values for the attribute are to be returned.

    Notes
    -----
    This function determines unique values using one of three strategies:

    1. **Exact Synchronization:** For fixed frequencies, it simulates the exact period where the frequency and attribute
    cycle align (LCM).
       * *Example:* ``attribute="hour", freq="2H"`` -> Returns even hours ``[0, 2, ..., 22]``.

    2. **Calendar Simulation:** For variable frequencies (e.g., Business Days), it simulates a 28-year cycle to
    guarantee capturing leap years and weekday shifts.
       * *Example:* ``attribute="day", freq="B"`` -> Returns ``[1..31]`` (ensures Feb 29th is eventually captured).

    3. **Heuristic Fallback:** If the simulation requires generating an excessive number of points (e.g., high-frequency
    data for low-frequency attributes), it assumes all theoretically possible values occur.
       * *Example:* ``attribute="month", freq="1min"`` -> Returns ``[1..12]`` immediately to save memory.

    Examples
    --------
    >>> from darts.utils.timeseries_generation import unique_datetime_values
    >>> from pandas.tseries.frequencies import to_offset
    >>> unique_datetime_values("hour", "15min", pd.Timestamp("2020-01-01"))
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23])
    >>> unique_datetime_values("minute", "15min", pd.Timestamp("2020-01-01"))
    array([0, 15, 30, 45])
    """
    # 1. Get the Natural Period of the attribute (~28 years as safe default)
    natural_period = ATTRIBUTE_PERIODS.get(attribute, FULL_CALENDAR_CYCLE)

    # 2. Try to convert frequency to Timedelta
    freq_td: Optional[pd.Timedelta] = None
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        if isinstance(offset, Tick):
            freq_td = pd.Timedelta(offset)

    except (ValueError, TypeError):
        # Handle raw strings that to_offset might not like, but to_timedelta might
        # e.g., "15min" is fine, but sometimes complex strings fail to_offset
        pass

    # Fallback: Try direct string-to-timedelta conversion if the above failed
    # This handles strings like "10us" if to_offset failed
    if freq_td is None:
        try:
            freq_td = pd.to_timedelta(freq)
        except (ValueError, TypeError):
            # If this fails, it is truly a variable frequency (e.g. 'M', 'B')
            pass

    # 3. Dynamic Duration Calculation
    if freq_td is not None:
        # How long until the Freq and the Attribute Period sync up?
        total_duration = _timedelta_lcm(freq_td, natural_period)
        # Check how many points this requires
        num_points = total_duration // freq_td

        # Safety fallback: If the interference pattern requires a large number of points
        if num_points > MAX_GENERATION_STEPS:
            return unique_datetime_values(attribute)

        # Otherwise, simulate exact LCM duration
        idx = pd.date_range(start=start, periods=num_points, freq=freq_td)

    else:
        # Variable frequency (e.g. 'BusinessDay')
        # We cannot calculate LCM easily. We fallback to the Safe Horizon (28 years).
        # 28 Years covers the synchronization of Weekdays, Leap Years, and Days.
        end_date = start + FULL_CALENDAR_CYCLE

        # Heuristic check for variable freqs:
        # If we are doing 'BusinessHour' over 28 years, that is too huge.
        # Estimate points: 28 years / rough estimate of freq.
        # If freq is unknown, we just run generation with a cap.
        idx = pd.date_range(start=start, end=end_date, freq=freq)

        if len(idx) > MAX_GENERATION_STEPS:
            return unique_datetime_values(attribute)

    # 4. Return unique values
    values = _get_datetime_attribute_values(attribute, idx)
    return np.unique(values).astype(int)


def unique_datetime_values(attribute: str) -> np.ndarray[tuple[int], int]:
    """Returns a sorted array of all the unique values that the given datetime attribute can take.

    Parameters
    ----------
    attribute
        An attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month", "weekday", "day",
        "hour", "minute", "second". See all available attributes in
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.

    Returns
    -------
    np.ndarray[tuple[int], int]
        Sorted array of all the unique values that the given datetime attribute can take.

    See Also
    --------
    unique_datetime_value_freq_aware: When the unique values are to be determined based on `freq` and `start`.

    Examples
    --------
    >>> from darts.utils.timeseries_generation import unique_datetime_values
    >>> from pandas.tseries.frequencies import to_offset
    >>> unique_datetime_values("month")
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    """
    raise_if_not(
        attribute in MAX_DATETIME_VALUES,
        f"Can't determine unique  values for attribute `{attribute}`, required for cyclic and one-hot encodings. "
        f"Supported datetime attribute: {list(MAX_DATETIME_VALUES.keys())}",
        logger,
    )
    return np.arange(MAX_DATETIME_VALUES[attribute])


def datetime_attribute_timeseries(
    time_index: Union[pd.DatetimeIndex, TimeSeries],
    attribute: str,
    one_hot: bool = False,
    one_hot_freq_aware: bool = False,
    cyclic: bool = False,
    cyclic_relative: bool = False,
    until: Optional[Union[int, str, pd.Timestamp]] = None,
    add_length: int = 0,
    dtype=np.float64,
    with_columns: Optional[Union[list[str], str]] = None,
    tz: Optional[str] = None,
) -> TimeSeries:
    """
    Returns a new TimeSeries with index `time_index` and one or more dimensions containing
    (optionally one-hot encoded or cyclic encoded) pd.DatatimeIndex attribute information derived from the index.

    1-indexed attributes are shifted to enforce 0-indexing across all the encodings.

    Parameters
    ----------
    time_index
        Either a `pd.DatetimeIndex` attribute which will serve as the basis of the new column(s), or
        a `TimeSeries` whose time axis will serve this purpose.
    attribute
        An attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month", "weekday", "day",
        "hour", "minute", "second". See all available attributes `here
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex>`__.
    one_hot
        Boolean value indicating whether to add the specified attribute as a one hot encoding
        (results in more columns).
    one_hot_freq_aware
        If `True`, the one-hot encoding infers which values are actually possible based on the frequency and start of
        the time index. If `False`, the encoding includes all possible values for the attribute. Only has an effect if
        `one_hot` is `True`.
    cyclic
        Boolean value indicating whether to add the specified attribute as a cyclic encoding.
        Alternative to one_hot encoding, enable only one of the two.
        (adds 2 columns, corresponding to sin and cos transformation)
    cyclic_relative
        Boolean value controlling the behavior of cyclic encoding for attributes with a variable maximum value
        (e.g., `day`, `dayofyear`, `week`). If `True`, the cyclic encoding uses the relative period based on the
        actual number of days or weeks in the current month or year. If `False`, the encoding uses the absolute
        maximum possible value for the attribute (e.g., 31 for days, 366 for days in a year, 53 for weeks).
    until
        Extend the time_index up until timestamp for datetime indexed series
        and int for range indexed series, should match or exceed forecasting window.
    add_length
        Extend the time_index by add_length, should match or exceed forecasting window.
        Set only one of until and add_length.
    dtype
        The desired NumPy dtype (np.float32 or np.float64) for the resulting series
    with_columns
        Optionally, specify the output component names.

        - If `one_hot` and `cyclic` are ``False``, must be a string
        - If `cyclic` is ``True``, must be a list of two strings. The first string for the sine, the second for the
          cosine component name.
        - If `one_hot` is ``True``, must be a list of strings of the same length as the generated one hot encoded
          features.
    tz
        Optionally, a time zone to convert the time index to before computing the attributes.

    Returns
    -------
    TimeSeries
        New datetime attribute TimeSeries instance.
    """

    time_index_ts, time_index = _process_time_index(
        time_index=time_index,
        tz=tz,
        until=until,
        add_length=add_length,
    )

    raise_if_not(
        hasattr(pd.DatetimeIndex, attribute)
        or (attribute in ["week", "weekofyear", "week_of_year"]),
        f"attribute `{attribute}` needs to be an attribute of pd.DatetimeIndex. "
        "See all available attributes in "
        "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex",
        logger,
    )

    raise_if(one_hot and cyclic, "set only one of one_hot or cyclic to true", logger)

    values = _get_datetime_attribute_values(attribute, time_index)
    if not one_hot and not cyclic:
        if with_columns is None:
            with_columns = attribute
        raise_if_not(
            isinstance(with_columns, str),
            "`with_columns` must be a string specifying the output component name.",
            logger=logger,
        )
        values_df = pd.DataFrame({with_columns: values})
    else:
        if one_hot:
            if one_hot_freq_aware:
                unique_values = unique_datetime_value_freq_aware(
                    attribute, time_index.freqstr, time_index[0]
                )
            else:
                unique_values = unique_datetime_values(attribute)
            values_df = pd.get_dummies(values)
            # fill missing columns (in case not all values appear in time_index)
            is_missing = np.isin(unique_values, values_df.columns.values, invert=True)
            # if there are attribute_range columns that are
            # not in values_df.columns.values
            if is_missing.any():
                dict_0 = {i: False for i in unique_values[is_missing]}
                # Make a dataframe from the dictionary and concatenate it
                # to the values values_df  in which the existing columns
                values_df = pd.concat(
                    [values_df, pd.DataFrame(dict_0, index=values_df.index)], axis=1
                ).sort_index(axis=1)
            else:
                values_df = values_df[unique_values]

            if with_columns is None:
                with_columns = [
                    f"{attribute}_{column_name}" for column_name in values_df.columns
                ]
            else:
                raise_if_not(
                    len(with_columns) == len(values_df.columns),
                    (
                        f"For the given case with `one_hot=True` and `one_hot_freq_aware={one_hot_freq_aware}`, "
                        f"`with_columns` must be a list of strings of length {values_df.columns}."
                    ),
                    logger=logger,
                )

            values_df.columns = with_columns
        else:
            unique_values = unique_datetime_values(attribute)
            if attribute in DATETIME_ATT_WITH_VARIABLE_MAX and cyclic_relative:
                if attribute == "day":
                    periods = time_index.days_in_month.values
                elif attribute in ("dayofyear", "day_of_year"):
                    periods = np.where(time_index.is_leap_year, 366, 365)
                elif attribute in ("week", "weekofyear", "week_of_year"):
                    periods = np.where(
                        (time_index.is_year_start & (time_index.weekday == 3))
                        | (
                            time_index.is_leap_year
                            & time_index.is_year_start
                            & (time_index.weekday == 2)
                        ),
                        53,
                        52,
                    )
                freq = 2 * np.pi * np.reciprocal(periods.astype(dtype))
            else:
                period = unique_values.max() + 1
                freq = 2 * np.pi / period

            if with_columns is None:
                with_columns = [attribute + "_sin", attribute + "_cos"]

            raise_if(
                len(with_columns) != 2,
                "`with_columns` must be a list of two strings when `cyclic=True`. "
                "The first string for the sine component name, the second for the cosine component name.",
                logger=logger,
            )
            values_df = pd.DataFrame({
                with_columns[0]: np.sin(freq * values),
                with_columns[1]: np.cos(freq * values),
            })

    return TimeSeries(
        times=time_index_ts,
        values=values_df.values.astype(dtype),
        components=values_df.columns,
        copy=False,
    )


def _build_forecast_series(
    points_preds: Union[np.ndarray, Sequence[np.ndarray]],
    input_series: TimeSeries,
    custom_columns: list[str] = None,
    with_static_covs: bool = True,
    with_hierarchy: bool = True,
    pred_start: Optional[Union[pd.Timestamp, int]] = None,
    time_index: Union[pd.DatetimeIndex, pd.RangeIndex] = None,
    copy: bool = False,
) -> TimeSeries:
    """
    Builds a forecast time series starting after the end of an input time series, with the
    correct time index (or after the end of the input series, if specified).

    Parameters
    ----------
    points_preds
        Forecasted values, can be either the target(s) or parameters of the likelihood model
    input_series
        TimeSeries used as input for the prediction
    custom_columns
        New names for the forecast TimeSeries, used when the number of components changes
    with_static_covs
        If set to `False`, do not copy the input_series `static_covariates` attribute
    with_hierarchy
        If set to `False`, do not copy the input_series `hierarchy` attribute
    pred_start
        Optionally, give a custom prediction start point. Only effective if `time_index` is `None`.
    time_index
        Optionally, the index to use for the forecast time series.
    copy
        If set to `True`, a copy of the input series is made. Otherwise, the input series is used as a view.

    Returns
    -------
    TimeSeries
        New TimeSeries instance starting after the input series
    """
    if time_index is None:
        time_index_length = (
            len(points_preds)
            if isinstance(points_preds, np.ndarray)
            else len(points_preds[0])
        )
        time_index = _generate_new_dates(
            time_index_length,
            input_series=input_series,
            start=pred_start,
        )
    values = (
        points_preds
        if isinstance(points_preds, np.ndarray)
        else np.stack(points_preds, axis=2)
    )
    return TimeSeries(
        times=time_index,
        values=values,
        freq=input_series.freq_str,
        components=input_series.columns if custom_columns is None else custom_columns,
        static_covariates=input_series.static_covariates if with_static_covs else None,
        hierarchy=input_series.hierarchy if with_hierarchy else None,
        metadata=input_series.metadata,
        copy=copy,
    )


def _build_forecast_series_from_schema(
    values: np.ndarray,
    schema: dict[str, Any],
    pred_start: Union[pd.Timestamp, int],
    predict_likelihood_parameters: bool,
    likelihood_component_names_fn: Optional[Callable] = None,
    copy: bool = False,
) -> TimeSeries:
    """
    Builds a forecast time series from predicted values and `TimeSeries` schema starting at `pred_start`.

    Parameters
    ----------
    values
        Forecasted values, can be either the target(s) or parameters of the likelihood model
    schema
        Schema of the predicted target `TimeSeries`.
    pred_start
        The prediction start time.
    predict_likelihood_parameters
        Whether the values represent predicted likelihood parameters.
    likelihood_component_names_fn
        A function to compute the likelihood parameter component names. Only effective when
        `predict_likelihood_parameters=True`.
    copy
        If set to `True`, a copy of the input series is made. Otherwise, the input series is used as a view.

    Returns
    -------
    TimeSeries
        A new TimeSeries instance.
    """
    time_index = generate_index(
        start=pred_start,
        freq=schema["time_freq"],
        length=len(values),
        name=schema["time_name"],
    )
    if predict_likelihood_parameters:
        if likelihood_component_names_fn is None:
            raise_log(
                ValueError(
                    "Must pass `likelihood_component_names_fn` with "
                    "`predict_likelihood_parameters=True`"
                ),
                logger=logger,
            )
        columns = likelihood_component_names_fn(components=schema["columns"])
        static_covariates = None
        hierarchy = None
    else:
        columns = schema["columns"]
        static_covariates = schema[STATIC_COV_TAG]
        hierarchy = schema[HIERARCHY_TAG]

    return TimeSeries(
        times=time_index,
        values=values,
        components=columns,
        static_covariates=static_covariates,
        hierarchy=hierarchy,
        metadata=schema[METADATA_TAG],
        copy=copy,
    )


def _generate_new_dates(
    n: int, input_series: TimeSeries, start: Optional[Union[pd.Timestamp, int]] = None
) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
    """
    Generates `n` new dates after the end of the specified series
    """
    if start is None:
        last = input_series.end_time()
        start = last + input_series.freq
    return generate_index(
        start=start,
        freq=input_series.freq,
        length=n,
        name=input_series._time_index.name,
    )


def _process_time_index(
    time_index: Union[TimeSeries, pd.DatetimeIndex],
    tz: Optional[str] = None,
    until: Optional[Union[int, str, pd.Timestamp]] = None,
    add_length: int = 0,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Extracts the time index, and optionally adds some time steps after the end of the index, and/or converts the
    index to another time zone.

    Returns a tuple of pd.DatetimeIndex with the first being the naive time index for generating a new TimeSeries,
    and the second being the one used for generating datetime attributes and holidays in a potentially different
    time zone.
    """
    if isinstance(time_index, TimeSeries):
        time_index = time_index.time_index

    if not isinstance(time_index, pd.DatetimeIndex):
        raise_log(
            ValueError(
                "`time_index` must be a pandas `DatetimeIndex` or a `TimeSeries` indexed with a `DatetimeIndex`."
            ),
            logger=logger,
        )
    if time_index.tz is not None:
        raise_log(
            ValueError("`time_index` must be time zone naive."),
            logger=logger,
        )
    time_index = _extend_time_index_until(
        time_index, until, add_length, time_index.name
    )

    # convert to another time zone
    if tz is not None:
        time_index_ = time_index.tz_localize("UTC").tz_convert(tz)
    else:
        time_index_ = time_index
    return time_index, time_index_
