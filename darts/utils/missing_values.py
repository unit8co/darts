"""
Utils for filling missing values
--------------------------------
"""

from typing import List, Optional, Union

from darts.logging import get_logger, raise_if, raise_if_not
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


def missing_values_ratio(series: TimeSeries) -> float:
    """
    Computes the ratio of missing values

    Parameters
    ----------
    series
        The time series to compute ratio on

    Returns
    -------
    float
        The ratio of missing values
    """

    return series.pd_dataframe().isnull().sum().mean() / len(series)


def fill_missing_values(
    series: TimeSeries, fill: Union[str, float] = "auto", **interpolate_kwargs
) -> TimeSeries:
    """
    Fills missing values in the provided time series

    Parameters
    ----------
    series
        The time series for which to fill missing values
    fill
        The value used to replace the missing values.
        If set to 'auto', will auto-fill missing values using the `pandas.Dataframe.interpolate()` method.
    interpolate_kwargs
        Keyword arguments for `pandas.Dataframe.interpolate()`, only used when fit is set to 'auto'.
        See `the documentation
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html>`_
        for the list of supported parameters.

    Returns
    -------
    TimeSeries
        A new TimeSeries with all missing values filled according to the rules above.
    """
    raise_if_not(
        isinstance(fill, str) or isinstance(fill, float),
        "`fill` should either be a string or a float",
        logger,
    )
    raise_if(
        isinstance(fill, str) and fill != "auto",
        "invalid string for `fill`: can only be set to 'auto'",
        logger,
    )

    if fill == "auto":
        return _auto_fill(series, **interpolate_kwargs)
    return _const_fill(series, fill)


def extract_subseries(
    series: TimeSeries, min_gap_size: Optional[int] = 1, mode: str = "all"
) -> List[TimeSeries]:
    """
    Partitions the series into a sequence of sub-series by using significant gaps of missing values

    Parameters
    ----------
    series
        The TimeSeries to partition into sub-series

    min_gap_size
        The minimum number of contiguous missing values to consider a gap as significant. Defaults to 1.

    mode
        Only for multivariate TimeSeries. The definition of a gap; presence of a NaN in any column ("any")
        or NaNs in all the columns ("all") for a given timestamp. Defaults to "all".

    Returns
    -------
    subseries
        A list of TimeSeries, sub-series without significant gaps of missing values

    See Also
    --------
    TimeSeries.gaps : return the gaps in the TimeSeries
    """

    # Remove null values from the series extremes
    series = series.strip()
    freq = series.freq

    if series.pd_dataframe().isna().sum().sum() == 0:
        return [series]

    # Get start/end times of sub-series without gaps of missing values
    gaps_df = series.gaps(mode=mode)
    if gaps_df.empty:
        return series
    else:
        gaps_df = gaps_df.query(f"gap_size>={min_gap_size}")
        start_times = [series.start_time()] + (gaps_df["gap_end"] + freq).to_list()
        end_times = (gaps_df["gap_start"] - freq).to_list() + [series.end_time() + freq]

        subseries = []
        for start, end in zip(start_times, end_times):
            subseries.append(series[start:end])

        return subseries


def _const_fill(series: TimeSeries, fill: float = 0) -> TimeSeries:
    """
    Fills the missing values of `series` with only the value provided (default zeroes).

    Parameters
    ----------
    series
        The TimeSeries to check for missing values.
    fill
        The value used to replace the missing values.

    Returns
    -------
    TimeSeries
        A TimeSeries, `series` with all missing values set to `fill`.
    """

    return TimeSeries.from_times_and_values(
        series.time_index,
        series.pd_dataframe().fillna(value=fill),
        freq=series.freq,
        columns=series.columns,
        static_covariates=series.static_covariates,
        hierarchy=series.hierarchy,
    )


def _auto_fill(series: TimeSeries, **interpolate_kwargs) -> TimeSeries:
    """
    This function fills the missing values in the TimeSeries `series`,
    using the `pandas.Dataframe.interpolate()` method.

    Parameters
    ----------
    series
        The time series
    interpolate_kwargs
        Keyword arguments for `pandas.Dataframe.interpolate()`.
        See `the documentation
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html>`_
        for the list of supported parameters.
    Returns
    -------
    TimeSeries
        A new TimeSeries with all missing values filled according to the rules above.
    """

    series_temp = series.pd_dataframe()

    # pandas interpolate wrapper, with chosen `method`
    if "limit_direction" not in interpolate_kwargs:
        interpolate_kwargs["limit_direction"] = "both"
    interpolate_kwargs["inplace"] = True
    series_temp.interpolate(**interpolate_kwargs)
    return TimeSeries.from_dataframe(
        series_temp,
        freq=series.freq,
        static_covariates=series.static_covariates,
        hierarchy=series.hierarchy,
    )
