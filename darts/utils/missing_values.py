"""
Utils for filling missing values
--------------------------------
"""

from ..timeseries import TimeSeries


def na_ratio(ts: TimeSeries) -> float:
    """
    Computes the ratio of missing values

    Parameters
    ----------
    ts
        The time series to compute ratio on
    Returns
    -------
    float
        The ratio of missing values
    """

    return ts.pd_dataframe().isnull().sum().mean() / len(ts)


def fillna(ts: TimeSeries, fill: float = 0) -> TimeSeries:
    """
    Fills the missing values of `ts` with only the value provided (default zeroes).

    Parameters
    ----------
    ts
        The TimeSeries to check for missing values.
    fill
        The value used to replace the missing values.

    Returns
    -------
    TimeSeries
        A TimeSeries, `ts` with all missing values set to `fill`.
    """

    return TimeSeries.from_times_and_values(ts.time_index(), ts.pd_dataframe().fillna(value=fill))


def auto_fillna(ts: TimeSeries,
                **interpolate_kwargs) -> TimeSeries:
    """
    This function fills the missing value in the TimeSeries `ts`,
    using the `pandas.Dataframe.interpolate()` method.

    Parameters
    ----------
    ts
        The time series
    interpolate_kwargs
        Keyword arguments  `pandas.Dataframe.interpolate()`.
        See `the documentation
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html>`_
        for the list of supported parameters.
    Returns
    -------
    TimeSeries
        A new TimeSeries with all missing values filled according to the rules above.
    """

    ts_temp = ts.pd_dataframe()

    # pandas interpolate wrapper, with chosen `method`
    if 'limit_direction' not in interpolate_kwargs:
        interpolate_kwargs['limit_direction'] = 'both'
    interpolate_kwargs['inplace'] = True
    ts_temp.interpolate(**interpolate_kwargs)

    return TimeSeries.from_times_and_values(ts.time_index(), ts_temp.values)
