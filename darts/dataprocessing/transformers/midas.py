"""
Mixed-data sampling (MIDAS) Transformer
------------------
"""
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from pandas import DatetimeIndex

from darts import TimeSeries
from darts.dataprocessing.transformers import (
    FittableDataTransformer,
    InvertibleDataTransformer,
)
from darts.logging import get_logger, raise_if, raise_if_not

logger = get_logger(__name__)


class MIDAS(FittableDataTransformer, InvertibleDataTransformer):
    def __init__(
        self,
        rule: str,
        strip: bool = True,
        name: str = "MIDASTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Mixed-data sampling transformer.

        A transformer that converts higher frequency time series to lower frequency using mixed-data sampling; see
        [1]_ for further details. This allows higher frequency covariates to be used whilst forecasting a lower
        frequency target series. For example, using monthly inputs to forecast a quarterly target.

        Notes
        -----
        The high input frequency should always relate in the same rate to the low target frequency. For
        example, there's always three months in quarter. However, the number of days in a month varies per month. So in
        the latter case a MIDAS transformation does not work and the transformer will raise an error.

        Parameters
        ----------
        rule
            The offset string or object representing target conversion. Passed on to the rule parameter in
            pandas.DataFrame.resample and therefore it is equivalent to it.
        strip
            Whether to strip -remove the NaNs from the start and the end of- the transformed series.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.dataprocessing.transformers import MIDAS
        >>> monthly_series = AirPassengersDataset().load()
        >>> midas = MIDAS(rule="QS")
        >>> quarterly_series = midas.transform(monthly_series)
        >>> print(quarterly_series.head())
        <TimeSeries (DataArray) (Month: 5, component: 3, sample: 1)>
        array([[[112.],
                [118.],
                [132.]],
        <BLANKLINE>
               [[129.],
                [121.],
                [135.]],
        <BLANKLINE>
               [[148.],
                [148.],
                [136.]],
        <BLANKLINE>
               [[119.],
                [104.],
                [118.]],
        <BLANKLINE>
               [[115.],
                [126.],
                [141.]]])
        Coordinates:
          * Month      (Month) datetime64[ns] 1949-01-01 1949-04-01 ... 1950-01-01
          * component  (component) object '#Passengers_0' ... '#Passengers_2'
        Dimensions without coordinates: sample
        Attributes:
            static_covariates:  None
            hierarchy:          None

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Mixed-data_sampling
        """
        self._rule = rule
        self._strip = strip
        # Original high frequency should be fitted on TimeSeries independently
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, global_fit=False)

    @staticmethod
    def ts_fit(
        series: Union[TimeSeries, Sequence[TimeSeries]],
        params: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """MIDAS needs the high frequency period name in order to easily reverse_transform
        TimeSeries, the parallelization is handled by `transform` and/or `inverse_transform`
        (see InvertibleDataTransformer.__init__() docstring).
        """
        is_single_series = isinstance(series, TimeSeries)

        if is_single_series:
            series = [series]

        fitted_params = [
            {
                "high_freq_name": ts.freq_str,
                "start": ts.start_time(),
                "end": ts.end_time(),
            }
            for ts in series
        ]

        return fitted_params[0] if is_single_series else fitted_params

    @staticmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
        """
        Transforms series from high to low frequency using a mixed-data sampling approach. Uses and relies on
        pandas.DataFrame.resample.

        Steps:
            (1) Transform series to pd.DataFrame and get frequency string for PeriodIndex
            (2) Downsample series and then upsample it again
            (3) Replace input series by unsampled series if it's not 'full'
            (4) Transform every column of the high frequency series into multiple columns for the low frequency series
            (5) Transform the low frequency series back into a TimeSeries
        """
        MIDAS._verify_series(series)
        rule, strip = params["fixed"]["_rule"], params["fixed"]["_strip"]

        # TimeSeries to pd.DataFrame
        series_df = series.pd_dataframe(copy=True)
        # TODO: get ride of the double copy?
        series_copy_df = series_df.copy()

        # get high frequency string that's suitable for PeriodIndex
        high_freq_datetime = series.freq_str
        high_freq_period = series_df.index.to_period().freqstr

        # downsample
        low_freq_series_df = series_df.resample(rule).last()
        # save the downsampled index
        low_index_datetime = low_freq_series_df.index

        # upsample again to get full range of high freq periods for every low freq period
        low_freq_series_df.index = low_index_datetime.to_period()
        high_freq_series_df = low_freq_series_df.resample(high_freq_period).last()

        # make sure the extension of the index matches the original index
        if "End" in str(series.freq):
            args_to_timestamp = {"freq": high_freq_period}
        else:
            args_to_timestamp = {"how": "start"}
        high_index_datetime = high_freq_series_df.index.to_timestamp(
            **args_to_timestamp
        )

        raise_if_not(
            low_freq_series_df.shape[0] < high_freq_series_df.shape[0],
            f"The target conversion should go from a high to a "
            f"low frequency, instead the targeted frequency is "
            f"{rule}, while the original frequency is {high_freq_datetime}.",
            logger,
        )

        # if necessary, expand the original series
        if len(high_index_datetime) > series_df.shape[0]:
            series_df = pd.DataFrame(
                np.nan, index=high_index_datetime, columns=series_copy_df.columns
            )
            series_df.loc[series_copy_df.index, :] = series_copy_df.values

        # make multiple low frequency columns out of the high frequency column(s)
        midas_df = _create_midas_df(
            series_df=series_df,
            low_index_datetime=low_index_datetime,
        )

        # back to TimeSeries
        midas_ts = TimeSeries.from_dataframe(
            midas_df,
            static_covariates=series.static_covariates,
        )
        if strip:
            midas_ts = midas_ts.strip()

        # components: comp0_0, comp1_0, comp0_1, comp1_1, ...
        return midas_ts

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries,
        params: Mapping[str, Any],
    ) -> TimeSeries:
        """
        Transforms series back to high frequency by retrieving the original high frequency and reshaping the values.
        When converting to/from anchorable offset [1]_, the original time index is not garanteed to be restored.

        Steps:
            (1) Reshape the values to eliminate the components introduced by the transform
            (2) Identify ratio between the low frequency and the original high frequency
            (3) Generate a new time index with the high frequency
            (4) When applicable, shift the time index back in time to adjust the start date

        References
        ----------
        .. [1] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets
        """
        MIDAS._verify_series(series)
        strip = params["fixed"]["_strip"]
        high_freq_name = params["fitted"]["high_freq_name"]
        orig_ts_start_time = params["fitted"]["start"]
        # orig_ts_end_time = params["fitted"]["end"]

        # retrieve the number of component introduced by midas
        n_midas_components = int(series.components[-1].split("_")[-1]) + 1
        series_n_components = series.n_components

        # original ts was univariate
        if n_midas_components == series_n_components:
            n_orig_components = 1
            series_values = series.values().flatten()
        else:
            n_orig_components = series_n_components // n_midas_components
            series_values = series.values().reshape((-1, n_orig_components))

        # retrieve original components name by removing the "_0" suffix
        component_names = [
            series.components[i][:-2] for i in range(0, n_orig_components)
        ]

        # adjust the start of the inverse transform output
        low_freq_timedelta = series.time_index[1] - series.time_index[0]
        transform_time_shift = series.time_index[0] - orig_ts_start_time

        # downsampling shift is smaller than low freq period
        if np.abs(transform_time_shift) <= low_freq_timedelta:
            start_time = orig_ts_start_time
        else:
            start_time = series.start_time()

        # account for the NaN
        left_nans = 0
        while np.isnan(series_values[left_nans]):
            left_nans += 1

        if strip and left_nans > 0:
            pass

        new_times = pd.date_range(
            start=start_time, periods=len(series_values), freq=high_freq_name
        )

        inversed_midas_ts = TimeSeries.from_times_and_values(
            times=new_times,
            values=series_values,
            freq=high_freq_name,
            columns=component_names,
            static_covariates=series.static_covariates,
        )

        if strip:
            inversed_midas_ts = inversed_midas_ts.strip()

        return inversed_midas_ts

    @staticmethod
    def _verify_series(series: TimeSeries):
        raise_if(
            series.is_probabilistic,
            "MIDAS Transformer cannot be applied to probabilistic/stochastic TimeSeries",
            logger,
        )

        raise_if_not(
            isinstance(series.time_index, pd.DatetimeIndex),
            "MIDAS input series must have a pd.Datetime index",
            logger,
        )


def _create_midas_df(
    series_df: pd.DataFrame,
    low_index_datetime: DatetimeIndex,
) -> pd.DataFrame:
    """
    Function creating the lower frequency dataframe out of a higher frequency dataframe.
    """
    # calculate the multiple
    n_high = series_df.shape[0]
    n_low = len(low_index_datetime)

    raise_if_not(
        n_high % n_low == 0,
        "The frequency of the high frequency input series should be an exact multiple of the targeted"
        "low frequency output. For example, you could go from a monthly series to a quarterly series.",
        logger,
    )

    multiple = n_high // n_low

    # set up integer index
    range_lst = list(range(n_high))
    col_names = list(series_df.columns)
    midas_lst = []

    # for every column we now create 'multiple' columns
    # by going through a column and picking every one in 'multiple' values
    for f in range(multiple):
        range_lst_tmp = range_lst[f:][0::multiple]
        series_tmp_df = series_df.iloc[range_lst_tmp, :]
        series_tmp_df.index = low_index_datetime
        col_names_tmp = [f"{col_name}_{f}" for col_name in col_names]
        rename_dict_tmp = dict(zip(col_names, col_names_tmp))
        midas_lst += [series_tmp_df.rename(columns=rename_dict_tmp)]

    return pd.concat(midas_lst, axis=1)
