"""
Mixed-data sampling (MIDAS) Transformer
------------------
"""
from typing import Iterator, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DateOffset, DatetimeIndex, Timedelta

from darts import TimeSeries
from darts.dataprocessing.transformers import BaseDataTransformer
from darts.logging import get_logger, raise_log
from darts.utils.utils import series2seq

logger = get_logger(__name__)


class MIDAS(BaseDataTransformer):
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
        super().__init__(name, n_jobs, verbose)
        self.rule = rule
        self.strip = strip

    def _transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries]]:

        series = series2seq(series)

        for s in series:
            yield s, self.rule, self.strip

    @staticmethod
    def ts_transform(
        series: TimeSeries,
        rule: Union[DateOffset, Timedelta, str],
        strip: bool = True,
    ) -> TimeSeries:
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
        high_freq_datetime = series.freq_str
        # TimeSeries to pd.DataFrame
        series_df = series.pd_dataframe()
        # get high frequency string that's suitable for PeriodIndex
        series_period_index_df = series_df.copy()
        series_period_index_df.index = series_df.index.to_period()
        high_freq_period = series_period_index_df.index.freqstr

        # downsample
        low_freq_series_df = series_df.resample(rule).last()
        low_index_datetime = low_freq_series_df.index
        low_freq_series_df.index = low_index_datetime.to_period()

        # upsample to get full range of high freq periods for every low freq period
        high_freq_series_df = (
            low_freq_series_df.resample(high_freq_period).bfill().ffill()
        )
        high_index_datetime = high_freq_series_df.index.to_timestamp()

        # check if user requested a transform from a high to a low frequency
        _assert_high_to_low_freq(
            high_freq_series_df=high_freq_series_df,
            low_freq_series_df=low_freq_series_df,
            rule=rule,
            high_freq=high_freq_datetime,
        )

        # if necessary, expand the original series
        if len(high_index_datetime) > series_df.shape[0]:
            series_df = pd.DataFrame(
                np.nan, index=high_index_datetime, columns=series_df.columns
            )
            series_df.loc[series_df.index, :] = series_df.values

        # make multiple low frequency columns out of the high frequency column(s)
        midas_df = _create_midas_df(
            series_df=series_df,
            low_index_datetime=low_index_datetime,
        )

        # back to TimeSeries
        midas_ts = TimeSeries.from_dataframe(midas_df)
        if strip:
            midas_ts = midas_ts.strip()

        return midas_ts


def _assert_high_to_low_freq(
    high_freq_series_df: pd.DataFrame,
    low_freq_series_df: pd.DataFrame,
    rule,
    high_freq,
):
    """
    Asserts that the lower frequency series really has a lower frequency then the assumed higher frequency series.
    """
    if not low_freq_series_df.shape[0] < high_freq_series_df.shape[0]:
        raise_log(
            ValueError(
                f"The target conversion should go from a high to a "
                f"low frequency, instead the targeted frequency is"
                f"{rule}, while the original frequency is {high_freq}."
            )
        )


def _create_midas_df(
    series_df: pd.DataFrame,
    low_index_datetime: DatetimeIndex,
) -> pd.DataFrame:
    """
    Function for actually creating the lower frequency dataframe out of a higher frequency dataframe.
    """
    # calculate the multiple
    n_high = series_df.shape[0]
    n_low = len(low_index_datetime)
    multiple = n_high / n_low

    if not multiple.is_integer():
        raise_log(
            ValueError(
                "The frequency of the high frequency input series should be an exact multiple of the targeted"
                "low frequency output. For example, you could go from a monthly series to a quarterly series."
            )
        )
    else:
        multiple = int(multiple)

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
        col_names_tmp = [col_name + f"_{f}" for col_name in col_names]
        rename_dict_tmp = dict(zip(col_names, col_names_tmp))
        midas_lst += [series_tmp_df.rename(columns=rename_dict_tmp)]

    return pd.concat(midas_lst, axis=1)
