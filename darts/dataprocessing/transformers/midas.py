"""
Mixed-data sampling (MIDAS) Transformer
------------------
"""
from typing import Union

import numpy as np
import pandas as pd
from pandas import DateOffset, DatetimeIndex, Timedelta

from darts import TimeSeries
from darts.dataprocessing.transformers import BaseDataTransformer
from darts.logging import get_logger, raise_log

logger = get_logger(__name__)


class MIDASTransformer(BaseDataTransformer):
    def __init__(
        self,
        rule: str,
        strip: bool = True,
        name: str = "MIDASTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """
        A transformer that converts higher frequency time series to lower frequency using mixed-data sampling.
        """
        super().__init__(name, n_jobs, verbose)
        self.rule = rule
        self.strip = strip

    @staticmethod
    def ts_transform(
        series: TimeSeries,
        rule: Union[DateOffset, Timedelta, str],
        strip: bool,
    ) -> TimeSeries:
        high_freq_datetime = series.freq_str
        series_df = series.pd_dataframe()
        series_copy_df = series_df.copy()
        series_df.index = series_df.index.to_period()
        high_freq_period = series_df.index.freqstr

        # ensure the length of the series is an exact multiple of the length of the targeted low frequency series
        # we do this by resampling from a high freq to a low freq and then back to high again (possibly adding NaNs)
        low_freq_series_df = series_df.resample(rule).last()
        low_index_datetime = low_freq_series_df.index.to_timestamp()
        high_freq_series_df = (
            low_freq_series_df.resample(high_freq_period).bfill().ffill()
        )
        high_index_datetime = high_freq_series_df.index.to_timestamp()

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
            series_df.loc[series_copy_df.index, :] = series_copy_df.values
        else:
            series_df = series_copy_df

        midas_df = _create_midas_df(
            series_df=series_df,
            low_freq_series_df=low_freq_series_df,
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
    """ "
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
    low_freq_series_df: pd.DataFrame,
    low_index_datetime: DatetimeIndex,
) -> pd.DataFrame:
    """
    Function for actually creating the lower frequency dataframe out of a higher frequency dataframe.
    """
    # calculate the multiple
    n_high = series_df.shape[0]
    n_low = low_freq_series_df.shape[0]
    multiple = int(n_high / n_low)

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


# from darts.datasets import AirPassengersDataset

# series = AirPassengersDataset().load()
