"""
Mixed-data sampling (MIDAS) Transformer
------------------
"""
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas import DatetimeIndex

from darts import TimeSeries
from darts.dataprocessing.transformers import (
    FittableDataTransformer,
    InvertibleDataTransformer,
)
from darts.logging import get_logger, raise_if, raise_if_not
from darts.timeseries import _finite_rows_boundaries
from darts.utils.timeseries_generation import generate_index

logger = get_logger(__name__)


class MIDAS(FittableDataTransformer, InvertibleDataTransformer):
    def __init__(
        self,
        low_freq: str,
        strip: bool = True,
        drop_static_covariates: bool = False,
        name: str = "MIDAS",
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

        For anchored low frequency, the transformed series must contain at least 2 samples in order to be
        able to retrieve the original time index.

        Parameters
        ----------
        low_freq
            The pd.DateOffset string alias corresponding to the target low
            frequency [2]_. Passed on to the `rule` parameter of pandas.DataFrame.resample().
        strip
            Whether to remove the NaNs from the start and the end of the transformed series.
        drop_static_covariates
            If set to `True`, the statics covariates of the input series won't be transferred to the output.
            Recommended for multivariate series with component-specific static covariates.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.dataprocessing.transformers import MIDAS
        >>> monthly_series = AirPassengersDataset().load()
        >>> print(monthly_series.time_index[:4])
        DatetimeIndex(['1949-01-01', '1949-02-01', '1949-03-01', '1949-04-01'], dtype='datetime64[ns]',
        name='Month', freq='MS')
        >>> print(monthly_series.values()[:4])
        [[112.], [118.], [132.], [129.]]

        >>> midas = MIDAS(low_freq="QS")
        >>> quarterly_series = midas.fit_transform(monthly_series)
        >>> print(quarterly_series.time_index[:3])
        DatetimeIndex(['1949-01-01', '1949-04-01', '1949-07-01'], dtype='datetime64[ns]', name='Month', freq='QS-JAN')
        >>> print(quarterly_series.values()[:3])
        [[112. 118. 132.], [129. 121. 135.], [148. 148. 136.]]

        >>> inversed_quaterly = midas.inverse_transform(quarterly_series)
        >>> print(inversed_quaterly.time_index[:4])
        DatetimeIndex(['1949-01-01', '1949-02-01', '1949-03-01', '1949-04-01'], dtype='datetime64[ns]',
        name='time', freq='MS')
        >>> print(inversed_quaterly.values()[:4])
        [[112.], [118.], [132.], [129.]]

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Mixed-data_sampling
        .. [2] https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects
        """
        self._low_freq = low_freq
        self._strip = strip
        self._drop_static_covariates = drop_static_covariates
        # Original high frequency should be fitted on TimeSeries independently
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, global_fit=False)

    @staticmethod
    def ts_fit(
        series: Union[TimeSeries, Sequence[TimeSeries]],
        params: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """MIDAS needs the high frequency period name in order to easily reverse_transform
        TimeSeries, the parallelization is handled by `transform` and/or `inverse_transform`
        (see InvertibleDataTransformer.__init__() docstring).
        """
        is_single_series = isinstance(series, TimeSeries)

        if is_single_series:
            series = [series]

        fitted_params = [
            {
                "high_freq": ts.freq_str,
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
        low_freq = params["fixed"]["_low_freq"]
        strip = params["fixed"]["_strip"]
        drop_static_covariates = params["fixed"]["_drop_static_covariates"]
        high_freq = params["fitted"]["high_freq"]
        MIDAS._verify_series(series, high_freq=high_freq)

        # TimeSeries to pd.DataFrame
        series_df = series.pd_dataframe(copy=True)
        # TODO: get ride of the double copy?
        series_copy_df = series_df.copy()

        # get high frequency string that's suitable for PeriodIndex
        high_freq_datetime = series.freq_str
        high_freq_period = series_df.index.to_period().freqstr

        # downsample
        low_freq_series_df = series_df.resample(rule=low_freq).last()
        # save the downsampled index
        low_index_datetime = low_freq_series_df.index

        # upsample again to get full range of high freq periods for every low freq period
        low_freq_series_df.index = low_index_datetime.to_period()
        high_freq_series_df = low_freq_series_df.resample(rule=high_freq_period).last()

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
            f"{low_freq}, while the original frequency is {high_freq_datetime}.",
            logger,
        )

        # if necessary, expand the original series
        if len(high_index_datetime) > series_df.shape[0]:
            series_df = pd.DataFrame(
                np.nan, index=high_index_datetime, columns=series_copy_df.columns
            )
            series_df.loc[series_copy_df.index, :] = series_copy_df.values

        n_high = series_df.shape[0]
        n_low = len(low_index_datetime)

        raise_if_not(
            n_high % n_low == 0,
            "The frequency of the high frequency input series should be an exact multiple of the targeted"
            "low frequency output. For example, you could go from a monthly series to a quarterly series.",
            logger,
        )

        multiple = n_high // n_low

        # make multiple low frequency columns out of the high frequency column(s)
        midas_df = MIDAS._create_midas_df(
            series_df=series_df,
            low_index_datetime=low_index_datetime,
            multiple=multiple,
        )

        new_static_covariates = MIDAS._process_static_covariates(
            static_covariates=series.static_covariates,
            index_or_multiple=multiple,
            drop_static_covariates=drop_static_covariates,
            inverse_transform=False,
        )

        # back to TimeSeries
        midas_ts = TimeSeries.from_dataframe(
            midas_df,
            static_covariates=new_static_covariates,
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
            (1) Reshape the values to flatten the components introduced by the transform
            (2) Eliminate the rows filled with NaNs, to facilitate time index adjustment
            (3) Retrieve the original components name
            (4) When applicable, shift the time index start back in time
            (5) Generate a new time index with the high frequency

        References
        ----------
        .. [1] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets
        """
        low_freq = params["fixed"]["_low_freq"]
        drop_static_covariates = params["fixed"]["_drop_static_covariates"]
        high_freq = params["fitted"]["high_freq"]
        orig_ts_start_time = params["fitted"]["start"]
        orig_ts_end_time = params["fitted"]["end"]
        MIDAS._verify_series(series, low_freq=low_freq)

        # retrieve the number of component introduced by midas
        n_midas_components = int(series.components[-1].split("_")[-1]) + 1
        series_n_components = series.n_components

        n_orig_components = series_n_components // n_midas_components
        # original ts was univariate
        if n_orig_components == 1:
            series_values = series.values(copy=False).flatten()
        else:
            series_values = series.values(copy=False).reshape((-1, n_orig_components))

        # retrieve original components name by removing the "_0" suffix
        component_names = [
            "_".join(series.components[i].split("_")[:-1])
            for i in range(0, n_orig_components)
        ]

        # remove the rows containing only NaNs at the extremities of the array, necessary to adjust the time index
        first_finite_row, last_finite_row = _finite_rows_boundaries(
            series_values, how="all"
        )
        # adding one to make the end bound inclusive
        series_values = series_values[first_finite_row : last_finite_row + 1]

        start_time = series.start_time()
        shift = 0
        # adjust the start if was shifted due to the frequency change
        if len(series.time_index) > 1:
            low_freq_timedelta = series.time_index[1] - series.time_index[0]
            transform_time_shift = series.time_index[0] - orig_ts_start_time
            # shift is caused by the low frequency anchoring, fitted and inversed ts have the same start
            if np.abs(transform_time_shift) <= low_freq_timedelta:
                start_time = orig_ts_start_time
            # shift is caused by the low frequency anchoring, inversed ts starts after the end of the fitted ts
            elif series.start_time() - orig_ts_end_time <= low_freq_timedelta:
                start_time = orig_ts_end_time
                shift = 1

        new_static_covariates = MIDAS._process_static_covariates(
            static_covariates=series.static_covariates,
            index_or_multiple=n_orig_components,
            drop_static_covariates=drop_static_covariates,
            inverse_transform=True,
        )

        inversed_midas_ts = TimeSeries.from_times_and_values(
            times=generate_index(
                start=start_time,
                length=len(series_values) + shift,
                freq=high_freq,
                name=series.time_index.name,
            )[shift:],
            values=series_values,
            freq=high_freq,
            columns=component_names,
            static_covariates=new_static_covariates,
        )

        return inversed_midas_ts

    @staticmethod
    def _verify_series(
        series: TimeSeries,
        high_freq: Optional[str] = None,
        low_freq: Optional[str] = None,
    ):
        """Some sanity checks on the input, the high_freq and low_freq arguments are mutually exclusive"""
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

        series_freq_str = series.freq_str
        input_freq = [series_freq_str]
        # flexibility on anchoring
        if isinstance(series_freq_str, str) and "-" in series_freq_str:
            input_freq.append(series_freq_str.split("-")[0])

        raise_if(
            high_freq is not None and high_freq not in input_freq,
            f"The frequency string of the series to transform must be identical to the fitted one, expected "
            f"{high_freq} but received {series.freq_str}.",
        )

        raise_if(
            low_freq is not None and low_freq not in input_freq,
            f"The frequency string of the series to inverse-transform must be identical to the fitted one, "
            f"expected {low_freq} but received {series.freq_str}.",
        )

    @staticmethod
    def _process_static_covariates(
        static_covariates: Union[None, pd.Series, pd.DataFrame],
        index_or_multiple: int,
        drop_static_covariates: bool,
        inverse_transform: bool,
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """If static covariates are component-specific, they must be reshaped appropriately.
        `index_or_multiple` has a different meaning depending on the transformation:
        - transform : multiple, to repeat the static covariates for the new components
        - inverse_transform : index, to remove the duplciated static covariates
        """
        if drop_static_covariates:
            return None
        elif (
            static_covariates is not None
            and static_covariates.index.name == "component"
        ):
            if inverse_transform:
                return static_covariates[:index_or_multiple]
            else:
                return pd.concat([static_covariates] * index_or_multiple)
        else:
            return static_covariates

    @staticmethod
    def _create_midas_df(
        series_df: pd.DataFrame,
        low_index_datetime: DatetimeIndex,
        multiple: int,
    ) -> pd.DataFrame:
        """
        Function creating the lower frequency dataframe out of a higher frequency dataframe.
        """
        # set up integer index
        cols_out = []
        midas_lst = []
        # for every column we now create 'multiple' columns
        # by going through a column and picking every one in 'multiple' values
        for f in range(multiple):
            cols_out += (series_df.columns + f"_{f}").tolist()
            midas_lst.append(series_df.iloc[f::multiple].reset_index(drop=True))
        transformed = pd.concat(midas_lst, axis=1)
        transformed.index = low_index_datetime
        transformed.columns = cols_out
        return transformed
