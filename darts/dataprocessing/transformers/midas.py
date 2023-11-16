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
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import _finite_rows_boundaries
from darts.utils.timeseries_generation import generate_index

logger = get_logger(__name__)


class MIDAS(FittableDataTransformer, InvertibleDataTransformer):
    def __init__(
        self,
        low_freq: str,
        strip: bool = True,
        how: str = "all",
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
        how
            Define if the entries containing `NaN` in all the components ('all') or in any of the components ('any')
            should be stripped. Default: 'all'
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
        self._how = how
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

        fitted_params = []
        low_freq = params["fixed"]["_low_freq"]
        for idx, ts in enumerate(series):
            high_freq = ts.freq_str
            if not pd.tseries.frequencies.is_subperiod(
                pd.tseries.frequencies.get_period_alias(high_freq),
                pd.tseries.frequencies.get_period_alias(low_freq),
            ):
                raise_log(
                    ValueError(
                        f"The frequency string of the series at index={idx} must be higher than the "
                        f"`low_freq` set at MIDAS creation. "
                        f"Received series frequency {high_freq} against `low_freq={low_freq}`"
                    ),
                    logger=logger,
                )
            fitted_params.append(
                {
                    "high_freq": high_freq,
                    "start": ts.start_time(),
                    "end": ts.end_time(),
                }
            )
        return fitted_params[0] if is_single_series else fitted_params

    @staticmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
        """
        Transforms series from high to low frequency using a mixed-data sampling approach. Uses and relies on
        pandas.DataFrame.resample.

        When converting to/from anchorable offset [1]_, the index is rolled backward if the series does not start on
        the anchor date to preserve all the values.

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
        df = pd.DataFrame(index=series.time_index)

        # get high frequency string that's suitable for PeriodIndex
        high_freq_period = df.index.to_period().freqstr

        # downsample
        resampled = df.resample(low_freq)
        low_freq_df = resampled.last()

        def get_upsampled(low_df: pd.DataFrame, high_period):
            """Upsample a single index DataFrame to a higher frequency"""
            low_df = low_df.copy(deep=True)
            low_df.index = low_df.index.to_period()
            return low_df.resample(rule=high_period).last()

        # first and last groups can be shorter than an entire lower freq period
        # we upsample them from the low to high frequency to get the expected number
        # higher freq time steps in one lower freq
        first_upsampled = get_upsampled(low_freq_df.iloc[:1], high_freq_period)
        last_upsampled = get_upsampled(low_freq_df.iloc[-1:], high_freq_period)

        # find unique sizes from: first group size + unique sizes of center groups + last group size
        sizes = np.unique(
            [len(first_upsampled)]
            + resampled.size()[1:-1].unique().tolist()
            + [len(last_upsampled)]
        )

        # MIDAS requires the high freq to be a round multiple of the low freq -> sizes must be identical
        if not len(sizes) == 1:
            raise_log(
                ValueError(
                    "The frequency of the input series should be an exact multiple of the targeted "
                    f"lower frequency output. Received series frequency `{high_freq}`, and lower frequency "
                    f"{low_freq}. E.g., a valid conversion would be from a monthly (high) to a quarterly "
                    f"(low) frequency."
                ),
                logger=logger,
            )

        # max size is the number of higher frequency time steps in one lower frequency period
        max_size = sizes[0]

        n_samples = series.n_samples
        n_cols = series.n_components
        n_times = len(series)

        sizes = resampled.size()
        n_midas_cols = max_size * n_cols

        arr = series.all_values(copy=False)
        time_index = low_freq_df.index
        if n_times < max_size:
            first_idx = first_upsampled.index.get_loc(df.index[0])
            last_idx = first_idx + n_times - 1

            start_chunk = np.empty((first_idx, 1, 1))
            start_chunk.fill(np.nan)
            end_chunk = np.empty((max_size - 1 - last_idx, 1, 1))
            end_chunk.fill(np.nan)
            arr = np.concatenate([start_chunk, arr, end_chunk])
            # arr has shape (n time steps, n components, n samples)
            # reshape to (1 time step, n midas components, n samples)
            arr = arr.reshape(1, n_midas_cols, n_samples)

            if strip:
                # results in an empty series
                arr = arr[0:0]
                time_index = time_index[0:0]
        else:
            # extract rows from higher frequency and add convert them to columns in the lower frequency
            # we can achieve this by extracting all windows with a size of `max_size`;
            # later on we stride to get only the relevant windows each `max_size` steps
            arr = np.lib.stride_tricks.sliding_window_view(
                arr, window_shape=max_size, axis=0
            )
            arr = arr.reshape(len(arr), n_midas_cols, n_samples)

            # the first resampled index might not have all dates from higher freq
            size_group_first = sizes.iloc[0]
            size_group_first = 0 if size_group_first == max_size else size_group_first
            components_group_first = size_group_first * n_cols
            first_group_arr = None
            if components_group_first and not strip:
                first_group_arr = np.empty(
                    (n_midas_cols - components_group_first, n_samples)
                )
                first_group_arr.fill(np.nan)
                first_group_arr = np.concatenate(
                    [first_group_arr, arr[0, :components_group_first]]
                )
                first_group_arr = np.expand_dims(first_group_arr, axis=0)

            # the last resampled index might not have all dates from higher freq
            size_group_last = sizes.iloc[-1]
            size_group_last = 0 if size_group_last == max_size else size_group_last
            components_group_last = size_group_last * n_cols
            last_group_arr = None
            if components_group_last and not strip:
                last_group_arr = np.empty(
                    (n_midas_cols - components_group_last, n_samples)
                )
                last_group_arr.fill(np.nan)
                last_group_arr = np.concatenate(
                    [arr[-1, -components_group_last:], last_group_arr]
                )
                last_group_arr = np.expand_dims(last_group_arr, axis=0)

            arr = arr[size_group_first::max_size]
            arr = np.concatenate(
                [
                    group_arr
                    for group_arr in [first_group_arr, arr, last_group_arr]
                    if group_arr is not None
                ]
            )

            if strip:
                first_idx = None if not components_group_first else 1
                last_idx = None if not components_group_last else -1
                time_index = time_index[first_idx:last_idx]

            # TODO: remove this
            arr = np.concatenate([arr[:, i::max_size] for i in range(max_size)], axis=1)

        ts = MIDAS._create_midas_df2(
            series=series,
            arr=arr,
            time_index=time_index,
            multiple=max_size,
            drop_static_covariates=drop_static_covariates,
            inverse_transform=False,
        )
        return ts

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries,
        params: Mapping[str, Any],
    ) -> TimeSeries:
        """
        Transforms series back to high frequency by retrieving the original high frequency and reshaping the values.

        When converting to/from anchorable offset [1]_, the index is rolled backward if the series does not start on
        the anchor date to preserve all the values.

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

        if len(series) == 0:
            # placeholders
            start_time = pd.Timestamp("2020-01-01")
            shift = 0
            series_values = np.empty((0, n_orig_components))
        else:
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
                start_to_start_shift = series.time_index[0] - orig_ts_start_time
                start_to_end_shift = series.time_index[0] - orig_ts_end_time
                # shift is caused by the low frequency anchoring, fitted and inversed ts have the same start
                if np.abs(start_to_start_shift) <= low_freq_timedelta:
                    start_time = orig_ts_start_time
                # shift is caused by the low frequency anchoring, inversed ts starts after the end of the fitted ts
                elif pd.Timedelta(0) < start_to_end_shift <= low_freq_timedelta:
                    start_time = orig_ts_end_time
                    shift = 1

        # time_index = generate_index(
        #     start=start_time,
        #     length=len(series_values) + shift,
        #     freq=high_freq,
        #     name=series.time_index.name,
        # )[shift:]
        #
        # inversed_midas_ts = MIDAS._create_midas_df2(
        #     series=series,
        #     arr=series_values,
        #     time_index=time_index,
        #     multiple=0,
        #     drop_static_covariates=drop_static_covariates,
        #     inverse_transform=True,
        # )

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
        if "-" in series_freq_str:
            input_freq.append(series_freq_str.split("-")[0])

        if high_freq is not None and high_freq not in input_freq:
            raise_log(
                ValueError(
                    f"The frequency string of the series to transform must be identical to the fitted one, expected "
                    f"{high_freq} but received {series_freq_str}."
                ),
                logger=logger,
            )
        if low_freq is not None and low_freq not in input_freq:
            raise_log(
                ValueError(
                    f"The frequency string of the series to inverse-transform must be identical to the fitted one, "
                    f"expected {low_freq} but received {series_freq_str}."
                ),
                logger=logger,
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

    @staticmethod
    def _process_static_covariates2(
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
    def _create_midas_df2(
        series: TimeSeries,
        arr: np.ndarray,
        time_index: Union[pd.DatetimeIndex, pd.RangeIndex],
        multiple: int,
        drop_static_covariates: bool,
        inverse_transform: bool,
    ) -> TimeSeries:
        """
        Function creating the lower frequency dataframe out of a higher frequency dataframe.
        """
        static_covariates = MIDAS._process_static_covariates2(
            static_covariates=series.static_covariates,
            index_or_multiple=multiple,
            drop_static_covariates=drop_static_covariates,
            inverse_transform=inverse_transform,
        )
        return TimeSeries.from_times_and_values(
            times=time_index,
            values=arr,
            # TODO: revert this to [f"{col}_{i}" for col in series.columns for i in range(multiple)]
            columns=[f"{col}_{i}" for i in range(multiple) for col in series.columns],
            static_covariates=static_covariates,
        )
