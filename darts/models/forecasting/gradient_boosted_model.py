"""
LightGBM Model
--------------

This is a LightGBM implementation of Gradient Boosted Trees algorightm.

To enable LightGBM support in Darts, follow the detailed install instructions for LightGBM in the README:
https://github.com/unit8co/darts/blob/master/README.md
"""

from darts.logging import get_logger
from typing import Union, Optional, Sequence, List, Tuple
from darts.models.forecasting.regression_model import RegressionModel
from darts.timeseries import TimeSeries
import lightgbm as lgb

logger = get_logger(__name__)


class LightGBMModel(RegressionModel):
    def __init__(self,
                 lags: Union[int, list] = None,
                 lags_past_covariates: Union[int, List[int]] = None,
                 lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
                 **kwargs):
        """ Light Gradient Boosted Model

        Parameters
        ----------
        lags
            Lagged target values used to predict the next time step. If an integer is given the last `lags` past lags
            are used (from -1 backward). Otherwise a list of integers with lags is required (each lag must be < 0).
        lags_past_covariates
            Number of lagged past_covariates values used to predict the next time step. If an integer is given the last
            `lags_past_covariates` past lags are used (inclusive, starting from lag -1). Otherwise a list of integers
            with lags < 0 is required.
        lags_future_covariates
            Number of lagged future_covariates values used to predict the next time step. If an tuple (past, future) is
            given the last `past` lags in the past are used (inclusive, starting from lag -1) along with the first
            `future` future lags (starting from 0 - the prediction time - up to `future - 1` included). Otherwise a list
            of integers with lags is required.
        **kwargs
            Additional keyword arguments passed to `lightgbm.LGBRegressor`.
        """
        self.kwargs = kwargs

        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            model=lgb.LGBMRegressor(
                **kwargs
            )
        )

    def __str__(self):
        return 'LGBModel(lags={}, lags_past={}, lags_future={})'.format(
            self.lags, self.lags_past_covariates, self.lags_future_covariates
        )

    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            max_samples_per_ts: Optional[int] = None,
            **kwargs) -> None:
        """
        Fits/trains the model using the provided list of features time series and the target time series.

        Parameters
        ----------
        series
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        past_covariates
            Optionally, a series or sequence of series specifying past-observed covariates
        future_covariates
            Optionally, a series or sequence of series specifying future-known covariates
        val_series
            TimeSeries or Sequence[TimeSeries] object containing the target values for evaluation dataset
        val_past_covariates
            Optionally, a series or sequence of series specifying past-observed covariates for evaluation dataset
        val_future_covariates : Union[TimeSeries, Sequence[TimeSeries]]
            Optionally, a series or sequence of series specifying future-known covariates for evaluation dataset
        max_samples_per_ts
            This is an integer upper bound on the number of tuples that can be produced
            per time series. It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        """

        if val_series is not None:

            kwargs['eval_set'] = self._create_lagged_data(
                target_series=val_series,
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
                max_samples_per_ts=max_samples_per_ts
                )

        super().fit(series=series,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    max_samples_per_ts=max_samples_per_ts,
                    **kwargs)
