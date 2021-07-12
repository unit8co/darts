"""
Regression Model
----------------

A `RegressionModel` forecasts future values of a target series based on lagged values of the target values
and possibly lags of an exogenous series. They can wrap around any regression model having a `fit()`
and `predict()` functions accepting tabularized data (e.g. scikit-learn regression models), and are using
`sklearn.linear_model.LinearRegression` by default.

Behind the scenes this model is tabularizing the time series data to make it work with regression models.
"""

import numpy as np
import pandas as pd

from inspect import signature
from typing import Union, Optional, Sequence

from darts.utils.data import SimpleInferenceDataset

from ..timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from .forecasting_model import GlobalForecastingModel
from ..logging import raise_if, raise_if_not, get_logger, raise_log
from darts.utils.data.lagged_dataset import LaggedDataset

logger = get_logger(__name__)


class RegressionModel(GlobalForecastingModel):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_covariates: Union[int, list] = None,
        model=None,
    ):
        """Regression Model

        Can be used to fit any scikit-learn-like regressor class to predict the target
        time series from lagged values.

        Parameters
        ----------
        lags : Union[int, list]
            Number of lagged target values used to predict the next time step. If an integer is given
            the last `lags` lags are used (inclusive). Otherwise a list of integers with lags is required.
            The integers must be strictly positive (>0).
        lags_covariates : Union[int, list, bool]
            Number of lagged exogenous values used to predict the next time step. If an integer is given
            the last `lags_covariates` lags are used (inclusive). Otherwise a list of integers with lags is required.
            The integers must be positive (>=0).
        model
            Scikit-learn-like model with `fit()` and `predict()` methods.
            Default: `sklearn.linear_model.LinearRegression(n_jobs=-1, fit_intercept=False)`
        """

        super().__init__()
        if model is None:
            model = LinearRegression()

        if not callable(getattr(model, "fit", None)):
            raise_log(
                Exception("Provided model object must have a fit() method", logger)
            )
        if not callable(getattr(model, "predict", None)):
            raise_log(
                Exception("Provided model object must have a predict() method", logger)
            )

        self.model = model
        self.lags = lags
        self.lags_covariates = lags_covariates
        self._fit_called = False

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts=None,
        **kwargs
    ) -> None:
        """Fits/trains the model using the provided list of features time series and the target time series.

        Parameters
        ----------
        series : Union[TimeSeries, Sequence[TimeSeries]]
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        covariates : Union[TimeSeries, Sequence[TimeSeries]], optional
            TimeSeries or Sequence[TimeSeries] object containing the exogenous values.
        max_samples_per_ts : int
            # TODO describe param
        """
        super().fit(series, covariates)

        raise_if(
            covariates is not None and self.lags_covariates is None,
            "`covariates` not None in `fit()` method call, but `lags_covariates` is None in constructor. ",
        )
        raise_if(
            covariates is None and self.lags_covariates is not None,
            "`covariates` is None in `fit()` method call, but `lags_covariates` is not None in constructor. ",
        )

        lagged_dataset = LaggedDataset(
            self.training_series,
            self.covariate_series,
            self.lags,
            self.lags_covariates,
            max_samples_per_ts,
        )
        # Prepare data
        training_x, training_y = lagged_dataset.get_data()

        self.model.fit(training_x, training_y, **kwargs)

        # TODO think about prediction data
        # if self.max_lag == 0:
        #     self.prediction_data = pd.DataFrame(
        #         columns=series.stack(other=covariates).columns()
        #     )
        # elif covariates is not None:
        #     self.prediction_data = series.stack(other=covariates)[-self.max_lag :]
        # else:
        #     self.prediction_data = series[-self.max_lag :]
        # print(self.prediction_data)
        # self._fit_called = True

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Forecasts values for `n` time steps after the end of the series.

        Parameters
        ----------
        n : int
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        """
        super().predict(n, series, covariates)
        # TODO: check if we have series or not. If so, use those, otherwise recycle the training data
        if series is None:
            pass
            # inference_data = SimpleInferenceDataset(series, covariates, n, input_chunk_length=)

        super().predict(n, series, covariates)
        pass

        # if self.max_lag != 0:
        #     prediction_data = self.prediction_data.copy()
        #     dummy_row = np.zeros(shape=(1, prediction_data.width))
        #     prediction_data = prediction_data.append_values(dummy_row)
        #
        # if covariates is not None and self.lags_covariates != [0]:
        #     required_start = (
        #         self.training_series.end_time() + self.training_series.freq()
        #     )
        #     raise_if_not(
        #         exog.start_time() == required_start,
        #         "`exog` first date must be equal to self.training_series.end_time()+1*freq. "
        #         + "Given: {}. Needed: {}.".format(exog.start_time(), required_start),
        #     )
        # if isinstance(exog, TimeSeries):
        #     exog = exog.pd_dataframe(copy=False)
        #
        # forecasts = [0] * n  # initializing the forecasts vector
        #
        # for i in range(n):
        #     # Prepare prediction data if for prediction at time `t` exog at time `t` is used
        #     if self.lags_covariates is not None and 0 in self.lags_covariates:
        #         append_row = [[0, *exog.iloc[i, :].values]]
        #         if self.max_lag == 0:
        #             prediction_data = pd.DataFrame(
        #                 append_row,
        #                 columns=self.prediction_data.columns,
        #                 index=[exog.index[i]],
        #             )
        #             prediction_data = TimeSeries(
        #                 prediction_data, freq=self.training_series.freq()
        #             )
        #         else:
        #             prediction_data = prediction_data[:-1]  # Remove last dummy row
        #             prediction_data = prediction_data.append_values(append_row)
        #
        #     # Make prediction
        #     target_data = prediction_data[[self.target_column]]
        #     exog_data = (
        #         prediction_data[self.exog_columns]
        #         if self.exog_columns is not None
        #         else None
        #     )
        #     forecasting_data, _ = self._create_training_data(
        #         series=target_data, exog=exog_data
        #     )
        #     if "series" in signature(self.model.fit).parameters:
        #         forecasting_data = TimeSeries(
        #             forecasting_data, freq=self.training_series.freq()
        #         )
        #         forecast = self.model.predict(
        #             n=len(forecasting_data), exog=forecasting_data, **kwargs
        #         )
        #         forecast = forecast.pd_dataframe().values
        #     else:
        #         forecast = self.model.predict(forecasting_data, **kwargs)
        #     forecast = forecast[0] if isinstance(forecast[0], np.ndarray) else forecast
        #
        #     # Prepare prediction data
        #     if self.max_lag > 0:
        #         prediction_data = prediction_data[:-1]  # Remove last dummy row
        #         append_row = (
        #             [[forecast[0], *exog.iloc[i, :].values]]
        #             if self.exog_columns is not None
        #             else [forecast]
        #         )
        #         prediction_data = prediction_data.append_values(append_row)
        #         prediction_data = prediction_data.append_values(dummy_row)[1:]
        #
        #     # Append forecast
        #     forecasts[i] = forecast[0]
        # return self._build_forecast_series(forecasts)

    def __str__(self):
        return self.model.__str__()
