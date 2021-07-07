"""
Regression Model
----------------

A `RegressionModel` forecasts future values of a target series based on lagged values of the target values
and possibly lags of an exoneous series. They can wrap around any regression model having a `fit()`
and `predict()` functions (e.g. scikit-learn regression models), and are using
`sklearn.linear_model.LinearRegression` by default.

Behind the scenes this model is tabularizing the time series data to make it work with regression models.
"""

import numpy as np
import pandas as pd

from inspect import signature
from typing import Union, Optional
from ..timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from .forecasting_model import ExtendedForecastingModel
from ..logging import raise_if, raise_if_not, get_logger, raise_log

logger = get_logger(__name__)


class RegressionModel(ExtendedForecastingModel):
    def __init__(self,
                 lags: Union[int, list] = None,
                 lags_exog: Union[int, list] = None,
                 model=None):
        """ Regression Model

        Can be used to fit any scikit-learn-like regressor class to predict the target
        time series from lagged values.

        Parameters
        ----------
        lags : Union[int, list]
            Number of lagged target values used to predict the next time step. If an integer is given
            the last `lags` lags are used (inclusive). Otherwise a list of integers with lags is required.
            The integers must be strictly positive (>0).
        lags_exog : Union[int, list, bool]
            Number of lagged exogenous values used to predict the next time step. If an integer is given
            the last `lags_exog` lags are used (inclusive). Otherwise a list of integers with lags is required.
            The integers must be positive (>=0).
        model
            Typically a scikit-learn model or a Darts `ExtendedForecastingModel` (models accepting `exog`) with
            `fit()` and `predict()` methods.
            Default: `sklearn.linear_model.LinearRegression(n_jobs=-1, fit_intercept=False)`
        """
        raise_if((lags is None) and (lags_exog is None),
            "At least one of `lags` or `lags_exog` must be not None."
        )
        raise_if_not(isinstance(lags, (int, list)) or lags is None,
            "`lags` must be of type int or list. Given: {}.".format(type(lags))
        )
        raise_if_not(isinstance(lags_exog, (int, list)) or lags_exog is None,
            "`lags_exog` must be of type int or list. Given: {}.".format(type(lags_exog))
        )
        raise_if(isinstance(lags, bool) or isinstance(lags_exog, bool),
            "`lags` and `lags_exog` must be of type int or list, not bool."
        )
        if model is None:
            model = LinearRegression()

        if (not callable(getattr(model, "fit", None))):
            raise_log(Exception('Provided model object must have a fit() method', logger))
        if (not callable(getattr(model, "predict", None))):
            raise_log(Exception('Provided model object must have a predict() method', logger))

        self.lags = lags
        if isinstance(self.lags, int):
            raise_if_not(self.lags > 0, "`lags` must be strictly positive. Given: {}.".format(self.lags))
            self.lags = list(range(1, self.lags+1))
        elif isinstance(self.lags, list):
            for lag in self.lags:
                raise_if(not isinstance(lag, int) or (lag <= 0), (
                    "Every element of `lags` must be a strictly positive integer. Given: {}.".format(self.lags))
                )

        self.lags_exog = lags_exog
        if self.lags_exog == 0:
            self.lags_exog = [0]
        elif isinstance(self.lags_exog, int):
            raise_if_not(self.lags_exog > 0, "`lags_exog` must be positive. Given: {}.".format(self.lags_exog))
            self.lags_exog = list(range(1, self.lags_exog+1))
        elif isinstance(self.lags_exog, list):
            for lag in self.lags_exog:
                raise_if(not isinstance(lag, int) or (lag < 0), (
                    "Every element of `lags_exog` must be a positive integer. Given: {}."
                    .format(self.lags_exog))
                )

        self.model = model
        if self.lags is not None and self.lags_exog is None:
            self.max_lag = max(self.lags)
        elif self.lags is None and self.lags_exog is not None:
            self.max_lag = max(self.lags_exog)
        else:
            self.max_lag = max([max(self.lags), max(self.lags_exog)])
        self._fit_called = False

    def fit(self, series: TimeSeries, exog: Optional[TimeSeries] = None, **kwargs) -> None:
        """ Fits/trains the model using the provided list of features time series and the target time series.

        Parameters
        ----------
        series : TimeSeries
            TimeSeries object containing the target values.
        exog : TimeSeries, optional
            TimeSeries object containing the exogenous values.
        """
        super().fit(series, exog)
        raise_if(exog is not None and self.lags_exog is None,
            "`exog` not None in `fit()` method call, but `lags_exog` is None in constructor. "
        )
        raise_if(exog is None and self.lags_exog is not None,
            "`exog` is None in `fit()` method call, but `lags_exog` is not None in constructor. "
        )
        self.exog_columns = exog.columns.values.tolist() if exog is not None else None
        self.nr_exog = exog.width*len(self.lags_exog) if exog is not None else 0

        # Rename column if necessary
        target_column = str(series.columns[0])
        if self.exog_columns is not None and target_column in self.exog_columns:
            series = TimeSeries.from_dataframe(series.pd_dataframe().rename({target_column: "Target"}))
        self.target_column = series.columns[0]

        # Prepare data
        training_x = self._create_training_data(series=series, exog=exog)
        training_y = series[training_x.time_index]

        # Fit model
        if "series" in signature(self.model.fit).parameters:
            self.model.fit(
                series=training_y,
                exog=training_x,
                **kwargs
            )
        else:
            self.model.fit(
                training_x.pd_dataframe(),
                training_y.pd_dataframe().values.ravel(),
                **kwargs
            )

        if self.max_lag == 0:
            self.prediction_data = pd.DataFrame(columns=series.stack(other=exog).columns)
        elif exog is not None:
            self.prediction_data = series.stack(other=exog)[-self.max_lag:]
        else:
            self.prediction_data = series[-self.max_lag:]

        self._fit_called = True

    def _create_training_data(self, series: TimeSeries, exog: TimeSeries = None):
        """ Create dataframe of exogenous and endogenous variables containing lagged values.

        Parameters
        ----------
        series : TimeSeries
            Target series.
        exog : TimeSeries, optional
            Exogenous variables.

        Returns
        -------
        TimeSeries
            TimeSeries with lagged values of target and exogenous variables.
        """
        raise_if(series.width > 1,
            "Series must not be multivariate. Pass exogenous variables to 'exog' parameter.",
            logger
        )
        training_data_list = []

        if self.lags is not None:
            lagged_data = self._create_lagged_data(series=series, lags=self.lags)
            training_data_list.append(lagged_data)

        if self.lags_exog is not None:
            for i, col in enumerate(exog.columns):
                lagged_data = self._create_lagged_data(
                    series=exog[col], lags=self.lags_exog
                )
                training_data_list.append(lagged_data)

        training_data = pd.concat(training_data_list, axis=1)[self.max_lag:]
        return TimeSeries.from_dataframe(training_data, fill_missing_dates=False, freq=series.freq)

    def _create_lagged_data(self, series: TimeSeries, lags: list):
        """ Creates a data frame where every lag is a new column.

        After creating this input it can be used in many regression models to make predictions
        based on previous available values for both the exogenous as well as endogenous variables.

        Parameters
        ----------
        series : TimeSeries
            Time series to be lagged.
        lags : list
            List if indexes.

        Returns
        -------
        pd.DataFrame
            Returns a data frame that contains lagged values.
        """
        lagged_series = series.pd_dataframe(copy=True)
        target_name = lagged_series.columns[0]
        for lag in lags:
            new_column_name = target_name + "_lag{}".format(lag)
            lagged_series[new_column_name] = lagged_series[target_name].shift(lag)
        lagged_series.drop(target_name, axis=1, inplace=True)
        return lagged_series

    def predict(self,
                n: int,
                exog: Optional[TimeSeries] = None,
                num_samples: int = 1,
                **kwargs):
        """ Forecasts values for `n` time steps after the end of the series.

        Parameters
        ----------
        n : int
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        exog : Optional[TimeSeries], optional
            The time series of exogenous variables which can be fed as input to the model. It must correspond to the
            exogenous time series that has been used with the `fit()` method for training. It also must be of length `n`
            and start one time step after the end of the training series.
        num_samples
            Currently this parameter is ignored for regression models.
        **kwargs
            Additional keyword arguments passed to the `predict` method of the model.

        Returns
        -------
        TimeSeries
            A time series containing the `n` next points after then end of the training series.
        """
        super().predict(n, exog, num_samples)

        if self.max_lag != 0:
            prediction_data = self.prediction_data.copy()
            dummy_row = np.zeros(shape=(1, prediction_data.width))
            prediction_data = prediction_data.append_values(dummy_row)

        if exog is not None and self.lags_exog != [0]:
            required_start = self.training_series.end_time()+self.training_series.freq
            raise_if_not(exog.start_time() == required_start,
                "`exog` first date must be equal to self.training_series.end_time()+1*freq. " +
                "Given: {}. Needed: {}.".format(exog.start_time(), required_start)
            )
        if isinstance(exog, TimeSeries):
            exog = exog.pd_dataframe(copy=False)

        forecasts = list(range(n))
        for i in range(n):

            # Prepare prediction data if for prediction at time `t` exog at time `t` is used
            if self.lags_exog is not None and 0 in self.lags_exog:
                append_row = [[0, *exog.iloc[i, :].values]]
                if self.max_lag == 0:
                    prediction_data = pd.DataFrame(
                        append_row, columns=self.prediction_data.columns, index=[exog.index[i]]
                    )
                    prediction_data = TimeSeries.from_dataframe(prediction_data,
                                                                fill_missing_dates=True,
                                                                freq=self.training_series.freq)
                else:
                    prediction_data = prediction_data[:-1]  # Remove last dummy row
                    prediction_data = prediction_data.append_values(append_row)

            # Make prediction
            target_data = prediction_data[[self.target_column]]
            exog_data = prediction_data[self.exog_columns] if self.exog_columns is not None else None
            forecasting_data = self._create_training_data(series=target_data, exog=exog_data).pd_dataframe()
            if "series" in signature(self.model.fit).parameters:
                forecasting_data = TimeSeries.from_dataframe(forecasting_data, freq=self.training_series.freq)
                forecast = self.model.predict(n=len(forecasting_data),
                                              exog=forecasting_data,
                                              **kwargs)
                forecast = forecast.pd_dataframe().values
            else:
                forecast = self.model.predict(forecasting_data, **kwargs)
            forecast = forecast[0] if isinstance(forecast[0], np.ndarray) else forecast

            # Prepare prediction data
            if self.max_lag > 0:
                prediction_data = prediction_data[:-1] # Remove last dummy row
                append_row = [[forecast[0], *exog.iloc[i, :].values]] if self.exog_columns is not None else [forecast]
                prediction_data = prediction_data.append_values(append_row)
                prediction_data = prediction_data.append_values(dummy_row)[1:]

            # Append forecast
            forecasts[i] = forecast[0]
        return self._build_forecast_series(np.array(forecasts))

    def __str__(self):
        return self.model.__str__()