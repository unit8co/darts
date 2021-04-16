"""
Regression Model Base Class
---------------------------

A regression model predicts values for a time series :math:`Y_t` as a function
of :math:`N` "features" time series :math:`X^i_t`:

.. math:: Y_t = f(X^1_t, ..., X^N_t),

where :math:`t` denotes the time step. Here, the function :math:`f()` is not necessarily linear.
"""

import numpy as np
import pandas as pd


from .. import metrics
from ..timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from .forecasting_model import ExtendedForecastingModel
from typing import List, Iterable, Union, Any, Callable, Optional
from ..logging import raise_if, raise_if_not, get_logger, raise_log
from ..utils import (
    _build_tqdm_iterator,
    _with_sanity_checks,
    _historical_forecasts_general_checks
)

logger = get_logger(__name__)


# TODO: Extend this to a "DynamicRegressiveModel" class, which acts on List[List[TimeSeries]].
# TODO: The first List[] would contain time-sliding lists of time series, letting the model
# TODO: be able to learn how to change weights over time. When len() of outer List[] is 0 it's a particular case
class RegressionModel(ExtendedForecastingModel):
    def __init__(self,
                 lags: Union[int, list] = None,
                 lags_exog: Union[int, list, bool] = None,
                 model=LinearRegression(n_jobs=-1, fit_intercept=False)):
        """ Regression Model

        Can be used to fit any scikit-learn-like regressor class to predict the target
        time series with lagged values.

        Parameters
        ----------
        lags : Union[int, list]
            Number of lagged target values used to predict the next time step. If an integer is given
            the last `lags` lags are used (inclusive). Otherwise a list of integers with lags.
        lags_exog : Union[int, list, bool]
            Number of lagged exogenous values used to predict the next time step. If an integer is given
            the last `lags_exog` lags are used (inclusive). Otherwise a list of integers with lags. If False,
            the value at time `t` is used which might lead to leakage. If True the same lags as for the
            target variable are used.
        model
            A regression model that implements `fit()` and `predict()` methods.
            Default: `sklearn.linear_model.LinearRegression(n_jobs=-1, fit_intercept=False)`
        """
        raise_if((lags is None) and (lags_exog is None),
            "At least one of 'lags' or 'lags_exog' must be not None."
        )
        raise_if_not(isinstance(lags, (int, list)) or lags is None,
            "`lags` must be of type int or list."
        )
        raise_if_not(isinstance(lags_exog, (int, list, bool)) or lags_exog is None,
            "`lags_exog` must be of type int, list or bool."
        )
        raise_if(lags is None and lags_exog is True,
            "`lags_exog` must not be True if `lags` is None."
        )

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
        if self.lags_exog is True:
            self.lags_exog = self.lags[:]
        elif self.lags_exog is False:
            self.lags_exog = [0]
        elif isinstance(self.lags_exog, int):
            raise_if_not(self.lags_exog > 0, "`lags_exog` must be strictly positive. Given: {}.".format(self.lags_exog))
            self.lags_exog = list(range(1, self.lags_exog+1))
        elif isinstance(self.lags_exog, list):
            for lag in self.lags_exog:
                raise_if(not isinstance(lag, int) or (lag < 0), (
                    "Every element of `lags_exog` must be a positive integer. Given: {}."
                    .format(self.lags_exog))
                )

        self.model = model
        if self.lags is not None and self.lags_exog is None:
            self.max_lag = int(np.max(self.lags))
        elif self.lags is None and self.lags_exog is not None:
            self.max_lag = int(np.max(self.lags_exog))
        else:
            self.max_lag = int(np.max([np.max(self.lags), np.max(self.lags_exog)]))
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
        self.target_column = series.columns()[0]
        self.exog_columns = exog.columns().values.tolist() if exog is not None else None
        if self.exog_columns is not None:
            if str(self.target_column) in self.exog_columns:
                series = TimeSeries(series.pd_dataframe().rename({self.target_column: "Target"}))
                self.target_column = series.columns()[0]
        self.training_data = self._create_training_data(series, exog)
        self.train_x =  self.training_data.pd_dataframe().drop(self.target_column, axis=1)
        self.train_y =  self.training_data.pd_dataframe()[self.target_column]
        self.nr_exog = exog.width*len(self.lags_exog) if exog is not None else 0

        self.model.fit(
            X=self.train_x,
            y=self.train_y,
            **kwargs
        )
        if self.max_lag == 0:
            self.prediction_data = pd.DataFrame(columns=series.stack(other=exog).columns())
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
        pd.dataframe
            Data frame with lagged values.
        """
        raise_if(series.width > 1,
            "Series must not be multivariate. Pass exogenous variables to 'exog' parameter.",
            logger
        )
        training_data = series.pd_dataframe()

        if self.lags is not None:
            lagged_data = self._create_lagged_data(series=series, lags=self.lags)
            training_data = pd.concat([training_data, lagged_data], axis=1)

        if self.lags_exog is not None:
            for col in exog.columns():
                lagged_data = self._create_lagged_data(
                    series=exog[col], lags=self.lags_exog
                )
                training_data = pd.concat([training_data, lagged_data], axis=1)

        return TimeSeries(training_data.dropna(), freq=series.freq())

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
        keep_current : bool
            If False, the current value (not-lagged) is dropped.

        Returns
        -------
        TYPE
            Returns a data frame that contains lagged values.
        """
        lagged_series = series.pd_dataframe(copy=True)
        target_name = lagged_series.columns[0]
        for lag in lags:
            new_column_name = target_name + "_lag{}".format(lag)
            lagged_series[new_column_name] = lagged_series[target_name].shift(lag)
        lagged_series.dropna(inplace=True)
        lagged_series.drop(target_name, axis=1, inplace=True)
        return lagged_series

    def predict(self, n: int, exog: Optional[TimeSeries] = None, **kwargs):
        super().predict(n, exog)

        if self.max_lag != 0:
            prediction_data = self.prediction_data.copy()
            dummy_row = np.zeros(shape=(1, prediction_data.width))
            prediction_data = prediction_data.append_values(dummy_row)

        if exog is not None:
            raise_if_not(exog.start_time() == self.training_series.end_time()+self.training_series.freq(),
                "`exog` first date must be equal to self.training_series.end_time()+1*freq. " +
                "Given: {}. Needed: {}.".format(exog.start_time(), self.training_series.end_time()+self.training_series.freq())
            )
        if isinstance(exog, TimeSeries):
            exog = exog.pd_dataframe()

        forecasts = list(range(n))
        for i in range(n):
            if self.lags_exog is not None and 0 in self.lags_exog:
                if self.max_lag == 0:
                    append_row = [[0, *exog.iloc[i, :].values.tolist()]]
                    prediction_data = pd.DataFrame(
                        append_row, columns=self.prediction_data.columns, index=[exog.index[i]]
                    )
                    prediction_data = TimeSeries(prediction_data, freq=self.training_series.freq())
                else:
                    prediction_data = prediction_data[:-1]
                    append_row = [[0, *exog.iloc[i, :].values.tolist()]]
                    prediction_data = prediction_data.append_values(append_row)
            target_data = prediction_data[[self.target_column]]
            if self.exog_columns is not None:
                exog_data = prediction_data[self.exog_columns]
            else:
                exog_data = None
            forecasting_data = self._create_training_data(target_data, exog=exog_data)
            forecasting_data = forecasting_data.pd_dataframe().iloc[:, 1:]
            forecast = self.model.predict(
                X=forecasting_data,
                **kwargs
            )
            if self.max_lag > 0:
                prediction_data = prediction_data[:-1]
                if self.exog_columns is not None:
                    append_row = [[forecast[0], *exog.iloc[i, :].values.tolist()]]
                    prediction_data = prediction_data.append_values(append_row)
                else:
                    prediction_data = prediction_data.append_values(forecast)
                prediction_data = prediction_data.append_values(dummy_row)[1:]
            forecasts[i] = forecast[0]
        return self._build_forecast_series(forecasts)

    def __str__(self):
        return model.__str__()