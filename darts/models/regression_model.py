"""
Regression Model
----------------

A `RegressionModel` forecasts future values of a target series based on lagged values of the target values
and possibly lags of an exogenous series. They can wrap around any regression model having a `fit()`
and `predict()` functions (e.g. scikit-learn regression models), and are using
`sklearn.linear_model.LinearRegression` by default.

Behind the scenes this model is tabularizing the time series data to make it work with regression models.
"""

import numpy as np
import pandas as pd

from inspect import signature
from typing import Union, Optional, Sequence
from ..timeseries import TimeSeries
from darts.utils.data.sequential_dataset import SequentialDataset
from sklearn.linear_model import LinearRegression
from .forecasting_model import GlobalForecastingModel
from ..logging import raise_if, raise_if_not, get_logger, raise_log

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
            Typically a scikit-learn model or a Darts `ExtendedForecastingModel` (models accepting `exog`) with
            `fit()` and `predict()` methods.
            Default: `sklearn.linear_model.LinearRegression(n_jobs=-1, fit_intercept=False)`
        """

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

        self.lags = lags
        if isinstance(self.lags, int):
            raise_if_not(
                self.lags > 0,
                "`lags` must be strictly positive. Given: {}.".format(self.lags),
            )
            self.lags = list(range(1, self.lags + 1))
        elif isinstance(self.lags, list):
            for lag in self.lags:
                raise_if(
                    not isinstance(lag, int) or (lag <= 0),
                    (
                        "Every element of `lags` must be a strictly positive integer. Given: {}.".format(
                            self.lags
                        )
                    ),
                )

        self.lags_covariates = lags_covariates
        if self.lags_covariates == 0:
            self.lags_covariates = [0]
        elif isinstance(self.lags_covariates, int):
            raise_if_not(
                self.lags_covariates > 0,
                "`lags_covariates` must be positive. Given: {}.".format(self.lags_covariates),
            )
            self.lags_covariates = list(range(1, self.lags_covariates + 1))
        elif isinstance(self.lags_covariates, list):
            for lag in self.lags_covariates:
                raise_if(
                    not isinstance(lag, int) or (lag < 0),
                    (
                        "Every element of `lags_covariates` must be a positive integer. Given: {}.".format(
                            self.lags_covariates
                        )
                    ),
                )

        self.model = model
        if self.lags is not None and self.lags_covariates is None:
            self.max_lag = max(self.lags)
        elif self.lags is None and self.lags_covariates is not None:
            self.max_lag = max(self.lags_covariates)
        else:
            self.max_lag = max([max(self.lags), max(self.lags_covariates)])

        self._fit_called = False

    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            **kwargs
            ) -> None:
        """Fits/trains the model using the provided list of features time series and the target time series.

        Parameters
        ----------
        series : Union[TimeSeries, Sequence[TimeSeries]]
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        covariate : Union[TimeSeries, Sequence[TimeSeries]], optional
            TimeSeries or Sequence[TimeSeries] object containing the exogenous values.
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

        # Prepare data
        training_x, training_y = self._create_training_data(series=series, covariates=covariates)

        # Fit model
        if "series" in signature(self.model.fit).parameters:
            # darts ExtendedForecastingModel
            self.model.fit(series=training_y, exog=training_x, **kwargs)
        else:
            # sklearn like model
            self.model.fit(
                training_x,
                training_y.values.ravel(),
                **kwargs
            )

        if self.max_lag == 0:
            self.prediction_data = pd.DataFrame(
                columns=series.stack(other=covariates).columns()
            )
        elif covariates is not None:
            self.prediction_data = series.stack(other=covariates)[-self.max_lag:]
        else:
            self.prediction_data = series[-self.max_lag:]
        print(self.prediction_data)
        self._fit_called = True

    def _create_training_data(self,
                              series: Union[TimeSeries, Sequence[TimeSeries]],
                              covariates: Union[TimeSeries, Sequence[TimeSeries]] = None):
        """Create dataframe of covariates and endogenous variables containing lagged values.

        Parameters
        ----------
        series : TimeSeries
            Target series.
        covariates : TimeSeries, optional
            Covariates variables.

        Returns
        -------
        TimeSeries
            TimeSeries with lagged values of target and exogenous variables.
        """

        sequential_dataset = SequentialDataset(
            target_series=series,
            covariates=covariates,
            input_chunk_length=self.max_lag,
            output_chunk_length=1,
            max_samples_per_ts=None,
        )

        X = []
        y = []

        for input_target, output_target, input_covariates in sequential_dataset:
            if self.lags is not None:
                lags_indices = np.array(self.lags) * (-1)
                input_target = input_target[lags_indices]
            if self.lags_covariates is not None:
                exog_lags_indices = np.array(self.lags_covariates) * (-1)
                input_covariates = input_covariates[exog_lags_indices]

            if input_covariates is not None:
                X.append(pd.DataFrame(np.concatenate((input_target, input_covariates), axis=None)))
            else:
                X.append(pd.DataFrame(input_target))
            y.append(pd.DataFrame(output_target))

        X = pd.concat(X, axis=1)
        y = pd.concat(y, axis=1)

        return X.T, y.T

    def predict(self,
                n: int,
                series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None
                ) -> Union[TimeSeries, Sequence[TimeSeries]]:     
        """Forecasts values for `n` time steps after the end of the series.

        Parameters
        ----------
        n : int
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        exog : Optional[TimeSeries], optional
            The time series of exogenous variables which can be fed as input to the model. It must correspond to the
            exogenous time series that has been used with the `fit()` method for training. It also must be of length `n`
            and start one time step after the end of the training series.
        **kwargs
            Additional keyword arguments passed to the `predict` method of the model.

        Returns
        -------
        TimeSeries
            A time series containing the `n` next points after then end of the training series.
        """
        """
        super().predict(n, exog)

        if self.max_lag != 0:
            prediction_data = self.prediction_data.copy()
            dummy_row = np.zeros(shape=(1, prediction_data.width))
            prediction_data = prediction_data.append_values(dummy_row)

        if exog is not None and self.lags_covariates != [0]:
            required_start = (
                self.training_series.end_time() + self.training_series.freq()
            )
            raise_if_not(
                exog.start_time() == required_start,
                "`exog` first date must be equal to self.training_series.end_time()+1*freq. "
                + "Given: {}. Needed: {}.".format(exog.start_time(), required_start),
            )
        if isinstance(exog, TimeSeries):
            exog = exog.pd_dataframe(copy=False)

        forecasts = [0] * n  # initializing the forecasts vector

        for i in range(n):
            # Prepare prediction data if for prediction at time `t` exog at time `t` is used
            if self.lags_covariates is not None and 0 in self.lags_covariates:
                append_row = [[0, *exog.iloc[i, :].values]]
                if self.max_lag == 0:
                    prediction_data = pd.DataFrame(
                        append_row,
                        columns=self.prediction_data.columns,
                        index=[exog.index[i]],
                    )
                    prediction_data = TimeSeries(
                        prediction_data, freq=self.training_series.freq()
                    )
                else:
                    prediction_data = prediction_data[:-1]  # Remove last dummy row
                    prediction_data = prediction_data.append_values(append_row)

            # Make prediction
            target_data = prediction_data[[self.target_column]]
            exog_data = (
                prediction_data[self.exog_columns]
                if self.exog_columns is not None
                else None
            )
            forecasting_data, _ = self._create_training_data(
                series=target_data, exog=exog_data
            )
            if "series" in signature(self.model.fit).parameters:
                forecasting_data = TimeSeries(
                    forecasting_data, freq=self.training_series.freq()
                )
                forecast = self.model.predict(
                    n=len(forecasting_data), exog=forecasting_data, **kwargs
                )
                forecast = forecast.pd_dataframe().values
            else:
                forecast = self.model.predict(forecasting_data, **kwargs)
            forecast = forecast[0] if isinstance(forecast[0], np.ndarray) else forecast

            # Prepare prediction data
            if self.max_lag > 0:
                prediction_data = prediction_data[:-1]  # Remove last dummy row
                append_row = (
                    [[forecast[0], *exog.iloc[i, :].values]]
                    if self.exog_columns is not None
                    else [forecast]
                )
                prediction_data = prediction_data.append_values(append_row)
                prediction_data = prediction_data.append_values(dummy_row)[1:]

            # Append forecast
            forecasts[i] = forecast[0]
        return self._build_forecast_series(forecasts)
        """
        return

    def __str__(self):
        return self.model.__str__()
