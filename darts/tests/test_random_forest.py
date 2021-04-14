"""
LightGBM
--------

Models for LightGBM (Light Gradient Boosting Machine) [1]_.
The implementations is wrapped around `lightgbm <https://lightgbm.readthedocs.io/en/latest/index.html>`_.

References
----------
.. [1] https://en.wikipedia.org/wiki/LightGBM
"""
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from typing import Optional, Union
from ..timeseries import TimeSeries
from ..logging import get_logger, raise_if, raise_if_not
from .forecasting_model import ExtendedForecastingModel

logger = get_logger(__name__)


class LightGBM(ExtendedForecastingModel):
    def __init__(self,
                 lags: Union[int, list],
                 lags_exog: Union[int, list, bool] = True,
                 num_leaves: Optional[int] = 128,
                 learning_rate: Optional[float] = 0.1,
                 n_estimators: Optional[int] = 100,
                 **kwargs):
        """ LightGBM

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
        num_leaves : int
            Maximum number of leaves in base learner.
        learning_rate : float
            Boosting learning rate.
        n_estimators : int
            Number of boosted trees.
        kwargs
            Additonal arguments for lightgbm.LGBMRegressor(...).
        """
        super().__init__()
        raise_if(isinstance(lags, int) and (lags < 0), "Lags must be positive integer.")
        raise_if(isinstance(lags, float), "Lags must be integer, not float.")

        self.lags = lags
        self.lags_exog = lags_exog
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.lgb_kwargs = kwargs
        self.model = LGBMRegressor(
            num_leaves=num_leaves, learning_rate=learning_rate,
            n_estimators=n_estimators, **kwargs
        )
        if isinstance(self.lags, int):
            self.lags = list(range(1, self.lags+1))

        if self.lags_exog is True:
            self.lags_exog = self.lags[:]
        elif self.lags_exog is False:
            self.lags_exog = [0]
        elif isinstance(self.lags_exog, int):
            self.lags_exog = list(range(1, self.lags_exog+1))
        self.max_lag = int(np.max([np.max(self.lags), np.max(self.lags_exog)]))

    def __str__(self):
        return 'LightGBM(lags={}, num_leaves={}, learning_rate={}, n_estimators={})'.format(
            self.lags, self.num_leaves, self.learning_rate, self.n_estimators
        )

    def fit(self, series: TimeSeries, exog: Optional[TimeSeries] = None, **kwargs):
        super().fit(series, exog)
        self.target_column = series.pd_dataframe().columns[0]
        self.exog_columns = exog.pd_dataframe().columns if exog is not None else None
        self.training_data = self._create_valid_data(series, exog)
        self.nr_exog = self.training_data.shape[1] - len(self.lags) - 1

        self.model.fit(
            X=self.training_data.drop(self.target_column, axis=1),
            y=self.training_data[self.target_column],
            **kwargs
        )
        if exog is not None:
            self.prediction_data = series.stack(other=exog)[-self.max_lag:]
        else:
            self.prediction_data = series[-self.max_lag:]

    def _create_valid_data(self, series: TimeSeries, exog: TimeSeries = None):
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
        training_data = self._create_lagged_data(series=series, lags=self.lags, keep_current=True)
        if exog is not None:
            for col in exog.pd_dataframe().columns:
                col_lagged_data = self._create_lagged_data(
                    series=exog[col], lags=self.lags_exog, keep_current=False
                )
                training_data = pd.concat([training_data, col_lagged_data], axis=1)
        return training_data.dropna()

    def _create_lagged_data(self, series: TimeSeries, lags: list, keep_current: bool):
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
        if not keep_current:
            lagged_series.drop(target_name, axis=1, inplace=True)
        return lagged_series

    def predict(self, n: int, exog: Optional[TimeSeries] = None, **kwargs):
        super().predict(n, exog)
        if isinstance(exog, TimeSeries):
            exog = exog.pd_dataframe()

        prediction_data = self.prediction_data.copy()
        dummy_row = np.zeros(shape=(1, prediction_data.width))
        prediction_data = prediction_data.append_values(dummy_row)
        forecasts = list(range(n))

        for i in range(n):
            target_data = TimeSeries(prediction_data.pd_dataframe()[[self.target_column]])
            if self.exog_columns is not None:
                exog_data = TimeSeries(prediction_data.pd_dataframe()[self.exog_columns])
            else:
                exog_data = None
            forecasting_data = self._create_valid_data(target_data, exog=exog_data).iloc[:, 1:]
            forecast = self.model.predict(
                X=forecasting_data,
                **kwargs
            )
            prediction_data = prediction_data[1:-1]
            if self.exog_columns is not None:
                append_row = [[forecast[0], *exog.iloc[i, :].values.tolist()]]
                prediction_data = prediction_data.append_values(append_row)
            else:
                prediction_data = prediction_data.append_values(forecast)
            prediction_data = prediction_data.append_values(dummy_row)
            forecasts[i] = forecast[0]
        return self._build_forecast_series(forecasts)

    @property
    def min_train_series_length(self) -> int:
        return 30
