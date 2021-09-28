"""
Facebook Prophet
----------------
"""

from typing import Optional
import logging
import numpy as np

from darts.timeseries import TimeSeries
from darts.models.forecasting.forecasting_model import DualCovariatesForecastingModel
import pandas as pd
from darts.logging import get_logger, execute_and_suppress_output
import prophet


logger = get_logger(__name__)
logger.level = logging.WARNING  # set to warning to suppress prophet logs


class Prophet(DualCovariatesForecastingModel):
    def __init__(self,
                 seasonal_periods: Optional[int] = None,
                 country_holidays: Optional[str] = None,
                 **prophet_kwargs):
        """ Facebook Prophet

        This class provides a basic wrapper around `Facebook Prophet <https://github.com/facebook/prophet>`_.
        It also supports country holidays.

        Parameters
        ----------
        seasonal_periods
            Optionally, some seasonal_periods, specifying a known seasonality, which will be added to prophet.
        country_holidays
            An optional country code, for which holidays can be taken into account by Prophet.

            See: https://github.com/dr-prodigy/python-holidays

            In addition to those countries, Prophet includes holidays for these
            countries: Brazil (BR), Indonesia (ID), India (IN), Malaysia (MY), Vietnam (VN),
            Thailand (TH), Philippines (PH), Turkey (TU), Pakistan (PK), Bangladesh (BD),
            Egypt (EG), China (CN), and Russia (RU).
        prophet_kwargs
            Some optional keyword arguments for Prophet.
            For information about the parameters see:
            `The Prophet source code <https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py>`_.

        """

        super().__init__()

        self.add_seasonalities = {}
        self.country_holidays = country_holidays
        self.seasonal_periods = seasonal_periods
        self.prophet_kwargs = prophet_kwargs
        self.model = None

    def __str__(self):
        return 'Prophet'

    def fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super().fit(series, future_covariates)
        series = self.training_series

        fit_df = pd.DataFrame(data={
            'ds': series.time_index,
            'y': series.univariate_values()
        })

        self.model = prophet.Prophet(**self.prophet_kwargs)

        # add user defined seasonalities (pre-fit)
        for seasonality in self.add_seasonalities:
            args, kwargs = self.add_seasonalities[seasonality]
            self.model.add_seasonality(*args, **kwargs)

        # add user defined seasonalities (at model creation)
        if self.seasonal_periods is not None:
            interval_length = self.freq_to_days(series.freq_str)
            self.model.add_seasonality(name=f'{self.seasonal_periods}_{series.freq_str}',
                                       period=self.seasonal_periods * interval_length,
                                       fourier_order=10)

        if future_covariates is not None:
            fit_df = fit_df.merge(future_covariates.pd_dataframe(), left_on='ds', right_index=True, how='left')
            for covariate in future_covariates.columns:
                self.model.add_regressor(covariate)

        # Input built-in country holidays
        if self.country_holidays is not None:
            self.model.add_country_holidays(self.country_holidays)

        execute_and_suppress_output(self.model.fit, logger, logging.WARNING, fit_df)

    def predict(self,
                n: int,
                future_covariates: Optional[TimeSeries] = None,
                num_samples: int = 1) -> TimeSeries:
        super().predict(n, future_covariates, num_samples)

        predict_df = self.generate_predict_df(n=n, future_covariates=future_covariates)

        if num_samples == 1:
            forecast = self.model.predict(predict_df)['yhat'].values
        else:
            forecast = np.expand_dims(self.stochastic_samples(predict_df, n_samples=num_samples), axis=1)

        return self._build_forecast_series(forecast)

    def predict_raw(self,
                    n: int,
                    future_covariates: Optional[TimeSeries] = None) -> pd.DataFrame:
        """Returns the output of the base Prophet model in form of a pandas DataFrame. Note however, that the outpu of
        method is not supported for further processing with the Darts API.

        Methods of the base Prophet model can be accessed with self.model.method() (i.e. self.model.plot_components())
        """
        super().predict(n, future_covariates, num_samples=1)

        predict_df = self.generate_predict_df(n=n, future_covariates=future_covariates)

        return self.model.predict(predict_df)

    def generate_predict_df(self,
                            n: int,
                            future_covariates: Optional[TimeSeries] = None) -> pd.DataFrame:
        """Returns a pandas DataFrame in the format required for Prophet.predict() with `n` dates after the end of
        the fitted TimeSeries"""

        predict_df = pd.DataFrame(data={'ds': self._generate_new_dates(n)})
        if future_covariates is not None:
            predict_df = predict_df.merge(future_covariates.pd_dataframe(), left_on='ds', right_index=True, how='left')
        return predict_df

    def stochastic_samples(self,
                           predict_df,
                           n_samples) -> np.ndarray:
        """Returns stochastic forecast of `n_samples` samples.
        This method is a replicate of Prophet.predict() which suspends simplification of stochastic samples to
        deterministic target values."""

        n_samples_default = self.model.uncertainty_samples
        self.model.uncertainty_samples = n_samples

        if self.model.history is None:
            raise Exception('Model has not been fit.')

        if predict_df is None:
            predict_df = self.model.history.copy()
        else:
            if predict_df.shape[0] == 0:
                raise ValueError('Dataframe has no rows.')
            predict_df = self.model.setup_dataframe(predict_df.copy())

        predict_df['trend'] = self.model.predict_trend(predict_df)

        forecast = self.model.sample_posterior_predictive(predict_df)

        self.model.uncertainty_samples = n_samples_default
        return forecast['yhat']

    def add_seasonality(self, name, period, fourier_order, **kwargs) -> None:
        """stores add_seasonality() calls for application in Prophet.fit()"""
        args = (name, period, fourier_order)
        self.add_seasonalities[name] = (args, kwargs)

    def _is_probabilistic(self) -> bool:
        return True

    @staticmethod
    def freq_to_days(freq: str) -> float:
        seconds_per_day = 86400
        if freq == ['A', 'BYS', 'BA', 'RE'] or freq.startswith(('A', 'BYS', 'BA', 'RE-')):  # year
            days = 365.25
        elif freq == ['Q', 'BQ', 'REQ'] or freq.startswith(('Q', 'BQ', 'REQ')):  # quarter
            days = 3 * 30.4375
        elif freq in ['M', 'BM', 'CBM', 'SM'] or freq.startswith(('M', 'BM', 'BS', 'CBM', 'SM')):  # month
            days = 30.4375
        elif freq in 'W' or freq.startswith('W-'):  # week
            days = 7.
        elif freq in ['D', 'B', 'C'] or freq.startswith(('D-', 'B-', 'C-')):  # day
            days = 1.
        elif freq in ['H', 'BH', 'CBH'] or freq.startswith(('H', 'BH', 'CBH')):  # hour
            days = 1/24
        elif freq in ['T', 'min'] or freq.startswith(('H', 'BH', 'CBH')):  # minute
            days = 1 / seconds_per_day
        elif freq == 'S' or freq.startswith('S'):  # second
            days = 1 / (seconds_per_day * 10**3)
        elif freq in ['L', 'ms'] or freq.startswith(('L', 'ms')):  # millisecond
            days = 1 / (seconds_per_day * 10**6)
        elif freq == ['U', 'us'] or freq.startswith(('U', 'us')):  # microsecond
            days = 1 / (seconds_per_day * 10**9)
        elif freq == 'N' or freq.startswith('N'):  # nanosecond
            days = 1 / (seconds_per_day * 10**12)
        else:  # pragma : no cover
            raise ValueError("freq {} not understood. Please report if you think this is in error.".format(freq))
        return days
