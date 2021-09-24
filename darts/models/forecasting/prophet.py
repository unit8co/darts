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
                 frequency: Optional[int] = None,
                 country_holidays: Optional[str] = None,
                 **prophet_kwargs):
        """ Facebook Prophet

        This class provides a basic wrapper around `Facebook Prophet <https://github.com/facebook/prophet>`_.
        It also supports country holidays.

        Parameters
        ----------
        frequency
            Optionally, some frequency, specifying a known seasonality, which will be added to prophet.
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

        self.country_holidays = country_holidays
        self.freq = frequency
        self.prophet_kwargs = prophet_kwargs
        self.model = None

    def __str__(self):
        return 'Prophet'

    def fit(self, series: TimeSeries, past_covariates: Optional[TimeSeries] = None):
        super().fit(series)
        series = self.training_series

        in_df = pd.DataFrame(data={
            'ds': series.time_index,
            'y': series.univariate_values()
        })

        # TODO: user-provided seasonalities, or "auto" based on stepduration
        self.model = prophet.Prophet(**self.prophet_kwargs)
        if self.freq is not None:
            if series.freq_str in ['MS', 'M', 'ME']:
                interval_length = 30.4375
            elif series.freq_str == 'Y':
                interval_length = 365.25
            else:
                interval_length = pd.to_timedelta(series.freq_str).days
            self.model.add_seasonality(name='custom', period=self.freq * interval_length,
                                       fourier_order=5)

        if past_covariates is not None:
            in_df = pd.concat([in_df, pd.DataFrame(past_covariates.all_values()[:, :, 0], columns=['year', 'month'])], axis=1)
            self.model.add_regressor('year')
            self.model.add_regressor('month')

        # Input built-in country holidays
        if self.country_holidays is not None:
            self.model.add_country_holidays(self.country_holidays)

        execute_and_suppress_output(self.model.fit, logger, logging.WARNING, in_df)

    def predict(self,
                n: int,
                num_samples: int = 1,
                future_covariates: Optional[TimeSeries] = None) -> TimeSeries:
        super().predict(n, num_samples)
        new_dates = self._generate_new_dates(n)
        new_dates_df = pd.DataFrame(data={'ds': new_dates})
        if future_covariates:
            new_dates_df = pd.concat([new_dates_df, pd.DataFrame(future_covariates.all_values()[:, :, 0], columns=['year', 'month'])],
                              axis=1)
        predictions = self.model.predict(new_dates_df)
        if num_samples == 1:
            forecast = predictions['yhat'].values
        else:
            forecast = np.expand_dims(self.stochastic_samples(new_dates_df, n_samples=num_samples), axis=1)
        return self._build_forecast_series(forecast)

    def stochastic_samples(self, predictions, n_samples) -> np.ndarray:
        """small hack to get stochastic samples"""
        predictions['t'] = (predictions['ds'] - self.model.start) / self.model.t_scale
        predictions['floor'] = 0
        return self.model.sample_posterior_predictive(predictions)['yhat']

    def _is_probabilistic(self) -> bool:
        return True
