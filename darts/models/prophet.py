"""
Facebook Prophet
----------------
"""

from typing import Optional
import logging

from ..timeseries import TimeSeries
from .forecasting_model import ForecastingModel
import pandas as pd
from ..logging import get_logger, execute_and_suppress_output
import prophet


logger = get_logger(__name__)
logger.level = logging.WARNING  # set to warning to suppress prophet logs


class Prophet(ForecastingModel):
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

    def fit(self, series: TimeSeries):
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

        # Input built-in country holidays
        if self.country_holidays is not None:
            self.model.add_country_holidays(self.country_holidays)

        execute_and_suppress_output(self.model.fit, logger, logging.WARNING, in_df)

    def predict(self,
                n: int,
                num_samples: int = 1) -> TimeSeries:
        super().predict(n, num_samples)
        new_dates = self._generate_new_dates(n)
        new_dates_df = pd.DataFrame(data={'ds': new_dates})

        predictions = self.model.predict(new_dates_df)

        forecast = predictions['yhat'][-n:].values
        return self._build_forecast_series(forecast)
